use crate::{ffi::*, weights::Weights};
use anyhow::{Context, Result, ensure};
use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    path::{Path, PathBuf},
    ptr::null_mut,
    sync::Mutex,
};

// Backend buffers and contexts own all tensor handles. Drop buffers before their contexts.
struct Owned {
    handle: H,
    free: unsafe extern "C" fn(H),
}
impl Owned {
    fn new(handle: H, free: unsafe extern "C" fn(H), name: &str) -> Result<Self> {
        ensure!(!handle.is_null(), "could not allocate {name}");
        Ok(Self { handle, free })
    }
}
impl Drop for Owned {
    fn drop(&mut self) {
        unsafe { (self.free)(self.handle) }
    }
}

pub struct Engine {
    graph: H,
    input: H,
    output: H,
    convolutions: Vec<ConvPlan>,
    _allocator: Owned,
    _buffer: Owned,
    _graph_ctx: Owned,
    _weights_ctx: Owned,
    backend: Owned,
    pub device: String,
}

struct ConvPlan {
    frames: usize,
    input: H,
    output: H,
    graph: H,
    _allocator: Owned,
}

fn allocate_graph(backend: H, graph: H) -> Result<Owned> {
    unsafe {
        let alloc = Owned::new(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)),
            ggml_gallocr_free,
            "graph allocator",
        )?;
        ensure!(
            ggml_gallocr_alloc_graph(alloc.handle, graph),
            "graph buffer allocation failed"
        );
        Ok(alloc)
    }
}

struct Builder<'a> {
    c: H,
    wc: H,
    model: &'a Weights,
    uploads: Vec<(H, Vec<f32>)>,
    native_uploads: Vec<(H, &'a [u8])>,
    f16_conv: bool,
    cpu: bool,
    handles: HashMap<String, H>,
}
impl Builder<'_> {
    fn weight(&mut self, name: &str, dims: &[i64], packed: bool) -> Result<H> {
        if let Some(&handle) = self.handles.get(name) {
            return Ok(handle);
        }
        let n = dims.iter().product::<i64>() as usize;
        if packed && let Some(bytes) = self.model.q8_matrix(name, n)? {
            ensure!(dims[0] % 32 == 0, "unaligned Q8_0 matrix {name}");
            let ty = if self.cpu { 0 } else { 8 };
            let t = unsafe { ggml_new_tensor(self.wc, ty, dims.len() as i32, dims.as_ptr()) };
            self.handles.insert(name.to_string(), t);
            if self.cpu {
                // Expand once at load time. FP32 GEMM is faster for these CPU matrix shapes.
                let values = bytes.chunks_exact(34).flat_map(|block| {
                    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                    block[2..].iter().map(move |v| (*v as i8) as f32 * scale)
                });
                self.uploads.push((t, values.collect()));
            } else {
                self.native_uploads.push((t, bytes));
            }
            return Ok(t);
        }
        let data = if packed {
            self.model.matrix(name, n)?
        } else {
            self.model.values(name, n)?
        };
        let t = unsafe { ggml_new_tensor(self.wc, 0, dims.len() as i32, dims.as_ptr()) };
        self.handles.insert(name.to_string(), t);
        self.uploads.push((t, data));
        Ok(t)
    }
    fn norm(&mut self, x: H, name: &str) -> Result<H> {
        let scale = self.weight(&format!("{name}.weight"), &[384], false)?;
        let bias = self.weight(&format!("{name}.bias"), &[384], false)?;
        unsafe {
            Ok(ggml_add(
                self.c,
                ggml_mul(self.c, ggml_norm(self.c, x, 1e-5), scale),
                bias,
            ))
        }
    }
    fn linear(&mut self, x: H, name: &str, ni: i64, no: i64, bias: bool) -> Result<H> {
        let w = self.weight(name, &[ni, no], true)?;
        let y = unsafe { ggml_mul_mat(self.c, w, x) };
        if !bias {
            return Ok(y);
        }
        let b = self.weight(&format!("{name}.bias"), &[no], false)?;
        Ok(unsafe { ggml_add(self.c, y, b) })
    }
    fn conv(&mut self, x: H, name: &str, channels: i64, tokens: i64, stride: i32) -> Result<H> {
        let key = format!("{name}.weight");
        let dims = [3, channels, 384];
        let w = if let Some(&t) = self.handles.get(&key) {
            t
        } else if self.f16_conv {
            let bytes = self.model.f16_values(&key, (3 * channels * 384) as usize)?;
            let t = unsafe { ggml_new_tensor(self.wc, 1, 3, dims.as_ptr()) };
            self.native_uploads.push((t, bytes));
            self.handles.insert(key, t);
            t
        } else {
            self.weight(&key, &dims, false)?
        };
        let b = self.weight(&format!("{name}.bias"), &[384], false)?;
        let c = self.c;
        unsafe {
            let cols = ggml_im2col(
                c,
                w,
                x,
                stride,
                0,
                1,
                0,
                1,
                0,
                false,
                i32::from(self.f16_conv),
            );
            let y = ggml_mul_mat(
                c,
                ggml_reshape_2d(c, w, 3 * channels, 384),
                ggml_reshape_2d(c, cols, 3 * channels, tokens),
            );
            Ok(ggml_gelu_erf(c, ggml_add(c, y, b)))
        }
    }
    fn mlp_linear(&mut self, x: H, name: &str, ni: i64, no: i64, rank: Option<i64>) -> Result<H> {
        let Some(rank) = rank else {
            return self.linear(x, name, ni, no, true);
        };
        let a = self.linear(x, &format!("{name}.a"), ni, rank, false)?;
        self.linear(a, &format!("{name}.b"), rank, no, true)
    }
    fn mlp(&mut self, x: H, stem: &str, width: i64, rank: Option<i64>) -> Result<H> {
        let a = self.norm(x, &format!("{stem}.mlp_ln"))?;
        let b = self.mlp_linear(a, &format!("{stem}.mlp0"), 384, width, rank)?;
        let b = unsafe { ggml_gelu_erf(self.c, b) };
        let y = self.mlp_linear(b, &format!("{stem}.mlp2"), width, 384, rank)?;
        Ok(unsafe { ggml_add(self.c, x, y) })
    }
    fn encoder_block(&mut self, x: H, stem: &str, tokens: i64) -> Result<H> {
        let a = self.norm(x, &format!("{stem}.attn_ln"))?;
        let c = self.c;
        let heads =
            |t| unsafe { ggml_permute(c, ggml_reshape_3d(c, t, 64, 6, tokens), 0, 2, 1, 3) };
        let q = heads(self.linear(a, &format!("{stem}.query"), 384, 384, true)?);
        let k = heads(self.linear(a, &format!("{stem}.key"), 384, 384, false)?);
        let v = heads(self.linear(a, &format!("{stem}.value"), 384, 384, true)?);
        let mixed = unsafe {
            let att = ggml_flash_attn_ext(c, q, k, v, null_mut(), 0.125, 0., 0.);
            ggml_prec_set_acc(att, 10);
            ggml_reshape_2d(c, att, 384, tokens)
        };
        let y = self.linear(mixed, &format!("{stem}.out"), 384, 384, true)?;
        self.mlp(
            unsafe { ggml_add(c, x, y) },
            stem,
            self.model.encoder_mlp_width,
            None,
        )
    }
    fn decoder_block(&mut self, x: H, audio: H, audio_t: H, index: usize) -> Result<H> {
        let c = self.c;
        let stem = format!("d{index}");
        let rank = self.model.decoder_rank;
        let (x, query) = if index == 0 && rank.is_some() {
            (x, self.weight("d0.fixed_kq", &[384, 6], false)?)
        } else {
            let a = self.norm(x, &format!("{stem}.attn_ln"))?;
            let s = if rank.is_some() {
                self.linear(a, &format!("{stem}.self"), 384, 384, true)?
            } else {
                let v = self.linear(a, &format!("{stem}.value"), 384, 384, true)?;
                self.linear(v, &format!("{stem}.out"), 384, 384, true)?
            };
            let x = unsafe { ggml_add(c, x, s) };
            let a = self.norm(x, &format!("{stem}.cross_ln"))?;
            let q = self.linear(a, &format!("{stem}.cross.query"), 384, 384, true)?;
            let k = self.weight(&format!("{stem}.cross.key"), &[384, 64, 6], true)?;
            let query = unsafe {
                let kt = ggml_cont(c, ggml_permute(c, k, 1, 0, 2, 3));
                ggml_reshape_2d(
                    c,
                    ggml_mul_mat(c, kt, ggml_reshape_3d(c, q, 64, 1, 6)),
                    384,
                    6,
                )
            };
            (x, query)
        };
        let v = self.weight(&format!("{stem}.cross.value"), &[384, 64, 6], true)?;
        let bias = self.weight(&format!("{stem}.cross.value.bias"), &[384], false)?;
        // One decoder query: q·(K·audio) = (Kᵀ·q)·audio. Apply each V head
        // after its weighted sum. This avoids projecting 1500 K/V vectors per layer.
        let mixed = unsafe {
            let probs = ggml_soft_max(c, ggml_scale(c, ggml_mul_mat(c, audio, query), 0.125));
            let pooled = ggml_reshape_3d(c, ggml_mul_mat(c, audio_t, probs), 384, 1, 6);
            ggml_add(
                c,
                ggml_reshape_2d(c, ggml_mul_mat(c, v, pooled), 384, 1),
                bias,
            )
        };
        let out = self.linear(mixed, &format!("{stem}.cross.out"), 384, 384, true)?;
        self.mlp(unsafe { ggml_add(c, x, out) }, &stem, 1536, rank)
    }
}

impl Engine {
    pub fn new(model: &Weights, backend_name: &str, threads: i32, plugins: &Path) -> Result<Self> {
        ensure!(threads > 0, "threads must be positive");
        let subdir = plugins.join(if backend_name == "CPU" {
            "cpu"
        } else {
            "vulkan"
        });
        let dir = if subdir.is_dir() {
            subdir
        } else {
            plugins.to_path_buf()
        }
        .canonicalize()?;
        let path = CString::new(
            dir.to_str()
                .context("non-UTF8 plugin path")?
                .trim_start_matches(r"\\?\"),
        )?;
        static LOADED: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
        let mut loaded = LOADED
            .lock()
            .map_err(|_| anyhow::anyhow!("ggml registry lock poisoned"))?;
        if !loaded.contains(&dir) {
            unsafe { ggml_backend_load_all_from_path(path.as_ptr()) };
            loaded.push(dir);
        }
        let name = CString::new(backend_name)?;
        unsafe {
            let backend = Owned::new(
                ggml_backend_init_by_name(name.as_ptr(), null_mut()),
                ggml_backend_free,
                backend_name,
            )?;
            drop(loaded);
            let b = backend.handle;
            let dev = ggml_backend_get_device(b);
            let device = CStr::from_ptr(ggml_backend_dev_description(dev))
                .to_string_lossy()
                .into_owned();
            if backend_name == "CPU" {
                let f = ggml_backend_reg_get_proc_address(
                    ggml_backend_dev_backend_reg(dev),
                    c"ggml_backend_set_n_threads".as_ptr(),
                );
                ensure!(!f.is_null(), "CPU thread control unavailable");
                std::mem::transmute::<H, unsafe extern "C" fn(H, i32)>(f)(b, threads);
            }
            let ctx = || {
                Owned::new(
                    ggml_init(Params {
                        size: 2 * 1024 * 1024,
                        buffer: null_mut(),
                        no_alloc: true,
                    }),
                    ggml_free,
                    "context",
                )
            };
            let wc = ctx()?;
            let gc = ctx()?;
            let c = gc.handle;
            let cpu = backend_name == "CPU";
            let compact = model.q8_matrix("e0.query", 384 * 384)?.is_some();
            let mut build = Builder {
                handles: HashMap::new(),
                c,
                wc: wc.handle,
                model,
                uploads: Vec::new(),
                native_uploads: Vec::new(),
                f16_conv: compact,
                cpu,
            };
            let mut convolutions = Vec::new();
            let mut conv_graphs = Vec::new();
            let mut build_conv = |frames| -> Result<(H, H)> {
                let mel = ggml_new_tensor(c, 0, 2, [frames, 80].as_ptr());
                ggml_set_input(mel);
                let y = build.conv(mel, "conv1", 80, frames, 1)?;
                let result = build.conv(
                    ggml_cont(c, ggml_transpose(c, y)),
                    "conv2",
                    384,
                    frames / 2,
                    2,
                )?;
                Ok((mel, result))
            };
            let (input, mut x) = if cpu {
                for frames in [512, 1024, 1792, 3000] {
                    let (mel, result) = build_conv(frames)?;
                    ggml_set_output(result);
                    let graph = ggml_new_graph(c);
                    ggml_build_forward_expand(graph, result);
                    conv_graphs.push((frames as usize, mel, result, graph));
                }
                let features = ggml_new_tensor(c, 0, 2, [384i64, 1500].as_ptr());
                ggml_set_input(features);
                (features, features)
            } else {
                build_conv(3000)?
            };
            x = ggml_add(c, x, build.weight("position", &[384, 1500], false)?);
            let mut tokens = 1500;
            for i in 0..4 {
                x = build.encoder_block(x, &format!("e{i}"), tokens)?;
                if model.pool_after == Some(i + 1) {
                    tokens /= 2;
                    // The two views select alternating time steps of the [384, time] tensor.
                    let frame_bytes = 384 * size_of::<f32>();
                    let even = ggml_view_2d(c, x, 384, tokens, 2 * frame_bytes, 0);
                    let odd = ggml_view_2d(c, x, 384, tokens, 2 * frame_bytes, frame_bytes);
                    x = ggml_scale(c, ggml_add(c, even, odd), 0.5);
                }
            }
            let audio = build.norm(x, "enc_ln")?;
            let audio_t = ggml_cont(c, ggml_transpose(c, audio));
            let initial = if model.decoder_rank.is_some() {
                "d0.fixed_x"
            } else {
                "sot"
            };
            x = build.weight(initial, &[384, 1], false)?;
            for i in 0..4 {
                x = build.decoder_block(x, audio, audio_t, i)?;
            }
            x = build.norm(x, "dec_ln")?;
            let output = build.linear(x, "head", 384, 99, false)?;
            ggml_set_output(output);
            let graph = ggml_new_graph(c);
            ggml_build_forward_expand(graph, output);
            for i in 0..ggml_graph_n_nodes(graph) {
                let node = ggml_graph_node(graph, i);
                ensure!(
                    ggml_backend_dev_supports_op(dev, node),
                    "{} unsupported on {device}",
                    CStr::from_ptr(ggml_op_desc(node)).to_string_lossy()
                );
            }
            let buffer = Owned::new(
                ggml_backend_alloc_ctx_tensors(wc.handle, b),
                ggml_backend_buffer_free,
                "weights",
            )?;
            for (t, data) in build.uploads {
                ggml_backend_tensor_set(t, data.as_ptr().cast(), 0, data.len() * 4);
            }
            for (t, bytes) in build.native_uploads {
                ggml_backend_tensor_set(t, bytes.as_ptr().cast(), 0, bytes.len());
            }
            for (frames, input, output, graph) in conv_graphs {
                convolutions.push(ConvPlan {
                    frames,
                    input,
                    output,
                    graph,
                    _allocator: allocate_graph(b, graph)?,
                });
            }
            let alloc = allocate_graph(b, graph)?;
            Ok(Self {
                convolutions,
                graph,
                input,
                output,
                _allocator: alloc,
                _buffer: buffer,
                _graph_ctx: gc,
                _weights_ctx: wc,
                backend,
                device,
            })
        }
    }
    pub fn logits(&mut self, mel: &[f32], active_frames: usize) -> Result<Vec<f32>> {
        ensure!(mel.len() == 80 * 3000, "expected 80 × 3000 Mel input");
        let mut out = vec![0f32; 99];
        unsafe {
            let features;
            let input = if self.convolutions.is_empty() {
                mel
            } else {
                let plan = self
                    .convolutions
                    .iter()
                    .find(|p| p.frames >= active_frames + 8 || p.frames == 3000)
                    .unwrap();
                let prefix: Vec<_> = mel
                    .chunks_exact(3000)
                    .flat_map(|band| band[..plan.frames].iter().copied())
                    .collect();
                ggml_backend_tensor_set(plan.input, prefix.as_ptr().cast(), 0, prefix.len() * 4);
                ensure!(
                    ggml_backend_graph_compute(self.backend.handle, plan.graph) == 0,
                    "CPU convolution failed"
                );
                let mut full = vec![0f32; 1500 * 384];
                let rows = plan.frames / 2;
                ggml_backend_tensor_get(plan.output, full.as_mut_ptr().cast(), 0, rows * 384 * 4);
                if rows < 1500 {
                    // Every interior receptive field beyond the audio is identical.
                    // Preserve the distinct last row, whose field crosses the true right edge.
                    full.copy_within((rows - 1) * 384..rows * 384, 1499 * 384);
                    for i in rows - 1..1499 {
                        full.copy_within((rows - 2) * 384..(rows - 1) * 384, i * 384);
                    }
                }
                features = full;
                &features
            };
            ggml_backend_tensor_set(
                self.input,
                input.as_ptr().cast(),
                0,
                std::mem::size_of_val(input),
            );
            ensure!(
                ggml_backend_graph_compute(self.backend.handle, self.graph) == 0,
                "backend computation failed"
            );
            ggml_backend_tensor_get(self.output, out.as_mut_ptr().cast(), 0, out.len() * 4);
        }
        ensure!(
            out.iter().all(|x| x.is_finite()),
            "non-finite language scores"
        );
        Ok(out)
    }
}
