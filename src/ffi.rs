//! C ABI of ggml from llama.cpp b12413. Tensor handles stay inside Engine.
#![allow(dead_code)]
use std::ffi::{c_char, c_int, c_void};
pub type H = *mut c_void;
#[repr(C)]
pub struct Params {
    pub size: usize,
    pub buffer: H,
    pub no_alloc: bool,
}

#[cfg_attr(windows, link(name = "ggml-base", kind = "raw-dylib"))]
#[cfg_attr(not(windows), link(name = "ggml-base"))]
unsafe extern "C" {
    pub fn ggml_init(p: Params) -> H;
    pub fn ggml_free(c: H);
    pub fn ggml_new_tensor(c: H, ty: c_int, nd: c_int, dims: *const i64) -> H;
    pub fn ggml_new_graph(c: H) -> H;
    pub fn ggml_set_input(t: H);
    pub fn ggml_set_output(t: H);
    pub fn ggml_build_forward_expand(graph: H, t: H);
    pub fn ggml_graph_n_nodes(graph: H) -> c_int;
    pub fn ggml_graph_node(graph: H, i: c_int) -> H;
    pub fn ggml_op_desc(t: H) -> *const c_char;
    pub fn ggml_add(c: H, a: H, b: H) -> H;
    pub fn ggml_mul(c: H, a: H, b: H) -> H;
    pub fn ggml_mul_mat(c: H, a: H, b: H) -> H;
    pub fn ggml_gelu_erf(c: H, a: H) -> H;
    pub fn ggml_norm(c: H, a: H, eps: f32) -> H;
    pub fn ggml_soft_max(c: H, a: H) -> H;
    pub fn ggml_scale(c: H, a: H, s: f32) -> H;
    pub fn ggml_reshape_2d(c: H, a: H, n0: i64, n1: i64) -> H;
    pub fn ggml_reshape_3d(c: H, a: H, n0: i64, n1: i64, n2: i64) -> H;
    pub fn ggml_transpose(c: H, a: H) -> H;
    pub fn ggml_cont(c: H, a: H) -> H;
    pub fn ggml_permute(c: H, a: H, x: c_int, y: c_int, z: c_int, w: c_int) -> H;
    pub fn ggml_im2col(
        c: H,
        kernel: H,
        a: H,
        sx: c_int,
        sy: c_int,
        px: c_int,
        py: c_int,
        dx: c_int,
        dy: c_int,
        is_2d: bool,
        ty: c_int,
    ) -> H;
    pub fn ggml_flash_attn_ext(
        c: H,
        q: H,
        k: H,
        v: H,
        mask: H,
        scale: f32,
        bias: f32,
        cap: f32,
    ) -> H;
    pub fn ggml_prec_set_acc(t: H, precision: c_int) -> bool;
    pub fn ggml_backend_alloc_ctx_tensors(c: H, backend: H) -> H;
    pub fn ggml_backend_buffer_free(b: H);
    pub fn ggml_backend_tensor_set(t: H, p: *const c_void, offset: usize, size: usize);
    pub fn ggml_backend_tensor_get(t: H, p: H, offset: usize, size: usize);
    pub fn ggml_backend_get_default_buffer_type(b: H) -> H;
    pub fn ggml_backend_get_device(b: H) -> H;
    pub fn ggml_backend_dev_description(d: H) -> *const c_char;
    pub fn ggml_backend_dev_backend_reg(d: H) -> H;
    pub fn ggml_backend_dev_supports_op(d: H, op: H) -> bool;
    pub fn ggml_backend_reg_get_proc_address(r: H, name: *const c_char) -> H;
    pub fn ggml_backend_graph_compute(b: H, graph: H) -> c_int;
    pub fn ggml_backend_free(b: H);
    pub fn ggml_gallocr_new(buffer_type: H) -> H;
    pub fn ggml_gallocr_alloc_graph(allocator: H, graph: H) -> bool;
    pub fn ggml_gallocr_free(allocator: H);
}
#[cfg_attr(windows, link(name = "ggml", kind = "raw-dylib"))]
#[cfg_attr(not(windows), link(name = "ggml"))]
unsafe extern "C" {
    pub fn ggml_backend_load_all_from_path(dir: *const c_char);
    pub fn ggml_backend_init_by_name(name: *const c_char, params: *const c_char) -> H;
}
