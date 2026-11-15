use anyhow::{Context, Result, bail};
use serde_json::json;
use speech_lid::{Backend, Detector, audio};
use std::{
    env,
    io::{self, BufRead, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

fn result(det: &mut Detector, pcm: &[f32]) -> Result<serde_json::Value> {
    let start = Instant::now();
    let Some(logits) = det.predict(pcm)? else {
        return Ok(json!({"status":"no_speech"}));
    };
    let ms = start.elapsed().as_secs_f64() * 1000.;
    let mut indices: Vec<_> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let best = indices[0];
    let max = logits[best];
    let sum = logits.iter().map(|x| (x - max).exp()).sum::<f32>();
    let ranked: Vec<_> = indices[..2]
        .iter()
        .map(|&i| {
            json!({"code":det.languages()[i],"index":i,
                                  "logit":logits[i],"probability":(logits[i]-max).exp()/sum})
        })
        .collect();
    Ok(
        json!({"status":"language","language":det.languages()[best],"ranked":ranked,
              "margin":(1.-(logits[indices[1]]-max).exp())/sum,"pcm_to_result_ms":ms,"device":det.device()}),
    )
}

fn main() -> Result<()> {
    let exe = env::current_exe()?;
    let dir = exe.parent().context("missing executable directory")?;
    let mut model = dir.join("lid10.gguf");
    let mut plugins = dir.to_path_buf();
    let mut backend = Backend::Cpu;
    let mut threads = std::thread::available_parallelism()?.get().min(8) as i32;
    let mut paths = Vec::new();
    let mut pcm_stream = false;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = PathBuf::from(args.next().context("missing --model value")?),
            "--plugins" => plugins = PathBuf::from(args.next().context("missing --plugins value")?),
            "--threads" => threads = args.next().context("missing thread count")?.parse()?,
            "--backend" => {
                backend = match args.next().as_deref() {
                    Some("cpu") => Backend::Cpu,
                    Some("vulkan") => Backend::Vulkan,
                    _ => bail!("choose cpu or vulkan"),
                }
            }
            "--pcm-stream" => pcm_stream = true,
            "--files-from" => {
                let list = std::fs::read_to_string(args.next().context("missing file list")?)?;
                paths.extend(list.lines().filter(|s| !s.is_empty()).map(str::to_owned));
            }
            "--help" | "-h" => {
                println!(
                    "speech-lid [--model lid10.gguf] [--backend cpu|vulkan] [--threads N] FILE...\n--files-from PATH reads one file path per line. --plugins DIR selects the backend directory.\nCPU uses up to 8 threads by default. --pcm-stream accepts repeated [u32 sample count, little-endian f32 mono 16 kHz PCM] on stdin."
                );
                return Ok(());
            }
            _ if arg.starts_with("--") => bail!("unknown option: {arg}"),
            _ => paths.push(arg),
        }
    }
    anyhow::ensure!(
        pcm_stream == paths.is_empty(),
        "supply audio files or --pcm-stream"
    );
    let mut det = Detector::load(&model, backend, threads, &plugins)?;
    if pcm_stream {
        let mut input = io::stdin().lock();
        let mut out = io::stdout().lock();
        while !input.fill_buf()?.is_empty() {
            let mut h = [0; 4];
            input.read_exact(&mut h)?;
            let n = u32::from_le_bytes(h) as usize;
            anyhow::ensure!(n <= 480000, "frame exceeds 30 seconds");
            let mut b = vec![0; 4 * n];
            input.read_exact(&mut b)?;
            let pcm: Vec<_> = b
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect();
            writeln!(out, "{}", result(&mut det, &pcm)?)?;
            out.flush()?;
        }
    } else {
        for path in paths {
            let start = Instant::now();
            let pcm = audio::read(Path::new(&path)).with_context(|| path.clone())?;
            let mut value = result(&mut det, &pcm)?;
            value["file"] = json!(path);
            value["file_to_result_ms"] = json!(start.elapsed().as_secs_f64() * 1000.);
            println!("{value}");
        }
    }
    Ok(())
}
