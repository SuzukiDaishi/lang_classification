use anyhow::{Result, bail, ensure};
use std::{f64::consts::PI, fs::File, io::ErrorKind, path::Path};
use symphonia::core::{
    audio::SampleBuffer, codecs::DecoderOptions, errors::Error, formats::FormatOptions,
    io::MediaSourceStream, meta::MetadataOptions, probe::Hint,
};

/// Decode at most 30 seconds and downmix channels with equal weights.
pub fn read(path: &Path) -> Result<Vec<f32>> {
    let stream = MediaSourceStream::new(Box::new(File::open(path)?), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut fmt = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions {
                enable_gapless: true,
                ..Default::default()
            },
            &MetadataOptions::default(),
        )?
        .format;
    let track = fmt
        .default_track()
        .ok_or_else(|| anyhow::anyhow!("missing audio track"))?;
    let id = track.id;
    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let mut pcm = Vec::new();
    let mut rate = track.codec_params.sample_rate.unwrap_or(0);
    loop {
        let packet = match fmt.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(e)) if e.kind() == ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != id {
            continue;
        }
        let decoded = decoder.decode(&packet)?;
        let spec = *decoded.spec();
        if rate == 0 {
            rate = spec.rate;
        }
        ensure!(
            spec.rate == rate && rate > 0,
            "sample rate changes within stream"
        );
        let channels = spec.channels.count();
        ensure!(channels > 0, "empty channel layout");
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buf.copy_interleaved_ref(decoded);
        pcm.extend(
            buf.samples()
                .chunks_exact(channels)
                .map(|s| s.iter().sum::<f32>() / channels as f32)
                .take(30 * rate as usize - pcm.len()),
        );
        if pcm.len() == 30 * rate as usize {
            break;
        }
    }
    if rate == 0 && !pcm.is_empty() {
        bail!("unknown sample rate");
    }
    Ok(if rate == 16000 || pcm.is_empty() {
        pcm
    } else {
        resample(&pcm, rate)
    })
}

fn resample(input: &[f32], rate: u32) -> Vec<f32> {
    // Band-limited interpolation with a Blackman window. Cache each fractional phase.
    let cutoff = 0.95 * (16000. / rate as f64).min(1.);
    let radius = (32. / cutoff).ceil() as i64;
    let mut phases = std::collections::HashMap::new();
    let mut padded = vec![0.; radius as usize - 1];
    padded.extend_from_slice(input);
    padded.resize(padded.len() + radius as usize + 1, 0.);
    #[cfg(target_arch = "x86_64")]
    let simd = is_x86_feature_detected!("avx");
    #[cfg(not(target_arch = "x86_64"))]
    let simd = false;
    let n = (input.len() as u64 * 16000 + rate as u64 / 2) / rate as u64;
    (0..n)
        .map(|i| {
            let pos = i * rate as u64;
            let center = (pos / 16000) as i64;
            let weights = phases.entry(pos % 16000).or_insert_with(|| {
                let frac = (pos % 16000) as f64 / 16000.;
                let mut w: Vec<f32> = (-radius + 1..=radius)
                    .map(|j| {
                        let x = j as f64 - frac;
                        let a = PI * x / radius as f64;
                        let sinc = if x.abs() < 1e-12 {
                            cutoff
                        } else {
                            (PI * x * cutoff).sin() / (PI * x)
                        };
                        (sinc * (0.42 + 0.5 * a.cos() + 0.08 * (2. * a).cos())) as f32
                    })
                    .collect();
                let sum = w.iter().sum::<f32>();
                for x in &mut w {
                    *x /= sum;
                }
                w
            });
            let from = center as usize;
            dot(&padded[from..from + weights.len()], weights, simd)
        })
        .collect()
}

#[inline]
fn dot(a: &[f32], b: &[f32], simd: bool) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if simd {
        // AVX is checked once per audio file and the two slices have equal length.
        return unsafe { dot_avx(a, b) };
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn dot_avx(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= a.len() {
            let x = _mm256_loadu_ps(a.as_ptr().add(i));
            let y = _mm256_loadu_ps(b.as_ptr().add(i));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(x, y));
            i += 8;
        }
        let mut lanes = [0.; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        lanes.iter().sum::<f32>() + a[i..].iter().zip(&b[i..]).map(|(x, y)| x * y).sum::<f32>()
    }
}
