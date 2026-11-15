#[cfg(feature = "audio-files")]
pub mod audio;
mod engine;
mod ffi;
mod weights;
use anyhow::{Result, ensure};
use realfft::{RealFftPlanner, RealToComplex};
use std::{path::Path, sync::Arc};

pub struct Detector {
    engine: engine::Engine,
    labels: Vec<String>,
    window: Vec<f32>,
    bands: Vec<Vec<(usize, f32)>>,
    fft: Arc<dyn RealToComplex<f32>>,
}

#[derive(Clone, Copy)]
pub enum Backend {
    Cpu,
    Vulkan,
}

impl Detector {
    /// Load once and reuse the detector for successive utterances. Plugins are the pinned ggml shared libraries.
    pub fn load(model: &Path, backend: Backend, threads: i32, plugin_dir: &Path) -> Result<Self> {
        let weights = weights::Weights::open(model)?;
        let window = weights.values("hann", 400)?;
        let bands = weights
            .values("mel", 80 * 201)?
            .chunks_exact(201)
            .map(|row| {
                row.iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, x)| *x != 0.)
                    .collect()
            })
            .collect();
        let engine = engine::Engine::new(
            &weights,
            match backend {
                Backend::Cpu => "CPU",
                Backend::Vulkan => "Vulkan0",
            },
            threads,
            plugin_dir,
        )?;
        Ok(Self {
            engine,
            labels: weights.labels,
            window,
            bands,
            fft: RealFftPlanner::new().plan_fft_forward(400),
        })
    }
    pub fn device(&self) -> &str {
        &self.engine.device
    }
    pub fn languages(&self) -> &[String] {
        &self.labels
    }
    /// Mono 16 kHz floating-point PCM. Output order matches languages(); empty or digital silence returns None.
    pub fn predict(&mut self, samples: &[f32]) -> Result<Option<Vec<f32>>> {
        let pcm = &samples[..samples.len().min(480000)];
        ensure!(pcm.iter().all(|s| s.is_finite()), "non-finite PCM");
        if !pcm.iter().any(|&x| x != 0.) {
            return Ok(None);
        }
        let mut mel = vec![-10f32; 80 * 3000];
        let mut input = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        let mut powers = [0f32; 201];
        let mut max = -10f32;
        // Frames wholly inside the right zero padding have known log energy.
        let active_frames = (pcm.len() + 200).div_ceil(160).min(3000);
        for t in 0..active_frames {
            for (j, value) in input.iter_mut().enumerate() {
                let idx = t as isize * 160 + j as isize - 200;
                let idx = if idx < 0 {
                    -idx
                } else if idx >= 480000 {
                    959998 - idx
                } else {
                    idx
                };
                *value = pcm.get(idx as usize).copied().unwrap_or(0.) * self.window[j];
            }
            self.fft
                .process_with_scratch(&mut input, &mut spectrum, &mut scratch)?;
            for (p, z) in powers.iter_mut().zip(&spectrum) {
                *p = z.norm_sqr();
            }
            for (b, band) in self.bands.iter().enumerate() {
                let energy = band
                    .iter()
                    .map(|&(i, w)| powers[i] * w)
                    .sum::<f32>()
                    .max(1e-10)
                    .log10();
                max = max.max(energy);
                mel[b * 3000 + t] = energy;
            }
        }
        for v in &mut mel {
            *v = (v.max(max - 8.) + 4.) * 0.25;
        }
        self.engine.logits(&mel, active_frames).map(Some)
    }
}
