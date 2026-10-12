//! Audio preprocessing: log-mel spectrograms in the style of the Universal Speech Model
//! frontend.

use std::f64::consts::PI;

use candle::{Device, Result, Tensor};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use serde::Deserialize;

fn default_feature_size() -> usize {
    128
}
fn default_sampling_rate() -> usize {
    16_000
}
fn default_frame_length() -> usize {
    320
}
fn default_hop_length() -> usize {
    160
}
fn default_fft_length() -> usize {
    512
}
fn default_max_frequency() -> f64 {
    8000.
}
fn default_mel_floor() -> f64 {
    1e-3
}
fn default_input_scale_factor() -> f64 {
    1.
}

/// Mirrors `preprocessor_config.json`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AudioFeatureExtractor {
    /// Number of mel bins.
    #[serde(default = "default_feature_size")]
    pub feature_size: usize,
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: usize,
    /// Analysis window, in samples.
    #[serde(default = "default_frame_length")]
    pub frame_length: usize,
    #[serde(default = "default_hop_length")]
    pub hop_length: usize,
    #[serde(default = "default_fft_length")]
    pub fft_length: usize,
    #[serde(default)]
    pub min_frequency: f64,
    #[serde(default = "default_max_frequency")]
    pub max_frequency: f64,
    #[serde(default = "default_mel_floor")]
    pub mel_floor: f64,
    #[serde(default)]
    pub preemphasis: f64,
    #[serde(default = "default_input_scale_factor")]
    pub input_scale_factor: f64,
}

impl Default for AudioFeatureExtractor {
    fn default() -> Self {
        Self {
            feature_size: default_feature_size(),
            sampling_rate: default_sampling_rate(),
            frame_length: default_frame_length(),
            hop_length: default_hop_length(),
            fft_length: default_fft_length(),
            min_frequency: 0.,
            max_frequency: default_max_frequency(),
            mel_floor: default_mel_floor(),
            preemphasis: 0.,
            input_scale_factor: default_input_scale_factor(),
        }
    }
}

/// Longest clip the reference processes, 30 seconds at 16 kHz. Longer ones are truncated.
pub const MAX_SAMPLES: usize = 480_000;
/// Waveforms are zero padded to a multiple of this many samples.
const PAD_TO_MULTIPLE_OF: usize = 128;

/// The log-mel spectrogram of a clip.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioFeatures {
    /// Row-major `(num_frames, feature_size)`.
    pub features: Vec<f32>,
    pub num_frames: usize,
    /// Frames whose analysis window lies entirely within the clip. They form a prefix; the
    /// rest only exist because of the padding and are zeroed.
    pub num_valid_frames: usize,
    /// Number of soft tokens the audio tower turns the valid frames into.
    pub num_soft_tokens: usize,
}

fn hertz_to_mel(hertz: f64) -> f64 {
    2595. * (1. + hertz / 700.).log10()
}

fn mel_to_hertz(mel: f64) -> f64 {
    700. * (10f64.powf(mel / 2595.) - 1.)
}

impl AudioFeatureExtractor {
    /// Triangular filters, evenly spaced on the HTK mel scale, without area normalization:
    /// `num_frequency_bins` rows of `feature_size` weights.
    fn mel_filters(&self) -> Vec<Vec<f64>> {
        let num_frequency_bins = self.fft_length / 2 + 1;
        let (mel_min, mel_max) = (
            hertz_to_mel(self.min_frequency),
            hertz_to_mel(self.max_frequency),
        );
        let filter_freqs: Vec<f64> = (0..self.feature_size + 2)
            .map(|i| {
                mel_to_hertz(
                    mel_min + (mel_max - mel_min) * i as f64 / (self.feature_size + 1) as f64,
                )
            })
            .collect();
        let nyquist = (self.sampling_rate / 2) as f64;
        (0..num_frequency_bins)
            .map(|bin| {
                let fft_freq = nyquist * bin as f64 / (num_frequency_bins - 1) as f64;
                (0..self.feature_size)
                    .map(|filter| {
                        let (lower, center, upper) = (
                            filter_freqs[filter],
                            filter_freqs[filter + 1],
                            filter_freqs[filter + 2],
                        );
                        let rising = (fft_freq - lower) / (center - lower);
                        let falling = (upper - fft_freq) / (upper - center);
                        rising.min(falling).max(0.)
                    })
                    .collect()
            })
            .collect()
    }

    /// Number of frames whose analysis window lies entirely within a clip of `num_samples`
    /// samples.
    fn num_valid_frames(&self, num_samples: usize) -> usize {
        // Frames start every `hop_length` samples of the clip padded by half a frame, and
        // span `frame_length + 1` samples.
        (self.frame_length / 2 + num_samples)
            .checked_sub(self.frame_length + 1)
            .map_or(0, |last_start| last_start / self.hop_length + 1)
    }

    /// Number of soft tokens the audio tower turns a clip of `num_samples` samples into:
    /// each of its two stride 2 convolutions keeps every other frame, starting from the
    /// first.
    pub fn num_soft_tokens(&self, num_samples: usize) -> usize {
        self.num_valid_frames(num_samples.min(MAX_SAMPLES))
            .div_ceil(2)
            .div_ceil(2)
    }

    /// Computes the features of a mono waveform sampled at `sampling_rate`, with samples in
    /// `[-1, 1]`. `padded_len` is the length the clip is zero padded to, for batching with
    /// longer clips; pass `None` for a clip processed on its own.
    pub fn extract(&self, samples: &[f32], padded_len: Option<usize>) -> Result<AudioFeatures> {
        let samples = &samples[..samples.len().min(MAX_SAMPLES)];
        if samples.is_empty() {
            candle::bail!("cannot process an empty audio clip")
        }
        let padded_len = padded_len
            .unwrap_or(samples.len())
            .clamp(samples.len(), MAX_SAMPLES.max(samples.len()))
            .next_multiple_of(PAD_TO_MULTIPLE_OF);

        // Semicausal padding: half a frame of silence up front centers the first frame at t=0.
        let pad_left = self.frame_length / 2;
        let mut waveform = vec![0f64; pad_left + padded_len];
        for (dst, &src) in waveform[pad_left..].iter_mut().zip(samples) {
            *dst = f64::from(src) * self.input_scale_factor;
        }

        // Frames span one more sample than the window, the extra one feeding the preemphasis.
        let frame_size = self.frame_length + 1;
        if waveform.len() < frame_size {
            candle::bail!("the audio clip is too short: {} samples", samples.len())
        }
        let num_frames = (waveform.len() - frame_size) / self.hop_length + 1;
        // A frame is valid when the last sample of its span is real audio.
        let num_valid_frames = self.num_valid_frames(samples.len());

        // Periodic Hann window.
        let window: Vec<f64> = (0..self.frame_length)
            .map(|n| 0.5 - 0.5 * (2. * PI * n as f64 / self.frame_length as f64).cos())
            .collect();
        let mel_filters = self.mel_filters();
        let fft = FftPlanner::<f64>::new().plan_fft_forward(self.fft_length);
        let mut buffer = vec![Complex::new(0f64, 0f64); self.fft_length];
        let mut features = vec![0f32; num_frames * self.feature_size];

        for frame in 0..num_valid_frames {
            let span = &waveform[frame * self.hop_length..][..frame_size];
            buffer.fill(Complex::new(0., 0.));
            for n in 0..self.frame_length {
                let sample = if self.preemphasis > 0. {
                    // HTK flavored preemphasis.
                    if n == 0 {
                        span[0] * (1. - self.preemphasis)
                    } else {
                        span[n] - self.preemphasis * span[n - 1]
                    }
                } else {
                    span[n]
                };
                buffer[n].re = sample * window[n];
            }
            fft.process(&mut buffer);
            let row = &mut features[frame * self.feature_size..][..self.feature_size];
            let mut mel = vec![0f64; self.feature_size];
            for (bin, filters) in mel_filters.iter().enumerate() {
                let magnitude = buffer[bin].norm();
                for (acc, weight) in mel.iter_mut().zip(filters) {
                    *acc += magnitude * weight;
                }
            }
            for (dst, energy) in row.iter_mut().zip(mel) {
                *dst = (energy + self.mel_floor).ln() as f32;
            }
        }
        Ok(AudioFeatures {
            features,
            num_frames,
            num_valid_frames,
            num_soft_tokens: self.num_soft_tokens(samples.len()),
        })
    }

    /// Stacks clips into the inputs of [`crate::audio::AudioTower::forward`]:
    /// `(batch, num_frames, feature_size)` features and a `(batch, num_frames)` mask. The
    /// clips must have been extracted with a common `padded_len`.
    pub fn collate(&self, clips: &[&AudioFeatures], device: &Device) -> Result<(Tensor, Tensor)> {
        let num_frames = clips.first().map_or(0, |clip| clip.num_frames);
        if clips.iter().any(|clip| clip.num_frames != num_frames) {
            candle::bail!("audio clips of a batch must be padded to a common length")
        }
        let features: Vec<f32> = clips
            .iter()
            .flat_map(|clip| clip.features.iter().copied())
            .collect();
        let mask: Vec<u8> = clips
            .iter()
            .flat_map(|clip| (0..num_frames).map(|frame| u8::from(frame < clip.num_valid_frames)))
            .collect();
        Ok((
            Tensor::from_vec(
                features,
                (clips.len(), num_frames, self.feature_size),
                device,
            )?,
            Tensor::from_vec(mask, (clips.len(), num_frames), device)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_counts_follow_the_clip_length() {
        let extractor = AudioFeatureExtractor::default();
        // 3.7 seconds: 59200 samples, padded to 59264. The window of the last frame reaches
        // into the padding.
        let features = extractor.extract(&vec![0.1; 59_200], None).unwrap();
        assert_eq!(features.num_frames, 370);
        assert_eq!(features.num_valid_frames, 369);
        assert_eq!(features.num_soft_tokens, 93);
        assert_eq!(extractor.num_soft_tokens(59_200), 93);
        // Padded to the length of a longer clip, the extra frames are invalid.
        let features = extractor.extract(&vec![0.1; 30_880], Some(59_200)).unwrap();
        assert_eq!(features.num_frames, 370);
        assert_eq!(features.num_valid_frames, 192);
        assert_eq!(features.num_soft_tokens, 48);
        assert!(features.features[192 * 128..]
            .iter()
            .all(|&value| value == 0.));
        assert_eq!(extractor.num_soft_tokens(100), 0);
    }
}
