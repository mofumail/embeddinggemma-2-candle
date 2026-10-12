//! The Gemma 4 audio tower, a conformer in the style of the Universal Speech Model: a
//! convolutional subsampling of the mel spectrogram followed by blocks of chunked local
//! attention with relative position biases, a causal depthwise convolution and two
//! feed-forward modules.

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, Activation, Conv2d, Conv2dConfig, Linear, VarBuilder};

use crate::config::AudioConfig;
use crate::layers::{ClippableLinear, RmsNorm};

/// The audio tower normalizes with the default epsilon, whatever `rms_norm_eps` says.
const RMS_NORM_EPS: f64 = 1e-6;

/// Clamps the activations to `+-limit`, a guard against overflows inherited from the
/// reference implementation.
fn clip(xs: &Tensor, limit: f64) -> Result<Tensor> {
    let limit = limit.min(f64::from(f32::MAX)) as f32;
    xs.clamp(-limit, limit)
}

/// Layer normalization over the last dimension, with a scale but no bias.
#[derive(Debug, Clone)]
struct ScaleOnlyLayerNorm {
    weight: Tensor,
    eps: f64,
}

impl Module for ScaleOnlyLayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?;
        let xs = xs.broadcast_sub(&xs.mean_keepdim(D::Minus1)?)?;
        let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
        xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?
            .broadcast_mul(&self.weight)?
            .to_dtype(dtype)
    }
}

#[derive(Debug, Clone)]
struct SubSampleConvLayer {
    conv: Conv2d,
    norm: ScaleOnlyLayerNorm,
}

impl SubSampleConvLayer {
    fn new(in_channels: usize, out_channels: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let conv = candle_nn::conv2d_no_bias(in_channels, out_channels, 3, cfg, vb.pp("conv"))?;
        let weight = vb
            .pp("norm")
            .get(out_channels, "weight")?
            .to_dtype(DType::F32)?;
        Ok(Self {
            conv,
            norm: ScaleOnlyLayerNorm { weight, eps },
        })
    }

    /// `xs` is `(batch, channels, time, mel)`, `mask` is `(batch, time)` with 1 for the valid
    /// frames. Both are halved along time.
    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, t) = mask.dims2()?;
        let xs = xs.broadcast_mul(&mask.reshape((b, 1, t, 1))?.to_dtype(xs.dtype())?)?;
        let xs = xs
            .apply(&self.conv)?
            .permute((0, 2, 3, 1))?
            .apply(&self.norm)?
            .permute((0, 3, 1, 2))?
            .relu()?;
        let kept: Vec<u32> = (0..t as u32).step_by(2).collect();
        let kept = Tensor::from_vec(kept, t.div_ceil(2), mask.device())?;
        Ok((xs, mask.index_select(&kept, 1)?))
    }
}

/// Downsamples the spectrogram 4x in time and in frequency, then flattens the frequency
/// axis into the model dimension.
#[derive(Debug, Clone)]
struct SubSampleConvProjection {
    layer0: SubSampleConvLayer,
    layer1: SubSampleConvLayer,
    input_proj_linear: Linear,
}

impl SubSampleConvProjection {
    fn new(cfg: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let &[channels0, channels1] = cfg.subsampling_conv_channels.as_slice() else {
            candle::bail!(
                "expected two subsampling convolutions, got {:?}",
                cfg.subsampling_conv_channels
            )
        };
        // The 128 mel bins are downsampled 4x, like the time axis.
        let proj_input_dim = (channels0 / 4) * channels1;
        Ok(Self {
            layer0: SubSampleConvLayer::new(1, channels0, cfg.rms_norm_eps, vb.pp("layer0"))?,
            layer1: SubSampleConvLayer::new(
                channels0,
                channels1,
                cfg.rms_norm_eps,
                vb.pp("layer1"),
            )?,
            input_proj_linear: linear_no_bias(
                proj_input_dim,
                cfg.hidden_size,
                vb.pp("input_proj_linear"),
            )?,
        })
    }

    fn forward(&self, input_features: &Tensor, mask: &Tensor) -> Result<(Tensor, Tensor)> {
        let xs = input_features.unsqueeze(1)?;
        let (xs, mask) = self.layer0.forward(&xs, mask)?;
        let (xs, mask) = self.layer1.forward(&xs, &mask)?;
        let (b, _, t, _) = xs.dims4()?;
        let xs = xs.permute((0, 2, 3, 1))?.reshape((b, t, ()))?;
        Ok((xs.apply(&self.input_proj_linear)?, mask))
    }
}

/// Geometry of the chunked local attention: queries are grouped in blocks of `chunk_size`,
/// each block attending to a window of `context_size` keys around itself.
#[derive(Debug, Clone, Copy)]
struct Chunking {
    chunk_size: usize,
    max_past_horizon: usize,
    max_future_horizon: usize,
}

impl Chunking {
    fn new(cfg: &AudioConfig) -> Self {
        Self {
            chunk_size: cfg.attention_chunk_size,
            max_past_horizon: cfg.attention_context_left.saturating_sub(1),
            max_future_horizon: cfg.attention_context_right,
        }
    }

    fn context_size(&self) -> usize {
        self.chunk_size + self.max_past_horizon + self.max_future_horizon
    }

    fn num_blocks(&self, seq_len: usize) -> usize {
        seq_len.div_ceil(self.chunk_size)
    }

    /// Position in the sequence of the key at slot `slot` of the context of block `block`,
    /// `None` for the slots that fall before the start of the sequence.
    fn key_position(&self, block: usize, slot: usize) -> Option<usize> {
        (block * self.chunk_size + slot).checked_sub(self.max_past_horizon)
    }

    /// The `(batch, 1, num_blocks, chunk_size, context_size)` mask of the scores to discard:
    /// keys that are padding, or outside of the query's own left/right context.
    fn blocked_mask(&self, valid: &[Vec<bool>], seq_len: usize, device: &Device) -> Result<Tensor> {
        let (num_blocks, context_size) = (self.num_blocks(seq_len), self.context_size());
        let mut masked =
            Vec::with_capacity(valid.len() * num_blocks * self.chunk_size * context_size);
        for valid in valid {
            for block in 0..num_blocks {
                for offset in 0..self.chunk_size {
                    let query = block * self.chunk_size + offset;
                    for slot in 0..context_size {
                        let allowed = query < seq_len
                            && self.key_position(block, slot).is_some_and(|key| {
                                let in_context = if key <= query {
                                    query - key < self.max_past_horizon
                                } else {
                                    key - query < self.max_future_horizon
                                };
                                key < seq_len && valid[key] && in_context
                            });
                        masked.push(u8::from(!allowed));
                    }
                }
            }
        }
        Tensor::from_vec(
            masked,
            (valid.len(), 1, num_blocks, self.chunk_size, context_size),
            device,
        )
    }
}

/// Sinusoidal embeddings of the relative positions `context_size / 2 ..= 0`.
fn relative_position_embeddings(cfg: &AudioConfig, device: &Device) -> Result<Tensor> {
    let context_size = Chunking::new(cfg).context_size();
    let num_timescales = cfg.hidden_size / 2;
    let log_increment = 10_000f64.ln() / (num_timescales.max(2) - 1) as f64;
    let num_positions = context_size / 2 + 1;
    let mut values = Vec::with_capacity(num_positions * cfg.hidden_size);
    for position in (0..num_positions).rev() {
        let scaled = |i: usize| position as f64 * (-(i as f64) * log_increment).exp();
        values.extend((0..num_timescales).map(|i| scaled(i).sin() as f32));
        values.extend((0..num_timescales).map(|i| scaled(i).cos() as f32));
    }
    Tensor::from_vec(values, (num_positions, 2 * num_timescales), device)
}

/// Matrix product of rank 5 tensors, which candle only implements up to rank 4.
fn matmul5(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let (b, heads, blocks, m, _) = lhs.dims5()?;
    let n = rhs.dim(D::Minus1)?;
    lhs.flatten(0, 1)?
        .matmul(&rhs.flatten(0, 1)?)?
        .reshape((b, heads, blocks, m, n))
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: ClippableLinear,
    k_proj: ClippableLinear,
    v_proj: ClippableLinear,
    post: ClippableLinear,
    relative_k_proj: Linear,
    /// `softplus(per_dim_scale)` times the query scale, in f32.
    query_scale: Tensor,
    key_scale: f64,
    softcap: f64,
    invalid_logits_value: f64,
    chunking: Chunking,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = hidden / num_heads;
        let linear =
            |name: &str| ClippableLinear::new(hidden, hidden, cfg.use_clipped_linears, vb.pp(name));
        let per_dim_scale = vb.get(head_dim, "per_dim_scale")?.to_dtype(DType::F32)?;
        // softplus(x) = ln(1 + e^x)
        let softplus = (per_dim_scale.exp()? + 1.)?.log()?;
        let ln2 = std::f64::consts::LN_2;
        let query_scale = (softplus * ((head_dim as f64).powf(-0.5) / ln2))?;
        Ok(Self {
            q_proj: linear("q_proj")?,
            k_proj: linear("k_proj")?,
            v_proj: linear("v_proj")?,
            post: linear("post")?,
            relative_k_proj: linear_no_bias(hidden, hidden, vb.pp("relative_k_proj"))?,
            query_scale,
            key_scale: (1. + std::f64::consts::E).ln() / ln2,
            softcap: cfg.attention_logit_cap,
            invalid_logits_value: cfg.attention_invalid_logits_value,
            chunking: Chunking::new(cfg),
            num_heads,
            head_dim,
        })
    }

    /// Splits `(batch, seq_len, heads, head_dim)` into `(batch, num_blocks, chunk_size, heads,
    /// head_dim)`, zero padding the last block.
    fn to_blocks(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, heads, head_dim) = xs.dims4()?;
        let chunk_size = self.chunking.chunk_size;
        let num_blocks = self.chunking.num_blocks(t);
        xs.pad_with_zeros(1, 0, num_blocks * chunk_size - t)?
            .reshape((b, num_blocks, chunk_size, heads, head_dim))
    }

    /// Extracts the overlapping context window of every block: `(batch, seq_len, heads,
    /// head_dim)` to `(batch, num_blocks, context_size, heads, head_dim)`.
    fn block_contexts(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, heads, head_dim) = xs.dims4()?;
        let Chunking {
            chunk_size,
            max_past_horizon,
            max_future_horizon,
        } = self.chunking;
        let context_size = self.chunking.context_size();
        let num_blocks = self.chunking.num_blocks(t);
        let xs = xs.pad_with_zeros(1, max_past_horizon, max_future_horizon + chunk_size - 1)?;
        let ids: Vec<u32> = (0..num_blocks)
            .flat_map(|block| (0..context_size).map(move |slot| (block * chunk_size + slot) as u32))
            .collect();
        let ids = Tensor::from_vec(ids, num_blocks * context_size, xs.device())?;
        xs.index_select(&ids, 1)?
            .reshape((b, num_blocks, context_size, heads, head_dim))
    }

    /// Realigns the position scores of a block, indexed by relative position, with the slots
    /// of the block's context. See appendix B of <https://arxiv.org/abs/1901.02860>.
    fn relative_shift(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, heads, num_blocks, block_size, num_positions) = xs.dims5()?;
        let context_size = self.chunking.context_size();
        xs.pad_with_zeros(D::Minus1, 0, context_size + 1 - num_positions)?
            .reshape((b, heads, num_blocks, block_size * (context_size + 1)))?
            .narrow(D::Minus1, 0, block_size * context_size)?
            .reshape((b, heads, num_blocks, block_size, context_size))
    }

    fn forward(
        &self,
        xs: &Tensor,
        position_embeddings: &Tensor,
        masked: &Tensor,
    ) -> Result<Tensor> {
        let dtype = xs.dtype();
        let (b, t, _) = xs.dims3()?;
        let (heads, head_dim) = (self.num_heads, self.head_dim);
        // The attention itself runs in f32.
        let split = |proj: &ClippableLinear| -> Result<Tensor> {
            xs.apply(proj)?
                .to_dtype(DType::F32)?
                .reshape((b, t, heads, head_dim))
        };
        let q = split(&self.q_proj)?.broadcast_mul(&self.query_scale)?;
        let k = (split(&self.k_proj)? * self.key_scale)?;
        let v = split(&self.v_proj)?;

        // (batch, heads, num_blocks, chunk_size, head_dim)
        let q = self.to_blocks(&q)?.permute((0, 3, 1, 2, 4))?.contiguous()?;
        let num_blocks = q.dim(2)?;
        let chunk_size = self.chunking.chunk_size;

        // Content scores: (batch, heads, num_blocks, chunk_size, context_size)
        let k = self
            .block_contexts(&k)?
            .permute((0, 3, 1, 4, 2))?
            .contiguous()?;
        let content = matmul5(&q, &k)?;

        // Position scores, against the embeddings of the relative positions.
        let num_positions = position_embeddings.dim(0)?;
        let relative_k = position_embeddings
            .to_dtype(dtype)?
            .apply(&self.relative_k_proj)?
            .to_dtype(DType::F32)?
            .reshape((num_positions, heads, head_dim))?
            .permute((1, 2, 0))?
            .contiguous()?;
        let position = q
            .reshape((b, heads, num_blocks * chunk_size, head_dim))?
            .broadcast_matmul(&relative_k)?
            .reshape((b, heads, num_blocks, chunk_size, num_positions))?;
        let position = self.relative_shift(&position)?;

        let scores = (((content + position)? / self.softcap)?.tanh()? * self.softcap)?;
        let invalid = Tensor::new(self.invalid_logits_value as f32, scores.device())?
            .broadcast_as(scores.shape())?;
        let scores = masked
            .broadcast_as(scores.shape())?
            .where_cond(&invalid, &scores)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;

        // (batch, heads, num_blocks, context_size, head_dim)
        let v = self
            .block_contexts(&v)?
            .permute((0, 3, 1, 2, 4))?
            .contiguous()?;
        matmul5(&weights, &v)?
            .permute((0, 2, 3, 1, 4))?
            .reshape((b, num_blocks * chunk_size, heads * head_dim))?
            .narrow(1, 0, t)?
            .to_dtype(dtype)?
            .apply(&self.post)
    }
}

#[derive(Debug, Clone)]
struct FeedForward {
    ffw_layer_1: ClippableLinear,
    ffw_layer_2: ClippableLinear,
    pre_layer_norm: RmsNorm,
    post_layer_norm: RmsNorm,
    act: Activation,
    gradient_clipping: f64,
    residual_weight: f64,
}

impl FeedForward {
    fn new(cfg: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let clipped = cfg.use_clipped_linears;
        Ok(Self {
            ffw_layer_1: ClippableLinear::new(hidden, hidden * 4, clipped, vb.pp("ffw_layer_1"))?,
            ffw_layer_2: ClippableLinear::new(hidden * 4, hidden, clipped, vb.pp("ffw_layer_2"))?,
            pre_layer_norm: RmsNorm::new(hidden, RMS_NORM_EPS, vb.pp("pre_layer_norm"))?,
            post_layer_norm: RmsNorm::new(hidden, RMS_NORM_EPS, vb.pp("post_layer_norm"))?,
            act: cfg.hidden_act,
            gradient_clipping: cfg.gradient_clipping,
            residual_weight: cfg.residual_weight,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = clip(xs, self.gradient_clipping)?
            .apply(&self.pre_layer_norm)?
            .apply(&self.ffw_layer_1)?
            .apply(&self.act)?
            .apply(&self.ffw_layer_2)?;
        let ys = clip(&ys, self.gradient_clipping)?.apply(&self.post_layer_norm)?;
        xs + (ys * self.residual_weight)?
    }
}

/// A gated linear unit feeding a causal depthwise convolution over time.
#[derive(Debug, Clone)]
struct LightConv1d {
    linear_start: ClippableLinear,
    linear_end: ClippableLinear,
    /// One `(1, 1, hidden_size)` tensor per tap of the depthwise kernel.
    kernel_taps: Vec<Tensor>,
    pre_layer_norm: RmsNorm,
    conv_norm: RmsNorm,
    act: Activation,
    gradient_clipping: f64,
}

impl LightConv1d {
    fn new(cfg: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let clipped = cfg.use_clipped_linears;
        let kernel_size = cfg.conv_kernel_size;
        let kernel = vb
            .pp("depthwise_conv1d")
            .get((hidden, 1, kernel_size), "weight")?;
        let kernel_taps = (0..kernel_size)
            .map(|tap| kernel.narrow(2, tap, 1)?.reshape((1, 1, hidden)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            linear_start: ClippableLinear::new(hidden, hidden * 2, clipped, vb.pp("linear_start"))?,
            linear_end: ClippableLinear::new(hidden, hidden, clipped, vb.pp("linear_end"))?,
            kernel_taps,
            pre_layer_norm: RmsNorm::new(hidden, cfg.rms_norm_eps, vb.pp("pre_layer_norm"))?,
            conv_norm: RmsNorm::new(hidden, cfg.rms_norm_eps, vb.pp("conv_norm"))?,
            act: cfg.hidden_act,
            gradient_clipping: cfg.gradient_clipping,
        })
    }

    /// The depthwise convolution, left padded so that it only looks at the past. With one
    /// kernel per channel it is a weighted sum of shifted copies of the `(batch, time,
    /// hidden_size)` input.
    fn causal_depthwise_conv(&self, xs: &Tensor) -> Result<Tensor> {
        let t = xs.dim(1)?;
        let padded = xs.pad_with_zeros(1, self.kernel_taps.len() - 1, 0)?;
        let mut ys: Option<Tensor> = None;
        for (tap, weight) in self.kernel_taps.iter().enumerate() {
            let term = padded.narrow(1, tap, t)?.broadcast_mul(weight)?;
            ys = Some(match ys {
                Some(ys) => (ys + term)?,
                None => term,
            });
        }
        ys.ok_or_else(|| candle::Error::Msg("the depthwise kernel is empty".to_string()))
    }
}

impl Module for LightConv1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = xs.apply(&self.pre_layer_norm)?.apply(&self.linear_start)?;
        let half = ys.dim(D::Minus1)? / 2;
        let gate = candle_nn::ops::sigmoid(&ys.narrow(D::Minus1, half, half)?)?;
        let ys = (ys.narrow(D::Minus1, 0, half)? * gate)?;
        let ys = self.causal_depthwise_conv(&ys)?;
        let ys = clip(&ys, self.gradient_clipping)?
            .apply(&self.conv_norm)?
            .apply(&self.act)?
            .apply(&self.linear_end)?;
        xs + ys
    }
}

#[derive(Debug, Clone)]
struct ConformerLayer {
    feed_forward1: FeedForward,
    feed_forward2: FeedForward,
    self_attn: Attention,
    lconv1d: LightConv1d,
    norm_pre_attn: RmsNorm,
    norm_post_attn: RmsNorm,
    norm_out: RmsNorm,
    gradient_clipping: f64,
}

impl ConformerLayer {
    fn new(cfg: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let norm = |name: &str| RmsNorm::new(cfg.hidden_size, RMS_NORM_EPS, vb.pp(name));
        Ok(Self {
            feed_forward1: FeedForward::new(cfg, vb.pp("feed_forward1"))?,
            feed_forward2: FeedForward::new(cfg, vb.pp("feed_forward2"))?,
            self_attn: Attention::new(cfg, vb.pp("self_attn"))?,
            lconv1d: LightConv1d::new(cfg, vb.pp("lconv1d"))?,
            norm_pre_attn: norm("norm_pre_attn")?,
            norm_post_attn: norm("norm_post_attn")?,
            norm_out: norm("norm_out")?,
            gradient_clipping: cfg.gradient_clipping,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        position_embeddings: &Tensor,
        masked: &Tensor,
    ) -> Result<Tensor> {
        let xs = self.feed_forward1.forward(xs)?;
        let ys = clip(&xs, self.gradient_clipping)?.apply(&self.norm_pre_attn)?;
        let ys = self.self_attn.forward(&ys, position_embeddings, masked)?;
        let ys = clip(&ys, self.gradient_clipping)?.apply(&self.norm_post_attn)?;
        let xs = (xs + ys)?
            .apply(&self.lconv1d)?
            .apply(&self.feed_forward2)?;
        clip(&xs, self.gradient_clipping)?.apply(&self.norm_out)
    }
}

/// The audio tower. Together with [`crate::MultimodalEmbedder`], it turns a log-mel
/// spectrogram into soft tokens for the text backbone, one for every 40 ms of audio.
#[derive(Debug, Clone)]
pub struct AudioTower {
    subsample_conv_projection: SubSampleConvProjection,
    position_embeddings: Tensor,
    layers: Vec<ConformerLayer>,
    output_proj: Linear,
    chunking: Chunking,
    dtype: DType,
}

impl AudioTower {
    pub fn new(cfg: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let vb_layers = vb.pp("layers");
        let layers = (0..cfg.num_hidden_layers)
            .map(|idx| ConformerLayer::new(cfg, vb_layers.pp(idx)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            subsample_conv_projection: SubSampleConvProjection::new(
                cfg,
                vb.pp("subsample_conv_projection"),
            )?,
            position_embeddings: relative_position_embeddings(cfg, vb.device())?,
            layers,
            output_proj: linear(cfg.hidden_size, cfg.output_size(), vb.pp("output_proj"))?,
            chunking: Chunking::new(cfg),
            dtype: vb.dtype(),
        })
    }

    /// Encodes a batch of log-mel spectrograms.
    ///
    /// - `input_features`: `(batch, num_frames, num_mel_bins)`.
    /// - `input_features_mask`: `(batch, num_frames)`, non-zero for the frames that are audio
    ///   rather than padding.
    ///
    /// Returns one `(num_soft_tokens, output_size)` tensor per clip, padding stripped.
    pub fn forward(
        &self,
        input_features: &Tensor,
        input_features_mask: &Tensor,
    ) -> Result<Vec<Tensor>> {
        let device = input_features.device();
        let mask = input_features_mask
            .to_dtype(DType::F32)?
            .ne(0f32)?
            .to_dtype(DType::F32)?;
        let (xs, mask) = self
            .subsample_conv_projection
            .forward(&input_features.to_dtype(self.dtype)?, &mask)?;
        let (_, t, _) = xs.dims3()?;
        let valid: Vec<Vec<bool>> = mask
            .to_vec2::<f32>()?
            .into_iter()
            .map(|row| row.into_iter().map(|value| value != 0.).collect())
            .collect();
        let masked = self.chunking.blocked_mask(&valid, t, device)?;

        let mut xs = xs;
        for layer in &self.layers {
            xs = layer.forward(&xs, &self.position_embeddings, &masked)?;
        }
        let xs = xs.apply(&self.output_proj)?;

        valid
            .iter()
            .enumerate()
            .map(|(idx, valid)| {
                let kept: Vec<u32> = (0..t as u32).filter(|&i| valid[i as usize]).collect();
                let len = kept.len();
                xs.get(idx)?
                    .index_select(&Tensor::from_vec(kept, len, device)?, 0)
            })
            .collect()
    }
}
