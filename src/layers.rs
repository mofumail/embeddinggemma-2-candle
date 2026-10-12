//! Building blocks shared by the text, vision and audio towers.

use candle::{DType, Module, Result, Tensor, D};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

/// RMS normalization as used across Gemma 4 and EmbeddingGemma 2: computed in f32 and scaled
/// by `weight` directly (unlike earlier Gemma checkpoints, which store `weight - 1`).
#[derive(Debug, Clone)]
pub(crate) struct RmsNorm {
    weight: Option<Tensor>,
    eps: f64,
}

impl RmsNorm {
    pub(crate) fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?.to_dtype(DType::F32)?;
        Ok(Self {
            weight: Some(weight),
            eps,
        })
    }

    pub(crate) fn without_scale(eps: f64) -> Self {
        Self { weight: None, eps }
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?;
        let mean_squared = (xs.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?;
        let xs = xs.broadcast_mul(&mean_squared.powf(-0.5)?)?;
        let xs = match &self.weight {
            Some(weight) => xs.broadcast_mul(weight)?,
            None => xs,
        };
        xs.to_dtype(dtype)
    }
}

pub(crate) fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let half = xs.dim(D::Minus1)? / 2;
    let x1 = xs.narrow(D::Minus1, 0, half)?;
    let x2 = xs.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

/// Repeats each key/value head `n_rep` times: `(b, kv_heads, t, d)` to `(b, kv_heads * n_rep, t, d)`.
pub(crate) fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let (b, kv_heads, t, d) = xs.dims4()?;
    xs.unsqueeze(2)?
        .broadcast_as((b, kv_heads, n_rep, t, d))?
        .reshape((b, kv_heads * n_rep, t, d))
}

/// `softmax(q k^T + mask) v`, with the softmax taken in f32 whatever the working dtype.
/// No `1/sqrt(d)` factor: queries and keys are RMS normalized beforehand.
pub(crate) fn attend(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let dtype = q.dtype();
    let weights = q
        .matmul(&k.transpose(2, 3)?.contiguous()?)?
        .to_dtype(DType::F32)?;
    let weights = match mask {
        Some(mask) => weights.broadcast_add(mask)?,
        None => weights,
    };
    let weights = candle_nn::ops::softmax_last_dim(&weights)?.to_dtype(dtype)?;
    weights.matmul(v)
}

/// A bias-free linear layer that optionally clamps its input and its output to ranges stored
/// in the checkpoint. The clamps are left over from quantization aware training.
#[derive(Debug, Clone)]
pub(crate) struct ClippableLinear {
    linear: Linear,
    clip: Option<Clip>,
}

#[derive(Debug, Clone)]
struct Clip {
    input: (Tensor, Tensor),
    output: (Tensor, Tensor),
}

impl ClippableLinear {
    pub(crate) fn new(
        in_dim: usize,
        out_dim: usize,
        use_clipped_linears: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let linear = linear_no_bias(in_dim, out_dim, vb.pp("linear"))?;
        let clip = if use_clipped_linears {
            let bound = |name: &str| vb.get((), name);
            Some(Clip {
                input: (bound("input_min")?, bound("input_max")?),
                output: (bound("output_min")?, bound("output_max")?),
            })
        } else {
            None
        };
        Ok(Self { linear, clip })
    }
}

fn clamp(xs: &Tensor, (min, max): &(Tensor, Tensor)) -> Result<Tensor> {
    xs.broadcast_maximum(min)?.broadcast_minimum(max)
}

impl Module for ClippableLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match &self.clip {
            None => self.linear.forward(xs),
            Some(clip) => clamp(
                &self.linear.forward(&clamp(xs, &clip.input)?)?,
                &clip.output,
            ),
        }
    }
}
