//! The Gemma 4 vision tower: a ViT over variable-resolution patch sequences, with learned
//! per-axis position embeddings, axial 2D RoPE and a spatial average pooling of the patches
//! into soft tokens.

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_no_bias, Activation, Linear, VarBuilder};

use crate::config::VisionConfig;
use crate::layers::{attend, repeat_kv, rotate_half, ClippableLinear, RmsNorm};

/// Position of a padding patch, on both axes.
pub const PADDING_POSITION: i64 = -1;

#[derive(Debug, Clone)]
struct PatchEmbedder {
    input_proj: Linear,
    /// `(2, position_embedding_size, hidden_size)`: one table per axis.
    position_embedding_table: Tensor,
}

impl PatchEmbedder {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_dim = 3 * cfg.patch_size * cfg.patch_size;
        Ok(Self {
            input_proj: linear_no_bias(patch_dim, cfg.hidden_size, vb.pp("input_proj"))?,
            position_embedding_table: vb.get(
                (2, cfg.position_embedding_size, cfg.hidden_size),
                "position_embedding_table",
            )?,
        })
    }

    fn forward(&self, pixel_values: &Tensor, patches: &PatchLayout) -> Result<Tensor> {
        let dtype = self.position_embedding_table.dtype();
        let device = pixel_values.device();
        let (b, n) = (patches.batch, patches.num_patches);
        // The image processor leaves the pixels in [0, 1], the model wants [-1, 1].
        let xs = ((pixel_values.to_dtype(DType::F32)? - 0.5)? * 2.)?.to_dtype(dtype)?;
        let xs = xs.apply(&self.input_proj)?;

        let axis_embeddings = |axis: usize| -> Result<Tensor> {
            let ids: Vec<u32> = patches
                .positions
                .iter()
                .map(|position| position[axis].max(0) as u32)
                .collect();
            let ids = Tensor::from_vec(ids, b * n, device)?;
            self.position_embedding_table
                .get(axis)?
                .index_select(&ids, 0)?
                .reshape((b, n, ()))
        };
        let position_embeddings = (axis_embeddings(0)? + axis_embeddings(1)?)?;
        let position_embeddings = match &patches.valid {
            Some(valid) => position_embeddings.broadcast_mul(&valid.to_dtype(dtype)?)?,
            None => position_embeddings,
        };
        xs + position_embeddings
    }
}

/// Axial 2D RoPE sharing its frequencies between both axes: the first half of each head
/// rotates with the x position, the second half with the y position.
#[derive(Debug, Clone)]
struct RotaryEmbedding {
    inv_freq: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &VisionConfig, device: &Device) -> Result<Self> {
        let spatial_dim = cfg.head_dim / 2;
        let theta = cfg.rope_parameters.rope_theta as f32;
        let inv_freq: Vec<f32> = (0..spatial_dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f32 / spatial_dim as f32))
            .collect();
        let len = inv_freq.len();
        Ok(Self {
            inv_freq: Tensor::from_vec(inv_freq, (1, 1, 1, len), device)?,
        })
    }

    /// Returns cos and sin tables of shape `(batch, 1, num_patches, head_dim)`.
    fn forward(&self, patches: &PatchLayout, dtype: DType) -> Result<(Tensor, Tensor)> {
        let (b, n) = (patches.batch, patches.num_patches);
        let positions: Vec<f32> = patches
            .positions
            .iter()
            .flat_map(|position| [position[0] as f32, position[1] as f32])
            .collect();
        let positions = Tensor::from_vec(positions, (b, n, 2, 1), self.inv_freq.device())?;
        let freqs = positions.broadcast_mul(&self.inv_freq)?;
        let layout = |table: Tensor| -> Result<Tensor> {
            let x = table.narrow(2, 0, 1)?.squeeze(2)?;
            let y = table.narrow(2, 1, 1)?.squeeze(2)?;
            Tensor::cat(&[&x, &x, &y, &y], D::Minus1)?
                .unsqueeze(1)?
                .to_dtype(dtype)
        };
        Ok((layout(freqs.cos()?)?, layout(freqs.sin()?)?))
    }
}

/// Rotates each axis' half of the head dimension independently.
fn apply_axial_rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = xs.dim(D::Minus1)? / 2;
    let rotate = |start: usize| -> Result<Tensor> {
        let xs = xs.narrow(D::Minus1, start, half)?;
        let cos = cos.narrow(D::Minus1, start, half)?;
        let sin = sin.narrow(D::Minus1, start, half)?;
        xs.broadcast_mul(&cos)? + rotate_half(&xs)?.broadcast_mul(&sin)?
    };
    Tensor::cat(&[rotate(0)?, rotate(half)?], D::Minus1)
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: ClippableLinear,
    k_proj: ClippableLinear,
    v_proj: ClippableLinear,
    o_proj: ClippableLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let (num_heads, num_kv_heads, head_dim) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let linear = |in_dim: usize, out_dim: usize, name: &str| {
            ClippableLinear::new(in_dim, out_dim, cfg.use_clipped_linears, vb.pp(name))
        };
        Ok(Self {
            q_proj: linear(hidden, num_heads * head_dim, "q_proj")?,
            k_proj: linear(hidden, num_kv_heads * head_dim, "k_proj")?,
            v_proj: linear(hidden, num_kv_heads * head_dim, "v_proj")?,
            o_proj: linear(num_heads * head_dim, hidden, "o_proj")?,
            q_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            v_norm: RmsNorm::without_scale(cfg.rms_norm_eps),
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, n, _) = xs.dims3()?;
        let split = |ys: Tensor, heads: usize| -> Result<Tensor> {
            ys.reshape((b, n, heads, self.head_dim))?.transpose(1, 2)
        };
        let q = split(self.q_proj.forward(xs)?, self.num_heads)?;
        let k = split(self.k_proj.forward(xs)?, self.num_kv_heads)?;
        let v = split(self.v_proj.forward(xs)?, self.num_kv_heads)?;

        let q = apply_axial_rope(&self.q_norm.forward(&q)?, cos, sin)?;
        let k = apply_axial_rope(&self.k_norm.forward(&k)?, cos, sin)?;
        let v = self.v_norm.forward(&v)?;

        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(k, n_rep)?.contiguous()?;
        let v = repeat_kv(v, n_rep)?.contiguous()?;

        attend(&q.contiguous()?, &k, &v, mask)?
            .transpose(1, 2)?
            .reshape((b, n, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: ClippableLinear,
    up_proj: ClippableLinear,
    down_proj: ClippableLinear,
    act: Activation,
}

impl Mlp {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let (hidden, intermediate) = (cfg.hidden_size, cfg.intermediate_size);
        let linear = |in_dim: usize, out_dim: usize, name: &str| {
            ClippableLinear::new(in_dim, out_dim, cfg.use_clipped_linears, vb.pp(name))
        };
        Ok(Self {
            gate_proj: linear(hidden, intermediate, "gate_proj")?,
            up_proj: linear(hidden, intermediate, "up_proj")?,
            down_proj: linear(intermediate, hidden, "down_proj")?,
            act: cfg.hidden_activation,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = xs.apply(&self.gate_proj)?.apply(&self.act)?;
        (gate * xs.apply(&self.up_proj)?)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct EncoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
}

impl EncoderLayer {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let norm = |name: &str| RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp(name));
        Ok(Self {
            self_attn: Attention::new(cfg, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: norm("input_layernorm")?,
            post_attention_layernorm: norm("post_attention_layernorm")?,
            pre_feedforward_layernorm: norm("pre_feedforward_layernorm")?,
            post_feedforward_layernorm: norm("post_feedforward_layernorm")?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let ys = self.input_layernorm.forward(xs)?;
        let ys = self.self_attn.forward(&ys, mask, cos, sin)?;
        let xs = (xs + ys.apply(&self.post_attention_layernorm)?)?;

        let ys = xs
            .apply(&self.pre_feedforward_layernorm)?
            .apply(&self.mlp)?
            .apply(&self.post_feedforward_layernorm)?;
        xs + ys
    }
}

/// What the tower derives from the patch positions of a batch.
struct PatchLayout {
    batch: usize,
    num_patches: usize,
    /// `(x, y)` of every patch, row major, [`PADDING_POSITION`] on both axes for padding.
    positions: Vec<[i64; 2]>,
    /// `(batch, num_patches, 1)`, 1 for real patches; `None` when nothing is padded.
    valid: Option<Tensor>,
    /// Number of real patches of every image.
    num_valid: Vec<usize>,
}

impl PatchLayout {
    fn new(positions: &Tensor, device: &Device) -> Result<Self> {
        let (batch, num_patches, axes) = positions.dims3()?;
        if axes != 2 {
            candle::bail!("patch positions must be (batch, num_patches, 2), got {axes} axes")
        }
        let positions: Vec<[i64; 2]> = positions
            .to_dtype(DType::I64)?
            .flatten_all()?
            .to_vec1::<i64>()?
            .chunks_exact(2)
            .map(|position| [position[0], position[1]])
            .collect();
        let is_valid: Vec<f32> = positions
            .iter()
            .map(|position| {
                let padding = position.iter().all(|&p| p == PADDING_POSITION);
                if padding {
                    0.
                } else {
                    1.
                }
            })
            .collect();
        let num_valid: Vec<usize> = is_valid
            .chunks(num_patches.max(1))
            .map(|image| image.iter().filter(|&&valid| valid != 0.).count())
            .collect();
        let valid = if num_valid.iter().any(|&count| count != num_patches) {
            Some(Tensor::from_vec(is_valid, (batch, num_patches, 1), device)?)
        } else {
            None
        };
        Ok(Self {
            batch,
            num_patches,
            positions,
            valid,
            num_valid,
        })
    }
}

/// The vision tower. Together with [`crate::MultimodalEmbedder`], it turns the patches of an
/// image into soft tokens for the text backbone.
#[derive(Debug, Clone)]
pub struct VisionTower {
    patch_embedder: PatchEmbedder,
    rotary_emb: RotaryEmbedding,
    layers: Vec<EncoderLayer>,
    standardization: Option<(Tensor, Tensor)>,
    hidden_size: usize,
    pooling_kernel_size: usize,
    dtype: DType,
}

impl VisionTower {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let vb_layers = vb.pp("encoder").pp("layers");
        let layers = (0..cfg.num_hidden_layers)
            .map(|idx| EncoderLayer::new(cfg, vb_layers.pp(idx)))
            .collect::<Result<Vec<_>>>()?;
        let standardization = if cfg.standardize {
            let get = |name: &str| vb.get(cfg.hidden_size, name)?.to_dtype(DType::F32);
            Some((get("std_bias")?, get("std_scale")?))
        } else {
            None
        };
        Ok(Self {
            patch_embedder: PatchEmbedder::new(cfg, vb.pp("patch_embedder"))?,
            rotary_emb: RotaryEmbedding::new(cfg, vb.device())?,
            layers,
            standardization,
            hidden_size: cfg.hidden_size,
            pooling_kernel_size: cfg.pooling_kernel_size,
            dtype: vb.dtype(),
        })
    }

    /// Encodes a batch of patchified images.
    ///
    /// - `pixel_values`: `(batch, num_patches, 3 * patch_size^2)`, in `[0, 1]`.
    /// - `positions`: `(batch, num_patches, 2)` integer `(x, y)` patch coordinates, with
    ///   [`PADDING_POSITION`] on both axes for the padding patches.
    ///
    /// `num_patches` has to be a multiple of `pooling_kernel_size^2`. Returns one
    /// `(num_soft_tokens, hidden_size)` tensor per image, padding stripped.
    pub fn forward(&self, pixel_values: &Tensor, positions: &Tensor) -> Result<Vec<Tensor>> {
        let device = pixel_values.device();
        let patches = PatchLayout::new(positions, device)?;
        let (b, n) = (patches.batch, patches.num_patches);
        let k_squared = self.pooling_kernel_size * self.pooling_kernel_size;
        if n % k_squared != 0 {
            candle::bail!("{n} patches cannot be pooled by {k_squared}")
        }
        let output_length = n / k_squared;

        let mut xs = self.patch_embedder.forward(pixel_values, &patches)?;
        let (cos, sin) = self.rotary_emb.forward(&patches, self.dtype)?;
        // (batch, 1, 1, num_patches): masks the padding keys with the lowest finite score.
        let mask = match &patches.valid {
            Some(valid) => Some(((1. - valid)? * f64::from(f32::MIN))?.reshape((b, 1, 1, n))?),
            None => None,
        };
        for layer in &self.layers {
            xs = layer.forward(&xs, mask.as_ref(), &cos, &sin)?;
        }

        // The pooling and the sqrt(hidden_size) scaling run in f32: the scaled activations can
        // exceed the range of half precision.
        let xs = xs.to_dtype(DType::F32)?;
        let xs = match &patches.valid {
            Some(valid) => xs.broadcast_mul(valid)?,
            None => xs,
        };
        let xs = self.pool(&xs, &patches, output_length)?;
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;
        if let Some((bias, scale)) = &self.standardization {
            xs = xs.broadcast_sub(bias)?.broadcast_mul(scale)?;
        }
        let xs = xs.to_dtype(self.dtype)?;

        (0..b)
            .map(|idx| {
                xs.get(idx)?
                    .narrow(0, 0, patches.num_valid[idx] / k_squared)
            })
            .collect()
    }

    /// Averages the patches of every `k x k` cell of the patch grid: `(batch, num_patches,
    /// hidden_size)` to `(batch, output_length, hidden_size)`.
    fn pool(&self, xs: &Tensor, patches: &PatchLayout, output_length: usize) -> Result<Tensor> {
        let (b, n, hidden) = xs.dims3()?;
        let k = self.pooling_kernel_size as i64;
        let mut cells = Vec::with_capacity(b * n);
        for image in patches.positions.chunks(n) {
            let width = image.iter().map(|position| position[0].max(0)).max();
            let cells_per_row = (width.unwrap_or(0) + 1) / k;
            for position in image {
                let (x, y) = (position[0].max(0), position[1].max(0));
                let cell = x / k + cells_per_row * (y / k);
                if cell as usize >= output_length {
                    candle::bail!(
                        "patch at ({x}, {y}) falls outside of the {output_length} pooled cells"
                    )
                }
                cells.push(cell as u32);
            }
        }
        let cells = Tensor::from_vec(cells, (b, n, 1), xs.device())?
            .broadcast_as((b, n, hidden))?
            .contiguous()?;
        let xs = (xs / (k * k) as f64)?;
        Tensor::zeros((b, output_length, hidden), DType::F32, xs.device())?
            .scatter_add(&cells, &xs, 1)
    }
}
