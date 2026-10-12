//! The EmbeddingGemma 2 text backbone.
//!
//! A Gemma style transformer run as an encoder: attention is bidirectional on every layer,
//! with five sliding-window layers for each full-attention one. Each layer is additionally
//! gated by a per-layer embedding (PLE) projected from the input embeddings.

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_b, linear_no_bias, Activation, Linear, VarBuilder};

use crate::config::{LayerGeometry, LayerType, TextConfig};
use crate::layers::{attend, repeat_kv, rotate_half, RmsNorm};

/// Cos and sin tables of shape `(seq_len, head_dim)` for positions `0..seq_len`.
fn rope_tables(
    head_dim: usize,
    theta: f64,
    seq_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let inv_freq: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / (theta as f32).powf(i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, head_dim / 2), device)?;
    let positions = Tensor::arange(0u32, seq_len as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((seq_len, 1))?;
    let freqs = positions.matmul(&inv_freq)?;
    let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
    Ok((emb.cos()?.to_dtype(dtype)?, emb.sin()?.to_dtype(dtype)?))
}

fn apply_rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    xs.broadcast_mul(cos)? + rotate_half(xs)?.broadcast_mul(sin)?
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &TextConfig, geometry: &LayerGeometry, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = geometry.num_key_value_heads;
        let head_dim = geometry.head_dim;
        let bias = cfg.attention_bias;
        let eps = cfg.rms_norm_eps;
        Ok(Self {
            q_proj: linear_b(hidden, num_heads * head_dim, bias, vb.pp("q_proj"))?,
            k_proj: linear_b(hidden, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?,
            v_proj: linear_b(hidden, num_kv_heads * head_dim, bias, vb.pp("v_proj"))?,
            o_proj: linear_b(num_heads * head_dim, hidden, bias, vb.pp("o_proj"))?,
            q_norm: RmsNorm::new(head_dim, eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(head_dim, eps, vb.pp("k_norm"))?,
            v_norm: RmsNorm::without_scale(eps),
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
        let (b, t, _) = xs.dims3()?;
        let split = |ys: Tensor, heads: usize| -> Result<Tensor> {
            ys.reshape((b, t, heads, self.head_dim))?.transpose(1, 2)
        };
        let q = split(self.q_proj.forward(xs)?, self.num_heads)?;
        let k = split(self.k_proj.forward(xs)?, self.num_kv_heads)?;
        let v = split(self.v_proj.forward(xs)?, self.num_kv_heads)?;

        let q = apply_rope(&self.q_norm.forward(&q)?, cos, sin)?;
        let k = apply_rope(&self.k_norm.forward(&k)?, cos, sin)?;
        let v = self.v_norm.forward(&v)?;

        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(k, n_rep)?.contiguous()?;
        let v = repeat_kv(v, n_rep)?.contiguous()?;

        attend(&q.contiguous()?, &k, &v, mask)?
            .transpose(1, 2)?
            .reshape((b, t, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act: Activation,
}

impl Mlp {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let (hidden, intermediate) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate_proj: linear_no_bias(hidden, intermediate, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden, intermediate, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate, hidden, vb.pp("down_proj"))?,
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

/// Produces the per-layer embeddings every encoder layer is gated with. EmbeddingGemma 2 uses
/// projection-only PLE: the signal is derived from the input embeddings alone.
#[derive(Debug, Clone)]
struct PerLayerEmbeddings {
    projection: Linear,
    norm: RmsNorm,
    scale: f64,
    num_layers: usize,
    dim: usize,
}

impl PerLayerEmbeddings {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size_per_layer_input;
        let num_layers = cfg.num_hidden_layers;
        Ok(Self {
            projection: linear_no_bias(
                cfg.hidden_size,
                num_layers * dim,
                vb.pp("per_layer_model_projection"),
            )?,
            norm: RmsNorm::new(dim, cfg.rms_norm_eps, vb.pp("per_layer_projection_norm"))?,
            scale: (cfg.hidden_size as f64).powf(-0.5),
            num_layers,
            dim,
        })
    }

    /// Maps `(b, t, hidden)` embeddings to `(b, t, num_layers, dim)`.
    fn forward(&self, inputs_embeds: &Tensor) -> Result<Tensor> {
        let (b, t, _) = inputs_embeds.dims3()?;
        let xs = (inputs_embeds.apply(&self.projection)? * self.scale)?;
        xs.reshape((b, t, self.num_layers, self.dim))?
            .apply(&self.norm)
    }
}

/// Mixes one layer's slice of the per-layer embeddings into the residual stream.
#[derive(Debug, Clone)]
struct PleBlock {
    per_layer_input_gate: Linear,
    per_layer_projection: Linear,
    post_per_layer_input_norm: RmsNorm,
    act: Activation,
}

impl PleBlock {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let (hidden, dim) = (cfg.hidden_size, cfg.hidden_size_per_layer_input);
        Ok(Self {
            per_layer_input_gate: linear_no_bias(hidden, dim, vb.pp("per_layer_input_gate"))?,
            per_layer_projection: linear_no_bias(dim, hidden, vb.pp("per_layer_projection"))?,
            post_per_layer_input_norm: RmsNorm::new(
                hidden,
                cfg.rms_norm_eps,
                vb.pp("post_per_layer_input_norm"),
            )?,
            act: cfg.hidden_activation,
        })
    }

    fn forward(&self, xs: &Tensor, per_layer_input: &Tensor) -> Result<Tensor> {
        let ys = xs.apply(&self.per_layer_input_gate)?.apply(&self.act)?;
        let ys = (ys * per_layer_input)?
            .apply(&self.per_layer_projection)?
            .apply(&self.post_per_layer_input_norm)?;
        xs + ys
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
    ple_block: PleBlock,
    layer_scalar: Tensor,
    geometry: LayerGeometry,
}

impl EncoderLayer {
    fn new(cfg: &TextConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let geometry = cfg.layer_geometry(layer_idx);
        let norm = |name: &str| RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp(name));
        Ok(Self {
            self_attn: Attention::new(cfg, &geometry, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: norm("input_layernorm")?,
            post_attention_layernorm: norm("post_attention_layernorm")?,
            pre_feedforward_layernorm: norm("pre_feedforward_layernorm")?,
            post_feedforward_layernorm: norm("post_feedforward_layernorm")?,
            ple_block: PleBlock::new(cfg, vb.pp("ple_block"))?,
            layer_scalar: vb.get(1, "layer_scalar")?,
            geometry,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        per_layer_input: &Tensor,
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
        let xs = (xs + ys)?;

        self.ple_block
            .forward(&xs, per_layer_input)?
            .broadcast_mul(&self.layer_scalar)
    }
}

/// Output of [`TextModel::forward_embeds_with_hidden_states`].
#[derive(Debug, Clone)]
pub struct TextModelOutput {
    /// Token embeddings of shape `(batch, seq_len, embedding_dim)`.
    pub last_hidden_state: Tensor,
    /// The input embeddings followed by the output of every layer, each `(batch, seq_len, hidden_size)`.
    pub hidden_states: Vec<Tensor>,
}

/// The text backbone, including the projection from `hidden_size` to `embedding_dim`.
#[derive(Debug, Clone)]
pub struct TextModel {
    embed_tokens: Tensor,
    embed_scale: Tensor,
    ple: PerLayerEmbeddings,
    layers: Vec<EncoderLayer>,
    norm: RmsNorm,
    embedding_projection: Linear,
    sliding_window: usize,
    dtype: DType,
    device: Device,
}

impl TextModel {
    pub fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        if cfg.layer_types.len() != cfg.num_hidden_layers {
            candle::bail!(
                "layer_types has {} entries for {} layers",
                cfg.layer_types.len(),
                cfg.num_hidden_layers
            )
        }
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let embed_tokens = vb
            .pp("embed_tokens")
            .get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        // The scale is rounded to the working dtype, as the reference does: in bf16 it is
        // 22.625 rather than sqrt(512) = 22.627.
        let embed_scale = Tensor::new((cfg.hidden_size as f32).sqrt(), &device)?.to_dtype(dtype)?;
        let vb_layers = vb.pp("layers");
        let layers = (0..cfg.num_hidden_layers)
            .map(|idx| EncoderLayer::new(cfg, idx, vb_layers.pp(idx)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens,
            embed_scale,
            ple: PerLayerEmbeddings::new(cfg, vb.pp("ple"))?,
            layers,
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?,
            embedding_projection: linear_no_bias(
                cfg.hidden_size,
                cfg.embedding_dim,
                vb.pp("embedding_projection"),
            )?,
            sliding_window: cfg.sliding_window,
            dtype,
            device,
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Looks up and scales the token embeddings: `(batch, seq_len)` ids to
    /// `(batch, seq_len, hidden_size)`.
    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, t) = input_ids.dims2()?;
        let hidden = self.embed_tokens.dim(1)?;
        self.embed_tokens
            .index_select(&input_ids.flatten_all()?, 0)?
            .reshape((b, t, hidden))?
            .broadcast_mul(&self.embed_scale)
    }

    /// Encodes token ids. See [`Self::forward_embeds`].
    pub fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.forward_embeds(&self.embed_tokens(input_ids)?, attention_mask)
    }

    /// Encodes `(batch, seq_len, hidden_size)` input embeddings into
    /// `(batch, seq_len, embedding_dim)` token embeddings.
    ///
    /// `attention_mask` is `(batch, seq_len)`, non-zero for the positions to attend to. It can
    /// be omitted when no sequence is padded.
    pub fn forward_embeds(
        &self,
        inputs_embeds: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let hidden_states = self.encode(inputs_embeds, attention_mask, None)?;
        self.project(&hidden_states)
    }

    /// Same as [`Self::forward_embeds`], also returning the output of every layer.
    pub fn forward_embeds_with_hidden_states(
        &self,
        inputs_embeds: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<TextModelOutput> {
        let mut hidden_states = Vec::with_capacity(self.layers.len() + 1);
        let last = self.encode(inputs_embeds, attention_mask, Some(&mut hidden_states))?;
        Ok(TextModelOutput {
            last_hidden_state: self.project(&last)?,
            hidden_states,
        })
    }

    fn project(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Projecting every token is equivalent to projecting after mean pooling.
        hidden_states
            .apply(&self.norm)?
            .apply(&self.embedding_projection)
    }

    fn encode(
        &self,
        inputs_embeds: &Tensor,
        attention_mask: Option<&Tensor>,
        mut trace: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        let (_, seq_len, _) = inputs_embeds.dims3()?;
        let per_layer_inputs = self.ple.forward(inputs_embeds)?;
        let masks = self.attention_masks(attention_mask, seq_len)?;

        // One pair of tables per attention geometry: the sliding and the full layers differ
        // in both head dimension and base frequency.
        let mut rope: HashMap<(usize, u64), (Tensor, Tensor)> = HashMap::new();
        for layer in &self.layers {
            let LayerGeometry {
                head_dim,
                rope_theta,
                ..
            } = layer.geometry;
            if let Entry::Vacant(entry) = rope.entry((head_dim, rope_theta.to_bits())) {
                entry.insert(rope_tables(
                    head_dim,
                    rope_theta,
                    seq_len,
                    self.dtype,
                    &self.device,
                )?);
            }
        }

        let mut xs = inputs_embeds.clone();
        if let Some(trace) = trace.as_deref_mut() {
            trace.push(xs.clone());
        }
        for (idx, layer) in self.layers.iter().enumerate() {
            let geometry = &layer.geometry;
            let (cos, sin) = &rope[&(geometry.head_dim, geometry.rope_theta.to_bits())];
            let mask = match geometry.layer_type {
                LayerType::FullAttention => masks.full.as_ref(),
                LayerType::SlidingAttention => masks.sliding.as_ref(),
            };
            let per_layer_input = per_layer_inputs.narrow(2, idx, 1)?.squeeze(2)?;
            xs = layer.forward(&xs, &per_layer_input, mask, cos, sin)?;
            if let Some(trace) = trace.as_deref_mut() {
                trace.push(xs.clone());
            }
        }
        Ok(xs)
    }

    /// Builds the additive f32 attention masks, shaped to broadcast against
    /// `(batch, heads, seq_len, seq_len)` attention scores.
    fn attention_masks(&self, attention_mask: Option<&Tensor>, seq_len: usize) -> Result<Masks> {
        // Masked scores get the lowest finite value rather than -inf, as in the reference: a
        // padded query whose whole window is padding then attends uniformly instead of
        // producing NaNs that would leak into the valid positions of the next layer.
        const MASKED: f32 = f32::MIN;
        // (batch, 1, 1, seq_len): masks the padded keys.
        let padding = match attention_mask {
            Some(mask) => {
                let (b, t) = mask.dims2()?;
                if t != seq_len {
                    candle::bail!("attention mask covers {t} positions, the inputs {seq_len}")
                }
                let mask = mask.to_dtype(DType::F32)?.to_device(&self.device)?;
                if mask.min_all()?.to_scalar::<f32>()? == 0. {
                    let on_false = Tensor::new(MASKED, &self.device)?.broadcast_as(mask.shape())?;
                    let additive = mask.ne(0f32)?.where_cond(&mask.zeros_like()?, &on_false)?;
                    Some(additive.reshape((b, 1, 1, t))?)
                } else {
                    None
                }
            }
            None => None,
        };
        // (1, 1, seq_len, seq_len): masks the keys outside of the inclusive window radius.
        let window = if seq_len > self.sliding_window + 1 {
            let radius = self.sliding_window as i64;
            let values: Vec<f32> = (0..seq_len as i64)
                .flat_map(|q| {
                    (0..seq_len as i64).map(
                        move |k| {
                            if (q - k).abs() <= radius {
                                0.
                            } else {
                                MASKED
                            }
                        },
                    )
                })
                .collect();
            Some(Tensor::from_vec(
                values,
                (1, 1, seq_len, seq_len),
                &self.device,
            )?)
        } else {
            None
        };
        let sliding = match (&padding, &window) {
            (Some(padding), Some(window)) => Some(padding.broadcast_minimum(window)?),
            (None, Some(window)) => Some(window.clone()),
            (padding, None) => padding.clone(),
        };
        Ok(Masks {
            full: padding,
            sliding,
        })
    }
}

struct Masks {
    full: Option<Tensor>,
    sliding: Option<Tensor>,
}
