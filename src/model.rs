//! The composite EmbeddingGemma 2 model: the vision and audio towers feed soft tokens to the
//! text backbone, which produces one embedding per token.

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use crate::audio::AudioTower;
use crate::config::Config;
use crate::layers::RmsNorm;
use crate::text::TextModel;
use crate::vision::VisionTower;

/// Projects the output of a tower into the embedding space of the text backbone.
#[derive(Debug, Clone)]
pub struct MultimodalEmbedder {
    embedding_pre_projection_norm: RmsNorm,
    embedding_projection: Linear,
}

impl MultimodalEmbedder {
    pub fn new(
        multimodal_hidden_size: usize,
        text_hidden_size: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            embedding_pre_projection_norm: RmsNorm::without_scale(eps),
            embedding_projection: linear_no_bias(
                multimodal_hidden_size,
                text_hidden_size,
                vb.pp("embedding_projection"),
            )?,
        })
    }
}

impl Module for MultimodalEmbedder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.embedding_pre_projection_norm)?
            .apply(&self.embedding_projection)
    }
}

/// Which towers to load next to the text backbone. Skipping the ones that are not needed
/// saves memory: the text backbone is 270M parameters, the vision tower 170M and the audio
/// tower 300M.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Modalities {
    /// Images and videos.
    pub vision: bool,
    pub audio: bool,
}

impl Modalities {
    pub const ALL: Self = Self {
        vision: true,
        audio: true,
    };
    pub const TEXT_ONLY: Self = Self {
        vision: false,
        audio: false,
    };
}

impl Default for Modalities {
    fn default() -> Self {
        Self::ALL
    }
}

/// Soft tokens to substitute for the placeholder tokens of the input ids. Each tensor is
/// `(num_soft_tokens, hidden_size)` and fills the placeholders of its modality in reading
/// order, across the whole batch.
#[derive(Debug, Clone, Copy, Default)]
pub struct SoftTokens<'a> {
    pub image: Option<&'a Tensor>,
    pub video: Option<&'a Tensor>,
    pub audio: Option<&'a Tensor>,
}

#[derive(Debug, Clone)]
pub struct EmbeddingGemma2Model {
    language_model: TextModel,
    vision: Option<(VisionTower, MultimodalEmbedder)>,
    audio: Option<(AudioTower, MultimodalEmbedder)>,
    pad_token_id: u32,
    image_token_id: u32,
    video_token_id: u32,
    audio_token_id: u32,
}

impl EmbeddingGemma2Model {
    /// Loads the model with every tower the checkpoint has.
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Self::with_modalities(cfg, Modalities::ALL, vb)
    }

    /// Loads the text backbone and the requested towers.
    pub fn with_modalities(cfg: &Config, modalities: Modalities, vb: VarBuilder) -> Result<Self> {
        let text_hidden_size = cfg.text_config.hidden_size;
        let vision = match &cfg.vision_config {
            Some(vision_cfg) if modalities.vision => Some((
                VisionTower::new(vision_cfg, vb.pp("vision_tower"))?,
                MultimodalEmbedder::new(
                    vision_cfg.hidden_size,
                    text_hidden_size,
                    vision_cfg.rms_norm_eps,
                    vb.pp("embed_vision"),
                )?,
            )),
            _ => None,
        };
        let audio = match &cfg.audio_config {
            Some(audio_cfg) if modalities.audio => Some((
                AudioTower::new(audio_cfg, vb.pp("audio_tower"))?,
                MultimodalEmbedder::new(
                    audio_cfg.output_size(),
                    text_hidden_size,
                    audio_cfg.rms_norm_eps,
                    vb.pp("embed_audio"),
                )?,
            )),
            _ => None,
        };
        Ok(Self {
            language_model: TextModel::new(&cfg.text_config, vb.pp("language_model"))?,
            vision,
            audio,
            pad_token_id: cfg.text_config.pad_token_id,
            image_token_id: cfg.image_token_id,
            video_token_id: cfg.video_token_id,
            audio_token_id: cfg.audio_token_id,
        })
    }

    pub fn language_model(&self) -> &TextModel {
        &self.language_model
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    pub fn has_audio(&self) -> bool {
        self.audio.is_some()
    }

    pub fn dtype(&self) -> DType {
        self.language_model.dtype()
    }

    pub fn device(&self) -> &Device {
        self.language_model.device()
    }

    /// Encodes patchified images, or video frames, into soft tokens: one
    /// `(num_soft_tokens, hidden_size)` tensor per image. See [`VisionTower::forward`] for
    /// the inputs.
    pub fn image_features(&self, pixel_values: &Tensor, positions: &Tensor) -> Result<Vec<Tensor>> {
        let Some((tower, embedder)) = &self.vision else {
            candle::bail!("the model was loaded without its vision tower")
        };
        tower
            .forward(pixel_values, positions)?
            .iter()
            .map(|features| embedder.forward(features))
            .collect()
    }

    /// Encodes log-mel spectrograms into soft tokens: one `(num_soft_tokens, hidden_size)`
    /// tensor per clip. See [`AudioTower::forward`] for the inputs.
    pub fn audio_features(
        &self,
        input_features: &Tensor,
        input_features_mask: &Tensor,
    ) -> Result<Vec<Tensor>> {
        let Some((tower, embedder)) = &self.audio else {
            candle::bail!("the model was loaded without its audio tower")
        };
        tower
            .forward(input_features, input_features_mask)?
            .iter()
            .map(|features| embedder.forward(features))
            .collect()
    }

    /// Computes the token embeddings, `(batch, seq_len, embedding_dim)`.
    ///
    /// - `input_ids`: `(batch, seq_len)` token ids, in which every soft token is announced by
    ///   a placeholder token of its modality.
    /// - `attention_mask`: `(batch, seq_len)`, non-zero for the tokens that are not padding.
    ///   Optional when nothing is padded.
    ///
    /// The sentence embedding is the L2 normalized mean of the token embeddings, see
    /// [`mean_pool`] and [`normalize`].
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        soft_tokens: SoftTokens,
    ) -> Result<Tensor> {
        let (b, t) = input_ids.dims2()?;
        let ids: Vec<u32> = input_ids
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let placeholders = [
            (self.image_token_id, soft_tokens.image, "image"),
            (self.video_token_id, soft_tokens.video, "video"),
            (self.audio_token_id, soft_tokens.audio, "audio"),
        ];

        // The placeholders are looked up as padding, their ids may be out of the vocabulary.
        let text_ids: Vec<u32> = ids
            .iter()
            .map(|id| {
                let is_placeholder = placeholders.iter().any(|(token_id, _, _)| token_id == id);
                if is_placeholder {
                    self.pad_token_id
                } else {
                    *id
                }
            })
            .collect();
        let text_ids = Tensor::from_vec(text_ids, (b, t), self.device())?;
        let mut inputs_embeds = self.language_model.embed_tokens(&text_ids)?;

        // Gather every position either from the text embeddings or from the soft tokens
        // appended after them.
        let mut sources: Vec<Tensor> = vec![inputs_embeds.flatten(0, 1)?];
        let mut gather: Vec<u32> = (0..(b * t) as u32).collect();
        let mut num_sources = b * t;
        for (token_id, features, name) in placeholders {
            let slots: Vec<usize> = (0..b * t).filter(|&i| ids[i] == token_id).collect();
            let num_features = match features {
                Some(features) => features.dim(0)?,
                None => 0,
            };
            if slots.len() != num_features {
                candle::bail!(
                    "{} {name} placeholder tokens for {num_features} {name} soft tokens",
                    slots.len()
                )
            }
            let Some(features) = features else { continue };
            for (k, slot) in slots.into_iter().enumerate() {
                gather[slot] = (num_sources + k) as u32;
            }
            num_sources += num_features;
            sources.push(features.to_dtype(inputs_embeds.dtype())?);
        }
        if sources.len() > 1 {
            let gather = Tensor::from_vec(gather, b * t, self.device())?;
            inputs_embeds = Tensor::cat(&sources, 0)?
                .index_select(&gather, 0)?
                .reshape((b, t, ()))?;
        }
        self.language_model
            .forward_embeds(&inputs_embeds, attention_mask)
    }
}

/// Averages `(batch, seq_len, dim)` token embeddings over the positions where the
/// `(batch, seq_len)` attention mask is non-zero. Returns f32 `(batch, dim)`.
pub fn mean_pool(token_embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let mask = attention_mask
        .to_dtype(DType::F32)?
        .to_device(token_embeddings.device())?
        .unsqueeze(D::Minus1)?;
    let summed = token_embeddings
        .to_dtype(DType::F32)?
        .broadcast_mul(&mask)?
        .sum(1)?;
    summed.broadcast_div(&mask.sum(1)?.clamp(1e-9f32, f32::MAX)?)
}

/// L2 normalizes `(batch, dim)` embeddings.
pub fn normalize(embeddings: &Tensor) -> Result<Tensor> {
    let norm = embeddings.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    embeddings.broadcast_div(&norm.clamp(1e-12f32, f32::MAX)?)
}

/// Truncates `(batch, dim)` Matryoshka embeddings to their first `dim` dimensions and
/// renormalizes them. The model is trained for 768, 512, 256 and 128 dimensions.
pub fn truncate(embeddings: &Tensor, dim: usize) -> Result<Tensor> {
    let full = embeddings.dim(D::Minus1)?;
    if dim == 0 || dim > full {
        candle::bail!("cannot truncate {full} dimensional embeddings to {dim} dimensions")
    }
    normalize(&embeddings.narrow(D::Minus1, 0, dim)?)
}
