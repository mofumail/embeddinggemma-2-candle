//! Configuration structs, deserialized from the checkpoint's `config.json`.

use std::collections::HashMap;

use candle_nn::Activation;
use serde::Deserialize;

/// Attention pattern of a text encoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    /// Bidirectional attention restricted to `|q - k| <= sliding_window`.
    SlidingAttention,
    /// Bidirectional attention over the whole sequence.
    FullAttention,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct RopeParams {
    pub rope_theta: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct TextRopeParameters {
    pub full_attention: RopeParams,
    pub sliding_attention: RopeParams,
}

impl Default for TextRopeParameters {
    fn default() -> Self {
        Self {
            full_attention: RopeParams {
                rope_theta: 1_000_000.0,
            },
            sliding_attention: RopeParams {
                rope_theta: 10_000.0,
            },
        }
    }
}

/// Per-layer overrides of the attention geometry. Full-attention layers are wider than the
/// sliding ones, so they carry their own head dimension and key/value head count.
#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
pub struct LayerOverride {
    pub head_dim: Option<usize>,
    pub num_key_value_heads: Option<usize>,
}

/// Attention geometry of one layer once the overrides are applied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerGeometry {
    pub layer_type: LayerType,
    pub head_dim: usize,
    pub num_key_value_heads: usize,
    pub rope_theta: f64,
}

fn default_vocab_size() -> usize {
    262_144
}
fn default_text_hidden_size() -> usize {
    512
}
fn default_text_intermediate_size() -> usize {
    2048
}
fn default_text_num_hidden_layers() -> usize {
    24
}
fn default_text_num_attention_heads() -> usize {
    4
}
fn default_text_num_key_value_heads() -> usize {
    2
}
fn default_text_head_dim() -> usize {
    256
}
fn default_gelu_tanh() -> Activation {
    Activation::GeluPytorchTanh
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_eos_token_id() -> u32 {
    1
}
fn default_sliding_window() -> usize {
    512
}
fn default_embedding_dim() -> usize {
    768
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TextConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_text_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_text_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_text_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_text_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_text_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_text_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_gelu_tanh")]
    pub hidden_activation: Activation,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub pad_token_id: u32,
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: u32,
    #[serde(default)]
    pub attention_bias: bool,
    /// Inclusive radius of the bidirectional window of the sliding layers.
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    pub layer_types: Vec<LayerType>,
    #[serde(default = "default_text_hidden_size")]
    pub hidden_size_per_layer_input: usize,
    #[serde(default = "default_embedding_dim")]
    pub embedding_dim: usize,
    #[serde(default)]
    pub rope_parameters: TextRopeParameters,
    /// Keyed by the layer index, zero padded in the released checkpoint (`"05"`).
    #[serde(default)]
    pub per_layer_config: HashMap<String, LayerOverride>,
}

impl TextConfig {
    /// Resolves the attention geometry of layer `layer_idx`.
    pub fn layer_geometry(&self, layer_idx: usize) -> LayerGeometry {
        let layer_type = self.layer_types[layer_idx];
        let overrides = self
            .per_layer_config
            .iter()
            .find(|(key, _)| key.parse::<usize>() == Ok(layer_idx))
            .map(|(_, overrides)| *overrides)
            .unwrap_or_default();
        let rope_theta = match layer_type {
            LayerType::FullAttention => self.rope_parameters.full_attention.rope_theta,
            LayerType::SlidingAttention => self.rope_parameters.sliding_attention.rope_theta,
        };
        LayerGeometry {
            layer_type,
            head_dim: overrides.head_dim.unwrap_or(self.head_dim),
            num_key_value_heads: overrides
                .num_key_value_heads
                .unwrap_or(self.num_key_value_heads),
            rope_theta,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct VisionRopeParameters {
    pub rope_theta: f64,
}

impl Default for VisionRopeParameters {
    fn default() -> Self {
        Self { rope_theta: 100.0 }
    }
}

fn default_vision_hidden_size() -> usize {
    768
}
fn default_vision_intermediate_size() -> usize {
    3072
}
fn default_vision_num_hidden_layers() -> usize {
    16
}
fn default_vision_num_heads() -> usize {
    12
}
fn default_vision_head_dim() -> usize {
    64
}
fn default_patch_size() -> usize {
    16
}
fn default_pooling_kernel_size() -> usize {
    3
}
fn default_position_embedding_size() -> usize {
    10_240
}

/// Configuration of the Gemma 4 vision tower.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_vision_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_vision_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_vision_num_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_vision_num_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_vision_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_gelu_tanh")]
    pub hidden_activation: Activation,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_pooling_kernel_size")]
    pub pooling_kernel_size: usize,
    #[serde(default = "default_position_embedding_size")]
    pub position_embedding_size: usize,
    #[serde(default)]
    pub rope_parameters: VisionRopeParameters,
    #[serde(default)]
    pub standardize: bool,
    #[serde(default)]
    pub use_clipped_linears: bool,
}

fn default_audio_hidden_size() -> usize {
    1024
}
fn default_audio_num_heads() -> usize {
    8
}
fn default_audio_num_hidden_layers() -> usize {
    12
}
fn default_attention_chunk_size() -> usize {
    12
}
fn default_attention_context_left() -> usize {
    13
}
fn default_attention_invalid_logits_value() -> f64 {
    -1e9
}
fn default_attention_logit_cap() -> f64 {
    50.0
}
fn default_conv_kernel_size() -> usize {
    5
}
fn default_gradient_clipping() -> f64 {
    1e10
}
fn default_residual_weight() -> f64 {
    0.5
}
fn default_subsampling_conv_channels() -> Vec<usize> {
    vec![128, 32]
}
fn default_silu() -> Activation {
    Activation::Silu
}
fn default_true() -> bool {
    true
}

/// Configuration of the Gemma 4 audio tower (a chunked-attention conformer).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AudioConfig {
    #[serde(default = "default_audio_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_audio_num_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_audio_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_silu")]
    pub hidden_act: Activation,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_attention_chunk_size")]
    pub attention_chunk_size: usize,
    #[serde(default = "default_attention_context_left")]
    pub attention_context_left: usize,
    #[serde(default)]
    pub attention_context_right: usize,
    #[serde(default = "default_attention_invalid_logits_value")]
    pub attention_invalid_logits_value: f64,
    #[serde(default = "default_attention_logit_cap")]
    pub attention_logit_cap: f64,
    #[serde(default = "default_conv_kernel_size")]
    pub conv_kernel_size: usize,
    #[serde(default = "default_gradient_clipping")]
    pub gradient_clipping: f64,
    #[serde(default = "default_residual_weight")]
    pub residual_weight: f64,
    #[serde(default = "default_subsampling_conv_channels")]
    pub subsampling_conv_channels: Vec<usize>,
    /// Width of the tower's output; falls back to `hidden_size` when absent.
    #[serde(default)]
    pub output_proj_dims: Option<usize>,
    #[serde(default = "default_true")]
    pub use_clipped_linears: bool,
}

impl AudioConfig {
    pub fn output_size(&self) -> usize {
        self.output_proj_dims.unwrap_or(self.hidden_size)
    }
}

fn default_image_token_id() -> u32 {
    258_880
}
fn default_audio_token_id() -> u32 {
    258_881
}
fn default_video_token_id() -> u32 {
    258_884
}
fn default_boi_token_id() -> u32 {
    255_999
}
fn default_eoi_token_id() -> u32 {
    258_882
}
fn default_boa_token_id() -> u32 {
    256_000
}
fn default_eoa_token_id() -> u32 {
    258_883
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    #[serde(default)]
    pub vision_config: Option<VisionConfig>,
    #[serde(default)]
    pub audio_config: Option<AudioConfig>,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: u32,
    #[serde(default = "default_audio_token_id")]
    pub audio_token_id: u32,
    #[serde(default = "default_video_token_id")]
    pub video_token_id: u32,
    /// Opens the soft tokens of an image or of a video frame.
    #[serde(default = "default_boi_token_id")]
    pub boi_token_id: u32,
    #[serde(default = "default_eoi_token_id")]
    pub eoi_token_id: u32,
    /// Opens the soft tokens of an audio clip.
    #[serde(default = "default_boa_token_id")]
    pub boa_token_id: u32,
    #[serde(default = "default_eoa_token_id", alias = "eoa_token_index")]
    pub eoa_token_id: u32,
}
