//! The end-to-end embedding pipeline: preprocessing, the model, pooling and normalization.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use image::DynamicImage;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::audio_features::{AudioFeatureExtractor, AudioFeatures, MAX_SAMPLES};
use crate::config::Config;
use crate::image_processor::{ImagePatches, ImageProcessor};
use crate::model::{mean_pool, normalize, EmbeddingGemma2Model, Modalities, SoftTokens};

/// Repository of the released checkpoint on the Hugging Face hub.
pub const DEFAULT_REPO: &str = "google/embeddinggemma-2";

/// Attention scores (batch x heads x seq_len^2) a batch may take, which bounds the memory of
/// the attention to a few GB. Batches of long inputs are split to stay under it.
const MAX_ATTENTION_SCORES: usize = 1 << 28;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Candle(#[from] candle::Error),
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error(transparent)]
    Hub(#[from] hf_hub::api::sync::ApiError),
    #[error("{}: {source}", path.display())]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{}: {source}", path.display())]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("unknown prompt {name:?}, the model defines: {known}")]
    UnknownPrompt { name: String, known: String },
    #[error("the model was loaded without its {0} tower")]
    MissingTower(&'static str),
    #[error("the tokenizer has no token with id {0}")]
    MissingToken(u32),
    #[error(
        "the input is {len} tokens long, over the limit of {max}; inputs with images, video or audio cannot be truncated"
    )]
    TooLong { len: usize, max: usize },
    #[error("audio must be sampled at {expected} Hz, got {actual} Hz")]
    SamplingRate { expected: usize, actual: usize },
    #[error("{0}")]
    InvalidInput(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;

/// One piece of an input.
#[derive(Debug, Clone)]
pub enum Part {
    Text(String),
    Image(DynamicImage),
    /// The frames of a video, already decoded and sampled; the reference samples one frame
    /// per second. Videos of more than `max_frames` frames are subsampled uniformly.
    Video(Vec<DynamicImage>),
    /// A mono waveform sampled at 16 kHz, with samples in `[-1, 1]`. Clips longer than 30
    /// seconds are truncated.
    Audio(Vec<f32>),
}

/// An input to embed: text, an image, a video, an audio clip, or any sequence of those,
/// which is embedded into a single vector.
#[derive(Debug, Clone, Default)]
pub struct Input {
    parts: Vec<Part>,
}

impl Input {
    pub fn new(parts: Vec<Part>) -> Self {
        Self { parts }
    }

    pub fn text(text: impl Into<String>) -> Self {
        Self::default().with_text(text)
    }

    pub fn image(image: DynamicImage) -> Self {
        Self::default().with_image(image)
    }

    pub fn video(frames: Vec<DynamicImage>) -> Self {
        Self::default().with_video(frames)
    }

    pub fn audio(samples: Vec<f32>) -> Self {
        Self::default().with_audio(samples)
    }

    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.parts.push(Part::Text(text.into()));
        self
    }

    pub fn with_image(mut self, image: DynamicImage) -> Self {
        self.parts.push(Part::Image(image));
        self
    }

    pub fn with_video(mut self, frames: Vec<DynamicImage>) -> Self {
        self.parts.push(Part::Video(frames));
        self
    }

    pub fn with_audio(mut self, samples: Vec<f32>) -> Self {
        self.parts.push(Part::Audio(samples));
        self
    }

    pub fn parts(&self) -> &[Part] {
        &self.parts
    }
}

impl From<&str> for Input {
    fn from(text: &str) -> Self {
        Self::text(text)
    }
}

impl From<String> for Input {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

impl From<DynamicImage> for Input {
    fn from(image: DynamicImage) -> Self {
        Self::image(image)
    }
}

/// How to load the model.
#[derive(Debug, Clone)]
pub struct Options {
    pub device: Device,
    /// Precision of the weights. Defaults to f32. bf16 halves the memory but needs a GPU;
    /// f16 is rejected, its range is too narrow for this model.
    pub dtype: DType,
    /// Towers to load. Loading only the ones in use saves memory.
    pub modalities: Modalities,
    /// Most inputs to run through the model at once. Batches of long inputs are smaller, to
    /// bound the memory of the attention.
    pub batch_size: usize,
    /// Longest input, in tokens. Longer texts are truncated; the reference implementation
    /// does not truncate, this is a guard for the quadratic cost of the attention.
    pub max_seq_len: usize,
    /// Revision of the hub repository.
    pub revision: Option<String>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
            modalities: Modalities::ALL,
            batch_size: 8,
            max_seq_len: 8192,
            revision: None,
        }
    }
}

impl Options {
    /// Runs on the first CUDA or Metal device when candle was built with support for it,
    /// on the CPU otherwise.
    pub fn with_best_device(mut self) -> Result<Self> {
        self.device = if candle::utils::cuda_is_available() {
            Device::new_cuda(0)?
        } else if candle::utils::metal_is_available() {
            Device::new_metal(0)?
        } else {
            Device::Cpu
        };
        Ok(self)
    }
}

#[derive(Debug, Default, Deserialize)]
struct SentenceTransformersConfig {
    #[serde(default)]
    prompts: HashMap<String, String>,
}

fn default_max_frames() -> usize {
    32
}

#[derive(Debug, Deserialize)]
struct VideoProcessorConfig {
    #[serde(flatten)]
    frames: ImageProcessor,
    #[serde(default = "default_max_frames")]
    max_frames: usize,
}

impl Default for VideoProcessorConfig {
    fn default() -> Self {
        Self {
            frames: ImageProcessor {
                max_soft_tokens: 140,
                ..Default::default()
            },
            max_frames: default_max_frames(),
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct ProcessorConfig {
    #[serde(default)]
    image_processor: ImageProcessor,
    #[serde(default)]
    video_processor: VideoProcessorConfig,
    #[serde(default)]
    feature_extractor: AudioFeatureExtractor,
}

/// The strings the placeholders of each modality are written with.
#[derive(Debug, Clone)]
struct SpecialTokens {
    boi: String,
    eoi: String,
    image: String,
    video: String,
    boa: String,
    eoa: String,
    audio: String,
}

/// EmbeddingGemma 2, from raw inputs to normalized embeddings.
pub struct EmbeddingGemma2 {
    model: EmbeddingGemma2Model,
    tokenizer: Tokenizer,
    image_processor: ImageProcessor,
    video_processor: ImageProcessor,
    max_frames: usize,
    audio_extractor: AudioFeatureExtractor,
    prompts: HashMap<String, String>,
    special_tokens: SpecialTokens,
    eos_token_id: u32,
    pad_token_id: u32,
    batch_size: usize,
    max_seq_len: usize,
    text_heads: usize,
    vision_heads: usize,
}

/// An input after preprocessing, ready to be batched.
struct Prepared {
    ids: Vec<u32>,
    /// Images and video frames, in the order of their placeholders.
    images: Vec<ImagePatches>,
    frames: Vec<ImagePatches>,
    /// Waveforms, in the order of their placeholders.
    audio: Vec<Vec<f32>>,
}

fn read(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_slice(&read(path)?).map_err(|source| Error::Json {
        path: path.to_path_buf(),
        source,
    })
}

/// Reads an optional file of the checkpoint, falling back to the defaults without it.
fn read_json_or_default<T: serde::de::DeserializeOwned + Default>(
    path: Option<&Path>,
) -> Result<T> {
    match path {
        Some(path) if path.exists() => read_json(path),
        _ => Ok(T::default()),
    }
}

/// Paths of the files of a checkpoint.
struct Files {
    config: PathBuf,
    weights: PathBuf,
    tokenizer: PathBuf,
    sentence_transformers_config: Option<PathBuf>,
    processor_config: Option<PathBuf>,
}

impl EmbeddingGemma2 {
    /// Loads the model from a repository of the Hugging Face hub, downloading it to the local
    /// cache if needed. See [`DEFAULT_REPO`].
    pub fn from_pretrained(repo_id: &str, options: Options) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = match &options.revision {
            Some(revision) => api.repo(hf_hub::Repo::with_revision(
                repo_id.to_string(),
                hf_hub::RepoType::Model,
                revision.clone(),
            )),
            None => api.model(repo_id.to_string()),
        };
        let files = Files {
            config: repo.get("config.json")?,
            weights: repo.get("model.safetensors")?,
            tokenizer: repo.get("tokenizer.json")?,
            sentence_transformers_config: repo.get("config_sentence_transformers.json").ok(),
            processor_config: repo.get("processor_config.json").ok(),
        };
        Self::from_files(files, options)
    }

    /// Loads the model from a local directory laid out like the hub repository.
    pub fn from_dir(dir: impl AsRef<Path>, options: Options) -> Result<Self> {
        let dir = dir.as_ref();
        let files = Files {
            config: dir.join("config.json"),
            weights: dir.join("model.safetensors"),
            tokenizer: dir.join("tokenizer.json"),
            sentence_transformers_config: Some(dir.join("config_sentence_transformers.json")),
            processor_config: Some(dir.join("processor_config.json")),
        };
        Self::from_files(files, options)
    }

    fn from_files(files: Files, options: Options) -> Result<Self> {
        if options.dtype == DType::F16 {
            return Err(Error::InvalidInput(
                "f16 overflows on this model, use bf16 or f32",
            ));
        }
        if options.dtype == DType::BF16 && options.device.is_cpu() {
            return Err(Error::InvalidInput(
                "candle has no bf16 matrix product on the CPU, use f32",
            ));
        }
        if options.batch_size == 0 || options.max_seq_len < 2 {
            return Err(Error::InvalidInput(
                "batch_size must be at least 1 and max_seq_len at least 2",
            ));
        }
        let config: Config = read_json(&files.config)?;
        let st_config: SentenceTransformersConfig =
            read_json_or_default(files.sentence_transformers_config.as_deref())?;
        let processor: ProcessorConfig = read_json_or_default(files.processor_config.as_deref())?;
        let tokenizer = Tokenizer::from_file(&files.tokenizer)?;

        // Safety: the weights are memory mapped, the file must not be modified while the
        // model is alive.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&files.weights], options.dtype, &options.device)?
        };
        let model = EmbeddingGemma2Model::with_modalities(&config, options.modalities, vb)?;

        let token = |id: u32| tokenizer.id_to_token(id).ok_or(Error::MissingToken(id));
        let special_tokens = SpecialTokens {
            boi: token(config.boi_token_id)?,
            eoi: token(config.eoi_token_id)?,
            image: token(config.image_token_id)?,
            video: token(config.video_token_id)?,
            boa: token(config.boa_token_id)?,
            eoa: token(config.eoa_token_id)?,
            audio: token(config.audio_token_id)?,
        };
        Ok(Self {
            model,
            tokenizer,
            image_processor: processor.image_processor,
            video_processor: processor.video_processor.frames,
            max_frames: processor.video_processor.max_frames,
            audio_extractor: processor.feature_extractor,
            prompts: st_config.prompts,
            special_tokens,
            eos_token_id: config.text_config.eos_token_id,
            pad_token_id: config.text_config.pad_token_id,
            batch_size: options.batch_size,
            max_seq_len: options.max_seq_len,
            text_heads: config.text_config.num_attention_heads,
            vision_heads: config
                .vision_config
                .as_ref()
                .map_or(1, |vision| vision.num_attention_heads),
        })
    }

    pub fn model(&self) -> &EmbeddingGemma2Model {
        &self.model
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Sampling rate audio clips have to be in, in Hz.
    pub fn sampling_rate(&self) -> usize {
        self.audio_extractor.sampling_rate
    }

    /// The task prompts the model was trained with, by name: `SearchQuery`, `Document`,
    /// `QuestionAnswering`, `Classification`, ...
    pub fn prompts(&self) -> &HashMap<String, String> {
        &self.prompts
    }

    /// Looks up a task prompt by name.
    pub fn prompt(&self, name: &str) -> Result<&str> {
        self.prompts.get(name).map(String::as_str).ok_or_else(|| {
            let mut known: Vec<&str> = self.prompts.keys().map(String::as_str).collect();
            known.sort_unstable();
            Error::UnknownPrompt {
                name: name.to_string(),
                known: known.join(", "),
            }
        })
    }

    /// Embeds texts with the task prompt named `prompt_name`. Returns normalized f32
    /// embeddings, `(texts.len(), embedding_dim)`.
    pub fn encode_text<S: AsRef<str>>(&self, texts: &[S], prompt_name: &str) -> Result<Tensor> {
        let prompt = self.prompt(prompt_name)?;
        let inputs: Vec<Input> = texts
            .iter()
            .map(|text| Input::text(text.as_ref()))
            .collect();
        self.encode(&inputs, Some(prompt))
    }

    /// Embeds search queries, with the `SearchQuery` prompt.
    pub fn encode_queries<S: AsRef<str>>(&self, queries: &[S]) -> Result<Tensor> {
        self.encode_text(queries, "SearchQuery")
    }

    /// Embeds untitled documents, with the `Document` prompt. Format documents that have a
    /// title as `title: {title} | text: {content}` and pass them to [`Self::encode`].
    pub fn encode_documents<S: AsRef<str>>(&self, documents: &[S]) -> Result<Tensor> {
        self.encode_text(documents, "Document")
    }

    /// Embeds inputs of any modality. `prompt` is written before each input; pass the task
    /// prompts (see [`Self::prompt`]) for text and `None` for images, video and audio, which
    /// were trained without one.
    ///
    /// Returns normalized f32 embeddings, `(inputs.len(), embedding_dim)`, on the device of
    /// the model. Their dot product is their cosine similarity; see [`crate::truncate`] for
    /// smaller Matryoshka embeddings.
    pub fn encode(&self, inputs: &[Input], prompt: Option<&str>) -> Result<Tensor> {
        if inputs.is_empty() {
            return Err(Error::InvalidInput("no inputs to encode"));
        }
        let prepared = inputs
            .iter()
            .map(|input| self.prepare(input, prompt))
            .collect::<Result<Vec<_>>>()?;

        // Batch inputs of similar lengths together, to limit the padding.
        let mut order: Vec<usize> = (0..prepared.len()).collect();
        order.sort_by_key(|&idx| std::cmp::Reverse(prepared[idx].ids.len()));
        let mut embeddings: Vec<Option<Tensor>> = vec![None; prepared.len()];
        let lengths: Vec<usize> = order.iter().map(|&idx| prepared[idx].ids.len()).collect();
        let mut start = 0;
        for size in batch_sizes(&lengths, self.text_heads, self.batch_size) {
            let batch = &order[start..start + size];
            start += size;
            let items: Vec<&Prepared> = batch.iter().map(|&idx| &prepared[idx]).collect();
            let batch_embeddings = self.encode_batch(&items)?;
            for (row, &idx) in batch.iter().enumerate() {
                embeddings[idx] = Some(batch_embeddings.get(row)?);
            }
        }
        let embeddings: Vec<Tensor> = embeddings.into_iter().flatten().collect();
        Ok(Tensor::stack(&embeddings, 0)?)
    }

    /// Cosine similarities between two sets of normalized embeddings, `(a.len(), b.len())`.
    pub fn similarity(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        Ok(a.matmul(&b.t()?)?)
    }

    /// Renders an input to token ids, with a run of placeholder tokens for every image,
    /// video frame and audio clip, and preprocesses its media.
    fn prepare(&self, input: &Input, prompt: Option<&str>) -> Result<Prepared> {
        let tokens = &self.special_tokens;
        let mut text = prompt.unwrap_or_default().to_string();
        let mut prepared = Prepared {
            ids: Vec::new(),
            images: Vec::new(),
            frames: Vec::new(),
            audio: Vec::new(),
        };
        for part in &input.parts {
            match part {
                Part::Text(part) => text.push_str(part),
                Part::Image(image) => {
                    if !self.model.has_vision() {
                        return Err(Error::MissingTower("vision"));
                    }
                    let patches = self.image_processor.preprocess(image)?;
                    text.push_str(&tokens.boi);
                    text.push_str(&tokens.image.repeat(patches.num_soft_tokens));
                    text.push_str(&tokens.eoi);
                    prepared.images.push(patches);
                }
                Part::Video(frames) => {
                    if !self.model.has_vision() {
                        return Err(Error::MissingTower("vision"));
                    }
                    if frames.is_empty() {
                        return Err(Error::InvalidInput("a video needs at least one frame"));
                    }
                    for idx in sample_uniformly(frames.len(), self.max_frames) {
                        let patches = self.video_processor.preprocess(&frames[idx])?;
                        text.push_str(&tokens.boi);
                        text.push_str(&tokens.video.repeat(patches.num_soft_tokens));
                        text.push_str(&tokens.eoi);
                        prepared.frames.push(patches);
                    }
                }
                Part::Audio(samples) => {
                    if !self.model.has_audio() {
                        return Err(Error::MissingTower("audio"));
                    }
                    let samples = &samples[..samples.len().min(MAX_SAMPLES)];
                    let num_soft_tokens = self.audio_extractor.num_soft_tokens(samples.len());
                    if num_soft_tokens == 0 {
                        return Err(Error::InvalidInput("the audio clip is too short"));
                    }
                    text.push_str(&tokens.boa);
                    text.push_str(&tokens.audio.repeat(num_soft_tokens));
                    text.push_str(&tokens.eoa);
                    prepared.audio.push(samples.to_vec());
                }
            }
        }
        let mut ids = self.tokenizer.encode(text, true)?.get_ids().to_vec();
        if ids.len() > self.max_seq_len {
            let has_media = !(prepared.images.is_empty()
                && prepared.frames.is_empty()
                && prepared.audio.is_empty());
            if has_media {
                return Err(Error::TooLong {
                    len: ids.len(),
                    max: self.max_seq_len,
                });
            }
            ids.truncate(self.max_seq_len - 1);
            ids.push(self.eos_token_id);
        }
        prepared.ids = ids;
        Ok(prepared)
    }

    fn encode_batch(&self, items: &[&Prepared]) -> Result<Tensor> {
        let device = self.model.device();
        let seq_len = items.iter().map(|item| item.ids.len()).max().unwrap_or(0);
        let mut ids = Vec::with_capacity(items.len() * seq_len);
        let mut mask = Vec::with_capacity(items.len() * seq_len);
        for item in items {
            let padding = seq_len - item.ids.len();
            ids.extend_from_slice(&item.ids);
            ids.extend(std::iter::repeat_n(self.pad_token_id, padding));
            mask.extend(std::iter::repeat_n(1u8, item.ids.len()));
            mask.extend(std::iter::repeat_n(0u8, padding));
        }
        let input_ids = Tensor::from_vec(ids, (items.len(), seq_len), device)?;
        let attention_mask = Tensor::from_vec(mask, (items.len(), seq_len), device)?;

        let images: Vec<&ImagePatches> = items.iter().flat_map(|item| &item.images).collect();
        let frames: Vec<&ImagePatches> = items.iter().flat_map(|item| &item.frames).collect();
        let audio: Vec<&[f32]> = items
            .iter()
            .flat_map(|item| item.audio.iter().map(Vec::as_slice))
            .collect();
        let image_tokens = self.vision_soft_tokens(&self.image_processor, &images)?;
        let video_tokens = self.vision_soft_tokens(&self.video_processor, &frames)?;
        let audio_tokens = self.audio_soft_tokens(&audio)?;

        let soft_tokens = SoftTokens {
            image: image_tokens.as_ref(),
            video: video_tokens.as_ref(),
            audio: audio_tokens.as_ref(),
        };
        let token_embeddings =
            self.model
                .forward(&input_ids, Some(&attention_mask), soft_tokens)?;
        Ok(normalize(&mean_pool(&token_embeddings, &attention_mask)?)?)
    }

    /// Runs images through the vision tower, a few at a time.
    fn vision_soft_tokens(
        &self,
        processor: &ImageProcessor,
        images: &[&ImagePatches],
    ) -> Result<Option<Tensor>> {
        if images.is_empty() {
            return Ok(None);
        }
        let lengths: Vec<usize> = images.iter().map(|image| image.num_patches()).collect();
        let mut soft_tokens = Vec::with_capacity(images.len());
        let mut start = 0;
        for size in batch_sizes(&lengths, self.vision_heads, self.batch_size) {
            let chunk = &images[start..start + size];
            start += size;
            let (pixel_values, positions) = processor.collate(chunk, self.model.device())?;
            soft_tokens.extend(self.model.image_features(&pixel_values, &positions)?);
        }
        Ok(Some(Tensor::cat(&soft_tokens, 0)?))
    }

    /// Runs audio clips through the audio tower, `batch_size` at a time.
    fn audio_soft_tokens(&self, clips: &[&[f32]]) -> Result<Option<Tensor>> {
        if clips.is_empty() {
            return Ok(None);
        }
        let mut soft_tokens = Vec::with_capacity(clips.len());
        for chunk in clips.chunks(self.batch_size) {
            let padded_len = chunk.iter().map(|clip| clip.len()).max();
            let features = chunk
                .iter()
                .map(|clip| self.audio_extractor.extract(clip, padded_len))
                .collect::<candle::Result<Vec<AudioFeatures>>>()?;
            let features: Vec<&AudioFeatures> = features.iter().collect();
            let (input_features, mask) = self
                .audio_extractor
                .collate(&features, self.model.device())?;
            soft_tokens.extend(self.model.audio_features(&input_features, &mask)?);
        }
        Ok(Some(Tensor::cat(&soft_tokens, 0)?))
    }
}

/// Splits sequences of the given lengths, in order, into consecutive batches of at most
/// `batch_size`, cut short where padding the batch to its longest sequence would take more
/// than [`MAX_ATTENTION_SCORES`]. A sequence over that budget is run on its own.
fn batch_sizes(lengths: &[usize], num_heads: usize, batch_size: usize) -> Vec<usize> {
    let mut sizes = Vec::new();
    let (mut size, mut longest) = (0, 0);
    for &len in lengths {
        let padded = longest.max(len);
        let scores = (size + 1) * num_heads * padded * padded;
        if size > 0 && (size == batch_size || scores > MAX_ATTENTION_SCORES) {
            sizes.push(size);
            (size, longest) = (1, len);
        } else {
            (size, longest) = (size + 1, padded);
        }
    }
    if size > 0 {
        sizes.push(size);
    }
    sizes
}

/// Indices of at most `max` out of `len` items, evenly spread from the first to the last.
fn sample_uniformly(len: usize, max: usize) -> Vec<usize> {
    if len <= max {
        return (0..len).collect();
    }
    if max == 1 {
        return vec![0];
    }
    (0..max).map(|i| i * (len - 1) / (max - 1)).collect()
}

#[cfg(test)]
mod tests {
    use super::{batch_sizes, sample_uniformly};

    #[test]
    fn batches_shrink_for_long_sequences() {
        // Short sequences fill the batches.
        assert_eq!(batch_sizes(&[100; 20], 4, 8), vec![8, 8, 4]);
        // Four heads over 8192 tokens are 2^28 scores: such a sequence runs alone.
        assert_eq!(
            batch_sizes(&[8192, 8192, 4096, 4096, 16], 4, 8),
            vec![1, 1, 3]
        );
        assert_eq!(batch_sizes(&[20_000, 10], 4, 8), vec![1, 1]);
        // Twelve heads over the 2520 patches of a full resolution image: three at a time.
        assert_eq!(batch_sizes(&[2520; 7], 12, 8), vec![3, 3, 1]);
        assert!(batch_sizes(&[], 4, 8).is_empty());
    }

    #[test]
    fn frames_are_sampled_from_first_to_last() {
        assert_eq!(sample_uniformly(3, 32), vec![0, 1, 2]);
        assert_eq!(sample_uniformly(10, 4), vec![0, 3, 6, 9]);
        assert_eq!(sample_uniformly(100, 32).len(), 32);
        assert_eq!(sample_uniformly(100, 32).last(), Some(&99));
    }
}
