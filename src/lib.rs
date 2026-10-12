//! A [candle](https://github.com/huggingface/candle) implementation of
//! [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), Google's multimodal
//! embedding model: text, images, video and audio are mapped into one 768-dimensional space.
//!
//! [`EmbeddingGemma2`] is the pipeline, from raw inputs to normalized embeddings:
//!
//! ```no_run
//! use embeddinggemma2::{EmbeddingGemma2, Input, Options, DEFAULT_REPO};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let embedder = EmbeddingGemma2::from_pretrained(DEFAULT_REPO, Options::default())?;
//!
//! // Text is embedded with a task prompt.
//! let queries = embedder.encode_queries(&["What causes the northern lights?"])?;
//! let documents = embedder.encode_documents(&[
//!     "Charged particles from the sun cause them.",
//!     "The stock market closed higher today.",
//! ])?;
//! // Images, video frames and audio are embedded without one, alone or mixed with text.
//! let photo = Input::image(image::open("aurora.jpg")?);
//! let images = embedder.encode(&[photo], None)?;
//!
//! // The embeddings are normalized: their dot products are cosine similarities.
//! let similarities = EmbeddingGemma2::similarity(&queries, &documents)?;
//! println!("{similarities}");
//! # Ok(())
//! # }
//! ```
//!
//! The layers underneath are public for callers that bring their own preprocessing:
//! [`EmbeddingGemma2Model`] combines the [`text`] backbone with the [`vision`] and [`audio`]
//! towers, and [`image_processor`] and [`audio_features`] prepare their inputs.

pub mod audio;
pub mod audio_features;
pub mod config;
pub mod embedder;
pub mod image_processor;
mod layers;
pub mod model;
pub mod text;
pub mod vision;

pub use config::Config;
pub use embedder::{EmbeddingGemma2, Error, Input, Options, Part, DEFAULT_REPO};
pub use model::{
    mean_pool, normalize, truncate, EmbeddingGemma2Model, Modalities, MultimodalEmbedder,
    SoftTokens,
};
