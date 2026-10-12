//! Ranks documents, images and audio clips against text queries.
//!
//! ```text
//! cargo run --release --example embed -- \
//!     --query "What causes the northern lights?" \
//!     --document "Charged particles from the sun cause auroras." \
//!     --document "The stock market closed higher today." \
//!     --image aurora.jpg --audio talk.wav
//! ```

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle::{DType, Device};
use clap::Parser;
use embeddinggemma2::{truncate, EmbeddingGemma2, Input, Modalities, Options, DEFAULT_REPO};

#[derive(Parser, Debug)]
struct Args {
    /// A search query; repeat for several.
    #[arg(long, required = true)]
    query: Vec<String>,

    /// A text document to rank; repeat for several.
    #[arg(long)]
    document: Vec<String>,

    /// An image file to rank; repeat for several.
    #[arg(long)]
    image: Vec<PathBuf>,

    /// A 16 kHz WAV file to rank; repeat for several.
    #[arg(long)]
    audio: Vec<PathBuf>,

    /// Truncate the embeddings to this many dimensions (768, 512, 256 or 128).
    #[arg(long)]
    dim: Option<usize>,

    /// Hub repository, or local directory, to load the model from.
    #[arg(long, default_value = DEFAULT_REPO)]
    model: String,

    /// Run on the CPU even when a GPU is available.
    #[arg(long)]
    cpu: bool,

    /// Load the weights in bf16 rather than f32.
    #[arg(long)]
    bf16: bool,
}

/// Reads a WAV file as mono samples in [-1, 1].
fn read_wav(path: &Path, sampling_rate: usize) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate as usize != sampling_rate {
        bail!(
            "expected audio sampled at {sampling_rate} Hz, got {} Hz",
            spec.sample_rate
        );
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|sample| sample as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    };
    let channels = usize::from(spec.channels);
    Ok(samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.document.is_empty() && args.image.is_empty() && args.audio.is_empty() {
        bail!("nothing to rank: pass --document, --image or --audio");
    }

    let mut options = Options {
        modalities: Modalities {
            vision: !args.image.is_empty(),
            audio: !args.audio.is_empty(),
        },
        ..Default::default()
    };
    if !args.cpu {
        options = options.with_best_device()?;
    }
    if args.bf16 {
        options.dtype = DType::BF16;
    }
    let device = match &options.device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "metal",
    };
    eprintln!("loading {} on {device} in {:?}", args.model, options.dtype);
    let embedder = if Path::new(&args.model).is_dir() {
        EmbeddingGemma2::from_dir(&args.model, options)?
    } else {
        EmbeddingGemma2::from_pretrained(&args.model, options)?
    };

    let mut labels: Vec<String> = Vec::new();
    let mut candidates = Vec::new();
    if !args.document.is_empty() {
        candidates.push(embedder.encode_documents(&args.document)?);
        labels.extend(args.document.iter().cloned());
    }
    // Images and audio are embedded without a prompt.
    let mut media = Vec::new();
    for path in &args.image {
        let image = image::open(path).with_context(|| format!("reading {}", path.display()))?;
        media.push(Input::image(image));
        labels.push(path.display().to_string());
    }
    for path in &args.audio {
        let samples = read_wav(path, embedder.sampling_rate())
            .with_context(|| format!("reading {}", path.display()))?;
        media.push(Input::audio(samples));
        labels.push(path.display().to_string());
    }
    if !media.is_empty() {
        candidates.push(embedder.encode(&media, None)?);
    }
    let mut candidates = candle::Tensor::cat(&candidates, 0)?;
    let mut queries = embedder.encode_queries(&args.query)?;
    if let Some(dim) = args.dim {
        candidates = truncate(&candidates, dim)?;
        queries = truncate(&queries, dim)?;
    }

    let similarities = EmbeddingGemma2::similarity(&queries, &candidates)?.to_vec2::<f32>()?;
    for (query, similarities) in args.query.iter().zip(similarities) {
        println!("{query}");
        let mut ranking: Vec<(f32, &String)> = similarities.into_iter().zip(&labels).collect();
        ranking.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (similarity, label) in ranking {
            println!("  {similarity:.4}  {label}");
        }
    }
    Ok(())
}
