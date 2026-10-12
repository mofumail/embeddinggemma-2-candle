//! Compares the candle implementation against activations dumped from the reference
//! `transformers` implementation by `scripts/dump_reference.py`.
//!
//! The tests need the checkpoint and the dumps, so they only run when
//! `EMBEDDINGGEMMA2_REFDATA` points at the directory holding the dumps.

use std::collections::HashMap;
use std::path::PathBuf;

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use embeddinggemma2::audio_features::AudioFeatureExtractor;
use embeddinggemma2::config::Config;
use embeddinggemma2::image_processor::ImageProcessor;
use embeddinggemma2::text::TextModel;
use embeddinggemma2::{
    mean_pool, normalize, EmbeddingGemma2, EmbeddingGemma2Model, Input, Options, SoftTokens,
    DEFAULT_REPO,
};

fn refdata() -> Option<PathBuf> {
    std::env::var_os("EMBEDDINGGEMMA2_REFDATA").map(PathBuf::from)
}

fn checkpoint() -> anyhow::Result<(Config, PathBuf)> {
    let repo = hf_hub::api::sync::Api::new()?.model("google/embeddinggemma-2".to_string());
    let config = serde_json::from_slice(&std::fs::read(repo.get("config.json")?)?)?;
    Ok((config, repo.get("model.safetensors")?))
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
    let diff = (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?.abs()?;
    Ok(diff.flatten_all()?.max(0)?.to_scalar::<f32>()?)
}

fn max_abs(a: &Tensor) -> anyhow::Result<f32> {
    Ok(a.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?)
}

fn load(path: PathBuf) -> anyhow::Result<HashMap<String, Tensor>> {
    Ok(candle::safetensors::load(path, &Device::Cpu)?)
}

#[test]
fn text_matches_reference() -> anyhow::Result<()> {
    let Some(refdata) = refdata() else {
        eprintln!("EMBEDDINGGEMMA2_REFDATA is not set, skipping");
        return Ok(());
    };
    let (config, weights) = checkpoint()?;
    let device = Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };
    let model = TextModel::new(&config.text_config, vb.pp("language_model"))?;

    for case in ["query", "doc", "batch", "long"] {
        let reference = load(refdata.join(format!("text_{case}.safetensors")))?;
        let input_ids = reference["input_ids"].to_dtype(DType::U32)?;
        let attention_mask = &reference["attention_mask"];
        let inputs_embeds = model.embed_tokens(&input_ids)?;
        let output =
            model.forward_embeds_with_hidden_states(&inputs_embeds, Some(attention_mask))?;

        // The reference reports the projected output in place of the last layer's.
        let num_layers = output.hidden_states.len() - 1;
        for (idx, hidden) in output.hidden_states.iter().take(num_layers).enumerate() {
            let expected = &reference[&format!("hidden_{idx:02}")];
            let diff = max_abs_diff(hidden, expected)?;
            let scale = max_abs(expected)?;
            println!("{case} hidden {idx:02}: max abs diff {diff:.3e} (scale {scale:.1})");
            assert!(
                diff <= 2e-3 * scale.max(1.),
                "{case}: hidden state {idx} diverges"
            );
        }
        let expected = &reference["last_hidden_state"];
        let diff = max_abs_diff(&output.last_hidden_state, expected)?;
        println!("{case} last_hidden_state: max abs diff {diff:.3e}");
        assert!(
            diff <= 2e-3 * max_abs(expected)?.max(1.),
            "{case}: output diverges"
        );
        let embedding = normalize(&mean_pool(&output.last_hidden_state, attention_mask)?)?;
        let cosine = min_cosine(&embedding, &reference["embedding"])?;
        println!("{case} embedding: cosine {cosine:.6}");
        assert!(cosine > 0.9999, "{case}: embedding diverges");
    }
    Ok(())
}

fn full_model() -> anyhow::Result<EmbeddingGemma2Model> {
    let (config, weights) = checkpoint()?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &Device::Cpu)? };
    Ok(EmbeddingGemma2Model::new(&config, vb)?)
}

/// Asserts that `actual` is within `tolerance` of `expected`, relative to its largest value.
fn assert_close(
    what: &str,
    actual: &Tensor,
    expected: &Tensor,
    tolerance: f32,
) -> anyhow::Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "{what}: shapes differ");
    let diff = max_abs_diff(actual, expected)?;
    let scale = max_abs(expected)?.max(1.);
    println!("{what}: max abs diff {diff:.3e} (scale {scale:.1})");
    assert!(diff <= tolerance * scale, "{what} diverges");
    Ok(())
}

/// Smallest cosine similarity between the rows of two batches of normalized embeddings.
fn min_cosine(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
    let cosines = (a * b)?.sum(1)?.to_vec1::<f32>()?;
    Ok(cosines.into_iter().fold(f32::INFINITY, f32::min))
}

/// Runs the model on the processor outputs stored in a reference dump and checks the token
/// embeddings and the sentence embeddings.
fn check_case(
    model: &EmbeddingGemma2Model,
    case: &str,
    reference: &HashMap<String, Tensor>,
    image: Option<&Tensor>,
    video: Option<&Tensor>,
    audio: Option<&Tensor>,
) -> anyhow::Result<()> {
    let attention_mask = &reference["attention_mask"];
    let soft_tokens = SoftTokens {
        image,
        video,
        audio,
    };
    let output = model.forward(&reference["input_ids"], Some(attention_mask), soft_tokens)?;
    assert_close(
        &format!("{case} last_hidden_state"),
        &output,
        &reference["last_hidden_state"],
        2e-3,
    )?;
    let embedding = normalize(&mean_pool(&output, attention_mask)?)?;
    let cosine = min_cosine(&embedding, &reference["embedding"])?;
    println!("{case} embedding: cosine {cosine:.6}");
    assert!(cosine > 0.9999, "{case}: embedding diverges");
    Ok(())
}

#[test]
fn vision_matches_reference() -> anyhow::Result<()> {
    let Some(refdata) = refdata() else {
        eprintln!("EMBEDDINGGEMMA2_REFDATA is not set, skipping");
        return Ok(());
    };
    let (config, weights) = checkpoint()?;
    let model = full_model()?;

    // The tower on its own.
    let reference = load(refdata.join("mm_image.safetensors"))?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &Device::Cpu)? };
    let tower = embeddinggemma2::vision::VisionTower::new(
        config.vision_config.as_ref().unwrap(),
        vb.pp("vision_tower"),
    )?;
    let tokens = tower.forward(&reference["pixel_values"], &reference["image_position_ids"])?;
    assert_close(
        "vision tower",
        &Tensor::cat(&tokens, 0)?,
        &reference["vision_tower_output"],
        2e-3,
    )?;

    for case in ["image", "image_batch", "image_text", "image_pair"] {
        let reference = load(refdata.join(format!("mm_{case}.safetensors")))?;
        let features =
            model.image_features(&reference["pixel_values"], &reference["image_position_ids"])?;
        let features = Tensor::cat(&features, 0)?;
        assert_close(
            &format!("{case} soft tokens"),
            &features,
            &reference["image_hidden_states"],
            2e-3,
        )?;
        check_case(&model, case, &reference, Some(&features), None, None)?;
    }

    let reference = load(refdata.join("mm_video.safetensors"))?;
    let features = model.image_features(
        &reference["pixel_values_videos"],
        &reference["video_position_ids"],
    )?;
    let features = Tensor::cat(&features, 0)?;
    check_case(&model, "video", &reference, None, Some(&features), None)?;
    Ok(())
}

#[test]
fn audio_matches_reference() -> anyhow::Result<()> {
    let Some(refdata) = refdata() else {
        eprintln!("EMBEDDINGGEMMA2_REFDATA is not set, skipping");
        return Ok(());
    };
    let (config, weights) = checkpoint()?;
    let model = full_model()?;

    let reference = load(refdata.join("mm_audio.safetensors"))?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &Device::Cpu)? };
    let tower = embeddinggemma2::audio::AudioTower::new(
        config.audio_config.as_ref().unwrap(),
        vb.pp("audio_tower"),
    )?;
    let tokens = tower.forward(
        &reference["input_features"],
        &reference["input_features_mask"],
    )?;
    assert_close(
        "audio tower",
        &tokens[0].unsqueeze(0)?,
        &reference["audio_tower_output"],
        2e-3,
    )?;

    for case in ["audio", "audio_batch", "audio_long", "audio_text"] {
        let reference = load(refdata.join(format!("mm_{case}.safetensors")))?;
        let features = model.audio_features(
            &reference["input_features"],
            &reference["input_features_mask"],
        )?;
        let features = Tensor::cat(&features, 0)?;
        assert_close(
            &format!("{case} soft tokens"),
            &features,
            &reference["audio_hidden_states"],
            2e-3,
        )?;
        check_case(&model, case, &reference, None, None, Some(&features))?;
    }
    Ok(())
}

#[test]
fn image_preprocessing_matches_reference() -> anyhow::Result<()> {
    let Some(refdata) = refdata() else {
        eprintln!("EMBEDDINGGEMMA2_REFDATA is not set, skipping");
        return Ok(());
    };
    let reference = load(refdata.join("mm_image.safetensors"))?;
    let processor = ImageProcessor::default();
    let patches = processor.preprocess(&image::open(refdata.join("image_a.png"))?)?;
    let (pixel_values, positions) = processor.collate(&[&patches], &Device::Cpu)?;

    // The reference pads every image to the full patch budget.
    let num_patches = patches.num_patches();
    let expected_pixels = reference["pixel_values"].narrow(1, 0, num_patches)?;
    let expected_positions = reference["image_position_ids"].narrow(1, 0, num_patches)?;
    assert_eq!(
        positions.to_vec3::<i64>()?,
        expected_positions.to_vec3::<i64>()?
    );
    // The resampling filters are not bit identical: compare the mean error.
    let mean_diff = (pixel_values - expected_pixels)?
        .abs()?
        .mean_all()?
        .to_scalar::<f32>()?;
    println!("pixel values: mean abs diff {mean_diff:.3e}");
    assert!(mean_diff < 4e-3);
    Ok(())
}

#[test]
fn audio_preprocessing_matches_reference() -> anyhow::Result<()> {
    let Some(refdata) = refdata() else {
        eprintln!("EMBEDDINGGEMMA2_REFDATA is not set, skipping");
        return Ok(());
    };
    let extractor = AudioFeatureExtractor::default();
    let clip_a = read_wav(refdata.join("audio_a.wav"))?;
    let clip_b = read_wav(refdata.join("audio_b.wav"))?;

    let reference = load(refdata.join("mm_audio_batch.safetensors"))?;
    let padded_len = clip_a.len().max(clip_b.len());
    let features_a = extractor.extract(&clip_a, Some(padded_len))?;
    let features_b = extractor.extract(&clip_b, Some(padded_len))?;
    let (features, mask) = extractor.collate(&[&features_a, &features_b], &Device::Cpu)?;
    assert_eq!(
        mask.to_dtype(DType::I64)?.to_vec2::<i64>()?,
        reference["input_features_mask"].to_vec2::<i64>()?
    );
    assert_close(
        "mel features",
        &features,
        &reference["input_features"],
        1e-4,
    )?;
    let num_placeholders = |row: usize| -> anyhow::Result<usize> {
        let ids = reference["input_ids"].get(row)?.to_vec1::<i64>()?;
        Ok(ids.into_iter().filter(|&id| id == 258_881).count())
    };
    assert_eq!(features_a.num_soft_tokens, num_placeholders(0)?);
    assert_eq!(features_b.num_soft_tokens, num_placeholders(1)?);
    Ok(())
}

fn read_wav(path: PathBuf) -> anyhow::Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let samples: Result<Vec<i16>, _> = reader.samples::<i16>().collect();
    Ok(samples?
        .into_iter()
        .map(|sample| f32::from(sample) / 32768.)
        .collect())
}

/// The whole pipeline, from raw inputs, against embeddings computed by sentence-transformers.
#[test]
fn pipeline_matches_sentence_transformers() -> anyhow::Result<()> {
    let Some(refdata) = refdata() else {
        eprintln!("EMBEDDINGGEMMA2_REFDATA is not set, skipping");
        return Ok(());
    };
    let embedder = EmbeddingGemma2::from_pretrained(DEFAULT_REPO, Options::default())?;
    let reference = load(refdata.join("sentence_transformers.safetensors"))?;
    let image = image::open(refdata.join("image_a.png"))?;
    let audio = read_wav(refdata.join("audio_a.wav"))?;

    let check = |what: &str, actual: &Tensor, expected: &Tensor, min: f32| -> anyhow::Result<()> {
        let cosine = min_cosine(actual, expected)?;
        println!("{what}: cosine {cosine:.6}");
        assert!(cosine > min, "{what} diverges");
        Ok(())
    };
    let query = embedder.encode_queries(&["What causes the northern lights?"])?;
    let document = embedder.encode_documents(&["Charged particles from the sun cause them."])?;
    check("query", &query, &reference["query"], 0.9999)?;
    check("document", &document, &reference["doc"], 0.9999)?;

    // Media go through this crate's own preprocessing, which is close to the reference's
    // but not bit identical.
    let mixed = Input::text("colorful shapes on a gradient").with_image(image.clone());
    let prompt = embedder.prompt("SearchQuery")?;
    check(
        "image",
        &embedder.encode(&[Input::image(image.clone())], None)?,
        &reference["image"],
        0.999,
    )?;
    check(
        "audio",
        &embedder.encode(&[Input::audio(audio.clone())], None)?,
        &reference["audio"],
        0.999,
    )?;
    check(
        "mixed",
        &embedder.encode(std::slice::from_ref(&mixed), None)?,
        &reference["mixed"],
        0.999,
    )?;
    check(
        "mixed, prompt",
        &embedder.encode(std::slice::from_ref(&mixed), Some(prompt))?,
        &reference["mixed_prompt"],
        0.999,
    )?;

    // Photos are downscaled to the patch budget, where the antialiasing of the resize matters.
    for name in ["image_large", "image_tall"] {
        let image = image::open(refdata.join(format!("{name}.png")))?;
        check(
            name,
            &embedder.encode(&[Input::image(image)], None)?,
            &reference[name],
            0.999,
        )?;
    }

    // Batching inputs of different lengths and modalities must not change their embeddings.
    let inputs = [
        Input::text("a short text"),
        mixed,
        Input::audio(audio[..20_000].to_vec()),
        Input::image(image::open(refdata.join("image_b.png"))?),
        Input::audio(audio).with_text("a rising tone"),
        Input::video(vec![image; 2]),
    ];
    let batched = embedder.encode(&inputs, None)?;
    assert_eq!(batched.dims(), [inputs.len(), 768]);
    for (idx, input) in inputs.iter().enumerate() {
        let alone = embedder.encode(std::slice::from_ref(input), None)?;
        check(
            &format!("batched input {idx}"),
            &batched.narrow(0, idx, 1)?,
            &alone,
            0.9999,
        )?;
    }
    Ok(())
}
