//! Compares the candle implementation against activations dumped from the reference
//! `transformers` implementation by `scripts/dump_reference.py`.
//!
//! The tests need the checkpoint and the dumps, so they only run when
//! `EMBEDDINGGEMMA2_REFDATA` points at the directory holding the dumps.

use std::collections::HashMap;
use std::path::PathBuf;

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use embeddinggemma2::config::Config;
use embeddinggemma2::text::TextModel;

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
            assert!(diff <= 2e-3 * scale.max(1.), "{case}: hidden state {idx} diverges");
        }
        let expected = &reference["last_hidden_state"];
        let diff = max_abs_diff(&output.last_hidden_state, expected)?;
        println!("{case} last_hidden_state: max abs diff {diff:.3e}");
        assert!(diff <= 2e-3 * max_abs(expected)?.max(1.), "{case}: output diverges");
    }
    Ok(())
}
