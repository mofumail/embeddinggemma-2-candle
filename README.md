# embeddinggemma-2

A [candle](https://github.com/huggingface/candle) implementation of
[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2), Google's
multimodal embedding model. Text, images, video frames and audio are embedded into one
768-dimensional space, in pure Rust.

```rust
use embeddinggemma2::{EmbeddingGemma2, Input, Options, DEFAULT_REPO};

let embedder = EmbeddingGemma2::from_pretrained(DEFAULT_REPO, Options::default().with_best_device()?)?;

let queries = embedder.encode_queries(&["What causes the northern lights?"])?;
let documents = embedder.encode_documents(&["Charged particles from the sun cause them."])?;
let images = embedder.encode(&[Input::image(image::open("aurora.jpg")?)], None)?;

// Embeddings are L2 normalized `(n, 768)` f32 tensors.
let similarities = EmbeddingGemma2::similarity(&queries, &documents)?;
```

The weights (1.5 GB) are downloaded from the Hugging Face hub on first use, into the usual
`~/.cache/huggingface` cache.

## Example

```sh
cargo run --release --example embed -- \
    --query "What causes the northern lights?" \
    --document "Charged particles from the sun cause them." \
    --document "The stock market closed higher today." \
    --image photo.jpg --audio clip.wav
```

```text
What causes the northern lights?
  0.7418  Charged particles from the sun cause them.
  0.5838  The stock market closed higher today.
  ...
```

Add `--features cuda` to run on an NVIDIA GPU, where `--bf16` loads the weights in half the
memory. `--dim 256` truncates the embeddings. The `metal`, `mkl` and `accelerate` features
forward to candle's backends of the same names; they have not been tested with this model.

## Usage notes

- **Prompts.** Text is embedded behind a task prompt. `encode_queries` and `encode_documents`
  use the `SearchQuery` and `Document` prompts; `encode_text(texts, "Classification")` takes any
  prompt of the model by name (`embedder.prompts()` lists them), and `encode(inputs, Some(prompt))`
  any string. Documents with a title are written `title: {title} | text: {content}`. Images,
  video and audio are embedded without a prompt.
- **Mixed inputs.** An `Input` is a sequence of parts embedded into a single vector:
  `Input::text("a caption").with_image(image)`.
- **Video** is passed as decoded frames, `Input::video(frames)`. The model was trained on one
  frame per second; more than 32 frames are subsampled uniformly.
- **Audio** is passed as mono 16 kHz samples in `[-1, 1]`, `Input::audio(samples)`, and is
  truncated to 30 seconds.
- **Matryoshka.** `truncate(&embeddings, 256)` keeps the first dimensions and renormalizes.
  The model is trained for 768, 512, 256 and 128.
- **Precision.** f32 by default. bf16 needs CUDA (candle has no bf16 matrix product on the
  CPU) and moved the similarities of the example above by up to 2e-3. f16 is rejected, the
  model overflows in it.
- **Memory.** `Options::modalities` skips the towers that are not needed: the text backbone
  is 270M parameters, the vision tower 170M and the audio tower 300M.
- **Long texts** are truncated to `Options::max_seq_len` tokens, 8192 by default. The reference
  implementation does not truncate; raise the limit to match it.

## Layout

| Module | |
| --- | --- |
| `embedder` | `EmbeddingGemma2`: tokenization, prompts, batching, pooling |
| `model` | `EmbeddingGemma2Model`: the towers feeding soft tokens to the text backbone |
| `text` | The text backbone: a bidirectional Gemma encoder with per-layer embeddings |
| `vision` | The Gemma 4 vision tower |
| `audio` | The Gemma 4 audio tower, a conformer |
| `image_processor`, `audio_features` | Preprocessing of images and audio |

The model modules follow the conventions of `candle-transformers` (a config struct
deserialized from `config.json`, layers built from a `VarBuilder`), and load the released
`model.safetensors` as is.

## Checking against the reference

`tests/parity.rs` compares every stage with the `transformers` implementation: the hidden
state after each layer of the text backbone, the outputs of both towers, the preprocessing,
and the final embeddings from raw inputs against `sentence-transformers`.

```sh
python scripts/dump_reference.py /path/to/refdata   # see the script for its dependencies
EMBEDDINGGEMMA2_REFDATA=/path/to/refdata cargo test --test parity
```

Without `EMBEDDINGGEMMA2_REFDATA` these tests are skipped. In f32 on CPU, the embeddings of
text and audio match the reference to a cosine similarity of 1.0000, as do those of images
given the reference's pixels. From raw images the similarity is 0.9999: the bicubic resize
of the `image` crate is not bit identical to torchvision's.
