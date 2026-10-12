"""Dumps reference activations of google/embeddinggemma-2 for `tests/parity.rs`.

Runs the `transformers` implementation in f32 on synthetic inputs and saves its inputs and
outputs as safetensors files, next to the images and audio clips they were computed from.

    uv venv ref
    uv pip install --python ref/bin/python torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python ref/bin/python "git+https://github.com/huggingface/transformers" \
        "sentence-transformers>=6.1.0" numpy pillow safetensors
    ref/bin/python scripts/dump_reference.py /path/to/refdata
    EMBEDDINGGEMMA2_REFDATA=/path/to/refdata cargo test --test parity
"""

import sys
import wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor

NAME = "google/embeddinggemma-2"
MODEL_KEYS = {
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_position_ids",
    "pixel_values_videos",
    "video_position_ids",
    "num_frames_per_video",
    "input_features",
    "input_features_mask",
}

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
processor = AutoProcessor.from_pretrained(NAME)
model = AutoModel.from_pretrained(NAME, dtype=torch.float32).eval()


def make_image(width, height, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    blue = 127 + 127 * np.sin(xx / 17.0 + seed) * np.cos(yy / 23.0)
    base = np.stack([255 * xx / width, 255 * yy / height, blue], -1)
    image = Image.fromarray(base.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for _ in range(12):
        x0, y0 = rng.integers(0, width - 40), rng.integers(0, height - 40)
        x1, y1 = x0 + rng.integers(20, width // 2), y0 + rng.integers(20, height // 2)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        shape = draw.ellipse if rng.random() < 0.5 else draw.rectangle
        shape([x0, y0, x1, y1], fill=color, outline=(0, 0, 0))
    return image


def make_audio(seconds, seed):
    """A rising tone over a gated 440 Hz note and some noise, as 16 kHz 16-bit PCM."""
    rng = np.random.default_rng(seed)
    n = int(16000 * seconds)
    t = np.arange(n) / 16000.0
    sweep = 0.4 * np.sin(2 * np.pi * np.cumsum(200 + 1500 * t / seconds) / 16000.0)
    note = 0.2 * np.sin(2 * np.pi * 440 * t) * (np.sin(2 * np.pi * 3 * t) > 0)
    x = sweep + note + 0.02 * rng.standard_normal(n)
    return (x.clip(-1, 1) * 32767).astype(np.int16)


def save_wav(path, pcm):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(pcm.tobytes())


def embed(output, enc):
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, dim=-1)


def dump(name, enc, extra=None, hidden_states=False):
    inputs = {k: v for k, v in enc.items() if k in MODEL_KEYS}
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=hidden_states)
    tensors = {k: v if torch.is_tensor(v) else torch.tensor(np.asarray(v)) for k, v in enc.items()}
    tensors["last_hidden_state"] = output.last_hidden_state
    tensors["embedding"] = embed(output, enc)
    if output.image_hidden_states is not None:
        tensors["image_hidden_states"] = output.image_hidden_states
    if output.audio_hidden_states is not None:
        tensors["audio_hidden_states"] = output.audio_hidden_states
    if hidden_states:
        # The last entry is the projected output rather than the output of the last layer.
        for i, hidden in enumerate(output.hidden_states):
            tensors[f"hidden_{i:02d}"] = hidden
    tensors.update(extra or {})
    tensors = {
        k: (v.to(torch.int64) if v.dtype == torch.bool else v).clone().contiguous()
        for k, v in tensors.items()
    }
    save_file(tensors, str(out / f"{name}.safetensors"))
    print(name, {k: tuple(v.shape) for k, v in tensors.items() if not k.startswith("hidden_")})


# --- text
long_document = " ".join(
    f"Sentence number {i}: the aurora borealis is caused by charged particles from the sun "
    "striking the upper atmosphere near the magnetic poles."
    for i in range(60)
)
texts = {
    "query": ["task: search result | query: What causes the northern lights?"],
    "doc": ["title: none | text: Charged particles from the sun cause them."],
    # Two lengths, to exercise the padding.
    "batch": [
        "task: search result | query: What causes the northern lights?",
        "title: none | text: Auroras appear when the solar wind disturbs the magnetosphere, "
        "and electrons precipitate into the thermosphere, exciting oxygen and nitrogen. "
        "極光は太陽風によって引き起こされます。",
    ],
    # Longer than the sliding window.
    "long": ["title: none | text: " + long_document],
}
for name, batch in texts.items():
    dump(f"text_{name}", processor(text=batch, return_tensors="pt", padding=True), hidden_states=True)

# --- images
image_a, image_b = make_image(640, 480, 1), make_image(300, 500, 2)
image_a.save(out / "image_a.png")
image_b.save(out / "image_b.png")
enc = processor(images=[image_a], return_tensors="pt")
with torch.no_grad():
    tower = model.vision_tower(pixel_values=enc["pixel_values"], pixel_position_ids=enc["image_position_ids"])
dump("mm_image", enc, {"vision_tower_output": tower.last_hidden_state})
dump("mm_image_batch", processor(images=[[image_a], [image_b]], return_tensors="pt"))
text = ["task: search result | query: colorful shapes <|image|> on a gradient"]
dump("mm_image_text", processor(text=text, images=[[image_a]], return_tensors="pt"))
text = ["<|image|> compared with <|image|>"]
dump("mm_image_pair", processor(text=text, images=[[image_b, image_a]], return_tensors="pt"))

# --- video, as already decoded frames
frames = [make_image(640, 360, 10 + i) for i in range(3)]
dump("mm_video", processor(videos=[frames], return_tensors="pt", do_sample_frames=False))

# --- audio
pcm_a, pcm_b, pcm_long = make_audio(3.7, 1), make_audio(1.93, 2), make_audio(15.0, 3)
save_wav(out / "audio_a.wav", pcm_a)
save_wav(out / "audio_b.wav", pcm_b)
wav_a, wav_b, wav_long = (pcm.astype(np.float32) / 32768.0 for pcm in (pcm_a, pcm_b, pcm_long))
enc = processor(audio=[wav_a], return_tensors="pt")
with torch.no_grad():
    tower = model.audio_tower(enc["input_features"], enc["input_features_mask"], return_dict=True)
dump("mm_audio", enc, {"audio_tower_output": tower.last_hidden_state})
dump("mm_audio_batch", processor(audio=[wav_a, wav_b], return_tensors="pt"))
dump("mm_audio_long", processor(audio=[wav_long], return_tensors="pt"))
text = ["task: search result | query: a rising tone <|audio|>"]
dump("mm_audio_text", processor(text=text, audio=[wav_b], return_tensors="pt"))

# --- sentence-transformers, the documented entry point, from raw inputs
st = SentenceTransformer(NAME, model_kwargs={"dtype": torch.float32})
# Photos are larger than the patch budget: these two are downscaled, unlike the ones above.
image_large, image_tall = make_image(3000, 2000, 3), make_image(1200, 4000, 4)
image_large.save(out / "image_large.png")
image_tall.save(out / "image_tall.png")
mixed = [{"text": "colorful shapes on a gradient", "image": image_a}]
embeddings = {
    "query": st.encode(["What causes the northern lights?"], prompt_name="SearchQuery", convert_to_tensor=True),
    "doc": st.encode(["Charged particles from the sun cause them."], prompt_name="Document", convert_to_tensor=True),
    "image": st.encode([image_a], convert_to_tensor=True),
    "image_large": st.encode([image_large], convert_to_tensor=True),
    "image_tall": st.encode([image_tall], convert_to_tensor=True),
    "audio": st.encode([{"array": wav_a, "sampling_rate": 16000}], convert_to_tensor=True),
    "mixed": st.encode(mixed, convert_to_tensor=True),
    "mixed_prompt": st.encode(mixed, prompt_name="SearchQuery", convert_to_tensor=True),
}
save_file({k: v.cpu().clone().contiguous() for k, v in embeddings.items()}, str(out / "sentence_transformers.safetensors"))
print("query/doc similarity", float(st.similarity(embeddings["query"], embeddings["doc"])))
