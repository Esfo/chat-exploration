# layer-visualization.py

from pathlib import Path
from collections import defaultdict, Counter
import json
import os
import re

import numpy as np
from gguf import GGUFReader


# Change this.
# Examples:
# MODEL_OR_PATH = "llama3:8b"
# MODEL_OR_PATH = "llama3:latest"
# MODEL_OR_PATH = "/Users/you/.ollama/models/blobs/sha256-abc..."
# MODEL_OR_PATH = "/path/to/folder/with/gguf"
MODEL_OR_PATH = "llama3:8b"

# Keep this False at first. True samples/dequantizes a tiny part of each tensor.
PROBE_VALUES = False

# Only used when PROBE_VALUES = True.
SAMPLE_ROWS = 2
MAX_1D_VALUES = 4096

# Use 0 to show every tensor in every layer.
MAX_TENSORS_PER_LAYER = 0


model_path = Path(MODEL_OR_PATH).expanduser()
ollama_models_dir = Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))

# Resolve MODEL_OR_PATH into an actual GGUF file.
if model_path.exists():
    if model_path.is_dir():
        ggufs = sorted(model_path.rglob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
        if not ggufs:
            raise FileNotFoundError(f"No .gguf files found inside folder: {model_path}")
        gguf_path = ggufs[0]
    else:
        gguf_path = model_path

else:
    manifests_dir = ollama_models_dir / "manifests"

    if not manifests_dir.exists():
        raise FileNotFoundError(f"No Ollama manifest folder found at: {manifests_dir}")

    matching_manifest = None

    for manifest_file in manifests_dir.rglob("*"):
        if not manifest_file.is_file():
            continue

        rel = manifest_file.relative_to(manifests_dir)
        parts = rel.parts

        possible_names = set()

        # Usually:
        # registry.ollama.ai/library/llama3/8b -> llama3:8b
        if len(parts) >= 4 and parts[1] == "library":
            possible_names.add(f"{parts[2]}:{parts[3]}")

        # Namespaced:
        # registry.ollama.ai/user/model/tag -> user/model:tag
        if len(parts) >= 4 and parts[1] != "library":
            possible_names.add(f"{parts[1]}/{parts[2]}:{parts[3]}")

        # Fallback:
        if len(parts) >= 2:
            possible_names.add(f"{parts[-2]}:{parts[-1]}")

        if MODEL_OR_PATH in possible_names:
            matching_manifest = manifest_file
            break

    if matching_manifest is None:
        visible_models = []

        for manifest_file in manifests_dir.rglob("*"):
            if not manifest_file.is_file():
                continue

            rel = manifest_file.relative_to(manifests_dir)
            parts = rel.parts

            if len(parts) >= 4 and parts[1] == "library":
                visible_models.append(f"{parts[2]}:{parts[3]}")
            elif len(parts) >= 4:
                visible_models.append(f"{parts[1]}/{parts[2]}:{parts[3]}")
            elif len(parts) >= 2:
                visible_models.append(f"{parts[-2]}:{parts[-1]}")

        print("Models I found:")
        for name in sorted(set(visible_models)):
            print("  -", name)

        raise FileNotFoundError(f"Could not find Ollama model: {MODEL_OR_PATH}")

    with open(matching_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    candidate_blobs = []

    if "config" in manifest and "digest" in manifest["config"]:
        candidate_blobs.append(manifest["config"]["digest"])

    for layer in manifest.get("layers", []):
        if "digest" in layer:
            candidate_blobs.append(layer["digest"])

    gguf_candidates = []

    for digest in candidate_blobs:
        algo, hash_part = digest.split(":", 1)
        blob_path = ollama_models_dir / "blobs" / f"{algo}-{hash_part}"

        if not blob_path.exists():
            continue

        with open(blob_path, "rb") as f:
            magic = f.read(4)

        if magic == b"GGUF":
            gguf_candidates.append(blob_path)

    if not gguf_candidates:
        raise FileNotFoundError(f"Found manifest but no GGUF blob inside it: {matching_manifest}")

    gguf_path = max(gguf_candidates, key=lambda p: p.stat().st_size)


reader = GGUFReader(str(gguf_path))

print()
print("GGUF file:")
print(gguf_path)
print()
print(f"File size: {gguf_path.stat().st_size / 1024**3:.2f} GB")
print(f"Total tensors: {len(reader.tensors)}")
print()


print("Metadata:")

metadata_keys = [
    "general.name",
    "general.architecture",
    "general.file_type",
    "llama.block_count",
    "llama.context_length",
    "llama.embedding_length",
    "llama.feed_forward_length",
    "llama.attention.head_count",
    "llama.attention.head_count_kv",
]

for key in metadata_keys:
    field = reader.fields.get(key)

    if field is None:
        continue

    try:
        value = field.contents()
    except Exception:
        value = field

    print(f"  {key}: {value}")


print()

total_bytes = sum(int(t.n_bytes) for t in reader.tensors)
total_elements = sum(int(t.n_elements) for t in reader.tensors)
tensor_types = Counter(str(getattr(t.tensor_type, "name", t.tensor_type)) for t in reader.tensors)

print(f"Tensor bytes on disk: {total_bytes / 1024**3:.2f} GB")
print(f"Logical elements: {total_elements:,}")

print("Tensor types:")
for tensor_type, count in sorted(tensor_types.items()):
    print(f"  {tensor_type}: {count}")

print()


layers = defaultdict(list)

for tensor in reader.tensors:
    match = re.match(r"blk\.(\d+)\.", tensor.name)

    if match:
        layer_id = int(match.group(1))
    else:
        layer_id = "global"

    layers[layer_id].append(tensor)


float_or_int_types = {
    "F32", "F16", "F64", "BF16",
    "I8", "I16", "I32", "I64",
}


print("Layer summary:")

for layer_id in sorted(layers.keys(), key=lambda x: -1 if x == "global" else x):
    tensors = layers[layer_id]

    layer_name = "global tensors" if layer_id == "global" else f"transformer block {layer_id}"
    layer_bytes = sum(int(t.n_bytes) for t in tensors)
    layer_elements = sum(int(t.n_elements) for t in tensors)
    layer_types = Counter(str(getattr(t.tensor_type, "name", t.tensor_type)) for t in tensors)

    print()
    print("=" * 100)
    print(layer_name)
    print(f"{len(tensors)} tensors | {layer_bytes / 1024**2:.2f} MB on disk | {layer_elements:,} logical values")
    print(f"types: {dict(layer_types)}")
    print("-" * 100)

    shown = 0

    for tensor in tensors:
        if MAX_TENSORS_PER_LAYER and shown >= MAX_TENSORS_PER_LAYER:
            hidden = len(tensors) - shown
            print(f"... {hidden} more tensors hidden")
            break

        shown += 1

        name = tensor.name
        shape = tuple(int(x) for x in tensor.shape)
        tensor_type = str(getattr(tensor.tensor_type, "name", tensor.tensor_type))
        size_mb = int(tensor.n_bytes) / 1024**2

        if name.endswith("token_embd.weight"):
            role = "token embedding"
        elif name.endswith("output.weight"):
            role = "lm head / output"
        elif name.endswith("output_norm.weight"):
            role = "final norm"
        elif name.endswith("attn_norm.weight"):
            role = "attention norm"
        elif name.endswith("attn_q.weight"):
            role = "attention q"
        elif name.endswith("attn_k.weight"):
            role = "attention k"
        elif name.endswith("attn_v.weight"):
            role = "attention v"
        elif name.endswith("attn_output.weight"):
            role = "attention output"
        elif name.endswith("ffn_norm.weight"):
            role = "ffn norm"
        elif name.endswith("ffn_gate.weight"):
            role = "ffn gate"
        elif name.endswith("ffn_up.weight"):
            role = "ffn up"
        elif name.endswith("ffn_down.weight"):
            role = "ffn down"
        else:
            role = "unknown"

        line = f"{name:<40} role={role:<18} shape={str(shape):<18} type={tensor_type:<8} size={size_mb:>8.2f} MB"

        if PROBE_VALUES:
            try:
                if tensor_type in float_or_int_types:
                    arr = tensor.data

                    if arr.ndim >= 2:
                        sample = np.asarray(arr[:SAMPLE_ROWS, :], dtype=np.float32).ravel()
                    else:
                        sample = np.asarray(arr[:MAX_1D_VALUES], dtype=np.float32).ravel()

                else:
                    from gguf.quants import dequantize

                    arr = tensor.data

                    if arr.ndim >= 2:
                        raw = arr[:SAMPLE_ROWS, :]
                    else:
                        raw = arr[:MAX_1D_VALUES]

                    sample = np.asarray(dequantize(raw, tensor.tensor_type), dtype=np.float32).ravel()

                sample = sample[np.isfinite(sample)]

                if sample.size:
                    line += (
                        f" | sample n={sample.size:,}"
                        f" mean={sample.mean():+.5f}"
                        f" std={sample.std():.5f}"
                        f" min={sample.min():+.5f}"
                        f" max={sample.max():+.5f}"
                    )
                else:
                    line += " | sample had no finite values"

            except Exception as e:
                line += f" | sample unavailable: {e}"

        print(line)

print()
print("Done.")
print("Use the tensor names above later when choosing one to plot.")
