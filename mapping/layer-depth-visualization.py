#layer-depth-visualization.py

from pathlib import Path
from collections import Counter
import json
import os
import re
import gc

import numpy as np
import matplotlib.pyplot as plt
from gguf import GGUFReader


#============================================================
#EDIT THESE
#============================================================

MODEL_OR_PATH = "llama3:8b"

#Pick one tensor role to compare across blocks.
#Good options:
#"attn_q.weight"
#"attn_k.weight"
#"attn_v.weight"
#"attn_output.weight"
#"ffn_gate.weight"
#"ffn_up.weight"
#"ffn_down.weight"
#"attn_norm.weight"
#"ffn_norm.weight"
TENSOR_SUFFIX = "attn_q.weight"

#Use specific blocks:
BLOCKS_TO_PLOT = [0, 8, 16, 24, 31]

#Or use every block:
#BLOCKS_TO_PLOT = "all"

#For 2D tensors, plot this row from each tensor.
ROW_INDEX = 0

#Plot this many values from the row.
VALUES_TO_PLOT = 250

#Start offset inside the row/vector.
START_AT = 0

#True makes each line show shape rather than absolute scale.
#Useful when lines are visually too close together.
NORMALIZE_EACH_LINE = False

#Optional: save the figure.
SAVE_FIGURE = False
OUTPUT_IMAGE = "layer-depth-plot.png"


#============================================================
#FIND GGUF FILE
#============================================================

model_path = Path(MODEL_OR_PATH).expanduser()
ollama_models_dir = Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))

if model_path.exists():
    if model_path.is_dir():
        ggufs = sorted(model_path.rglob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
        if not ggufs:
            raise FileNotFoundError(f"No .gguf files found in folder: {model_path}")
        gguf_path = ggufs[0]
    else:
        gguf_path = model_path

else:
    manifests_dir = ollama_models_dir / "manifests"

    if not manifests_dir.exists():
        raise FileNotFoundError(f"No Ollama manifests folder found at: {manifests_dir}")

    matching_manifest = None

    for manifest_file in manifests_dir.rglob("*"):
        if not manifest_file.is_file():
            continue

        rel = manifest_file.relative_to(manifests_dir)
        parts = rel.parts
        possible_names = set()

        if len(parts) >= 4 and parts[1] == "library":
            possible_names.add(f"{parts[2]}:{parts[3]}")

        if len(parts) >= 4 and parts[1] != "library":
            possible_names.add(f"{parts[1]}/{parts[2]}:{parts[3]}")

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

    candidate_digests = []

    if "config" in manifest and "digest" in manifest["config"]:
        candidate_digests.append(manifest["config"]["digest"])

    for layer in manifest.get("layers", []):
        if "digest" in layer:
            candidate_digests.append(layer["digest"])

    gguf_candidates = []

    for digest in candidate_digests:
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


#============================================================
#OPEN MODEL
#============================================================

reader = GGUFReader(str(gguf_path))

print()
print("GGUF file:")
print(gguf_path)
print()
print(f"File size: {gguf_path.stat().st_size / 1024**3:.2f} GB")
print(f"Total tensors: {len(reader.tensors)}")
print()


#============================================================
#FIND MATCHING TENSORS
#============================================================

all_block_ids = []

for tensor in reader.tensors:
    match = re.match(r"blk\.(\d+)\.", tensor.name)
    if match:
        all_block_ids.append(int(match.group(1)))

all_block_ids = sorted(set(all_block_ids))

if BLOCKS_TO_PLOT == "all":
    block_ids = all_block_ids
else:
    block_ids = BLOCKS_TO_PLOT

wanted_names = {
    block_id: f"blk.{block_id}.{TENSOR_SUFFIX}"
    for block_id in block_ids
}

tensors_by_name = {
    tensor.name: tensor
    for tensor in reader.tensors
}

missing = [
    name
    for name in wanted_names.values()
    if name not in tensors_by_name
]

if missing:
    print("Missing tensors:")
    for name in missing:
        print("  -", name)

    print()
    print("Available tensor suffixes in blk.0:")

    suffixes = []
    for tensor in reader.tensors:
        if tensor.name.startswith("blk.0."):
            suffixes.append(tensor.name.replace("blk.0.", ""))

    for suffix in sorted(suffixes):
        print("  -", suffix)

    raise KeyError(f"Some requested tensors do not exist for suffix: {TENSOR_SUFFIX}")


#============================================================
#DEQUANTIZE ONE TENSOR AT A TIME AND PLOT
#============================================================

float_or_int_types = {
    "F32", "F16", "F64", "BF16",
    "I8", "I16", "I32", "I64",
}

plt.figure(figsize=(14, 7))

print(f"Plotting tensor role across depth: {TENSOR_SUFFIX}")
print()

for block_id in block_ids:
    tensor_name = wanted_names[block_id]
    tensor = tensors_by_name[tensor_name]

    tensor_type = str(getattr(tensor.tensor_type, "name", tensor.tensor_type))
    logical_shape = tuple(int(x) for x in tensor.shape)

    print(f"Loading blk.{block_id}: {tensor_name}")
    print(f"  shape={logical_shape}, type={tensor_type}, disk_size={int(tensor.n_bytes) / 1024**2:.2f} MB")

    if tensor_type in float_or_int_types:
        values = np.asarray(tensor.data, dtype=np.float32)

    else:
        from gguf.quants import dequantize
        values = np.asarray(dequantize(tensor.data, tensor.tensor_type), dtype=np.float32)

    if values.ndim == 1 and np.prod(logical_shape) == values.size:
        values = values.reshape(logical_shape)

    if values.ndim >= 2:
        row = values[ROW_INDEX % values.shape[0]]
        plotted = np.asarray(row[START_AT:START_AT + VALUES_TO_PLOT], dtype=np.float32)
        source_desc = f"row {ROW_INDEX % values.shape[0]}"

    else:
        plotted = np.asarray(values[START_AT:START_AT + VALUES_TO_PLOT], dtype=np.float32)
        source_desc = "vector"

    plotted = plotted[np.isfinite(plotted)]

    if plotted.size == 0:
        print("  skipped: no finite values")
        continue

    if NORMALIZE_EACH_LINE:
        std = plotted.std()
        if std > 0:
            plotted = (plotted - plotted.mean()) / std
        else:
            plotted = plotted - plotted.mean()

    print(
        f"  plotted {source_desc}, n={plotted.size:,}, "
        f"mean={plotted.mean():+.6f}, "
        f"std={plotted.std():.6f}, "
        f"min={plotted.min():+.6f}, "
        f"max={plotted.max():+.6f}"
    )

    plt.plot(plotted, linewidth=1, label=f"blk.{block_id}")

    del values
    del plotted
    gc.collect()


title = f"{TENSOR_SUFFIX} across transformer depth"

if NORMALIZE_EACH_LINE:
    title += " normalized"

plt.title(title)
plt.xlabel("Weight index within sampled row/vector")
plt.ylabel("Weight value" if not NORMALIZE_EACH_LINE else "Normalized value")
plt.legend()
plt.tight_layout()

if SAVE_FIGURE:
    plt.savefig(OUTPUT_IMAGE, dpi=180)
    print()
    print(f"Saved figure to: {OUTPUT_IMAGE}")

plt.show()
