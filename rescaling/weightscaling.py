#!/usr/bin/env python3

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType


def run(args):
    result = subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(args)
            + "\n\nSTDOUT:\n"
            + result.stdout
            + "\nSTDERR:\n"
            + result.stderr
        )

    return result.stdout


def require_command(command):
    if shutil.which(command) is None:
        raise RuntimeError(f"Missing command in PATH: {command}")


def resolve(path):
    return Path(path).expanduser().resolve()


def get_source_gguf_from_ollama(model_name):
    modelfile = run(["ollama", "show", "--modelfile", model_name])

    for line in modelfile.splitlines():
        line = line.strip()

        if not line.startswith("FROM "):
            continue

        candidate = line.removeprefix("FROM ").strip().strip('"')
        path = resolve(candidate)

        if path.exists():
            return path

    raise RuntimeError(f"Could not find local GGUF path for Ollama model: {model_name}")


def dequantize_to_f16(llama_quantize, source_gguf, f16_gguf):
    run([
        llama_quantize,
        "--allow-requantize",
        str(source_gguf),
        str(f16_gguf),
        "F16",
    ])


def scale_gguf_in_place(path, scaling_index):
    above_scale = 1.0 + scaling_index
    below_scale = 1.0 - scaling_index

    float_types = {
        GGMLQuantizationType.F16,
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F64,
    }

    reader = GGUFReader(str(path), mode="r+")

    changed_tensors = 0
    changed_values = 0

    for tensor in reader.tensors:
        if tensor.tensor_type not in float_types:
            continue

        arr = tensor.data
        weights = arr.astype(np.float32, copy=False)

        mean_abs = float(np.mean(np.abs(weights)))
        if mean_abs == 0.0:
            continue

        mask = np.abs(weights) > mean_abs
        edited = np.where(
            mask,
            weights * above_scale,
            weights * below_scale,
        )

        arr[...] = edited.astype(arr.dtype)

        changed_tensors += 1
        changed_values += arr.size

    reader.data.flush()

    if changed_tensors == 0:
        raise RuntimeError("No float tensors were changed.")

    print(f"changed_tensors={changed_tensors}")
    print(f"changed_values={changed_values}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--scaling-index", type=float, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--llama-quantize", default="llama-quantize")

    args = parser.parse_args()

    if not args.output_file.endswith(".gguf"):
        raise ValueError("--output-file must end in .gguf")

    if args.scaling_index < 0.0 or args.scaling_index >= 1.0:
        raise ValueError("--scaling-index must be >= 0.0 and < 1.0")

    require_command("ollama")
    require_command(args.llama_quantize)

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_stem = Path(args.output_file).stem
    build_dir = output_dir / "build" / output_stem
    build_dir.mkdir(parents=True, exist_ok=True)

    f16_gguf = build_dir / f"{output_stem}.f16.gguf"
    output_gguf = output_dir / args.output_file

    if output_gguf.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_gguf}")

    if f16_gguf.exists():
        f16_gguf.unlink()

    if output_gguf.exists():
        output_gguf.unlink()

    source_gguf = get_source_gguf_from_ollama(args.model)

    print(f"source_gguf={source_gguf}")
    print(f"f16_gguf={f16_gguf}")
    print(f"output_gguf={output_gguf}")

    dequantize_to_f16(args.llama_quantize, source_gguf, f16_gguf)

    shutil.copy2(f16_gguf, output_gguf)

    scale_gguf_in_place(output_gguf, args.scaling_index)

    print(f"wrote={output_gguf}")


if __name__ == "__main__":
    main()
