"""Resolve and prepare GGUF model sources for the HF backend.

The user works from Ollama-provided GGUF models (no Hugging Face downloads), the
same way ``/rescaling`` does. Activation capture needs a PyTorch model, but
``transformers`` can load a GGUF checkpoint directly and dequantize it in memory
into a real ``LlamaForCausalLM`` we can attach forward hooks to — so no HF
download and usually no on-disk conversion are required.

This module mirrors the resolution logic in ``rescaling/weightscaling.py``:

  * resolve an Ollama model name (e.g. ``llama3:8b``) to its local GGUF blob via
    ``ollama show --modelfile`` and the ``FROM`` line,
  * optionally dequantize that GGUF to an F16 GGUF with ``llama-quantize`` (the
    same ``dequantize_to_f16`` step ``/rescaling`` uses) as a robust fallback for
    quantization types the in-memory loader can't read directly.

Nothing here imports torch/transformers; it only locates and prepares files.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _run(args: list[str]) -> str:
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n" + " ".join(args)
            + "\n\nSTDOUT:\n" + result.stdout + "\nSTDERR:\n" + result.stderr
        )
    return result.stdout


def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _ollama_models_dir() -> Path:
    return Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))


def get_source_gguf_from_ollama_store(model_name: str) -> Path:
    """Resolve an Ollama model's GGUF blob from the local store, no server needed.

    Ollama stores models under ``$OLLAMA_MODELS`` (default ``~/.ollama/models``)
    as a manifest plus content-addressed blobs. We parse the manifest for the
    layer whose mediaType marks it as the model weights and map its digest to the
    blob file. This works even when ``ollama serve`` is not running.
    """
    models_dir = _ollama_models_dir()

    #Parse "[registry/]namespace/name:tag" with Ollama's defaults.
    name, _, tag = model_name.partition(":")
    tag = tag or "latest"
    parts = name.split("/")
    if len(parts) == 1:
        registry, namespace, repo = "registry.ollama.ai", "library", parts[0]
    elif len(parts) == 2:
        registry, namespace, repo = "registry.ollama.ai", parts[0], parts[1]
    else:
        registry, namespace, repo = parts[0], parts[1], "/".join(parts[2:])

    manifest = models_dir / "manifests" / registry / namespace / repo / tag
    if not manifest.exists():
        raise FileNotFoundError(
            f"Ollama manifest not found for {model_name!r} at {manifest}. "
            "Pull it first (e.g. `ollama pull llama3:8b`) or set OLLAMA_MODELS."
        )

    data = json.loads(manifest.read_text())
    digest = None
    for layer in data.get("layers", []):
        if "model" in layer.get("mediaType", ""):
            digest = layer["digest"]
            break
    if digest is None:
        raise RuntimeError(f"No model layer in Ollama manifest: {manifest}")

    blob = models_dir / "blobs" / digest.replace(":", "-")
    if not blob.exists():
        raise FileNotFoundError(f"Ollama blob missing for {model_name!r}: {blob}")
    return _resolve(str(blob))


def get_source_gguf_from_ollama(model_name: str) -> Path:
    """Return the local GGUF blob backing an Ollama model.

    Tries the on-disk store first (no server required); falls back to
    ``ollama show --modelfile`` (mirrors /rescaling) if the store layout can't be
    resolved. Surfaces a combined, actionable error if both fail.
    """
    store_error = None
    try:
        return get_source_gguf_from_ollama_store(model_name)
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError) as exc:
        store_error = exc

    try:
        modelfile = _run(["ollama", "show", "--modelfile", model_name])
    except (RuntimeError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"Could not resolve GGUF for Ollama model {model_name!r}.\n"
            f"  - local store: {store_error}\n"
            f"  - `ollama show`: {exc}\n"
            "Start the server (`ollama serve`) or `ollama pull` the model, "
            "or pass --model with a direct .gguf path."
        ) from exc

    for line in modelfile.splitlines():
        line = line.strip()
        if not line.startswith("FROM "):
            continue
        candidate = line.removeprefix("FROM ").strip().strip('"')
        path = _resolve(candidate)
        if path.exists():
            return path
    raise RuntimeError(f"Could not find local GGUF path for Ollama model: {model_name}")


@dataclass
class GGUFSource:
    """A resolved GGUF file ready to hand to transformers' gguf loader."""

    gguf_path: Path

    @property
    def directory(self) -> str:
        return str(self.gguf_path.parent)

    @property
    def filename(self) -> str:
        return self.gguf_path.name


def looks_like_gguf_request(model_path: str) -> bool:
    """True if ``model_path`` should be treated as GGUF/Ollama rather than HF dir.

    HF directories contain a ``config.json``; anything else (a ``.gguf`` file, an
    Ollama model name like ``llama3:8b``, or a directory of ``.gguf`` files) is
    routed through the GGUF path.
    """
    p = Path(model_path).expanduser()
    if p.is_dir() and (p / "config.json").exists():
        return False  # genuine HF checkpoint directory
    if str(model_path).endswith(".gguf"):
        return True
    if p.is_dir():
        return any(p.glob("*.gguf"))
    #Not an existing path: treat as an Ollama model name (e.g. "llama3:8b").
    return True


def resolve_gguf_source(model_path: str) -> GGUFSource:
    """Resolve ``model_path`` to a concrete GGUF file."""
    p = Path(model_path).expanduser()
    if str(model_path).endswith(".gguf") and p.exists():
        return GGUFSource(_resolve(model_path))
    if p.is_dir():
        ggufs = sorted(p.glob("*.gguf"), key=lambda f: f.stat().st_size, reverse=True)
        if ggufs:
            return GGUFSource(_resolve(str(ggufs[0])))
        raise FileNotFoundError(f"No .gguf files in directory: {p}")
    #Otherwise, an Ollama model name.
    return GGUFSource(get_source_gguf_from_ollama(model_path))


def dequantize_to_f16(source_gguf: Path, out_dir: Path,
                      llama_quantize: str = "llama-quantize") -> GGUFSource:
    """Dequantize a GGUF to F16, mirroring /rescaling's dequantize_to_f16.

    Used as a fallback when the in-memory loader cannot read the source quant
    type. Caches the F16 output so repeated runs skip the work.
    """
    if shutil.which(llama_quantize) is None:
        raise RuntimeError(
            f"Missing command in PATH: {llama_quantize}. Install llama.cpp tools "
            "or load the GGUF directly without --dequantize-f16."
        )
    out_dir = _resolve(str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    f16 = out_dir / f"{source_gguf.stem}.f16.gguf"
    if not f16.exists():
        _run([llama_quantize, "--allow-requantize", str(source_gguf), str(f16), "F16"])
    return GGUFSource(f16)
