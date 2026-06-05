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


def get_source_gguf_from_ollama(model_name: str) -> Path:
    """Return the local GGUF blob backing an Ollama model (mirrors /rescaling)."""
    modelfile = _run(["ollama", "show", "--modelfile", model_name])
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
