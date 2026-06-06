"""Stage 0 — initialize the library.

Creates the directory skeleton, mints the model and run IDs, and writes the
manifest set (library/model/architecture/tokenizer/extraction_config/
artifact_versions) that makes the whole library reproducible and auditable.
"""

from __future__ import annotations

from .. import ids
from ..config import ExtractionConfig, all_subdirs
from ..manifest import Library
from ..model_backend import ModelBackend
from . import register


@register("init", 0, requires=[], produces=["manifest/library.json"])
def run(library: Library, backend: ModelBackend, config: ExtractionConfig | None = None,
        **kwargs):
    config = config or ExtractionConfig()

    #Create the full directory skeleton up front so every stage can assume it.
    for d in all_subdirs():
        (library.root / d).mkdir(parents=True, exist_ok=True)

    arch = backend.architecture()
    #Parameter count from config geometry avoids loading weights at init time.
    approx_params = _approx_param_count(arch)
    mid = ids.model_id(
        name=str(config.model_path), architecture=arch.model_type,
        param_count=approx_params,
    )
    rid = ids.run_id(mid, config.to_dict())

    library.write_json("manifest/library.json", {
        "schema_version": "1.0.0",
        "model_id": mid,
        "run_id": rid,
        "library_kind": "MIBL",
    })
    library.write_json("manifest/model.json", {
        "model_id": mid,
        "model_path": config.model_path,
        "approx_param_count": approx_params,
        "reference_ollama_model": "llama3:8b",
        "backend": config.backend,
    })
    library.write_json("manifest/architecture.json", arch.to_dict())
    library.write_json("manifest/tokenizer.json", {
        "tokenizer_path": config.tokenizer_path,
        "vocab_size": arch.vocab_size,
    })
    library.write_json("manifest/extraction_config.json", config.to_dict())
    library.write_json("manifest/artifact_versions.json", {})

    library.log("init", "library initialized", model_id=mid, run_id=rid,
                num_layers=arch.num_layers)
    library.update_artifact_versions("init")
    return {"model_id": mid, "run_id": rid, "num_layers": arch.num_layers}


def _approx_param_count(arch) -> int:
    """Estimate parameter count from geometry (avoids loading weights)."""
    per_layer = (
        4 * arch.hidden_size * arch.hidden_size           # attn proj (approx)
        + 3 * arch.hidden_size * arch.intermediate_size   # mlp
    )
    embed = 2 * arch.vocab_size * arch.hidden_size
    return per_layer * arch.num_layers + embed
