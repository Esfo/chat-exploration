"""Library handle: manifests, run log, and the artifact index.

A ``Library`` is the single object every stage receives. It knows where the
library lives on disk, exposes resolved paths for each artifact, reads/writes
the JSON manifests that make the library reproducible, appends to the run log,
and maintains ``catalog/artifact_index.parquet`` so downstream pipelines can
discover which files exist, what schema version they carry, and how they are
partitioned.

The manifests directly satisfy the plan's "Manifests" artifact class:
library.json, model.json, architecture.json, tokenizer.json,
extraction_config.json, artifact_versions.json, and run_log.jsonl.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import schemas
from .config import ExtractionConfig
from .storage import write_parquet, read_parquet


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Library:
    """A handle to a Model Internal Behavior Library on disk."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    #--- path helpers -----------------------------------------------------
    def path(self, rel: str) -> Path:
        return self.root / rel

    def exists(self) -> bool:
        return (self.root / "manifest" / "library.json").exists()

    #--- manifest read/write ---------------------------------------------
    def write_json(self, rel: str, data: dict[str, Any]) -> None:
        p = self.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2, sort_keys=True))

    def read_json(self, rel: str) -> dict[str, Any]:
        return json.loads(self.path(rel).read_text())

    def model_id(self) -> str:
        return self.read_json("manifest/model.json")["model_id"]

    def run_id(self) -> str:
        return self.read_json("manifest/library.json")["run_id"]

    def config(self) -> ExtractionConfig:
        return ExtractionConfig.from_dict(self.read_json("manifest/extraction_config.json"))

    #--- run log ----------------------------------------------------------
    def log(self, stage: str, message: str, **fields: Any) -> None:
        entry = {"ts": _now(), "stage": stage, "message": message, **fields}
        log_path = self.path("manifest/run_log.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")

    #--- artifact index ---------------------------------------------------
    def register_artifact(
        self,
        rel: str,
        artifact_class: str,
        stage: str,
        row_count: int = -1,
        partition_keys: list[str] | None = None,
        fmt: str = "parquet",
    ) -> None:
        """Record (or update) an artifact in catalog/artifact_index.parquet."""
        idx_path = self.path("catalog/artifact_index.parquet")
        records: dict[str, dict[str, Any]] = {}
        if idx_path.exists():
            for row in read_parquet(idx_path).to_pylist():
                records[row["artifact_path"]] = row
        records[rel] = {
            "artifact_path": rel,
            "artifact_class": artifact_class,
            "format": fmt,
            "stage": stage,
            "schema_version": schemas.SCHEMA_VERSION,
            "row_count": int(row_count),
            "partition_keys": partition_keys or [],
            "created_at": _now(),
            "committed": True,
        }
        write_parquet(list(records.values()), idx_path, schema=schemas.ARTIFACT_INDEX)

    def commit_table(
        self,
        rel: str,
        rows,
        artifact_class: str,
        stage: str,
        partition_cols: list[str] | None = None,
    ) -> int:
        """Write a schema-checked Parquet artifact and register it atomically."""
        schema = schemas.SCHEMA_REGISTRY.get(rel)
        n = write_parquet(rows, self.path(rel), schema=schema, partition_cols=partition_cols)
        self.register_artifact(
            rel, artifact_class, stage, row_count=n, partition_keys=partition_cols
        )
        return n

    def update_artifact_versions(self, stage: str) -> None:
        """Stamp the stage as completed in artifact_versions.json."""
        rel = "manifest/artifact_versions.json"
        versions = self.read_json(rel) if self.path(rel).exists() else {}
        versions[stage] = {"schema_version": schemas.SCHEMA_VERSION, "completed_at": _now()}
        self.write_json(rel, versions)
