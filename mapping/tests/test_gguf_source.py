"""Tests for GGUF source resolution (no torch / no real model needed)."""

from __future__ import annotations

import json

from atlas import gguf_source


def test_hf_dir_is_not_gguf(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert gguf_source.looks_like_gguf_request(str(tmp_path)) is False


def test_gguf_file_is_gguf(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF")
    assert gguf_source.looks_like_gguf_request(str(f)) is True
    src = gguf_source.resolve_gguf_source(str(f))
    assert src.filename == "model.gguf"
    assert src.directory == str(tmp_path)


def test_dir_with_gguf_picks_largest(tmp_path):
    small = tmp_path / "small.gguf"
    big = tmp_path / "big.gguf"
    small.write_bytes(b"x")
    big.write_bytes(b"x" * 100)
    assert gguf_source.looks_like_gguf_request(str(tmp_path)) is True
    assert gguf_source.resolve_gguf_source(str(tmp_path)).filename == "big.gguf"


def test_ollama_name_treated_as_gguf():
    #A bare model name that is not an existing path is routed to the Ollama path.
    assert gguf_source.looks_like_gguf_request("llama3:8b") is True


def test_ollama_store_resolution_without_server(tmp_path, monkeypatch):
    #Build a fake ~/.ollama/models layout: manifest + content-addressed blob.
    models = tmp_path / "models"
    manifest_dir = models / "manifests" / "registry.ollama.ai" / "library" / "llama3"
    manifest_dir.mkdir(parents=True)
    digest = "sha256:abc123"
    blob = models / "blobs" / "sha256-abc123"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"GGUF-weights")
    (manifest_dir / "8b").write_text(json.dumps({
        "layers": [
            {"mediaType": "application/vnd.ollama.image.license", "digest": "sha256:lic"},
            {"mediaType": "application/vnd.ollama.image.model", "digest": digest},
        ]
    }))
    monkeypatch.setenv("OLLAMA_MODELS", str(models))

    resolved = gguf_source.get_source_gguf_from_ollama_store("llama3:8b")
    assert resolved == blob.resolve()

