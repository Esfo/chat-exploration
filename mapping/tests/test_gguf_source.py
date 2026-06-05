"""Tests for GGUF source resolution (no torch / no real model needed)."""

from __future__ import annotations

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
