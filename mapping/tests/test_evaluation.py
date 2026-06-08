"""Tests for the self-checking evaluation harness (atlas evaluate / compare).

The pure-text metric/check/format logic is asserted directly; the orchestration
(``evaluate_model`` / ``compare_models``) is driven end-to-end against the
torch-free ``FakeBackend`` so the full report + comparison artifacts are produced
and schema-valid without a real model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from atlas.evaluation.eval_pack import load_eval_pack, validate_item, EvalPackError
from atlas.evaluation.behavior import analyze_generation
from atlas.evaluation.checks import run_checks
from atlas.evaluation.chat_format import encode_sample, render_prompt_text, ASSISTANT_TAG
from atlas.evaluation.loss import assistant_loss
from atlas.evaluation.evaluate import evaluate_model, EvalConfig
from atlas.evaluation.compare import compare_models, _decide
from atlas.evaluation import schemas


EVALS_DIR = Path(__file__).resolve().parent.parent / "evals"


#--------------------------------------------------------------------------
#eval pack format
#--------------------------------------------------------------------------

def test_sample_packs_load_and_validate():
    for name in ("chat_basic.jsonl", "chat_multi_turn.jsonl", "refusal_boundaries.jsonl"):
        items = load_eval_pack(EVALS_DIR / name)
        assert items
        assert all(i.id for i in items)


def test_multi_turn_answer_and_prompt_split():
    items = load_eval_pack(EVALS_DIR / "chat_multi_turn.jsonl")
    item = items[0]
    #Ends with a user turn -> answer is the reference (None here), prompt is all.
    assert not item.ends_with_assistant
    assert item.prompt_messages()[-1]["role"] == "user"


def test_held_out_assistant_is_scored_as_answer():
    item = validate_item({
        "id": "x", "category": "c",
        "messages": [{"role": "user", "content": "hi"},
                     {"role": "assistant", "content": "hello there"}],
    })
    assert item.ends_with_assistant
    assert item.answer_text() == "hello there"
    assert item.prompt_messages() == [{"role": "user", "content": "hi"}]


def test_validation_rejects_bad_items():
    with pytest.raises(EvalPackError):
        validate_item({"id": "x", "messages": []})
    with pytest.raises(EvalPackError):
        validate_item({"id": "x", "messages": [{"role": "bot", "content": "y"}]})
    with pytest.raises(EvalPackError):
        validate_item({"messages": [{"role": "user", "content": "y"}]})


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "dup.jsonl"
    p.write_text(
        '{"id": "a", "category": "c", "messages": [{"role": "user", "content": "x"}]}\n'
        '{"id": "a", "category": "c", "messages": [{"role": "user", "content": "y"}]}\n')
    with pytest.raises(EvalPackError):
        load_eval_pack(p)


#--------------------------------------------------------------------------
#behaviour / degeneration metrics
#--------------------------------------------------------------------------

def test_clean_answer_is_not_degenerate():
    m = analyze_generation("Overfitting is when a model memorizes the training data.",
                           stopped_on_eos=True)
    assert not m.degenerate
    assert not m.empty_output
    assert m.unique_token_ratio > 0.5


def test_repetition_is_flagged():
    text = "the cat the cat the cat the cat the cat the cat the cat"
    m = analyze_generation(text, stopped_on_eos=True)
    assert m.repetition_5gram_rate > 0.2
    assert m.degenerate


def test_role_leak_and_loop_flagged():
    m = analyze_generation("Sure.\n" + ASSISTANT_TAG + "\nmore", stopped_on_eos=True)
    assert m.contains_assistant_role_loop
    assert m.degenerate
    assert not m.stopped_cleanly


def test_empty_output_flagged():
    m = analyze_generation("   ", stopped_on_eos=True)
    assert m.empty_output and m.malformed_output and m.degenerate


#--------------------------------------------------------------------------
#trait checks
#--------------------------------------------------------------------------

def test_must_include_check():
    m = analyze_generation("Your dog's name is Orbit.", stopped_on_eos=True)
    ok = run_checks("Your dog's name is Orbit.", {"must_include": ["Orbit"]}, m)
    assert ok.must_include_pass and ok.behavior_pass
    bad = run_checks("I forget.", {"must_include": ["Orbit"]},
                     analyze_generation("I forget.", stopped_on_eos=True))
    assert not bad.must_include_pass and not bad.behavior_pass
    assert any(r.startswith("missing:") for r in bad.reason_codes)


def test_should_not_answer_check():
    text = "Step 1: insert the pick into the lock."
    m = analyze_generation(text, stopped_on_eos=True)
    res = run_checks(text, {"should_answer": False, "must_not_include": ["insert the pick"]}, m)
    assert not res.behavior_pass
    assert "answered_when_should_refuse" in res.reason_codes


def test_max_words_check():
    text = " ".join(["word"] * 50)
    m = analyze_generation(text, stopped_on_eos=True)
    res = run_checks(text, {"max_words": 10}, m)
    assert not res.length_pass and "too_long" in res.reason_codes


#--------------------------------------------------------------------------
#loss
#--------------------------------------------------------------------------

def test_assistant_loss_masks_prompt(fake_backend):
    msgs = [{"role": "user", "content": "What is overfitting?"}]
    lm = assistant_loss(fake_backend, msgs, "Overfitting is memorizing training data.")
    assert lm.tokens_scored > 0
    assert lm.assistant_loss > 0
    assert lm.sample_length > lm.tokens_scored  # prompt tokens are not scored
    assert lm.assistant_perplexity > 1.0


def test_render_prompt_ends_with_assistant_marker():
    text = render_prompt_text([{"role": "user", "content": "hi"}])
    assert text.rstrip().endswith(ASSISTANT_TAG)


#--------------------------------------------------------------------------
#evaluate orchestration
#--------------------------------------------------------------------------

def test_evaluate_model_end_to_end(fake_backend, tmp_path):
    summary = evaluate_model(fake_backend, EVALS_DIR / "chat_basic.jsonl",
                             out_dir=tmp_path, model_id="fake_a",
                             cfg=EvalConfig(max_new_tokens=16))
    assert summary["n_items"] == 4
    assert summary["recommendation"] in ("PASS", "WARNING", "FAIL")
    assert 0.0 <= summary["metrics"]["behavior_pass_rate"] <= 1.0
    assert "direct_qa" in summary["category_pass_rates"]

    base = Path(summary["_artifacts"]["dir"])
    assert (base / "eval_summary.json").exists()
    assert (base / "generations.jsonl").exists()
    table = pq.read_table(base / "eval_by_item.parquet")
    assert table.num_rows == 4
    assert set(schemas.EVAL_BY_ITEM.names) == set(table.schema.names)


#--------------------------------------------------------------------------
#compare orchestration
#--------------------------------------------------------------------------

def test_decide_prefers_lower_loss_and_passing():
    a = {"assistant_loss": 3.0, "behavior_pass": True, "degenerate": False,
         "empty_output": False, "malformed_output": False}
    b = {"assistant_loss": 5.0, "behavior_pass": False, "degenerate": False,
         "empty_output": False, "malformed_output": False}
    winner, decision, conf, reasons = _decide(a, b)
    assert winner == "model_a"
    assert decision in ("A_STRONG_WIN", "A_WEAK_WIN")


def test_decide_both_bad():
    a = {"assistant_loss": 3.0, "behavior_pass": False, "degenerate": True,
         "empty_output": True, "malformed_output": True}
    b = {"assistant_loss": 4.0, "behavior_pass": False, "degenerate": True,
         "empty_output": True, "malformed_output": True}
    _, decision, _, _ = _decide(a, b)
    assert decision == "BOTH_BAD"


def test_decide_metrics_disagree():
    #A has lower loss but B passes checks and A is not degenerate -> conflict.
    a = {"assistant_loss": 2.0, "behavior_pass": False, "degenerate": False,
         "empty_output": False, "malformed_output": False}
    b = {"assistant_loss": 5.0, "behavior_pass": True, "degenerate": False,
         "empty_output": False, "malformed_output": False}
    _, decision, _, _ = _decide(a, b)
    assert decision == "METRICS_DISAGREE"


def test_compare_models_end_to_end(fake_backend, fake_config, tmp_path):
    from tests.conftest import FakeBackend
    backend_a = fake_backend
    backend_b = FakeBackend(fake_config)
    summary = compare_models(backend_a, backend_b, EVALS_DIR / "chat_basic.jsonl",
                             out_dir=tmp_path, model_a_id="fake_a", model_b_id="fake_b",
                             cfg=EvalConfig(max_new_tokens=16))
    assert summary["overall_winner"] in ("model_a", "model_b", "tie")
    wr = summary["win_rate"]
    assert abs(wr["model_a"] + wr["model_b"] + wr["tie"] - 1.0) < 1e-9

    base = Path(summary["_artifacts"]["dir"])
    for f in ("comparison_summary.json", "comparison_by_item.parquet",
              "comparison_by_category.parquet", "regression_cases.jsonl",
              "improvement_cases.jsonl", "both_bad_cases.jsonl"):
        assert (base / f).exists(), f
    item_tbl = pq.read_table(base / "comparison_by_item.parquet")
    assert item_tbl.num_rows == 4
    assert set(schemas.COMPARISON_BY_ITEM.names) == set(item_tbl.schema.names)


def test_identical_models_mostly_tie(fake_backend, fake_config, tmp_path):
    from tests.conftest import FakeBackend
    summary = compare_models(fake_backend, FakeBackend(fake_config),
                             EVALS_DIR / "chat_basic.jsonl", out_dir=tmp_path,
                             model_a_id="same1", model_b_id="same2",
                             cfg=EvalConfig(max_new_tokens=16))
    #Two identical fake backends should not produce a decisive overall winner.
    assert summary["overall_winner"] == "tie"
