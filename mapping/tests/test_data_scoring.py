"""Tests for training-data scoring + filtering (atlas score-data / filter-data)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from atlas.evaluation.dataset import load_dataset
from atlas.evaluation.data_quality import quality_metrics
from atlas.evaluation.dedup import group_sizes, exact_hash
from atlas.evaluation.labeling import label_sample
from atlas.evaluation.category_recommend import recommend_categories
from atlas.evaluation.score_data import score_dataset, _percentile_ranks
from atlas.evaluation.filter_data import filter_dataset, route
from atlas.evaluation import schemas


def _write(tmp_path, name, rows):
    p = tmp_path / name
    with p.open("w") as fh:
        for r in rows:
            fh.write((r if isinstance(r, str) else json.dumps(r)) + "\n")
    return p


def _msgs(u, a):
    return {"messages": [{"role": "user", "content": u},
                         {"role": "assistant", "content": a}]}


#--------------------------------------------------------------------------
#dataset loader (tolerant)
#--------------------------------------------------------------------------

def test_loader_is_tolerant(tmp_path):
    p = _write(tmp_path, "d.jsonl", [
        _msgs("hi", "hello"),
        "{not valid json",
        {"prompt": "q?", "response": "a."},
    ])
    rows = load_dataset(p)
    assert len(rows) == 3
    assert not rows[1].valid_json
    #flat prompt/response normalised into turns
    assert rows[2].has_user_message and rows[2].has_assistant_message


#--------------------------------------------------------------------------
#quality metrics
#--------------------------------------------------------------------------

def test_quality_flags_non_answer_and_artifact(tmp_path):
    rows = load_dataset(_write(tmp_path, "d.jsonl", [
        _msgs("Explain X.", "ok"),                       # non-answer
        _msgs("Explain X.", "<div>click here</div> &nbsp;"),  # artifact
        _msgs("What is 2+2?", "Two plus two equals four, a basic arithmetic fact."),
    ]))
    q0 = quality_metrics(rows[0])
    q1 = quality_metrics(rows[1])
    q2 = quality_metrics(rows[2])
    assert q0.assistant_is_non_answer
    assert q1.assistant_contains_artifact
    assert q2.has_actual_information and not q2.assistant_is_non_answer


def test_repeats_prompt_detected(tmp_path):
    rows = load_dataset(_write(tmp_path, "d.jsonl", [
        _msgs("the quick brown fox jumps", "the quick brown fox jumps"),
    ]))
    assert quality_metrics(rows[0]).assistant_repeats_prompt


#--------------------------------------------------------------------------
#dedup
#--------------------------------------------------------------------------

def test_dedup_group_sizes():
    texts = ["hello world how are you", "hello world how are you", "totally different text here"]
    exact, near, ek, nk = group_sizes(texts)
    assert exact[0] == 2 and exact[1] == 2 and exact[2] == 1
    assert ek[0] == ek[1] != ek[2]


def test_near_duplicate_bucket():
    texts = ["the cat sat on the mat today",
             "the cat sat on the mat today please",  # near dup
             "quantum chromodynamics describes strong force"]
    _, near, _, nk = group_sizes(texts)
    assert near[0] >= 2 and nk[0] == nk[1]


#--------------------------------------------------------------------------
#labeling decision function
#--------------------------------------------------------------------------

def _base(**over):
    m = {
        "valid_json": True, "valid_message_roles": True, "has_user_message": True,
        "has_assistant_message": True, "empty_field_count": 0,
        "assistant_word_count": 20, "weird_character_ratio": 0.0,
        "assistant_repetition_5gram_rate": 0.0, "role_leakage": False,
        "assistant_is_non_answer": False, "assistant_repeats_prompt": False,
        "assistant_contains_artifact": False, "has_actual_information": True,
        "assistant_loss": 3.0, "loss_percentile": 0.5,
        "exact_dup_group_size": 1, "near_dup_group_size": 1,
        "is_duplicate_secondary": False, "category_frequency_percentile": 0.5,
        "model_weak": False, "too_short": False, "too_long": False,
    }
    m.update(over)
    return m


def test_label_drop_for_junk():
    r = label_sample(_base(assistant_is_non_answer=True, loss_percentile=0.95))
    assert r["action"] == "DROP" and "non_answer" in r["reason_codes"]


def test_label_keep_hard_high_loss_clean_rare():
    r = label_sample(_base(loss_percentile=0.9, category_frequency_percentile=0.05))
    assert r["action"] == "KEEP_HARD"
    assert "rare_category" in r["reason_codes"]


def test_high_loss_clean_is_not_dropped():
    #The core rule: high loss + clean != DROP.
    r = label_sample(_base(loss_percentile=0.99))
    assert r["action"] in ("KEEP_HARD",)


def test_label_fix_for_broken_roles():
    r = label_sample(_base(valid_message_roles=False))
    assert r["action"] == "FIX" and "bad_roles" in r["reason_codes"]


def test_label_downsample_near_dup():
    r = label_sample(_base(near_dup_group_size=4))
    assert r["action"] == "DOWNSAMPLE"


def test_label_drop_exact_duplicate_secondary():
    r = label_sample(_base(exact_dup_group_size=2, is_duplicate_secondary=True))
    assert r["action"] == "DROP" and "exact_duplicate" in r["reason_codes"]


def test_label_keep_clean():
    assert label_sample(_base())["action"] == "KEEP"


#--------------------------------------------------------------------------
#percentile ranks
#--------------------------------------------------------------------------

def test_percentile_ranks_basic():
    ranks = _percentile_ranks([1.0, 2.0, 3.0])
    assert ranks[0] == 0.0 and ranks[2] == 1.0


#--------------------------------------------------------------------------
#category recommendations
#--------------------------------------------------------------------------

def test_category_collect_more():
    stats = {"multi_turn": {"sample_count": 200, "clean_count": 100,
                            "duplicate_rate": 0.1, "mean_quality": 0.9, "drop_count": 5}}
    recs = recommend_categories(stats, {"multi_turn": 0.61})
    assert recs[0]["recommendation"] == "COLLECT_MORE"
    assert recs[0]["target_new_examples"] > 0


def test_category_downsample():
    stats = {"greetings": {"sample_count": 8000, "clean_count": 8000,
                           "duplicate_rate": 0.44, "mean_quality": 0.95, "drop_count": 0}}
    recs = recommend_categories(stats, {"greetings": 0.02})
    assert recs[0]["recommendation"] == "DOWNSAMPLE"
    assert recs[0]["drop_fraction"] > 0


#--------------------------------------------------------------------------
#score-data end to end
#--------------------------------------------------------------------------

def test_score_dataset_end_to_end(fake_backend, tmp_path):
    rows = [
        _msgs("What is overfitting?", "Overfitting is memorizing training data so it fails on new examples."),
        _msgs("What is overfitting?", "Overfitting is memorizing training data so it fails on new examples."),  # dup
        _msgs("Explain X.", "ok"),                                   # non-answer -> DROP
        _msgs("Say something", "blah blah blah blah blah blah blah blah blah blah"),  # repetition
        "{broken json",                                             # invalid -> DROP/FIX
    ]
    for i, r in enumerate(rows):
        if isinstance(r, dict):
            r["category"] = "direct_qa"
    p = _write(tmp_path, "train.jsonl", rows)
    summary = score_dataset(fake_backend, p, out_dir=tmp_path)

    assert summary["n_samples"] == 5
    counts = summary["action_counts"]
    assert sum(counts.values()) == 5
    assert counts.get("DROP", 0) >= 1

    base = Path(summary["_artifacts"]["dir"])
    for f in ("sample_scores.parquet", "sample_recommendations.jsonl",
              "category_recommendations.json", "drop_ids.txt", "keep_ids.txt",
              "review_ids.txt", "collect_more.json", "source_rows.jsonl"):
        assert (base / f).exists(), f
    table = pq.read_table(base / "sample_scores.parquet")
    assert table.num_rows == 5
    assert set(schemas.SAMPLE_SCORES.names) == set(table.schema.names)


#--------------------------------------------------------------------------
#filter-data
#--------------------------------------------------------------------------

def test_route_modes():
    assert route("DROP", 0.9, "conservative") == "drop"
    assert route("DROP", 0.6, "conservative") == "review"   # low conf -> review
    assert route("DROP", 0.6, "normal") == "drop"
    assert route("DOWNSAMPLE", 0.7, "normal") == "keep"
    assert route("DOWNSAMPLE", 0.7, "aggressive") == "drop"
    assert route("FIX", 0.7, "normal") == "review"
    assert route("KEEP_HARD", 0.7, "conservative") == "keep"


def test_filter_dataset_end_to_end(fake_backend, tmp_path):
    rows = [
        _msgs("What is overfitting?", "Overfitting is memorizing training data so it fails on new examples."),
        _msgs("Explain X.", "ok"),  # DROP
    ]
    for r in rows:
        r["category"] = "direct_qa"
    p = _write(tmp_path, "train.jsonl", rows)
    summary = score_dataset(fake_backend, p, out_dir=tmp_path)
    scores = Path(summary["_artifacts"]["dir"]) / "sample_recommendations.jsonl"

    res = filter_dataset(scores, mode="normal",
                         keep_out=tmp_path / "keep.jsonl",
                         drop_out=tmp_path / "drop.jsonl",
                         review_out=tmp_path / "review.jsonl",
                         dataset_path=p)
    total = sum(res["counts"].values())
    assert total == 2
    #Uses persisted source_rows when no dataset given.
    res2 = filter_dataset(scores, mode="conservative",
                          keep_out=tmp_path / "k2.jsonl",
                          drop_out=tmp_path / "d2.jsonl")
    assert sum(res2["counts"].values()) == 2
