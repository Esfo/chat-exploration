"""``atlas score-data`` — score every training row and label it for action.

This is the stage that decides what training data is worth keeping. For each row
it computes model-free text-quality metrics, exact/near-duplicate group sizes,
and (with the model) assistant-only loss, then assigns one action label per row
via :mod:`labeling`. Dataset-relative context — loss percentile, category
frequency percentile, per-category weakness — is computed across the whole
dataset so "high loss" and "rare category" mean *relative to this corpus*.

Outputs (under ``<out>/data_scores/``):
    sample_scores.parquet          - one row per training sample (the drill-down)
    sample_recommendations.jsonl   - {sample_id, action, confidence, reason_codes, metrics}
    category_recommendations.json  - COLLECT_MORE / CLEAN_EXISTING / DOWNSAMPLE / HOLD
    drop_ids.txt / keep_ids.txt / review_ids.txt
    collect_more.json
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..storage import write_parquet
from .chat_format import encode_sample
from .data_quality import quality_metrics
from .dataset import load_dataset, TrainRow
from .dedup import group_sizes
from .labeling import label_sample
from .category_recommend import recommend_categories
from .loss import assistant_loss
from . import schemas


@dataclass
class ScoreConfig:
    too_short_words: int = 3
    too_long_tokens: int = 2048
    #A category is "weak" when its mean assistant loss is at/above the dataset
    #median — i.e. the model is comparatively worse there.
    pass


def _percentile_ranks(values: list[float]) -> list[float]:
    """Fractional rank in [0,1] for each value (ties share the average rank).
    NaNs map to NaN. 0 = smallest, 1 = largest."""
    finite = [(v, i) for i, v in enumerate(values) if isinstance(v, (int, float)) and math.isfinite(v)]
    ranks = [float("nan")] * len(values)
    if not finite:
        return ranks
    order = sorted(finite, key=lambda t: t[0])
    n = len(order)
    for rank, (_, i) in enumerate(order):
        ranks[i] = rank / (n - 1) if n > 1 else 0.5
    return ranks


def score_dataset(backend, dataset_path, *, out_dir, cfg: ScoreConfig | None = None,
                  eval_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or ScoreConfig()
    rows = load_dataset(dataset_path)
    n = len(rows)

    #--- per-row text quality + token length + assistant loss ------------
    qms = [quality_metrics(r) for r in rows]
    full_texts = [r.last_user_text() + "\n" + r.assistant_text() for r in rows]
    exact_sizes, near_sizes, exact_keys, near_keys = group_sizes(full_texts)

    losses: list[float] = []
    token_lengths: list[int] = []
    for r in rows:
        answer = r.assistant_text()
        if answer and r.prompt_messages() and r.has_assistant_message and r.valid_message_roles:
            try:
                lm = assistant_loss(backend, r.prompt_messages(), answer)
                losses.append(lm.assistant_loss)
                token_lengths.append(lm.sample_length)
            except Exception:  # noqa: BLE001 — a broken row stays scored as junk
                losses.append(float("nan"))
                token_lengths.append(len(encode_sample(backend.tokenizer, r.prompt_messages(), answer).full_ids)
                                     if r.prompt_messages() else 0)
        else:
            losses.append(float("nan"))
            token_lengths.append(0)

    loss_pcts = _percentile_ranks(losses)

    #--- category frequency percentile + per-category weakness -----------
    cat_counts: dict[str, int] = {}
    for r in rows:
        c = _category(r)
        cat_counts[c] = cat_counts.get(c, 0) + 1
    #Rare = small count -> low percentile. Rank categories by count.
    cat_rank = _percentile_ranks([float(cat_counts[_category(r)]) for r in rows])

    finite_losses = [l for l in losses if math.isfinite(l)]
    median_loss = sorted(finite_losses)[len(finite_losses) // 2] if finite_losses else float("nan")
    cat_loss_sum: dict[str, float] = {}
    cat_loss_n: dict[str, int] = {}
    for r, l in zip(rows, losses):
        if math.isfinite(l):
            c = _category(r)
            cat_loss_sum[c] = cat_loss_sum.get(c, 0.0) + l
            cat_loss_n[c] = cat_loss_n.get(c, 0) + 1
    cat_weak = {c: (cat_loss_sum[c] / cat_loss_n[c] >= median_loss)
                for c in cat_loss_sum} if finite_losses else {}

    #--- secondary-duplicate flags (all but first occurrence) ------------
    seen_exact: set[str] = set()
    secondary = []
    for k in exact_keys:
        secondary.append(k in seen_exact)
        seen_exact.add(k)

    #--- assemble per-row metrics, label, and score rows -----------------
    score_rows: list[dict[str, Any]] = []
    recs: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        q = qms[i].to_dict()
        cat = _category(r)
        too_short = q["assistant_word_count"] < cfg.too_short_words
        too_long = token_lengths[i] > cfg.too_long_tokens
        metrics = {
            **q,
            "assistant_loss": losses[i],
            "loss_percentile": loss_pcts[i],
            "token_length": token_lengths[i],
            "exact_dup_group_size": exact_sizes[i],
            "near_dup_group_size": near_sizes[i],
            "is_duplicate_secondary": secondary[i],
            "category_frequency_percentile": cat_rank[i],
            "model_weak": cat_weak.get(cat, False),
            "too_short": too_short,
            "too_long": too_long,
        }
        label = label_sample(metrics)
        score_rows.append(_score_row(r.id, cat, label, metrics))
        recs.append({
            "sample_id": r.id,
            "action": label["action"],
            "confidence": label["confidence"],
            "reason_codes": label["reason_codes"],
            "metrics": {
                "assistant_loss": _clean(losses[i]),
                "loss_percentile": _clean(loss_pcts[i]),
                "repetition_5gram_rate": _clean(q["assistant_repetition_5gram_rate"]),
                "exact_dup_group_size": exact_sizes[i],
                "near_dup_group_size": near_sizes[i],
                "token_length": token_lengths[i],
                "category_frequency_percentile": _clean(cat_rank[i]),
            },
        })

    #--- category recommendations ----------------------------------------
    cat_stats = _category_stats(rows, score_rows, exact_sizes, near_sizes)
    eval_fail = _eval_failures(eval_summary)
    cat_recs = recommend_categories(cat_stats, eval_fail)
    collect_more = [c for c in cat_recs if c["recommendation"] == "COLLECT_MORE"]

    #--- write outputs ---------------------------------------------------
    base = Path(out_dir) / "data_scores"
    base.mkdir(parents=True, exist_ok=True)
    write_parquet(score_rows, base / "sample_scores.parquet", schema=schemas.SAMPLE_SCORES)
    _write_jsonl(base / "sample_recommendations.jsonl", recs)
    (base / "category_recommendations.json").write_text(json.dumps(cat_recs, indent=2))
    (base / "collect_more.json").write_text(json.dumps(collect_more, indent=2))
    _write_ids(base / "drop_ids.txt", recs, {"DROP"})
    _write_ids(base / "keep_ids.txt", recs, {"KEEP", "KEEP_HARD"})
    _write_ids(base / "review_ids.txt", recs, {"REVIEW", "FIX"})
    #Persist the exact source lines so `filter-data` / `export-data-filter` can
    #reconstruct the kept/dropped datasets without re-reading the original file.
    _write_jsonl(base / "source_rows.jsonl",
                 [{"sample_id": r.id, "raw_text": r.raw_text} for r in rows])

    summary = _summary(score_rows, cat_recs, n)
    summary["_artifacts"] = {"dir": str(base)}
    (base / "score_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _category(r: TrainRow) -> str:
    if isinstance(r.raw, dict) and r.raw.get("category"):
        return str(r.raw["category"])
    return "uncategorized"


def _clean(v):
    return None if isinstance(v, float) and not math.isfinite(v) else v


def _score_row(rid, cat, label, m) -> dict[str, Any]:
    return {
        "sample_id": rid, "category": cat,
        "action": label["action"], "confidence": label["confidence"],
        "assistant_loss": m["assistant_loss"], "loss_percentile": m["loss_percentile"],
        "token_length": int(m["token_length"]),
        "assistant_word_count": int(m["assistant_word_count"]),
        "exact_dup_group_size": int(m["exact_dup_group_size"]),
        "near_dup_group_size": int(m["near_dup_group_size"]),
        "is_duplicate_secondary": bool(m["is_duplicate_secondary"]),
        "repetition_5gram_rate": m["assistant_repetition_5gram_rate"],
        "weird_character_ratio": m["weird_character_ratio"],
        "role_leakage": bool(m["role_leakage"]),
        "assistant_is_non_answer": bool(m["assistant_is_non_answer"]),
        "assistant_repeats_prompt": bool(m["assistant_repeats_prompt"]),
        "assistant_contains_artifact": bool(m["assistant_contains_artifact"]),
        "has_actual_information": bool(m["has_actual_information"]),
        "too_short": bool(m["too_short"]), "too_long": bool(m["too_long"]),
        "valid_json": bool(m["valid_json"]),
        "valid_message_roles": bool(m["valid_message_roles"]),
        "category_frequency_percentile": m["category_frequency_percentile"],
        "model_weak": bool(m["model_weak"]),
        "reason_codes": label["reason_codes"],
    }


def _category_stats(rows, score_rows, exact_sizes, near_sizes):
    stats: dict[str, dict[str, Any]] = {}
    for r, sr, ex in zip(rows, score_rows, exact_sizes):
        c = sr["category"]
        s = stats.setdefault(c, {"sample_count": 0, "clean_count": 0, "drop_count": 0,
                                 "_dup": 0, "_qual": 0.0})
        s["sample_count"] += 1
        if sr["action"] in ("KEEP", "KEEP_HARD"):
            s["clean_count"] += 1
        if sr["action"] == "DROP":
            s["drop_count"] += 1
        if ex > 1:
            s["_dup"] += 1
        #Cheap per-row quality proxy: 1 minus failure flags.
        bad = sum([sr["role_leakage"], sr["assistant_is_non_answer"],
                   sr["assistant_repeats_prompt"], sr["assistant_contains_artifact"],
                   not sr["has_actual_information"]])
        s["_qual"] += max(0.0, 1.0 - bad / 5.0)
    for c, s in stats.items():
        n = s["sample_count"]
        s["duplicate_rate"] = s.pop("_dup") / n if n else 0.0
        s["mean_quality"] = s.pop("_qual") / n if n else 0.0
    return stats


def _eval_failures(eval_summary):
    if not eval_summary:
        return None
    cats = eval_summary.get("category_pass_rates", {})
    return {c: 1.0 - v.get("behavior_pass_rate", 0.0) for c, v in cats.items()}


def _summary(score_rows, cat_recs, n):
    counts: dict[str, int] = {}
    for sr in score_rows:
        counts[sr["action"]] = counts.get(sr["action"], 0) + 1
    reason_counts: dict[str, int] = {}
    for sr in score_rows:
        for rc in sr["reason_codes"]:
            fam = rc.split("=", 1)[0]
            reason_counts[fam] = reason_counts.get(fam, 0) + 1
    return {
        "n_samples": n,
        "action_counts": counts,
        "drop_reason_counts": dict(sorted(reason_counts.items(),
                                          key=lambda kv: kv[1], reverse=True)),
        "category_recommendations": cat_recs,
    }


def _write_jsonl(path, rows):
    with Path(path).open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _write_ids(path, recs, actions):
    ids = [r["sample_id"] for r in recs if r["action"] in actions]
    Path(path).write_text("\n".join(ids) + ("\n" if ids else ""))
