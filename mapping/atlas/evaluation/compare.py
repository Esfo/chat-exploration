"""``atlas compare`` — run two models over the same eval pack and decide.

Both models see exactly the same prompts. For each item we keep both losses, both
generations, both check results, then pick a per-item winner using more than just
loss: a model wins when it has lower assistant loss *and* passes at least as many
behaviour checks *and* does not have worse degeneration. Disagreements between the
loss signal and the behaviour signal are surfaced as ``METRICS_DISAGREE`` rather
than silently resolved.

Outputs (under ``<out>/``):
    eval_runs/<model_a>/...                 - full per-model eval runs
    eval_runs/<model_b>/...
    comparisons/<a>_vs_<b>/
        comparison_summary.json
        comparison_by_item.parquet
        comparison_by_category.parquet
        regression_cases.jsonl
        improvement_cases.jsonl
        both_bad_cases.jsonl
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..storage import write_parquet
from .evaluate import evaluate_model, EvalConfig, _slug
from .eval_pack import load_eval_pack
from . import schemas

#How much lower assistant loss must be to count as a meaningful difference.
_LOSS_EPS = 0.05
_STRONG_LOSS = 0.5


def _decide(a, b):
    """Return (winner, decision, confidence, reason_codes) for one item.

    ``a`` and ``b`` are the per-item eval rows for model A and model B.
    """
    reasons: list[str] = []
    la, lb = a["assistant_loss"], b["assistant_loss"]
    pa, pb = a["behavior_pass"], b["behavior_pass"]
    da, db = a["degenerate"], b["degenerate"]

    a_bad = da or (a["empty_output"] or a["malformed_output"])
    b_bad = db or (b["empty_output"] or b["malformed_output"])
    if a_bad and b_bad:
        return "tie", "BOTH_BAD", 0.5, ["both_degenerate"]

    have_loss = math.isfinite(la) and math.isfinite(lb)
    loss_delta = (la - lb) if have_loss else 0.0  # positive => B better
    loss_favors = 0
    if have_loss:
        if loss_delta > _LOSS_EPS:
            loss_favors = +1  # B
            reasons.append("b_lower_assistant_loss")
        elif loss_delta < -_LOSS_EPS:
            loss_favors = -1  # A
            reasons.append("a_lower_assistant_loss")

    behave_favors = 0
    if pa and not pb:
        behave_favors = -1
        reasons.append("a_passes_more_checks")
    elif pb and not pa:
        behave_favors = +1
        reasons.append("b_passes_more_checks")
    #Degeneration is a behaviour signal too.
    if a_bad and not b_bad:
        behave_favors = +1
        reasons.append("a_degenerate")
    elif b_bad and not a_bad:
        behave_favors = -1
        reasons.append("b_degenerate")

    #Combine. If the two signals point the same way -> clear win; if one is
    #neutral, the other decides; if they conflict -> METRICS_DISAGREE.
    signals = [s for s in (loss_favors, behave_favors) if s != 0]
    if not signals:
        return "tie", "TIE", 0.4, reasons or ["no_meaningful_difference"]
    if loss_favors and behave_favors and loss_favors != behave_favors:
        return "tie", "METRICS_DISAGREE", 0.3, reasons

    direction = signals[0]  # both agree or only one present
    strong = (abs(loss_delta) >= _STRONG_LOSS) and (loss_favors == direction) and \
        (behave_favors in (0, direction))
    conf = 0.85 if strong else 0.6
    if direction > 0:
        return "model_b", ("B_STRONG_WIN" if strong else "B_WEAK_WIN"), conf, reasons
    return "model_a", ("A_STRONG_WIN" if strong else "A_WEAK_WIN"), conf, reasons


def compare_models(backend_a, backend_b, eval_pack, *, out_dir: str | Path,
                   model_a_id: str | None = None, model_b_id: str | None = None,
                   cfg: EvalConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EvalConfig()
    items = load_eval_pack(eval_pack) if isinstance(eval_pack, (str, Path)) else list(eval_pack)
    model_a_id = model_a_id or getattr(backend_a, "model_path", "model_a")
    model_b_id = model_b_id or getattr(backend_b, "model_path", "model_b")

    sum_a = evaluate_model(backend_a, items, out_dir=out_dir, model_id=model_a_id, cfg=cfg)
    sum_b = evaluate_model(backend_b, items, out_dir=out_dir, model_id=model_b_id, cfg=cfg)

    rows_a = _load_rows(out_dir, model_a_id)
    rows_b = _load_rows(out_dir, model_b_id)
    by_id_b = {r["item_id"]: r for r in rows_b}

    comparison_id = f"cmp.{_slug(str(model_a_id))}_vs_{_slug(str(model_b_id))}"
    item_rows: list[dict[str, Any]] = []
    regressions, improvements, both_bad = [], [], []
    for a in rows_a:
        b = by_id_b.get(a["item_id"])
        if b is None:
            continue
        winner, decision, conf, reasons = _decide(a, b)
        la, lb = a["assistant_loss"], b["assistant_loss"]
        delta = (la - lb) if (math.isfinite(la) and math.isfinite(lb)) else float("nan")
        row = {
            "comparison_id": comparison_id,
            "item_id": a["item_id"], "category": a["category"],
            "difficulty": a["difficulty"],
            "model_a_assistant_loss": la, "model_b_assistant_loss": lb,
            "assistant_loss_delta": delta,
            "model_a_behavior_pass": a["behavior_pass"],
            "model_b_behavior_pass": b["behavior_pass"],
            "model_a_degenerate": a["degenerate"],
            "model_b_degenerate": b["degenerate"],
            "winner": winner, "decision": decision, "confidence": conf,
            "reason_codes": reasons,
        }
        item_rows.append(row)
        case = {
            "item_id": a["item_id"], "category": a["category"], "decision": decision,
            "model_a_generated_text": a["generated_text"][:400],
            "model_b_generated_text": b["generated_text"][:400],
            "model_a_assistant_loss": la, "model_b_assistant_loss": lb,
            "reason_codes": reasons,
        }
        #Regression = A passed but B fails / B newly degenerate.
        if decision in ("A_STRONG_WIN", "A_WEAK_WIN"):
            regressions.append(case)
        elif decision in ("B_STRONG_WIN", "B_WEAK_WIN"):
            improvements.append(case)
        elif decision == "BOTH_BAD":
            both_bad.append(case)

    summary = _summarize_comparison(
        comparison_id, item_rows, str(model_a_id), str(model_b_id), sum_a, sum_b)
    cat_rows = _by_category(comparison_id, item_rows)

    base = Path(out_dir) / "comparisons" / f"{_slug(str(model_a_id))}_vs_{_slug(str(model_b_id))}"
    base.mkdir(parents=True, exist_ok=True)
    write_parquet(item_rows, base / "comparison_by_item.parquet",
                  schema=schemas.COMPARISON_BY_ITEM)
    write_parquet(cat_rows, base / "comparison_by_category.parquet",
                  schema=schemas.COMPARISON_BY_CATEGORY)
    (base / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    _write_jsonl(base / "regression_cases.jsonl", regressions)
    _write_jsonl(base / "improvement_cases.jsonl", improvements)
    _write_jsonl(base / "both_bad_cases.jsonl", both_bad)
    summary["_artifacts"] = {"dir": str(base)}
    return summary


def _load_rows(out_dir, model_id):
    import pyarrow.parquet as pq
    p = Path(out_dir) / "eval_runs" / _slug(str(model_id)) / "eval_by_item.parquet"
    return pq.read_table(p).to_pylist()


def _write_jsonl(path, rows):
    with Path(path).open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _rate(flags):
    flags = list(flags)
    return (sum(1 for f in flags if f) / len(flags)) if flags else 0.0


def _by_category(comparison_id, item_rows):
    groups: dict[str, list[dict]] = {}
    for r in item_rows:
        groups.setdefault(r["category"], []).append(r)
    out = []
    for cat, grp in groups.items():
        deltas = [r["assistant_loss_delta"] for r in grp
                  if math.isfinite(r["assistant_loss_delta"])]
        out.append({
            "comparison_id": comparison_id, "category": cat, "n_items": len(grp),
            "a_win_rate": _rate(r["winner"] == "model_a" for r in grp),
            "b_win_rate": _rate(r["winner"] == "model_b" for r in grp),
            "tie_rate": _rate(r["winner"] == "tie" for r in grp),
            "assistant_loss_delta_mean": (sum(deltas) / len(deltas)) if deltas else float("nan"),
            "behavior_pass_delta": _rate(r["model_b_behavior_pass"] for r in grp)
            - _rate(r["model_a_behavior_pass"] for r in grp),
        })
    return out


def _summarize_comparison(comparison_id, item_rows, model_a_id, model_b_id,
                          sum_a, sum_b):
    n = len(item_rows)
    a_wins = sum(1 for r in item_rows if r["winner"] == "model_a")
    b_wins = sum(1 for r in item_rows if r["winner"] == "model_b")
    ties = n - a_wins - b_wins
    deltas = [r["assistant_loss_delta"] for r in item_rows
              if math.isfinite(r["assistant_loss_delta"])]
    deltas_sorted = sorted(deltas)
    overall = "tie"
    if b_wins > a_wins and (b_wins - a_wins) / max(n, 1) >= 0.1:
        overall = "model_b"
    elif a_wins > b_wins and (a_wins - b_wins) / max(n, 1) >= 0.1:
        overall = "model_a"

    regressions = [r["item_id"] for r in item_rows
                   if r["decision"] in ("A_STRONG_WIN", "A_WEAK_WIN")]
    improvements = [r["item_id"] for r in item_rows
                    if r["decision"] in ("B_STRONG_WIN", "B_WEAK_WIN")]
    return {
        "comparison_id": comparison_id,
        "model_a": model_a_id, "model_b": model_b_id,
        "n_items": n,
        "overall_winner": overall,
        "win_rate": {"model_a": a_wins / max(n, 1), "model_b": b_wins / max(n, 1),
                     "tie": ties / max(n, 1)},
        "assistant_loss_delta_mean": (sum(deltas) / len(deltas)) if deltas else None,
        "assistant_loss_delta_median": (deltas_sorted[len(deltas_sorted) // 2]
                                        if deltas_sorted else None),
        "behavior_pass_delta": (sum_b["metrics"]["behavior_pass_rate"]
                                - sum_a["metrics"]["behavior_pass_rate"]),
        "repetition_failure_delta": (sum_b["metrics"]["repetition_failure_rate"]
                                     - sum_a["metrics"]["repetition_failure_rate"]),
        "format_failure_delta": (sum_b["metrics"]["format_failure_rate"]
                                 - sum_a["metrics"]["format_failure_rate"]),
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
        "decision_counts": _decision_counts(item_rows),
        "model_a_recommendation": sum_a["recommendation"],
        "model_b_recommendation": sum_b["recommendation"],
    }


def _decision_counts(item_rows):
    counts: dict[str, int] = {}
    for r in item_rows:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    return counts
