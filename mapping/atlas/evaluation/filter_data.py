"""``atlas filter-data`` / ``atlas export-data-filter`` — split a dataset.

Given the per-row recommendations from ``score-data``, route each original source
line to a kept / dropped / review-needed file. The default mode is **conservative**
on purpose: early in a cleaning loop you do not want the system deleting
hard-but-useful data, so conservative only drops high-confidence DROP rows and
sends everything uncertain to review.

Modes:
  conservative  drop only high-confidence DROP; FIX/REVIEW -> review; rest kept.
  normal        drop DROP; FIX/REVIEW -> review; KEEP/KEEP_HARD/DOWNSAMPLE kept.
  aggressive    additionally drop DOWNSAMPLE rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONSERVATIVE_DROP_CONF = 0.85


def route(action: str, confidence: float, mode: str) -> str:
    """Return 'keep' | 'drop' | 'review' for one row under ``mode``."""
    if action == "DROP":
        if mode == "conservative":
            return "drop" if confidence >= CONSERVATIVE_DROP_CONF else "review"
        return "drop"
    if action in ("FIX", "REVIEW"):
        return "review"
    if action == "DOWNSAMPLE":
        return "drop" if mode == "aggressive" else "keep"
    #KEEP, KEEP_HARD, COLLECT_MORE_LIKE_THIS
    return "keep"


def _load_recs(scores_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in scores_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[r["sample_id"]] = r
    return out


def _source_lines(dataset_path, scores_path: Path):
    """Yield ``(sample_id, raw_text)`` in original order.

    Prefers an explicit ``--dataset`` file; otherwise falls back to the
    ``source_rows.jsonl`` persisted next to the scores by ``score-data``.
    """
    if dataset_path:
        from .dataset import load_dataset
        for r in load_dataset(dataset_path):
            yield r.id, r.raw_text
        return
    src = scores_path.parent / "source_rows.jsonl"
    if not src.exists():
        raise FileNotFoundError(
            f"no --dataset given and {src} not found; re-run score-data or pass --dataset")
    for line in src.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        yield o["sample_id"], o["raw_text"]


def filter_dataset(scores_path, *, mode: str = "conservative", keep_out, drop_out,
                   review_out=None, dataset_path=None) -> dict[str, Any]:
    scores_path = Path(scores_path)
    recs = _load_recs(scores_path)
    counts = {"keep": 0, "drop": 0, "review": 0, "unscored": 0}

    keep_fh = Path(keep_out).open("w")
    drop_fh = Path(drop_out).open("w")
    review_fh = Path(review_out).open("w") if review_out else None
    try:
        for sample_id, raw in _source_lines(dataset_path, scores_path):
            rec = recs.get(sample_id)
            if rec is None:
                #Unscored rows are kept (fail-safe) but counted.
                counts["unscored"] += 1
                keep_fh.write(raw + "\n")
                continue
            dest = route(rec["action"], rec.get("confidence", 0.0), mode)
            counts[dest] += 1
            if dest == "keep":
                keep_fh.write(raw + "\n")
            elif dest == "drop":
                drop_fh.write(raw + "\n")
            else:
                (review_fh or keep_fh).write(raw + "\n")
    finally:
        keep_fh.close()
        drop_fh.close()
        if review_fh:
            review_fh.close()

    return {"mode": mode, "counts": counts,
            "keep_out": str(keep_out), "drop_out": str(drop_out),
            "review_out": str(review_out) if review_out else None}
