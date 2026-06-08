"""Category-level data-collection recommendations (Part 9 of the plan).

Dropping rows is only half the feedback loop; the other half is *"you need more
data like this."* For each category we combine training-side signals (sample
count, clean fraction, duplicate rate, mean quality) with — when available —
eval-side signals (the category's failure rate from an ``atlas evaluate`` run) to
emit one of: COLLECT_MORE, CLEAN_EXISTING, DOWNSAMPLE, or HOLD, each with a
human-readable reason and a target count / drop fraction.
"""

from __future__ import annotations

from typing import Any

#A category is "weak" on eval when it fails at least this fraction of items.
EVAL_FAIL_HIGH = 0.4
#"Low" / "huge" clean-sample-count thresholds (tune per dataset scale).
LOW_CLEAN = 300
HUGE_COUNT = 5000
HIGH_DUP_RATE = 0.3
POOR_QUALITY = 0.5  # mean per-row quality score below this is "poor"


def recommend_categories(cat_stats: dict[str, dict[str, Any]],
                         eval_fail: dict[str, float] | None = None
                         ) -> list[dict[str, Any]]:
    """Return one recommendation record per category.

    ``cat_stats[name]`` must provide: ``sample_count``, ``clean_count``,
    ``duplicate_rate``, ``mean_quality`` (0-1), ``drop_count``. ``eval_fail`` maps
    category -> failure rate (0-1) from an eval run, or is ``None``.
    """
    out = []
    for cat, s in sorted(cat_stats.items()):
        count = s["sample_count"]
        clean = s["clean_count"]
        dup_rate = s["duplicate_rate"]
        quality = s["mean_quality"]
        fail = (eval_fail or {}).get(cat)

        rec, reason, extra = _decide(cat, count, clean, dup_rate, quality, fail)
        rec_row = {"category": cat, "recommendation": rec, "reason": reason,
                   "sample_count": count, "clean_count": clean,
                   "duplicate_rate": round(dup_rate, 3),
                   "mean_quality": round(quality, 3)}
        if fail is not None:
            rec_row["eval_failure_rate"] = round(fail, 3)
        rec_row.update(extra)
        out.append(rec_row)
    return out


def _decide(cat, count, clean, dup_rate, quality, fail):
    high_fail = fail is not None and fail >= EVAL_FAIL_HIGH

    if high_fail and clean < LOW_CLEAN:
        target = max(LOW_CLEAN, clean * 4)
        return ("COLLECT_MORE",
                f"Model fails {fail:.0%} of eval prompts and only {clean} clean "
                f"training examples exist.",
                {"target_new_examples": int(target - clean)})
    if high_fail and count >= LOW_CLEAN and quality < POOR_QUALITY:
        return ("CLEAN_EXISTING",
                f"Model fails {fail:.0%} of eval prompts; {count} examples exist "
                f"but mean quality is {quality:.2f}.", {})
    if dup_rate >= HIGH_DUP_RATE and count >= HUGE_COUNT and (
            fail is None or fail < EVAL_FAIL_HIGH):
        drop_fraction = round(min(0.9, dup_rate), 2)
        return ("DOWNSAMPLE",
                f"Duplicate rate is {dup_rate:.0%} across {count} overrepresented "
                f"examples.", {"drop_fraction": drop_fraction})
    #No eval data: fall back to pure quality/dup heuristics.
    if fail is None:
        if quality < POOR_QUALITY and count >= LOW_CLEAN:
            return ("CLEAN_EXISTING",
                    f"Mean quality {quality:.2f} across {count} examples; no eval "
                    f"signal available.", {})
        if dup_rate >= HIGH_DUP_RATE and count >= HUGE_COUNT:
            return ("DOWNSAMPLE",
                    f"Duplicate rate {dup_rate:.0%} across {count} examples.",
                    {"drop_fraction": round(min(0.9, dup_rate), 2)})
    return ("HOLD",
            f"{clean} clean examples; stable signal." if fail is None
            else f"{clean} clean examples; eval failure {fail:.0%} acceptable.", {})
