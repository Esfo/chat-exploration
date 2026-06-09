"""Assign a final action label to a training row (Part 7-8 of the plan).

Every row gets exactly one action — KEEP, KEEP_HARD, DOWNSAMPLE, FIX, DROP,
COLLECT_MORE_LIKE_THIS, or REVIEW — with a confidence and traceable reason codes.

The central rule the plan insists on: **do not drop a row just because its loss is
high.** High loss is dropped only when it co-occurs with quality failures
(malformed / repetition / role leak / non-answer / duplicate). A high-loss row
that is clean, informative, and from a rare category is KEEP_HARD — those are the
examples that teach the model new things, and deleting them is how you
accidentally lobotomise the dataset.

``label_sample`` consumes a flat metrics dict assembled by ``score_data`` (text
quality + dedup group sizes + assistant loss + dataset-relative percentiles) so
this module stays a pure decision function that is trivial to unit-test.
"""

from __future__ import annotations

from typing import Any

#Loss at/above this dataset percentile counts as "high loss".
HIGH_LOSS_PCT = 0.80
#Category at/below this frequency percentile counts as "rare" (underrepresented).
RARE_CATEGORY_PCT = 0.20
#Repetition / weirdness thresholds for a quality failure.
REP_FAIL = 0.20
WEIRD_FAIL = 0.05


def _quality_failures(m: dict[str, Any]) -> list[str]:
    """Hard text-quality failures that make a row junk regardless of loss."""
    fails = []
    if m.get("assistant_word_count", 0) == 0 or not m.get("has_assistant_message", True):
        fails.append("empty_assistant_answer")
    if m.get("assistant_is_non_answer"):
        fails.append("non_answer")
    if m.get("assistant_repeats_prompt"):
        fails.append("assistant_repeats_prompt")
    if m.get("assistant_repetition_5gram_rate", 0.0) > REP_FAIL:
        fails.append("high_repetition_rate")
    if m.get("role_leakage"):
        fails.append("role_leakage")
    if m.get("assistant_contains_artifact"):
        fails.append("dataset_artifact")
    if m.get("weird_character_ratio", 0.0) > WEIRD_FAIL:
        fails.append("weird_characters")
    if m.get("too_long"):
        fails.append("extremely_long")
    return fails


def _format_broken(m: dict[str, Any]) -> list[str]:
    """Recoverable structural problems: the *idea* may be fine but the row is
    mis-formatted (bad roles, missing fields) — a FIX, not a DROP."""
    broken = []
    if not m.get("valid_json", True):
        broken.append("invalid_json")
    if not m.get("valid_message_roles", True):
        broken.append("bad_roles")
    if not m.get("has_user_message", True):
        broken.append("missing_user_message")
    if m.get("empty_field_count", 0) > 0:
        broken.append("empty_field")
    return broken


def label_sample(m: dict[str, Any]) -> dict[str, Any]:
    """Return ``{action, confidence, reason_codes}`` for one row's metrics."""
    reasons: list[str] = []
    quality_fails = _quality_failures(m)
    broken = _format_broken(m)

    loss = m.get("assistant_loss")
    loss_pct = m.get("loss_percentile")
    high_loss = loss_pct is not None and loss_pct >= HIGH_LOSS_PCT
    cat_pct = m.get("category_frequency_percentile")
    rare = cat_pct is not None and cat_pct <= RARE_CATEGORY_PCT
    near_dup = m.get("near_dup_group_size", 1) or 1
    exact_dup = m.get("exact_dup_group_size", 1) or 1
    is_secondary = m.get("is_duplicate_secondary", False)
    too_easy = m.get("model_weak") is False and loss_pct is not None and loss_pct <= 0.2

    #--- DROP: junk, or high loss combined with quality failure ----------
    #An exact duplicate beyond the first occurrence is removable noise.
    if is_secondary and exact_dup > 1:
        reasons.append("exact_duplicate")
        return _result("DROP", 0.9, reasons + ["duplicate_secondary"])
    if quality_fails:
        #Junk on its own merits. High loss raises confidence but isn't required.
        conf = 0.9 if (high_loss or len(quality_fails) >= 2) else 0.75
        if high_loss:
            reasons.append("high_loss")
        return _result("DROP", conf, reasons + quality_fails)

    #--- FIX: clean idea, broken structure -------------------------------
    if broken:
        return _result("FIX", 0.7, reasons + broken)

    #--- KEEP_HARD: high loss but clean + valuable -----------------------
    if high_loss and m.get("has_actual_information", True):
        reasons.append("high_assistant_loss")
        reasons.append("clean_format")
        reasons.append("passes_quality_checks")
        if rare:
            reasons.append("rare_category")
            return _result("KEEP_HARD", 0.85, reasons)
        return _result("KEEP_HARD", 0.7, reasons)

    #--- DOWNSAMPLE: clean but redundant / too easy ----------------------
    if near_dup >= 3 or (is_secondary and near_dup > 1):
        reasons.append("near_duplicate")
        return _result("DOWNSAMPLE", 0.75, reasons + [f"near_dup_group_size={near_dup}"])
    if too_easy:
        reasons.append("very_easy_for_model")
        return _result("DOWNSAMPLE", 0.6, reasons)

    #--- COLLECT_MORE_LIKE_THIS: clean, rare, model not already strong ---
    if rare and m.get("model_weak", False):
        reasons.append("rare_category")
        reasons.append("model_weak_here")
        return _result("COLLECT_MORE_LIKE_THIS", 0.65, reasons)

    #--- REVIEW: borderline / low-confidence -----------------------------
    if high_loss and not m.get("has_actual_information", True):
        reasons.append("high_loss_low_information")
        return _result("REVIEW", 0.5, reasons)

    return _result("KEEP", 0.8, reasons + ["clean_useful_on_format"])


def _result(action: str, confidence: float, reasons: list[str]) -> dict[str, Any]:
    return {"action": action, "confidence": round(float(confidence), 3),
            "reason_codes": reasons}
