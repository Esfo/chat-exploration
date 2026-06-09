"""PyArrow schemas for the evaluation output tables.

These live in the evaluation package rather than the main ``atlas.schemas``
registry on purpose: the eval/compare artifacts are produced by standalone
commands that operate on a model checkpoint, not by the resumable extraction
``run-all`` pipeline, so they must not participate in its schema-fingerprint
staleness machinery. Tables are written via ``storage.write_parquet`` with the
explicit schema below.
"""

from __future__ import annotations

import pyarrow as pa


def _schema(fields):
    return pa.schema([pa.field(n, t) for n, t in fields])


EVAL_BY_ITEM = _schema([
    ("eval_run_id", pa.string()),
    ("model_id", pa.string()),
    ("item_id", pa.string()),
    ("category", pa.string()),
    ("difficulty", pa.string()),
    ("full_loss", pa.float64()),
    ("assistant_loss", pa.float64()),
    ("user_prompt_loss", pa.float64()),
    ("assistant_perplexity", pa.float64()),
    ("tokens_scored", pa.int64()),
    ("generated_text", pa.string()),
    ("length_words", pa.int64()),
    ("stopped_cleanly", pa.bool_()),
    ("empty_output", pa.bool_()),
    ("malformed_output", pa.bool_()),
    ("contains_user_role_leak", pa.bool_()),
    ("contains_assistant_role_loop", pa.bool_()),
    ("repetition_3gram_rate", pa.float64()),
    ("repetition_5gram_rate", pa.float64()),
    ("unique_token_ratio", pa.float64()),
    ("degenerate", pa.bool_()),
    ("behavior_pass", pa.bool_()),
    ("format_pass", pa.bool_()),
    ("must_include_pass", pa.bool_()),
    ("reason_codes", pa.list_(pa.string())),
])

COMPARISON_BY_ITEM = _schema([
    ("comparison_id", pa.string()),
    ("item_id", pa.string()),
    ("category", pa.string()),
    ("difficulty", pa.string()),
    ("model_a_assistant_loss", pa.float64()),
    ("model_b_assistant_loss", pa.float64()),
    ("assistant_loss_delta", pa.float64()),  # a - b (positive => b better)
    ("model_a_behavior_pass", pa.bool_()),
    ("model_b_behavior_pass", pa.bool_()),
    ("model_a_degenerate", pa.bool_()),
    ("model_b_degenerate", pa.bool_()),
    ("winner", pa.string()),
    ("decision", pa.string()),
    ("confidence", pa.float64()),
    ("reason_codes", pa.list_(pa.string())),
])

COMPARISON_BY_CATEGORY = _schema([
    ("comparison_id", pa.string()),
    ("category", pa.string()),
    ("n_items", pa.int64()),
    ("a_win_rate", pa.float64()),
    ("b_win_rate", pa.float64()),
    ("tie_rate", pa.float64()),
    ("assistant_loss_delta_mean", pa.float64()),
    ("behavior_pass_delta", pa.float64()),
])

SAMPLE_SCORES = _schema([
    ("sample_id", pa.string()),
    ("category", pa.string()),
    ("action", pa.string()),
    ("confidence", pa.float64()),
    ("assistant_loss", pa.float64()),
    ("loss_percentile", pa.float64()),
    ("token_length", pa.int64()),
    ("assistant_word_count", pa.int64()),
    ("exact_dup_group_size", pa.int64()),
    ("near_dup_group_size", pa.int64()),
    ("is_duplicate_secondary", pa.bool_()),
    ("repetition_5gram_rate", pa.float64()),
    ("weird_character_ratio", pa.float64()),
    ("role_leakage", pa.bool_()),
    ("assistant_is_non_answer", pa.bool_()),
    ("assistant_repeats_prompt", pa.bool_()),
    ("assistant_contains_artifact", pa.bool_()),
    ("has_actual_information", pa.bool_()),
    ("too_short", pa.bool_()),
    ("too_long", pa.bool_()),
    ("valid_json", pa.bool_()),
    ("valid_message_roles", pa.bool_()),
    ("category_frequency_percentile", pa.float64()),
    ("model_weak", pa.bool_()),
    ("reason_codes", pa.list_(pa.string())),
])
