"""Per-row text-quality metrics for training-data scoring.

These are the cleanliness + usefulness signals from the plan (Part 6) that decide
whether a training row is worth keeping — computed purely from the row's text so
they are fast and unit-testable. Model-dependent signals (assistant-only loss)
are added separately in ``score_data`` from the loss module; everything here
needs no model.

The headline distinction the labeller later relies on: *bad data* (malformed,
non-answer, repeats the prompt, scrape artifacts) is removable, but *hard data*
(clean, informative, just difficult) must be kept even when its loss is high.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

from .behavior import analyze_generation
from .dataset import TrainRow

_WORD = re.compile(r"\w+", re.UNICODE)
#Obvious dataset/scrape artifacts that mark a row as junk.
_ARTIFACT_PATTERNS = (
    "<div", "</div>", "&nbsp;", "&amp;", "http://", "https://", "[deleted]",
    "[removed]", "click here", "read more", "\\u00", "����", "<!--",
)
_HTML_RE = re.compile(r"</?[a-zA-Z][^>]{0,40}>")


def _words(text: str) -> list[str]:
    return _WORD.findall(text)


def _weird_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = sum(1 for c in text if ord(c) > 0x2000 and not c.isspace() and not c.isalnum())
    return bad / len(text)


def _markup_noise_score(text: str) -> float:
    if not text:
        return 0.0
    return min(1.0, len(_HTML_RE.findall(text)) / max(1, len(text.split())) * 4)


def _is_non_answer(text: str) -> bool:
    w = _words(text)
    return len(w) <= 2


def _repeats_prompt(answer: str, user: str) -> bool:
    """The assistant largely echoes the user prompt back."""
    a, u = answer.strip().lower(), user.strip().lower()
    if not a or not u or len(u) < 12:
        return False
    if a == u:
        return True
    aw, uw = set(_words(a)), set(_words(u))
    if not aw:
        return False
    overlap = len(aw & uw) / len(aw)
    #High word overlap AND the answer adds little of its own.
    return overlap > 0.85 and len(aw - uw) <= 2


@dataclass
class QualityMetrics:
    valid_json: bool
    valid_message_roles: bool
    has_user_message: bool
    has_assistant_message: bool
    empty_field_count: int
    assistant_word_count: int
    weird_character_ratio: float
    markup_noise_score: float
    role_leakage: bool
    assistant_is_non_answer: bool
    assistant_repeats_prompt: bool
    assistant_contains_artifact: bool
    assistant_repetition_5gram_rate: float
    assistant_unique_token_ratio: float
    has_actual_information: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def quality_metrics(row: TrainRow) -> QualityMetrics:
    """Compute the model-free text-quality metrics for one training row."""
    answer = row.assistant_text()
    user = row.last_user_text()
    low = answer.lower()

    empty_fields = sum(1 for m in row.messages if not m.get("content", "").strip())
    bm = analyze_generation(answer, stopped_on_eos=True)
    role_leak = bm.contains_user_role_leak or bm.contains_assistant_role_loop \
        or bool(re.search(r"(?i)\b(user|assistant)\s*:", answer))
    artifact = any(p in low for p in _ARTIFACT_PATTERNS)
    aw = _words(answer)

    return QualityMetrics(
        valid_json=row.valid_json,
        valid_message_roles=row.valid_message_roles,
        has_user_message=row.has_user_message,
        has_assistant_message=row.has_assistant_message,
        empty_field_count=empty_fields,
        assistant_word_count=len(aw),
        weird_character_ratio=_weird_char_ratio(answer),
        markup_noise_score=_markup_noise_score(answer),
        role_leakage=role_leak,
        assistant_is_non_answer=_is_non_answer(answer),
        assistant_repeats_prompt=_repeats_prompt(answer, user),
        assistant_contains_artifact=artifact,
        assistant_repetition_5gram_rate=bm.repetition_5gram_rate,
        assistant_unique_token_ratio=bm.unique_token_ratio,
        has_actual_information=len(aw) >= 4 and not bm.degenerate,
    )
