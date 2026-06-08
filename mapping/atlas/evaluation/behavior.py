"""Output-behaviour and degeneration metrics over generated text.

Loss tells you whether the model assigns probability to the right text;
generation tells you what the model actually *does*. For small models the most
important failures are degenerations — repeating a phrase, looping both sides of
the conversation, emitting empty or malformed output — and these are cheap to
detect from the decoded string. Everything here is pure text analysis (no model,
no torch) so it is fully unit-testable and reused by both ``evaluate`` and
``compare``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

from .chat_format import USER_TAG, ASSISTANT_TAG

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _ngram_repeat_rate(tokens: list[str], n: int) -> float:
    """Fraction of n-grams that are repeats of an earlier n-gram. 0 = all unique,
    approaching 1 = highly repetitive."""
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def _repeated_line_count(text: str) -> int:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    seen: dict[str, int] = {}
    for ln in lines:
        seen[ln] = seen.get(ln, 0) + 1
    return sum(c - 1 for c in seen.values() if c > 1)


@dataclass
class BehaviorMetrics:
    length_words: int
    length_chars: int
    empty_output: bool
    stopped_cleanly: bool
    contains_user_role_leak: bool
    contains_assistant_role_loop: bool
    repeated_line_count: int
    repetition_3gram_rate: float
    repetition_5gram_rate: float
    unique_token_ratio: float
    malformed_output: bool
    non_answer: bool
    degenerate: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#Short stock phrases that signal the model dodged rather than answered.
_NON_ANSWER_PATTERNS = (
    "i don't know", "i do not know", "i cannot help", "i can't help",
    "as an ai", "i'm not sure", "i am not sure", "no comment",
)


def analyze_generation(text: str, *, stopped_on_eos: bool) -> BehaviorMetrics:
    """Compute behaviour + degeneration metrics for one generated string."""
    stripped = text.strip()
    words = _words(text)
    n_words = len(words)
    empty = n_words == 0

    role_leak = USER_TAG in text or re.search(r"(?i)\buser\s*:", text) is not None
    #A role loop: the model starts producing another assistant turn (it is writing
    #both sides of the conversation forever).
    assistant_loop = (text.count(ASSISTANT_TAG) >= 1) or (
        len(re.findall(r"(?i)\bassistant\s*:", text)) >= 1)

    rep3 = _ngram_repeat_rate(words, 3)
    rep5 = _ngram_repeat_rate(words, 5)
    unique_ratio = (len(set(words)) / n_words) if n_words else 0.0
    rep_lines = _repeated_line_count(text)

    low = stripped.lower()
    non_answer = (not empty) and (
        n_words <= 2 or any(p in low for p in _NON_ANSWER_PATTERNS))

    #Malformed: empty, or contains control/garbage runs, or collapses to one token.
    malformed = empty or bool(re.search(r"(.)\1{12,}", text)) or (
        n_words >= 8 and unique_ratio < 0.15)

    degenerate = (
        empty or malformed or role_leak or assistant_loop
        or rep5 > 0.2 or rep_lines >= 2 or unique_ratio < 0.25 and n_words >= 8
    )

    return BehaviorMetrics(
        length_words=n_words,
        length_chars=len(text),
        empty_output=empty,
        stopped_cleanly=bool(stopped_on_eos) and not role_leak and not assistant_loop,
        contains_user_role_leak=role_leak,
        contains_assistant_role_loop=assistant_loop,
        repeated_line_count=rep_lines,
        repetition_3gram_rate=rep3,
        repetition_5gram_rate=rep5,
        unique_token_ratio=unique_ratio,
        malformed_output=malformed,
        non_answer=non_answer,
        degenerate=bool(degenerate),
    )
