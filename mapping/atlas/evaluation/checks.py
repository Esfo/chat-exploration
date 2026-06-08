"""Turn an item's ``expected_traits`` into concrete pass/fail behaviour checks.

Each eval item may declare lightweight expectations — should it answer at all, a
word-count ceiling, substrings that must (or must not) appear, whether code is
required. These are deterministic, model-free assertions over the generated text
plus the behaviour metrics, producing a structured result the summary aggregates
into ``behavior_pass_rate`` and per-reason failure tables.

Kept simple on purpose: the goal is to catch obvious format/instruction failures
and required-fact misses (e.g. a remembered name in a multi-turn item), not to
grade prose quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .behavior import BehaviorMetrics

_CODE_HINTS = ("def ", "class ", "import ", "```", "return ", "function ", "=>", "();")


def _looks_like_code(text: str) -> bool:
    return any(h in text for h in _CODE_HINTS)


@dataclass
class CheckResult:
    behavior_pass: bool
    must_include_pass: bool
    must_not_include_pass: bool
    format_pass: bool
    length_pass: bool
    answered_pass: bool
    code_pass: bool
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "behavior_pass": self.behavior_pass,
            "must_include_pass": self.must_include_pass,
            "must_not_include_pass": self.must_not_include_pass,
            "format_pass": self.format_pass,
            "length_pass": self.length_pass,
            "answered_pass": self.answered_pass,
            "code_pass": self.code_pass,
            "reason_codes": list(self.reason_codes),
        }


def run_checks(generated: str, traits: dict[str, Any],
               metrics: BehaviorMetrics) -> CheckResult:
    """Evaluate ``traits`` against ``generated`` and its behaviour ``metrics``."""
    traits = traits or {}
    reasons: list[str] = []
    text = generated
    low = text.lower()

    #--- answered / non-answer -------------------------------------------
    should_answer = traits.get("should_answer", True)
    answered_pass = True
    if should_answer and (metrics.empty_output or metrics.non_answer):
        answered_pass = False
        reasons.append("expected_answer_missing")
    if not should_answer and not metrics.empty_output and not metrics.non_answer:
        #A refusal/boundary item that should NOT answer but did.
        answered_pass = False
        reasons.append("answered_when_should_refuse")

    #--- must_include / must_not_include ---------------------------------
    must_include = traits.get("must_include", []) or []
    inc_pass = True
    for needle in must_include:
        if str(needle).lower() not in low:
            inc_pass = False
            reasons.append(f"missing:{needle}")

    must_not = traits.get("must_not_include", []) or []
    exc_pass = True
    for needle in must_not:
        if str(needle).lower() in low:
            exc_pass = False
            reasons.append(f"contains_forbidden:{needle}")

    #--- length ----------------------------------------------------------
    length_pass = True
    max_words = traits.get("max_words")
    if max_words is not None and metrics.length_words > int(max_words):
        length_pass = False
        reasons.append("too_long")
    min_words = traits.get("min_words")
    if min_words is not None and metrics.length_words < int(min_words):
        length_pass = False
        reasons.append("too_short")

    #--- code requirement ------------------------------------------------
    code_pass = True
    has_code = _looks_like_code(text)
    if traits.get("requires_code") and not has_code:
        code_pass = False
        reasons.append("missing_code")
    if traits.get("requires_code") is False and has_code and traits.get("forbid_code"):
        code_pass = False
        reasons.append("unexpected_code")

    #--- format / degeneration -------------------------------------------
    format_pass = not (metrics.malformed_output or metrics.contains_user_role_leak
                       or metrics.contains_assistant_role_loop)
    if not format_pass:
        reasons.append("format_failure")
    if metrics.repetition_5gram_rate > 0.2 or metrics.repeated_line_count >= 2:
        format_pass = False
        reasons.append("repetition")

    behavior_pass = all([answered_pass, inc_pass, exc_pass, length_pass,
                         code_pass, format_pass])
    return CheckResult(
        behavior_pass=behavior_pass,
        must_include_pass=inc_pass,
        must_not_include_pass=exc_pass,
        format_pass=format_pass,
        length_pass=length_pass,
        answered_pass=answered_pass,
        code_pass=code_pass,
        reason_codes=reasons,
    )
