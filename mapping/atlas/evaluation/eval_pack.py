"""Evaluation-pack format: the standard JSONL eval-dataset contract.

An eval pack is a ``.jsonl`` file where each line is one eval item. The format is
kept deliberately separate from training data so eval prompts never leak into the
training set. Each item carries an id, a category/difficulty/tags taxonomy, the
conversation ``messages`` so far, optional ``expected_traits`` that drive the
behaviour checks, and an optional ``reference_answer`` used for loss scoring and
similarity.

Two shapes are supported (both validated here):

  * single-turn: ``messages`` ends with a user turn; the model must answer.
  * multi-turn:  ``messages`` interleaves user/assistant turns and ends with a
    user turn whose answer the model must produce (optionally checked against
    ``must_include`` traits such as a remembered name).

Items may also end with an assistant turn (a held-out reference answer); in that
case that assistant content is the answer scored for loss and the prompt is the
preceding turns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


VALID_ROLES = {"system", "user", "assistant"}


@dataclass
class EvalItem:
    """One evaluation item, parsed and normalised."""

    id: str
    category: str
    messages: list[dict[str, str]]
    difficulty: str = "unknown"
    tags: list[str] = field(default_factory=list)
    expected_traits: dict[str, Any] = field(default_factory=dict)
    reference_answer: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ends_with_assistant(self) -> bool:
        return bool(self.messages) and self.messages[-1]["role"] == "assistant"

    def prompt_messages(self) -> list[dict[str, str]]:
        """The turns shown to the model before it must answer."""
        if self.ends_with_assistant:
            return self.messages[:-1]
        return self.messages

    def answer_text(self) -> str | None:
        """The text scored for assistant-only loss.

        A held-out assistant turn is its own content; otherwise the reference
        answer (if any) stands in for the target assistant turn. Returns ``None``
        when there is nothing to score (generation-only item).
        """
        if self.ends_with_assistant:
            return self.messages[-1]["content"]
        return self.reference_answer


class EvalPackError(ValueError):
    """Raised when an eval item or pack is malformed."""


def validate_item(obj: dict[str, Any], *, line: int | None = None) -> EvalItem:
    """Validate one raw item dict and return a normalised :class:`EvalItem`."""

    where = f" (line {line})" if line is not None else ""

    def fail(msg: str) -> None:
        raise EvalPackError(f"{msg}{where}")

    if not isinstance(obj, dict):
        fail("eval item must be a JSON object")
    if not obj.get("id"):
        fail("eval item missing 'id'")
    messages = obj.get("messages")
    if not isinstance(messages, list) or not messages:
        fail(f"item {obj.get('id')!r} has no 'messages'")
    norm_messages = []
    for m in messages:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            fail(f"item {obj.get('id')!r} has a malformed message")
        if m["role"] not in VALID_ROLES:
            fail(f"item {obj.get('id')!r} has invalid role {m['role']!r}")
        norm_messages.append({"role": str(m["role"]), "content": str(m["content"])})
    if not any(m["role"] == "user" for m in norm_messages):
        fail(f"item {obj.get('id')!r} has no user turn")

    return EvalItem(
        id=str(obj["id"]),
        category=str(obj.get("category", "uncategorized")),
        messages=norm_messages,
        difficulty=str(obj.get("difficulty", "unknown")),
        tags=list(obj.get("tags", []) or []),
        expected_traits=dict(obj.get("expected_traits", {}) or {}),
        reference_answer=obj.get("reference_answer"),
        raw=obj,
    )


def load_eval_pack(path: str | Path) -> list[EvalItem]:
    """Load and validate a ``.jsonl`` eval pack into a list of items.

    Blank lines and ``#`` comment lines are skipped so packs can be annotated.
    Item ids must be unique within a pack.
    """
    path = Path(path)
    if not path.exists():
        raise EvalPackError(f"eval pack not found: {path}")
    items: list[EvalItem] = []
    seen: set[str] = set()
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise EvalPackError(f"invalid JSON on line {n}: {e}") from e
        item = validate_item(obj, line=n)
        if item.id in seen:
            raise EvalPackError(f"duplicate eval id {item.id!r} (line {n})")
        seen.add(item.id)
        items.append(item)
    if not items:
        raise EvalPackError(f"eval pack {path} contains no items")
    return items
