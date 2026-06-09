"""Tolerant loader for training-data JSONL rows.

Unlike eval packs (which must be well-formed), training data is exactly what we
are trying to *audit*, so the loader never raises on a bad row: it records the
format problem on the row itself and keeps going, so a malformed line becomes a
scored, droppable sample rather than a crash. Each row is normalised into the
same ``user`` / ``assistant`` turn shape the rest of the harness uses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_ROLES = {"system", "user", "assistant"}


@dataclass
class TrainRow:
    id: str
    messages: list[dict[str, str]]
    raw: dict[str, Any] = field(default_factory=dict)
    format_errors: list[str] = field(default_factory=list)
    line: int = -1
    raw_text: str = ""

    @property
    def valid_json(self) -> bool:
        return "invalid_json" not in self.format_errors

    @property
    def valid_message_roles(self) -> bool:
        return bool(self.messages) and all(
            m.get("role") in VALID_ROLES for m in self.messages)

    @property
    def has_user_message(self) -> bool:
        return any(m.get("role") == "user" for m in self.messages)

    @property
    def has_assistant_message(self) -> bool:
        return any(m.get("role") == "assistant" for m in self.messages)

    def prompt_messages(self) -> list[dict[str, str]]:
        """Turns up to (but excluding) the final assistant answer."""
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "assistant":
                return self.messages[:i]
        return self.messages

    def assistant_text(self) -> str:
        for m in reversed(self.messages):
            if m.get("role") == "assistant":
                return m.get("content", "")
        return ""

    def last_user_text(self) -> str:
        for m in reversed(self.prompt_messages()):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""


def _normalize(obj: dict[str, Any], rid: str) -> tuple[list[dict[str, str]], list[str]]:
    errors: list[str] = []
    messages = obj.get("messages")
    norm: list[dict[str, str]] = []
    #Also accept the flat {"prompt": ..., "response": ...} shape some datasets use.
    if not isinstance(messages, list):
        if "prompt" in obj or "response" in obj:
            if obj.get("prompt"):
                norm.append({"role": "user", "content": str(obj["prompt"])})
            if obj.get("response"):
                norm.append({"role": "assistant", "content": str(obj["response"])})
            if not norm:
                errors.append("no_messages")
            return norm, errors
        errors.append("no_messages")
        return norm, errors
    for m in messages:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            errors.append("malformed_message")
            continue
        if m["role"] not in VALID_ROLES:
            errors.append("invalid_role")
        norm.append({"role": str(m.get("role", "")), "content": str(m.get("content", ""))})
    return norm, errors


def load_dataset(path: str | Path) -> list[TrainRow]:
    """Load a training ``.jsonl`` into tolerant :class:`TrainRow` objects."""
    path = Path(path)
    rows: list[TrainRow] = []
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        rid = f"train_{n:06d}"
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            rows.append(TrainRow(id=rid, messages=[], raw={},
                                 format_errors=["invalid_json"], line=n, raw_text=s))
            continue
        if isinstance(obj, dict) and obj.get("id"):
            rid = str(obj["id"])
        messages, errors = _normalize(obj if isinstance(obj, dict) else {}, rid)
        rows.append(TrainRow(id=rid, messages=messages,
                             raw=obj if isinstance(obj, dict) else {},
                             format_errors=errors, line=n, raw_text=s))
    return rows
