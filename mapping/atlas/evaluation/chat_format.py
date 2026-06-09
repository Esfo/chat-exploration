"""Render conversations to text and locate the assistant-answer token span.

For conversation training and evaluation, total next-token loss is too blunt:
what matters is how well the model predicts the *assistant* answer given the user
turn. To compute that we need to know which tokens belong to the assistant
answer. This module renders messages with explicit role markers and tokenizes the
prompt and answer separately so their boundary is exact, then concatenates the
ids — a deterministic, tokenizer-agnostic construction that works equally for the
real HF tokenizer and the test fake.

The role-marker template is intentionally simple and self-contained rather than
depending on a model's bundled chat template, so eval results are comparable
across models that share a tokenizer family. The same markers are reused by the
behaviour checks to detect role-leak / role-loop degeneration.
"""

from __future__ import annotations

from dataclasses import dataclass

USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"
SYSTEM_TAG = "<|system|>"

_TAGS = {"user": USER_TAG, "assistant": ASSISTANT_TAG, "system": SYSTEM_TAG}


def _encode(tokenizer, text: str, add_special: bool) -> list[int]:
    """Tokenize ``text`` to a list of ids, controlling special tokens when
    the tokenizer supports it (HF) and degrading gracefully when it does not
    (the test fake)."""
    try:
        out = tokenizer(text, add_special_tokens=add_special)
    except TypeError:
        out = tokenizer(text)
    return list(out["input_ids"])


def render_prompt_text(prompt_messages: list[dict[str, str]]) -> str:
    """Render the prompt turns, ending with an open assistant marker so the
    model (or the reference answer) continues from there."""
    parts = []
    for m in prompt_messages:
        tag = _TAGS.get(m["role"], USER_TAG)
        parts.append(f"{tag}\n{m['content']}\n")
    parts.append(f"{ASSISTANT_TAG}\n")
    return "".join(parts)


@dataclass
class EncodedSample:
    prompt_ids: list[int]
    answer_ids: list[int]

    @property
    def full_ids(self) -> list[int]:
        return self.prompt_ids + self.answer_ids

    @property
    def answer_start(self) -> int:
        """Index of the first assistant-answer token within ``full_ids``."""
        return len(self.prompt_ids)


def encode_prompt(tokenizer, prompt_messages: list[dict[str, str]]) -> list[int]:
    """Token ids for the prompt portion (BOS/special tokens included)."""
    return _encode(tokenizer, render_prompt_text(prompt_messages), add_special=True)


def encode_sample(tokenizer, prompt_messages: list[dict[str, str]],
                  answer_text: str) -> EncodedSample:
    """Encode a full ``prompt + assistant answer`` sample with an exact boundary.

    The answer is tokenized without special tokens so concatenation does not
    inject a spurious BOS mid-sequence; the boundary index is therefore exactly
    ``len(prompt_ids)``.
    """
    prompt_ids = encode_prompt(tokenizer, prompt_messages)
    answer_ids = _encode(tokenizer, "\n" + answer_text, add_special=False) if answer_text \
        else []
    return EncodedSample(prompt_ids=prompt_ids, answer_ids=answer_ids)
