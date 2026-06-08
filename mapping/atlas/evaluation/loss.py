"""Assistant-only, full, and prompt loss + perplexity for a chat sample.

Given a prompt and a target assistant answer, we run one forward pass over the
concatenated ids, read the per-token log-probabilities, and split them into the
user-prompt span and the assistant-answer span. ``assistant_loss`` — the mean
negative log-likelihood over only the assistant tokens — is the headline metric,
because it directly measures conversational answer quality rather than how
predictable the (model-independent) user prompt was.

All losses are natural-log nats per token; perplexity is ``exp(loss)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any

from .chat_format import encode_sample


@dataclass
class LossMetrics:
    full_loss: float
    assistant_loss: float
    user_prompt_loss: float
    assistant_perplexity: float
    tokens_scored: int
    sample_length: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def assistant_loss(backend, prompt_messages, answer_text) -> LossMetrics:
    """Compute the loss family for one ``prompt + answer`` chat sample.

    ``backend.token_logprobs(ids)`` returns ``log P(ids[i+1] | ids[:i+1])`` for
    ``i`` in ``0..len(ids)-2``; so the log-prob that *predicts* token at position
    ``k`` sits at index ``k-1``. The assistant answer occupies positions
    ``[answer_start, len(full))``, predicted by logprob indices
    ``[answer_start-1, len(full)-1)``.
    """
    enc = encode_sample(backend.tokenizer, prompt_messages, answer_text or "")
    full = enc.full_ids
    logprobs = backend.token_logprobs(full)  # length len(full) - 1

    start = enc.answer_start
    n_answer = len(enc.answer_ids)
    #Predictions for the answer tokens: indices start-1 .. start-1+n_answer-1.
    assistant_lp = logprobs[start - 1: start - 1 + n_answer] if start >= 1 else logprobs[:n_answer]
    user_lp = logprobs[: max(start - 1, 0)]

    full_loss = -_mean(logprobs)
    a_loss = -_mean(assistant_lp) if assistant_lp else float("nan")
    u_loss = -_mean(user_lp) if user_lp else float("nan")
    ppl = math.exp(a_loss) if a_loss == a_loss and a_loss < 50 else float("inf")
    return LossMetrics(
        full_loss=full_loss,
        assistant_loss=a_loss,
        user_prompt_loss=u_loss,
        assistant_perplexity=ppl,
        tokens_scored=len(assistant_lp),
        sample_length=len(full),
    )
