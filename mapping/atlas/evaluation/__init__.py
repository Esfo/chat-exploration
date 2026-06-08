"""Self-checking model evaluation (Phases 1-2 of the evaluation plan).

This package turns the atlas pipeline from a pure model-internals warehouse into
an evaluation system that answers the practical questions: *is this model
healthy*, and *is model B better than model A*. It is intentionally decoupled
from the extraction ``Library`` and stage registry — ``atlas evaluate`` and
``atlas compare`` operate on a model checkpoint directly, the way the plan's
target workflow invokes them.

Layout:

  eval_pack   - the JSONL evaluation-dataset format + loader/validator
  chat_format - render conversations to text + locate the assistant answer span
  loss        - assistant-only / full / prompt loss + perplexity
  behavior    - output-behaviour and degeneration metrics over generated text
  checks      - turn an item's ``expected_traits`` into pass/fail behaviour checks
  evaluate    - run one model over an eval pack -> eval_summary.json + by-item table
  compare     - run two models over the same pack -> per-item winner + summaries
  schemas     - pyarrow schemas for the by-item / comparison tables

The model-dependent pieces go through a tiny backend surface (``token_logprobs``
and ``generate_greedy``) so the whole harness is exercised against the torch-free
fake backend in the test suite.
"""

from __future__ import annotations

from .eval_pack import EvalItem, load_eval_pack, validate_item  # noqa: F401
from .evaluate import evaluate_model, EvalConfig  # noqa: F401
from .compare import compare_models  # noqa: F401
