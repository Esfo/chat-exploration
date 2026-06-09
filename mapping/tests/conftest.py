"""Test fixtures: a torch-free fake transformer backend.

The real backend needs torch + a multi-gigabyte Llama checkpoint, neither of
which belongs in CI. ``FakeBackend`` implements the same surface as
``atlas.model_backend.ModelBackend`` over a tiny deterministic "model" so the
entire extraction pipeline can run end-to-end in milliseconds and assert that
every artifact is produced and schema-valid.

The fake activations are a deterministic function of token id and unit index, so
co-firing structure genuinely exists and clustering produces non-trivial groups.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas.config import ExtractionConfig
from atlas.model_backend import Architecture


class FakeTokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0

    def __call__(self, text, truncation=True, max_length=64, return_attention_mask=False):
        ids = [(ord(c) % (self.vocab_size - 1)) + 1 for c in text]
        if truncation:
            ids = ids[:max_length]
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(chr(33 + (i % 90)) for i in ids)


class FakeBackend:
    """Minimal stand-in for ModelBackend with a tiny deterministic model."""

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.device = "cpu"
        self._arch = Architecture(
            model_type="llama_fake", num_layers=3, hidden_size=16,
            intermediate_size=32, num_attention_heads=4, num_key_value_heads=4,
            head_dim=4, vocab_size=64,
        )
        self._model = None
        self._tok = FakeTokenizer(self._arch.vocab_size)

    #--- API parity -------------------------------------------------------
    def load_config_only(self):
        return self._arch

    def load(self):
        return None

    @property
    def tokenizer(self):
        return self._tok

    def architecture(self):
        return self._arch

    def param_count(self):
        a = self._arch
        return a.num_layers * (4 * a.hidden_size ** 2 + 3 * a.hidden_size * a.intermediate_size)

    def iter_named_tensors(self):
        for name, shape in self._tensor_specs():
            yield name, shape, "float32"

    def get_tensor(self, name):
        for n, shape in self._tensor_specs():
            if n == name:
                rng = np.random.default_rng(abs(hash(name)) % (2 ** 32))
                return rng.standard_normal(shape).astype(np.float32)
        raise KeyError(name)

    #--- evaluation primitives (parity with ModelBackend) -----------------
    @property
    def eos_id(self):
        return self._arch.vocab_size - 1

    def token_logprobs(self, token_ids):
        """Deterministic per-token log-probs: each next token gets a log-prob that
        is a smooth function of the (prev, next) id pair, so losses are stable and
        vary across samples without needing torch."""
        token_ids = list(token_ids)
        if len(token_ids) < 2:
            return []
        out = []
        for prev, nxt in zip(token_ids[:-1], token_ids[1:]):
            lp = -2.0 - 0.5 * abs(float(np.sin(0.13 * prev + 0.07 * nxt)))
            out.append(lp)
        return out

    def generate_greedy(self, prompt_ids, max_new_tokens=128, eos_ids=None):
        """Deterministic 'generation': emit a short varied id sequence derived
        from the prompt, then stop with eos. Decodes (via FakeTokenizer) to a
        non-degenerate string so behaviour checks exercise their happy path."""
        prompt_ids = list(prompt_ids)
        seed = sum(prompt_ids) % 53 + 7
        n = min(max_new_tokens, 12)
        ids = [((seed * (i + 3)) % (self._arch.vocab_size - 2)) + 1 for i in range(n)]
        ids.append(self.eos_id)
        return ids

    def capture(self, input_ids, attention_mask, layer_callback):
        a = self._arch
        input_ids = np.asarray(input_ids)
        b, t = input_ids.shape
        for layer in range(a.num_layers):
            #MLP neuron acts: structured function of token id and neuron index.
            j = np.arange(a.intermediate_size)
            tok = input_ids[:, :, None].astype(np.float64)
            mlp = np.sin(0.1 * tok * (j + 1) + layer) ** 2  # [B, T, inter] >= 0
            heads = np.arange(a.num_attention_heads)
            attn = np.abs(np.cos(0.2 * tok * (heads + 1) + layer))  # [B, T, heads]
            layer_callback(layer, {"mlp": mlp, "attn_head_norm": attn})

    #--- helpers ----------------------------------------------------------
    def _tensor_specs(self):
        a = self._arch
        specs = []
        for i in range(a.num_layers):
            p = f"model.layers.{i}."
            specs += [
                (p + "self_attn.q_proj.weight", (a.hidden_size, a.hidden_size)),
                (p + "self_attn.k_proj.weight", (a.head_dim * a.num_key_value_heads, a.hidden_size)),
                (p + "self_attn.v_proj.weight", (a.head_dim * a.num_key_value_heads, a.hidden_size)),
                (p + "self_attn.o_proj.weight", (a.hidden_size, a.hidden_size)),
                (p + "mlp.gate_proj.weight", (a.intermediate_size, a.hidden_size)),
                (p + "mlp.up_proj.weight", (a.intermediate_size, a.hidden_size)),
                (p + "mlp.down_proj.weight", (a.hidden_size, a.intermediate_size)),
                (p + "input_layernorm.weight", (a.hidden_size,)),
                (p + "post_attention_layernorm.weight", (a.hidden_size,)),
            ]
        specs += [
            ("model.embed_tokens.weight", (a.vocab_size, a.hidden_size)),
            ("model.norm.weight", (a.hidden_size,)),
            ("lm_head.weight", (a.vocab_size, a.hidden_size)),
        ]
        return specs


@pytest.fixture
def fake_config():
    return ExtractionConfig(
        model_path="/fake/Meta-Llama-3-8B", tokenizer_path="/fake/Meta-Llama-3-8B",
        target_tokens=2000, max_sequence_length=24, batch_size=2,
        signature_dim=32, bitset_positions=256, reservoir_size=32,
        edges_top_k=8, min_cluster_size=2, edge_min_score=0.0,
    )


@pytest.fixture
def fake_backend(fake_config):
    return FakeBackend(fake_config)
