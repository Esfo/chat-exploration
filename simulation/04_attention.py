"""
STAGE 4 — SELF-ATTENTION (forward)
==================================

Source: `attention_forward()` in training.py

WHAT THIS STAGE DOES
--------------------
Attention is how each token gathers information from the OTHER tokens. Every
token asks a question (a "query"), every token advertises what it has (a "key"),
and carries content to share (a "value"). A token's output is a weighted average
of all values, where the weights come from how well its query matches each key.

THE Q / K / V PROJECTIONS
-------------------------
    q = h @ Wq      "what am I looking for?"
    k = h @ Wk      "what do I offer for matching?"
    v = h @ Wv      "what information do I carry?"
Each of Wq,Wk,Wv is (d_model, d_model), so q,k,v keep shape (batch, seq, d_model).

SPLITTING INTO HEADS
--------------------
d_model is sliced into n_heads independent sub-spaces of width head_dim, so the
model can attend to several kinds of relationships at once.
    reshape  (batch, seq, d_model) -> (batch, seq, n_heads, head_dim)
    transpose                       -> (batch, n_heads, seq, head_dim)
Now each head is its own little (seq, head_dim) attention problem.

SCORES, SCALING, MASK, SOFTMAX
------------------------------
    scores = q @ k^T            shape (batch, heads, seq, seq): every token vs every token
    scores /= sqrt(head_dim)    dot products grow with width; rescale so softmax isn't saturated
    causal mask                 set scores[i,j] = -1e9 for j>i so a token can't see the FUTURE
    probs = softmax(scores)     each row becomes attention weights summing to 1

GATHER VALUES AND PROJECT OUT
-----------------------------
    heads = probs @ v           weighted average of value vectors, (batch,heads,seq,head_dim)
    recombine heads             back to (batch, seq, d_model)
    out = combined @ Wo         final mix, (batch, seq, d_model)

WHERE IT GOES AFTERWARDS
------------------------
`out` is added to the residual stream (h = h + out). The `cache` tuple
(h, qh, kh, vh, probs, combined) is stored so the backward pass (stage 9) can
reuse these exact intermediate values instead of recomputing them.
"""

import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

# toy dims
batch, seq, d_model = 1, 6, 8
n_heads, head_dim = 2, 4
rng = np.random.default_rng(1)

h  = rng.normal(0, 1, size=(batch, seq, d_model))
Wq = rng.normal(0, 1/math.sqrt(d_model), size=(d_model, d_model))
Wk = rng.normal(0, 1/math.sqrt(d_model), size=(d_model, d_model))
Wv = rng.normal(0, 1/math.sqrt(d_model), size=(d_model, d_model))
Wo = rng.normal(0, 1/math.sqrt(d_model), size=(d_model, d_model))

print("=" * 70)
print("SELF-ATTENTION SIMULATION")
print("=" * 70)
print(f"\nhidden state h: {h.shape}  (batch, seq, d_model)")

q = h @ Wq; k = h @ Wk; v = h @ Wv
print(f"q,k,v after projection: {q.shape}")

qh = q.reshape(batch, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
kh = k.reshape(batch, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
vh = v.reshape(batch, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
print(f"split into heads: {qh.shape}  (batch, n_heads, seq, head_dim)")

scores = qh @ kh.transpose(0, 1, 3, 2)
print(f"scores = q @ k^T: {scores.shape}  (each token scored against every token)")
scores = scores / math.sqrt(head_dim)

future_mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
scores_masked = scores.copy()
scores_masked[:, :, future_mask] = -1e9
print("\ncausal mask applied: upper triangle (the future) set to -1e9")

probs = softmax(scores_masked)
print(f"attention probs: {probs.shape}, each row sums to 1")
print("\nhead 0 attention matrix (row i = how token i distributes attention):")
print(np.round(probs[0, 0], 3))
print("NOTE: it is lower-triangular — token 0 attends only to itself, etc.")

heads = probs @ vh
combined = heads.transpose(0, 2, 1, 3).reshape(batch, seq, d_model)
out = combined @ Wo
print(f"\nweighted values -> recombine -> Wo  =>  out {out.shape}")

# --- VISUALIZATION ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

im0 = axes[0].imshow(scores[0, 0], cmap="viridis")
axes[0].set_title("raw scores q@k^T / sqrt(d)\n(before masking)")
axes[0].set_xlabel("key token (j)"); axes[0].set_ylabel("query token (i)")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

masked_display = np.ma.masked_where(future_mask, probs[0, 0])
im1 = axes[1].imshow(probs[0, 0], cmap="magma", vmin=0, vmax=1)
axes[1].set_title("attention weights (head 0)\ncausal: lower-triangular")
axes[1].set_xlabel("key token (j)"); axes[1].set_ylabel("query token (i)")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

im2 = axes[2].imshow(probs[0, 1], cmap="magma", vmin=0, vmax=1)
axes[2].set_title("attention weights (head 1)\ndifferent head, different pattern")
axes[2].set_xlabel("key token (j)"); axes[2].set_ylabel("query token (i)")
plt.colorbar(im2, ax=axes[2], fraction=0.046)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "visualizations", "04_attention.png")
plt.savefig(out_path, dpi=110)
print(f"\nVisualization saved -> {out_path}")
