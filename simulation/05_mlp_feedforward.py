"""
STAGE 5 — MLP / FEED-FORWARD BLOCK
==================================

Source: the MLP lines inside `forward()` in training.py:
    pre_activation = h @ W1 + b1
    mlp_hidden     = np.maximum(pre_activation, 0)   # ReLU
    mlp_output     = mlp_hidden @ W2 + b2

WHAT THIS STAGE DOES
--------------------
Attention moves information BETWEEN tokens. The MLP processes each token
INDEPENDENTLY, giving the model room to transform/combine the features it just
gathered. It is applied to every position with the same weights.

THE "EXPAND, NON-LINEAR, SHRINK" SHAPE
--------------------------------------
    W1 : (d_model, d_ff)   expands each token vector to a wider space (d_ff = 4*d_model here)
    ReLU                   the non-linearity — without it, two linear layers would
                           collapse into a single linear layer and gain nothing
    W2 : (d_ff, d_model)   projects back down to d_model so it fits the residual stream

The wide hidden layer gives the network many "feature detectors"; ReLU lets each
one switch on or off; W2 recombines the active ones.

WHY ReLU?
---------
    ReLU(x) = max(x, 0)   negatives -> 0, positives pass through unchanged.
Cheap, and its gradient is dead simple: 1 where the input was positive, 0 where
it was negative (this becomes the `(pre_activation > 0)` mask in backprop).

SHAPES
------
    h              : (batch, seq, d_model)
    pre_activation : (batch, seq, d_ff)     after W1
    mlp_hidden     : (batch, seq, d_ff)     after ReLU (same shape, some zeros)
    mlp_output     : (batch, seq, d_model)  after W2, ready for the residual add

WHERE IT GOES AFTERWARDS
------------------------
mlp_output is added back: h = h_before_mlp + mlp_output (stage 6). pre_activation
and mlp_hidden are cached for backprop (the ReLU mask needs pre_activation).
"""

import numpy as np
import math
import matplotlib.pyplot as plt

batch, seq, d_model = 1, 5, 8
d_ff = 32
rng = np.random.default_rng(1)

h  = rng.normal(0, 1, size=(batch, seq, d_model))
W1 = rng.normal(0, 1/math.sqrt(d_model), size=(d_model, d_ff)); b1 = np.zeros(d_ff)
W2 = rng.normal(0, 1/math.sqrt(d_ff),    size=(d_ff, d_model)); b2 = np.zeros(d_model)

print("=" * 70)
print("MLP / FEED-FORWARD SIMULATION")
print("=" * 70)
print(f"\ninput h: {h.shape}")

pre_activation = h @ W1 + b1
print(f"after W1 (expand): {pre_activation.shape}  d_model {d_model} -> d_ff {d_ff}")

mlp_hidden = np.maximum(pre_activation, 0)
killed = (pre_activation <= 0).mean() * 100
print(f"after ReLU: {mlp_hidden.shape}  ({killed:.0f}% of values clamped to 0)")

mlp_output = mlp_hidden @ W2 + b2
print(f"after W2 (shrink): {mlp_output.shape}  d_ff {d_ff} -> d_model {d_model}")

print("\nReLU on the first token's hidden vector (before -> after):")
print(f"  before: {np.round(pre_activation[0,0], 2)}")
print(f"  after : {np.round(mlp_hidden[0,0], 2)}   (negatives became 0)")

# --- VISUALIZATION ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

xs = np.linspace(-3, 3, 200)
axes[0].plot(xs, np.maximum(xs, 0), color="#4C72B0", lw=2)
axes[0].axhline(0, color="k", lw=0.6); axes[0].axvline(0, color="k", lw=0.6)
axes[0].set_title("ReLU(x) = max(x, 0)\nthe non-linearity")
axes[0].set_xlabel("input"); axes[0].set_ylabel("output")

axes[1].bar(range(d_ff), pre_activation[0, 0], color="#C44E52", label="pre-ReLU")
axes[1].bar(range(d_ff), mlp_hidden[0, 0], color="#55A868", alpha=0.8, label="post-ReLU")
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_title("Hidden layer (one token)\nReLU zeroes the negatives")
axes[1].set_xlabel("d_ff neuron"); axes[1].legend()

widths = [d_model, d_ff, d_model]
labels = ["h\n(d_model)", "hidden\n(d_ff, ReLU)", "out\n(d_model)"]
axes[2].bar([0, 1, 2], widths, color=["#4C72B0", "#DD8452", "#4C72B0"])
for i, w in enumerate(widths):
    axes[2].text(i, w + 1, str(w), ha="center")
axes[2].set_xticks([0, 1, 2]); axes[2].set_xticklabels(labels)
axes[2].set_title("expand -> non-linear -> shrink")
axes[2].set_ylabel("vector width")

plt.tight_layout()
plt.show()
