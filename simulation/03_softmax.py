"""
STAGE 3 — SOFTMAX
=================

Source: `softmax()` in training.py

WHAT THIS STAGE DOES
--------------------
Softmax turns a row of arbitrary real-valued scores ("logits") into a
probability distribution: all outputs are positive and they sum to exactly 1.
It is used in TWO different places in this model:
  1. inside attention, to turn match-scores into attention weights
  2. at the output, to turn next-token scores into a probability per token

THE MATH
--------
    softmax(x)_i = exp(x_i) / sum_j exp(x_j)

exp() makes everything positive and exaggerates differences (a slightly bigger
logit becomes a much bigger probability). Dividing by the sum normalises.

THE NUMERICAL-STABILITY TRICK
-----------------------------
exp(1000) overflows to infinity in floating point. But softmax is unchanged if
you subtract any constant from every logit first (the constant cancels in the
ratio). So we subtract the row maximum:
        x = x - max(x)
Now the largest exponent is exp(0) = 1, nothing overflows, and the answer is
mathematically identical. This is the `x - np.max(...)` line in the source.

SHAPES
------
Operates on the LAST axis (axis=-1) with keepdims=True so it broadcasts back.
Input shape == output shape; only the values along the last axis change.

WHERE IT GOES AFTERWARDS
------------------------
Its output feeds cross-entropy (stage 7) and attention (stage 4). Its gradient
has an especially clean form when combined with cross-entropy (see stage 7).
"""

import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # stability shift
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

print("=" * 70)
print("SOFTMAX SIMULATION")
print("=" * 70)

logits = np.array([2.0, 1.0, 0.1, -1.0, 3.5])
print(f"\nraw logits      : {logits}")
p = softmax(logits)
print(f"probabilities   : {np.round(p, 4)}")
print(f"sum of probs     = {p.sum():.6f}  (always 1)")
print(f"argmax logit idx = {logits.argmax()}, gets highest prob {p.max():.4f}")

print("\nSTABILITY DEMONSTRATION")
big = np.array([1000.0, 1001.0, 1002.0])
naive = np.exp(big) / np.sum(np.exp(big))     # overflows
print(f"  naive exp(1000+) -> {naive}  (nan/inf: overflow!)")
print(f"  stable softmax   -> {np.round(softmax(big),4)}  (correct)")

print("\nTEMPERATURE INTUITION (dividing logits before softmax)")
for T in (0.5, 1.0, 2.0):
    print(f"  T={T}: {np.round(softmax(logits / T), 3)}  "
          f"({'sharper' if T < 1 else 'flatter' if T > 1 else 'baseline'})")

# --- VISUALIZATION ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].bar(range(len(logits)), logits, color="#C44E52")
axes[0].axhline(0, color="k", lw=0.8)
axes[0].set_title("INPUT: raw logits\n(any real number)")
axes[0].set_xlabel("class"); axes[0].set_ylabel("score")

axes[1].bar(range(len(p)), p, color="#55A868")
axes[1].set_title("OUTPUT: probabilities\n(positive, sum to 1)")
axes[1].set_xlabel("class"); axes[1].set_ylabel("probability"); axes[1].set_ylim(0, 1)

xs = np.linspace(-4, 4, 200)
for T, c in zip((0.5, 1.0, 2.0), ("#4C72B0", "#000000", "#DD8452")):
    # 2-class softmax of [x, 0] to show the squashing S-curve at each temperature
    probs = softmax(np.stack([xs / T, np.zeros_like(xs)], axis=-1))[:, 0]
    axes[2].plot(xs, probs, color=c, label=f"T={T}")
axes[2].set_title("softmax squashes scores\ninto (0,1); T changes sharpness")
axes[2].set_xlabel("logit difference"); axes[2].set_ylabel("P(class 0)"); axes[2].legend()

plt.tight_layout()
plt.show()
