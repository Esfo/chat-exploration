"""
STAGE 7 — CROSS-ENTROPY LOSS (and its gradient)
===============================================

Source: `cross_entropy()` in training.py

WHAT THIS STAGE DOES
--------------------
This is the SCORE the whole model is trying to improve. For each position the
model outputs a probability distribution over the vocabulary; cross-entropy asks
"how much probability did you put on the ACTUALLY-correct next token?" and
turns that into a single number to minimise.

THE MATH
--------
    loss = -mean( log( p[correct_token] ) )
  - if the model gave the correct token probability ~1, log(1)=0  -> loss ~0 (great)
  - if it gave it probability ~0, log(0)=-inf -> huge loss (terrible)
  - the 1e-12 inside the log just prevents log(0) from being literally infinite.

THE BEAUTIFUL GRADIENT
----------------------
Softmax and cross-entropy are derived together, and the gradient of the loss
with respect to the LOGITS collapses to something astonishingly simple:

    dlogits = probs           (start with the predicted distribution)
    dlogits[correct] -= 1     (subtract 1 from the true class)
    dlogits /= N              (average over all N = batch*seq predictions)

Intuition: "(what you predicted) minus (what was true)". If you predicted 0.7 on
the correct token, its gradient is 0.7-1 = -0.3, telling the logit to go UP. A
wrong token you gave 0.2 has gradient +0.2, telling its logit to go DOWN.

SHAPES, STEP BY STEP
--------------------
    logits  : (batch, seq, vocab)
    probs   : (batch, seq, vocab)            via softmax
    flatten : (batch*seq, vocab)             treat every position as one prediction
    targets : (batch*seq,)                   the correct token id per prediction
    loss    : scalar
    dlogits : (batch, seq, vocab)            reshaped back, ready to backprop

WHERE IT GOES AFTERWARDS
------------------------
dlogits is the SEED of backpropagation — the very first gradient. Stage 8 pushes
it back through the output head and every transformer block.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x); return e / np.sum(e, axis=-1, keepdims=True)

def cross_entropy(logits, targets):
    probs = softmax(logits)
    b, s, v = logits.shape
    flat_probs = probs.reshape(b * s, v)
    flat_targets = targets.reshape(b * s)
    correct = flat_probs[np.arange(b * s), flat_targets]
    loss = -np.mean(np.log(correct + 1e-12))
    dlogits = flat_probs.copy()
    dlogits[np.arange(b * s), flat_targets] -= 1
    dlogits /= b * s
    return loss, dlogits.reshape(b, s, v), probs

print("=" * 70)
print("CROSS-ENTROPY LOSS SIMULATION")
print("=" * 70)

# one prediction over a vocab of 6, true token is index 2
vocab = 6
logits = np.array([[[0.5, 1.0, 2.5, 0.2, -0.5, 0.1]]])
targets = np.array([[2]])
loss, dlogits, probs = cross_entropy(logits, targets)

print(f"\nlogits : {logits.ravel()}")
print(f"probs  : {np.round(probs.ravel(), 3)}  (from softmax)")
print(f"true token = index {targets.ravel()[0]}, model gave it p={probs.ravel()[2]:.3f}")
print(f"loss = -log({probs.ravel()[2]:.3f}) = {loss:.4f}")

print("\ngradient dlogits = probs, then -1 on the true class:")
print(f"  dlogits : {np.round(dlogits.ravel(), 3)}")
print("  true class (idx 2) is NEGATIVE  -> push its logit UP")
print("  all others POSITIVE             -> push their logits DOWN")

print("\nHOW LOSS DEPENDS ON THE PROBABILITY OF THE TRUE TOKEN:")
for ptrue in (0.99, 0.5, 0.1, 0.01):
    print(f"  p(correct)={ptrue:<5} -> loss = {-np.log(ptrue):.3f}")

# --- VISUALIZATION ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

idx = np.arange(vocab)
axes[0].bar(idx, probs.ravel(), color="#4C72B0")
axes[0].bar(2, probs.ravel()[2], color="#55A868", label="true token")
axes[0].set_title("Predicted distribution\n(green = correct token)")
axes[0].set_xlabel("token id"); axes[0].set_ylabel("probability"); axes[0].legend()

colors = ["#55A868" if i == 2 else "#C44E52" for i in idx]
axes[1].bar(idx, dlogits.ravel(), color=colors)
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_title("Gradient dlogits = probs - onehot\n(green<0: push up; red>0: push down)")
axes[1].set_xlabel("token id"); axes[1].set_ylabel("d loss / d logit")

p = np.linspace(0.001, 1, 200)
axes[2].plot(p, -np.log(p), color="#8172B3", lw=2)
axes[2].scatter([probs.ravel()[2]], [loss], color="#C44E52", zorder=5,
                label=f"this example\nloss={loss:.2f}")
axes[2].set_title("loss = -log(p_correct)\nconfident & right -> ~0")
axes[2].set_xlabel("probability on correct token"); axes[2].set_ylabel("loss"); axes[2].legend()

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "visualizations", "07_cross_entropy_loss.png")
plt.savefig(out, dpi=110)
print(f"\nVisualization saved -> {out}")
