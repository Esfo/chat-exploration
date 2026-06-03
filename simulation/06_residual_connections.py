"""
STAGE 6 — RESIDUAL CONNECTIONS
==============================

Source: the two `h = ... + ...` lines per layer in `forward()`:
    h = h_before_attention + attention_output
    h = h_before_mlp       + mlp_output

WHAT THIS STAGE DOES
--------------------
Instead of replacing the hidden state with each block's output, we ADD the
output to the input:   h_new = h_old + block(h_old).
The block only has to learn a *correction* ("residual") to the running state,
not reinvent it from scratch. The unchanged copy of h_old is the "residual
stream" — a highway that runs straight through the whole network.

WHY THIS MATTERS: GRADIENT FLOW
-------------------------------
Because of the `+`, the backward pass splits the incoming gradient and sends a
copy DIRECTLY back to h_old (the derivative of (a+b) w.r.t. a is 1). So even in
a deep stack, gradients always have a clean, undiminished path back to early
layers. Without residuals, gradients get repeatedly multiplied by weight
matrices and tend to vanish (shrink to ~0) or explode. This simulation shows
that difference numerically.

SHAPES
------
The add is elementwise, so both sides must match: (batch, seq, d_model). That is
exactly why the MLP's W2 and attention's Wo project back DOWN to d_model — so the
result can re-enter the residual stream.

WHERE IT GOES AFTERWARDS
------------------------
The summed h continues to the next sub-block / layer. In backprop the gradient
dh is reused twice: once down the block path, once straight down the residual
path, and the two are added (you'll see `dh = dh + ...` in stage 8).
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
d = 32
n_layers = 40   # deliberately deep to expose vanishing gradients

# Simulate backward gradient magnitude through many layers, with vs without residuals.
# Each "block" is a random linear map; we track ||gradient|| as it flows backward.
Ws = [rng.normal(0, 1/np.sqrt(d), size=(d, d)) for _ in range(n_layers)]

def backprop_norms(residual):
    g = rng.normal(0, 1, size=(d,))           # gradient arriving at the top
    norms = [np.linalg.norm(g)]
    for W in reversed(Ws):
        block_grad = g @ W.T                  # gradient through the block's matrix
        g = (g + block_grad) if residual else block_grad   # residual adds the highway copy
        norms.append(np.linalg.norm(g))
    return norms

print("=" * 70)
print("RESIDUAL CONNECTION SIMULATION")
print("=" * 70)

no_res = backprop_norms(residual=False)
with_res = backprop_norms(residual=True)

print(f"\nGradient norm after flowing back through {n_layers} layers:")
print(f"  WITHOUT residuals: {no_res[0]:.3f} -> {no_res[-1]:.6f}   "
      f"(x{no_res[-1]/no_res[0]:.1e})  vanishes")
print(f"  WITH residuals   : {with_res[0]:.3f} -> {with_res[-1]:.3f}   "
      f"(x{with_res[-1]/with_res[0]:.2f})  preserved")

print("\nForward view: residual = identity highway + a small correction")
h_old = rng.normal(0, 1, size=(d,))
correction = 0.1 * rng.normal(0, 1, size=(d,))   # block output is a small nudge
h_new = h_old + correction
print(f"  ||h_old||={np.linalg.norm(h_old):.3f}, ||correction||={np.linalg.norm(correction):.3f}, "
      f"||h_new||={np.linalg.norm(h_new):.3f}")

# --- VISUALIZATION ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

axes[0].plot(no_res, "o-", color="#C44E52", label="no residual (vanishes)", ms=3)
axes[0].plot(with_res, "o-", color="#55A868", label="with residual (stable)", ms=3)
axes[0].set_yscale("log")
axes[0].set_title("Gradient magnitude flowing backward\nthrough a deep stack")
axes[0].set_xlabel("layers traversed (backward)"); axes[0].set_ylabel("||gradient|| (log scale)")
axes[0].legend()

axes[1].bar(["h_old", "correction", "h_new = h_old+corr"],
            [np.linalg.norm(h_old), np.linalg.norm(correction), np.linalg.norm(h_new)],
            color=["#4C72B0", "#DD8452", "#8172B3"])
axes[1].set_title("Forward: block learns only a\nsmall correction to the stream")
axes[1].set_ylabel("vector norm")

plt.tight_layout()
plt.show()
