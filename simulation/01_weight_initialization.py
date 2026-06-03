"""
STAGE 1 — WEIGHT INITIALIZATION
================================

Source: `init_params()` in training.py

WHAT THIS STAGE DOES
--------------------
Before any learning can happen, every trainable matrix needs starting numbers.
You cannot start them all at zero: if every weight is identical, every neuron
computes the same thing and receives the same gradient, so they can never
become different from each other ("symmetry breaking" fails). So we sample
small random numbers from a normal (Gaussian) distribution.

THE KEY IDEA: SCALE THE RANDOMNESS BY 1/sqrt(fan_in)
----------------------------------------------------
When a layer computes   y = x @ W   each output number is a sum of `fan_in`
products (fan_in = number of input features = rows of W). If you add up many
random numbers, the result's variance grows with how many you add. To keep the
size of activations stable as they pass through the network, each weight is
drawn with standard deviation 1/sqrt(fan_in). That way the variance of the sum
stays ~constant no matter how wide the layer is.

  - tok_emb / pos_emb : scale 0.02 (a small fixed constant, the GPT convention)
  - Wq,Wk,Wv,Wo,W1    : scale 1/sqrt(d_model)
  - W2                : scale 1/sqrt(d_ff)      (its input is d_ff wide)
  - Wout              : scale 1/sqrt(d_model)
  - all biases        : start at exactly 0 (biases don't cause symmetry problems)

WHERE THE SHAPES COME FROM
--------------------------
  tok_emb : (vocab_size, d_model)        one row per token  -> its meaning vector
  pos_emb : (context_length, d_model)    one row per position
  Wq/Wk/Wv/Wo : (d_model, d_model)       mix a token vector into a new token vector
  W1 : (d_model, d_ff)                   expand each token vector
  W2 : (d_ff, d_model)                   shrink it back
  Wout : (d_model, vocab_size)           score every possible next token

WHERE THESE GO AFTERWARDS
-------------------------
The returned dict `p` is the model. Every other stage reads from it (forward
pass) and every gradient writes a correction back into it (the optimizer).
"""

import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# --- toy config (small so we can actually look at the numbers) ---
vocab_size = 50
d_model = 32
d_ff = 128          # = d_model * 4
context_length = 16

rng = np.random.default_rng(1)   # seed 1 -> repeatable, exactly like training.py

print("=" * 70)
print("WEIGHT INITIALIZATION SIMULATION")
print("=" * 70)

def show(name, arr, scale):
    print(f"\n{name}")
    print(f"  shape           = {arr.shape}")
    print(f"  target std dev  = {scale:.5f}")
    print(f"  measured std    = {arr.std():.5f}   (should be close to target)")
    print(f"  measured mean   = {arr.mean():+.5f}  (should be near 0)")

tok_emb = rng.normal(0.0, 0.02, size=(vocab_size, d_model))
show("tok_emb  (token meaning table)", tok_emb, 0.02)

Wq = rng.normal(0.0, 1/math.sqrt(d_model), size=(d_model, d_model))
show("Wq  (query projection, scaled by 1/sqrt(d_model))", Wq, 1/math.sqrt(d_model))

W2 = rng.normal(0.0, 1/math.sqrt(d_ff), size=(d_ff, d_model))
show("W2  (MLP down-projection, scaled by 1/sqrt(d_ff))", W2, 1/math.sqrt(d_ff))

print("\nWHY DIFFERENT SCALES?")
print(f"  W1 input width = d_model = {d_model}  -> scale 1/sqrt({d_model}) = {1/math.sqrt(d_model):.4f}")
print(f"  W2 input width = d_ff    = {d_ff} -> scale 1/sqrt({d_ff}) = {1/math.sqrt(d_ff):.4f}")
print("  Wider input -> smaller weights, so the summed output keeps a stable size.")

# --- DEMONSTRATION: what happens to activation size with right vs wrong scale ---
x = rng.normal(0, 1, size=(1000, d_model))          # fake activations, std = 1
good_W = rng.normal(0, 1/math.sqrt(d_model), size=(d_model, d_model))
bad_W  = rng.normal(0, 1.0,                  size=(d_model, d_model))  # unscaled
good_out = x @ good_W
bad_out  = x @ bad_W
print("\nACTIVATION-SIZE DEMO (input std = 1.0):")
print(f"  output std with 1/sqrt(d_model) scaling : {good_out.std():.3f}  (stays ~1, healthy)")
print(f"  output std with unscaled weights        : {bad_out.std():.3f}  (blows up!)")

# --- VISUALIZATION ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].hist(tok_emb.ravel(), bins=50, color="#4C72B0", alpha=0.85)
axes[0].axvline(0, color="k", lw=0.8)
axes[0].set_title("Initial weights are a tight\nGaussian centred on 0")
axes[0].set_xlabel("weight value"); axes[0].set_ylabel("count")

axes[1].hist(good_out.ravel(), bins=60, color="#55A868", alpha=0.8, label="1/sqrt(fan_in) scaled")
axes[1].hist(bad_out.ravel(),  bins=60, color="#C44E52", alpha=0.5, label="unscaled")
axes[1].set_title("Why the scale matters:\nactivation size after one layer")
axes[1].set_xlabel("activation value"); axes[1].legend()

# show how std target shrinks as fan_in grows
fan_ins = np.arange(8, 1025)
axes[2].plot(fan_ins, 1/np.sqrt(fan_ins), color="#8172B3")
for f in (d_model, d_ff):
    axes[2].scatter([f], [1/math.sqrt(f)], zorder=5)
    axes[2].annotate(f"fan_in={f}", (f, 1/math.sqrt(f)), textcoords="offset points", xytext=(8, 6))
axes[2].set_title("Init std = 1/sqrt(fan_in)\nwider layers start smaller")
axes[2].set_xlabel("fan_in (input width)"); axes[2].set_ylabel("std dev of weights")

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "visualizations", "01_weight_initialization.png")
plt.savefig(out, dpi=110)
print(f"\nVisualization saved -> {out}")
