"""
STAGE 2 — EMBEDDINGS (token + position)
=======================================

Source: the first lines of `forward()` in training.py:
    token_vectors    = p["tok_emb"][x]
    position_vectors = p["pos_emb"][positions][None, :, :]
    h = token_vectors + position_vectors

WHAT THIS STAGE DOES
--------------------
The model cannot do math on the integer token ID 7. It needs a vector. The
embedding step is a LOOKUP: each token ID indexes one row of a learned table.

  tok_emb has shape (vocab_size, d_model). tok_emb[x] uses the integer array x
  to gather rows. This is "fancy indexing": for an x of shape (batch, seq) the
  result is (batch, seq, d_model) — every ID has been swapped for its row.

WHY ADD POSITION EMBEDDINGS?
----------------------------
Attention (next stage) is order-blind: by itself it treats a sentence as a bag
of tokens. "dog bites man" and "man bites dog" would look identical. So we add a
second learned table indexed by POSITION (0,1,2,...). Now the vector fed forward
encodes BOTH "what the token is" AND "where it sits".

  pos_emb has shape (context_length, d_model). We take the first `seq_len` rows,
  then add a leading axis with [None, :, :] -> shape (1, seq_len, d_model) so it
  BROADCASTS across every example in the batch (same positions for all rows).

SHAPES, STEP BY STEP
--------------------
  x                : (batch, seq)              integer IDs
  tok_emb          : (vocab, d_model)          learned table
  tok_emb[x]       : (batch, seq, d_model)     gathered token vectors
  pos_emb[:seq]    : (seq, d_model)            learned positions
  [None,:,:]       : (1, seq, d_model)         broadcastable
  h = sum          : (batch, seq, d_model)     <- this is the hidden state

WHERE IT GOES AFTERWARDS
------------------------
`h` is the running "hidden state" carried through every transformer block. In
the backward pass, gradients flow all the way back here: dh is scattered back
into the exact rows of tok_emb/pos_emb that were looked up (see stage 8).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

vocab_size = 12
d_model = 8
rng = np.random.default_rng(1)

tok_emb = rng.normal(0, 0.02, size=(vocab_size, d_model))
pos_emb = rng.normal(0, 0.02, size=(16, d_model))

# a tiny batch of 2 sequences, 5 tokens each
x = np.array([[3, 7, 7, 1, 9],
              [2, 5, 8, 7, 0]])
batch, seq = x.shape

print("=" * 70)
print("EMBEDDING SIMULATION")
print("=" * 70)
print(f"\ninput token IDs x, shape {x.shape}:\n{x}")

token_vectors = tok_emb[x]
print(f"\nafter tok_emb[x]: shape {token_vectors.shape}")
print("each integer was replaced by its d_model-wide row.")
print(f"token ID {x[0,0]} -> vector {np.round(token_vectors[0,0], 3)}")
print("NOTE both 7's map to the SAME vector (same row of the table):")
print(f"  x[0,1]=7 -> {np.round(token_vectors[0,1],3)}")
print(f"  x[0,2]=7 -> {np.round(token_vectors[0,2],3)}")

positions = np.arange(seq)
position_vectors = pos_emb[positions][None, :, :]
print(f"\nposition vectors, shape {position_vectors.shape} (leading 1 broadcasts over batch)")

h = token_vectors + position_vectors
print(f"\nh = token meaning + position meaning, shape {h.shape}")
print("the two 7's are now DIFFERENT because they sit at different positions:")
print(f"  position 1 copy of 7 -> {np.round(h[0,1],3)}")
print(f"  position 2 copy of 7 -> {np.round(h[0,2],3)}")

# --- VISUALIZATION ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

im0 = axes[0].imshow(token_vectors[0], aspect="auto", cmap="coolwarm")
axes[0].set_title("token_vectors (seq 0)\nrows=positions, cols=d_model")
axes[0].set_yticks(range(seq)); axes[0].set_yticklabels([f"id={i}" for i in x[0]])
axes[0].set_xlabel("embedding dim"); plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(position_vectors[0], aspect="auto", cmap="coolwarm")
axes[1].set_title("position_vectors\nsame for every batch row")
axes[1].set_yticks(range(seq)); axes[1].set_yticklabels([f"pos={i}" for i in range(seq)])
axes[1].set_xlabel("embedding dim"); plt.colorbar(im1, ax=axes[1], fraction=0.046)

im2 = axes[2].imshow(h[0], aspect="auto", cmap="coolwarm")
axes[2].set_title("h = token + position\nfed into the transformer")
axes[2].set_yticks(range(seq))
axes[2].set_yticklabels([f"id{x[0,i]}@p{i}" for i in range(seq)])
axes[2].set_xlabel("embedding dim"); plt.colorbar(im2, ax=axes[2], fraction=0.046)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "visualizations", "02_embeddings.png")
plt.savefig(out, dpi=110)
print(f"\nVisualization saved -> {out}")
