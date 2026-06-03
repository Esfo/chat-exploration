"""
ATTENTION SIMULATION
====================

an interactable, shrunk-down version of ../training.py:attention_forward.

the goal is to *watch* one attention block work: type a short sequence, then
step through queries/keys/values -> scores -> causal mask -> softmax weights
-> weighted sum of values -> output projection, with every intermediate drawn
as a heatmap.

FAITHFULNESS
------------
the operations here are identical to training.py:attention_forward:
    q,k,v = h @ Wq, h @ Wk, h @ Wv
    split into heads
    scores = q @ k^T / sqrt(head_dim)
    mask out the future (token i cannot see tokens after it)
    weights = softmax(scores)
    heads = weights @ v
    out = recombine(heads) @ Wo
the ONLY thing changed from the real model is the size of the numbers:
training.py runs d_model=256, head_dim=64, seq_len=128. that is impossible to
read on screen, so this sim uses tiny dimensions (set in CONFIG below). the
code path is the same; there is no LayerNorm here, exactly as in training.py.

HOW TO USE
----------
this is a `# %%` cell script: open it in an editor that renders cells (VS Code,
Jupyter) and run cells top to bottom, or run the whole file. the knobs in the
CONFIG cell are the interaction surface -- change SENTENCE, HEAD_TO_VIEW,
USE_MASK, USE_SCALING and re-run to see the effect.
"""

# %%
# ---------------------------------------------------------------------------
# CONFIG  --  these are the knobs. change them and re-run.
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from _shared import softmax, show_matrix

#the sequence the block attends over. keep it short so the matrices stay readable.
#each whitespace-separated word becomes one token (a deliberately trivial tokenizer;
#the real tokenizer lives in training.py and gets its own simulation later).
SENTENCE = "the cat sat on the mat"

#tiny stand-ins for the real Config values (training.py: d_model=256, head_dim=64).
#small enough that every number fits on screen.
D_MODEL = 8
HEAD_DIM = 4
#n_heads is derived exactly like Config.n_heads: d_model // head_dim
N_HEADS = D_MODEL // HEAD_DIM
assert D_MODEL % HEAD_DIM == 0

#which attention head to inspect in the per-head plots (0 .. N_HEADS-1)
HEAD_TO_VIEW = 0

#toggles so you can SEE what each step is for:
#  turn the mask off and watch tokens illegally attend to the future
#  turn scaling off and watch the scores blow up / softmax get spiky
USE_MASK = True
USE_SCALING = True

#fixed seed so the random weights (and thus the plots) are reproducible
SEED = 1

tokens = SENTENCE.split()
SEQ_LEN = len(tokens)
print(f"tokens ({SEQ_LEN}): {tokens}")
print(f"d_model={D_MODEL}  head_dim={HEAD_DIM}  n_heads={N_HEADS}")


# %%
# ---------------------------------------------------------------------------
# EMBEDDINGS  --  hidden state = token meaning + position meaning
# mirrors training.py:forward (tok_emb[x] + pos_emb[positions])
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)

#each distinct word gets its own learned vector. repeated words ("the") share a row,
#exactly like the token embedding table is shared by ID in training.py.
vocab = {tok: i for i, tok in enumerate(dict.fromkeys(tokens))}
token_ids = np.array([vocab[t] for t in tokens])

#0.02 init scale, same as init_params in training.py
tok_emb = rng.normal(0.0, 0.02, size=(len(vocab), D_MODEL))
pos_emb = rng.normal(0.0, 0.02, size=(SEQ_LEN, D_MODEL))

token_vectors = tok_emb[token_ids]
position_vectors = pos_emb[np.arange(SEQ_LEN)]

#hidden state h: this is what attention actually consumes
h = token_vectors + position_vectors

show_matrix(h, title="hidden state h  (token + position)  [seq_len x d_model]",
            row_labels=tokens, col_labels=[f"d{i}" for i in range(D_MODEL)])
plt.show()


# %%
# ---------------------------------------------------------------------------
# Q / K / V PROJECTIONS
# q = "what each token is looking for"
# k = "what each token offers for matching"
# v = "what information each token can pass along"
# mirrors training.py:attention_forward (h @ Wq, h @ Wk, h @ Wv)
# ---------------------------------------------------------------------------
#weight init scale 1/sqrt(d_model), same as init_params
scale = 1.0 / np.sqrt(D_MODEL)
Wq = rng.normal(0.0, scale, size=(D_MODEL, D_MODEL))
Wk = rng.normal(0.0, scale, size=(D_MODEL, D_MODEL))
Wv = rng.normal(0.0, scale, size=(D_MODEL, D_MODEL))
Wo = rng.normal(0.0, scale, size=(D_MODEL, D_MODEL))

q = h @ Wq
k = h @ Wk
v = h @ Wv

fig, axes = plt.subplots(1, 3, figsize=(14, 0.6 * SEQ_LEN + 2))
for ax, mat, name in zip(axes, (q, k, v), ("queries q", "keys k", "values v")):
    show_matrix(mat, title=f"{name}  [seq_len x d_model]", row_labels=tokens, ax=ax)
plt.tight_layout()
plt.show()


# %%
# ---------------------------------------------------------------------------
# SPLIT INTO HEADS
# d_model is sliced into N_HEADS independent head_dim-wide subspaces.
# shape: seq_len x d_model  ->  n_heads x seq_len x head_dim
# (training.py reshapes to batch x heads x seq x head_dim; we drop the batch dim)
# ---------------------------------------------------------------------------
def split_heads(mat):
    #seq_len x d_model -> seq_len x n_heads x head_dim -> n_heads x seq_len x head_dim
    return mat.reshape(SEQ_LEN, N_HEADS, HEAD_DIM).transpose(1, 0, 2)

qh = split_heads(q)
kh = split_heads(k)
vh = split_heads(v)
print(f"per-head shapes: q {qh.shape}  (n_heads, seq_len, head_dim)")


# %%
# ---------------------------------------------------------------------------
# SCORES  --  how much each query matches each key
# scores = q @ k^T,  then divide by sqrt(head_dim)   (toggle USE_SCALING)
# ---------------------------------------------------------------------------
#raw dot-product scores for the head we're inspecting
raw_scores = qh[HEAD_TO_VIEW] @ kh[HEAD_TO_VIEW].T

if USE_SCALING:
    #dot products grow with width; dividing keeps softmax from going spiky.
    scores = raw_scores / np.sqrt(HEAD_DIM)
    scaling_note = f"scaled by 1/sqrt({HEAD_DIM})"
else:
    scores = raw_scores
    scaling_note = "UNSCALED (USE_SCALING=False)"

show_matrix(scores,
            title=f"head {HEAD_TO_VIEW} scores  {scaling_note}\nrow=query token, col=key token",
            row_labels=tokens, col_labels=tokens, cmap="coolwarm")
plt.show()


# %%
# ---------------------------------------------------------------------------
# CAUSAL MASK  --  a token may not attend to the future
# token i can only see tokens 0..i. mirrors training.py's np.triu future_mask
# with the forbidden positions set to a huge negative number before softmax.
# (toggle USE_MASK to watch the model illegally peek ahead)
# ---------------------------------------------------------------------------
future_mask = np.triu(np.ones((SEQ_LEN, SEQ_LEN), dtype=bool), k=1)

masked_scores = scores.copy()
if USE_MASK:
    masked_scores[future_mask] = -1e9
    mask_note = "future masked to -1e9"
else:
    mask_note = "NO MASK (USE_MASK=False) -- tokens can see the future!"

#show the mask itself, then the scores after masking
fig, axes = plt.subplots(1, 2, figsize=(13, 0.6 * SEQ_LEN + 2))
show_matrix(future_mask.astype(float),
            title="future_mask  (1 = forbidden)", row_labels=tokens,
            col_labels=tokens, cmap="Greys", ax=axes[0])
show_matrix(masked_scores,
            title=f"scores after masking  ({mask_note})", row_labels=tokens,
            col_labels=tokens, cmap="coolwarm", ax=axes[1])
plt.tight_layout()
plt.show()


# %%
# ---------------------------------------------------------------------------
# ATTENTION WEIGHTS  --  softmax turns scores into a probability distribution
# each row sums to 1: "where does this query spend its attention?"
# uses the exact softmax from training.py (via _shared)
# ---------------------------------------------------------------------------
weights = softmax(masked_scores)

show_matrix(weights,
            title=f"head {HEAD_TO_VIEW} attention weights  (each row sums to 1)\n"
                  f"row=query token, col=attended token",
            row_labels=tokens, col_labels=tokens, cmap="viridis")
plt.show()

#sanity: every row is a distribution
print("row sums (should all be 1.0):", np.round(weights.sum(axis=1), 4))


# %%
# ---------------------------------------------------------------------------
# WEIGHTED SUM OF VALUES + OUTPUT PROJECTION
# heads = weights @ v   (per head),   then recombine and project through Wo.
# this is the block's output, identical in form to training.py:attention_forward.
# ---------------------------------------------------------------------------
#do the full multi-head computation (all heads), not just the viewed one,
#so the recombination + Wo matches the real forward pass.
all_scores = qh @ kh.transpose(0, 2, 1)
if USE_SCALING:
    all_scores = all_scores / np.sqrt(HEAD_DIM)
if USE_MASK:
    all_scores[:, future_mask] = -1e9
all_weights = softmax(all_scores)

heads = all_weights @ vh                       # n_heads x seq_len x head_dim
combined = heads.transpose(1, 0, 2).reshape(SEQ_LEN, D_MODEL)
out = combined @ Wo                            # the attention block output

fig, axes = plt.subplots(1, 2, figsize=(13, 0.6 * SEQ_LEN + 2))
show_matrix(combined, title="recombined heads  [seq_len x d_model]",
            row_labels=tokens, ax=axes[0])
show_matrix(out, title="attention output  (combined @ Wo)",
            row_labels=tokens, ax=axes[1])
plt.tight_layout()
plt.show()

print("attention output shape:", out.shape, "(matches input h shape)")
