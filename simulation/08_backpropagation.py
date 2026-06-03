"""
STAGE 8 — BACKPROPAGATION (the whole chain)
===========================================

Source: `backward()` in training.py (plus attention_backward, stage 9)

WHAT THIS STAGE DOES
--------------------
Backpropagation answers ONE question for every weight: "if I nudge you a tiny
bit, how much does the loss change?" That number is the gradient. We get it by
applying the chain rule, walking BACKWARD through exactly the operations the
forward pass did, reusing the cached intermediates.

THE CORE RULES (each forward op has a matching backward op)
-----------------------------------------------------------
  forward  y = x @ W           backward  dW = x^T @ dy ,   dx = dy @ W^T
  forward  y = x + b           backward  db = sum(dy over batch/seq)
  forward  y = ReLU(x)         backward  dx = dy * (x > 0)
  forward  y = a + b (residual)backward  da = dy ,  db = dy   (gradient COPIES to both)
  forward  embedding lookup    backward  scatter-add dy into the rows that were used

WHY x^T @ dy FOR THE WEIGHT GRADIENT?
-------------------------------------
W connects every input feature to every output feature. dW[i,j] must accumulate
"input i" times "incoming gradient on output j", summed over all the (batch*seq)
examples that flowed through. The matrix product x^T @ dy does exactly that sum.
That is why we FLATTEN (batch,seq,d) -> (batch*seq, d) first: so one matmul sums
the gradient contributions of every token position at once.

TRACING ONE GRADIENT'S SHAPE JOURNEY (output head, Wout: d_model x vocab)
-------------------------------------------------------------------------
  dlogits : (batch, seq, vocab)          seed from cross-entropy (stage 7)
  h       : (batch, seq, d_model)        cached input to the head
  flatten both -> (B*S, vocab), (B*S, d_model)
  grads["Wout"] = h_flat^T @ dlogits_flat   ->  (d_model, vocab)   <- matches Wout!
  grads["bout"] = sum(dlogits over batch,seq) -> (vocab,)          <- matches bout!
  dh = dlogits @ Wout^T                       -> (batch,seq,d_model) keeps flowing down

This script builds a SMALL real network (embed -> 1 transformer block -> head),
runs backward, and then PROVES the gradients are correct with a numerical
gradient check (finite differences). If analytic ≈ numerical, the math is right.
"""

import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ---------- tiny model ----------
rng = np.random.default_rng(1)
vocab, d_model, d_ff, seq, batch = 7, 8, 16, 4, 2
n_heads, head_dim = 2, 4

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x); return e / np.sum(e, axis=-1, keepdims=True)

def init():
    p = {}
    p["tok_emb"] = rng.normal(0, 0.02, (vocab, d_model))
    p["pos_emb"] = rng.normal(0, 0.02, (seq, d_model))
    for L in range(1):
        for nm in ("Wq", "Wk", "Wv", "Wo"):
            p[f"{L}.{nm}"] = rng.normal(0, 1/math.sqrt(d_model), (d_model, d_model))
        p[f"{L}.W1"] = rng.normal(0, 1/math.sqrt(d_model), (d_model, d_ff)); p[f"{L}.b1"] = np.zeros(d_ff)
        p[f"{L}.W2"] = rng.normal(0, 1/math.sqrt(d_ff), (d_ff, d_model));     p[f"{L}.b2"] = np.zeros(d_model)
    p["Wout"] = rng.normal(0, 1/math.sqrt(d_model), (d_model, vocab)); p["bout"] = np.zeros(vocab)
    return p

def attn_fwd(h, p):
    B, S, D = h.shape
    q = h @ p["0.Wq"]; k = h @ p["0.Wk"]; v = h @ p["0.Wv"]
    qh = q.reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)
    kh = k.reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)
    vh = v.reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)
    scores = (qh @ kh.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
    mask = np.triu(np.ones((S, S), bool), 1)
    scores[:, :, mask] = -1e9
    probs = softmax(scores)
    heads = probs @ vh
    combined = heads.transpose(0, 2, 1, 3).reshape(B, S, D)
    out = combined @ p["0.Wo"]
    return out, (h, qh, kh, vh, probs, combined)

def attn_bwd(dout, cache, p):
    h, qh, kh, vh, probs, combined = cache
    B, S, D = h.shape
    g = {}
    g["0.Wo"] = combined.reshape(-1, D).T @ dout.reshape(-1, D)
    dcomb = dout @ p["0.Wo"].T
    dheads = dcomb.reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)
    dprobs = dheads @ vh.transpose(0, 1, 3, 2)
    dvh = probs.transpose(0, 1, 3, 2) @ dheads
    dscores = probs * (dprobs - np.sum(dprobs * probs, axis=-1, keepdims=True))
    dscores /= math.sqrt(head_dim)
    dqh = dscores @ kh
    dkh = dscores.transpose(0, 1, 3, 2) @ qh
    dq = dqh.transpose(0, 2, 1, 3).reshape(B, S, D)
    dk = dkh.transpose(0, 2, 1, 3).reshape(B, S, D)
    dv = dvh.transpose(0, 2, 1, 3).reshape(B, S, D)
    fh = h.reshape(-1, D)
    g["0.Wq"] = fh.T @ dq.reshape(-1, D)
    g["0.Wk"] = fh.T @ dk.reshape(-1, D)
    g["0.Wv"] = fh.T @ dv.reshape(-1, D)
    dh = dq @ p["0.Wq"].T + dk @ p["0.Wk"].T + dv @ p["0.Wv"].T
    return dh, g

def forward(x, p):
    B, S = x.shape
    h = p["tok_emb"][x] + p["pos_emb"][np.arange(S)][None]
    h_before_attn = h
    ao, acache = attn_fwd(h, p)
    h = h_before_attn + ao
    h_before_mlp = h
    pre = h @ p["0.W1"] + p["0.b1"]
    hid = np.maximum(pre, 0)
    mo = hid @ p["0.W2"] + p["0.b2"]
    h = h_before_mlp + mo
    logits = h @ p["Wout"] + p["bout"]
    return logits, h, (acache, pre, hid, h_before_mlp)

def loss_and_seed(logits, y):
    probs = softmax(logits)
    B, S, V = logits.shape
    fp = probs.reshape(B * S, V); ft = y.reshape(B * S)
    loss = -np.mean(np.log(fp[np.arange(B * S), ft] + 1e-12))
    d = fp.copy(); d[np.arange(B * S), ft] -= 1; d /= B * S
    return loss, d.reshape(B, S, V)

def backward(x, y, logits, h, cache, p):
    acache, pre, hid, h_before_mlp = cache
    loss, dlogits = loss_and_seed(logits, y)
    g = {n: np.zeros_like(v) for n, v in p.items()}
    B, S, D = h.shape
    # --- output head ---
    g["Wout"] = h.reshape(-1, D).T @ dlogits.reshape(-1, vocab)
    g["bout"] = np.sum(dlogits, axis=(0, 1))
    dh = dlogits @ p["Wout"].T
    # --- MLP block (through its residual) ---
    d_mo = dh
    g["0.W2"] += hid.reshape(-1, d_ff).T @ d_mo.reshape(-1, D)
    g["0.b2"] += np.sum(d_mo, axis=(0, 1))
    d_hid = d_mo @ p["0.W2"].T
    d_pre = d_hid * (pre > 0)                 # ReLU mask
    g["0.W1"] += h_before_mlp.reshape(-1, D).T @ d_pre.reshape(-1, d_ff)
    g["0.b1"] += np.sum(d_pre, axis=(0, 1))
    dh = dh + d_pre @ p["0.W1"].T             # residual copy + block path
    # --- attention block (through its residual) ---
    d_ao = dh
    dh_attn, ag = attn_bwd(d_ao, acache, p)
    for n, gg in ag.items():
        g[n] += gg
    dh = dh + dh_attn
    # --- embeddings ---
    g["pos_emb"][:S] += np.sum(dh, axis=0)
    np.add.at(g["tok_emb"], x.reshape(-1), dh.reshape(-1, D))
    return loss, g

# ---------- run it ----------
print("=" * 70)
print("BACKPROPAGATION SIMULATION")
print("=" * 70)
p = init()
x = rng.integers(0, vocab, (batch, seq))
y = rng.integers(0, vocab, (batch, seq))
logits, h, cache = forward(x, p)
loss, grads = backward(x, y, logits, h, cache, p)
print(f"\nforward loss = {loss:.4f}")
print("\nGRADIENT SHAPES MATCH PARAMETER SHAPES (they must, to update them):")
for nm in ("tok_emb", "pos_emb", "0.Wq", "0.W1", "0.b1", "Wout", "bout"):
    print(f"  {nm:9s}: param {str(p[nm].shape):14s} grad {str(grads[nm].shape):14s}"
          f" {'OK' if p[nm].shape == grads[nm].shape else 'MISMATCH'}")

# ---------- numerical gradient check: the proof ----------
print("\nNUMERICAL GRADIENT CHECK (finite differences vs our analytic gradient)")
print("perturb each weight by +/-eps, measure dLoss/dW, compare to backprop:")
eps = 1e-5
rel_errors = {}
for nm in ("Wout", "0.W2", "0.Wq", "tok_emb", "pos_emb", "bout"):
    W = p[nm]
    idx = tuple(np.random.randint(0, s) for s in W.shape)
    orig = W[idx]
    W[idx] = orig + eps; lp, _, _ = forward(x, p); lp, _ = loss_and_seed(lp, y)
    W[idx] = orig - eps; lm, _, _ = forward(x, p); lm, _ = loss_and_seed(lm, y)
    W[idx] = orig
    numeric = (lp - lm) / (2 * eps)
    analytic = grads[nm][idx]
    rel = abs(numeric - analytic) / (abs(numeric) + abs(analytic) + 1e-12)
    rel_errors[nm] = rel
    print(f"  {nm:9s} analytic={analytic:+.6e}  numeric={numeric:+.6e}  rel.err={rel:.2e}")
print("\nrel.err ~1e-6 or smaller => the hand-derived backprop math is CORRECT.")

# --- VISUALIZATION ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

names = list(rel_errors.keys())
axes[0].bar(names, [rel_errors[n] for n in names], color="#55A868")
axes[0].set_yscale("log"); axes[0].axhline(1e-4, color="#C44E52", ls="--", label="1e-4 threshold")
axes[0].set_title("Gradient check: relative error\n(tiny = correct)")
axes[0].set_ylabel("rel. error (log)"); axes[0].tick_params(axis="x", rotation=45); axes[0].legend()

# gradient magnitude per parameter group = "how strong is the learning signal here"
mags = {n: np.abs(grads[n]).mean() for n in p}
axes[1].barh(list(mags.keys()), list(mags.values()), color="#4C72B0")
axes[1].set_title("Mean |gradient| per parameter\n(the learning signal reaching each)")
axes[1].set_xlabel("mean |grad|")

im = axes[2].imshow(grads["Wout"], aspect="auto", cmap="coolwarm")
axes[2].set_title("grad of Wout (d_model x vocab)\n= h^T @ dlogits")
axes[2].set_xlabel("vocab"); axes[2].set_ylabel("d_model")
plt.colorbar(im, ax=axes[2], fraction=0.046)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "visualizations", "08_backpropagation.png")
plt.savefig(out, dpi=110)
print(f"\nVisualization saved -> {out}")
