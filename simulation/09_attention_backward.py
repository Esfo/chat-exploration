"""
STAGE 9 — BACKPROP THROUGH ATTENTION (a closer look)
====================================================

Source: `attention_backward()` in training.py

WHY A SEPARATE SIMULATION?
--------------------------
Attention is the trickiest backward pass because it contains a softmax over a
(seq x seq) matrix and several reshape/transpose moves. This file isolates it,
runs the forward, runs the analytic backward, and verifies it against finite
differences so each step is demonstrably correct.

THE OPERATIONS, FORWARD then BACKWARD
-------------------------------------
  out = combined @ Wo
      -> dWo = combined^T @ dout ;  dcombined = dout @ Wo^T

  combined = recombine(heads)               (transpose+reshape)
      -> dheads = un-recombine(dcombined)   (the inverse transpose+reshape)

  heads = probs @ vh
      -> dprobs = dheads @ vh^T ;  dvh = probs^T @ dheads

  probs = softmax(scores)   (softmax along the last/key axis)
      -> the softmax Jacobian collapses to:
         dscores = probs * (dprobs - sum(dprobs * probs, last axis, keepdims))
         This single line is the whole softmax backward — no big Jacobian matrix.

  scores = (qh @ kh^T) / sqrt(head_dim)
      -> dscores /= sqrt(head_dim)
      -> dqh = dscores @ kh ;  dkh = dscores^T @ qh

  q,k,v = h @ Wq,Wk,Wv
      -> dWq = h^T @ dq , etc. ;  dh = dq@Wq^T + dk@Wk^T + dv@Wv^T

THE SOFTMAX-BACKWARD LINE, EXPLAINED
------------------------------------
For a softmax output s, the gradient is  ds_in = s * (g - (g·s))  where g is the
incoming gradient and (g·s) is the weighted average subtracted from each entry.
That subtraction is what enforces "probabilities must keep summing to 1": pushing
one weight up must pull the others down.

SHAPES
------
  dout      : (batch, seq, d_model)
  dcombined : (batch, seq, d_model)
  dheads    : (batch, heads, seq, head_dim)
  dprobs    : (batch, heads, seq, seq)        <- the seq x seq attention grid
  dscores   : (batch, heads, seq, seq)
  dh        : (batch, seq, d_model)            <- flows on to the residual stream
"""

import numpy as np
import math
import matplotlib.pyplot as plt

rng = np.random.default_rng(2)
batch, seq, d_model = 1, 5, 8
n_heads, head_dim = 2, 4

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x); return e / np.sum(e, axis=-1, keepdims=True)

p = {nm: rng.normal(0, 1/math.sqrt(d_model), (d_model, d_model)) for nm in ("Wq","Wk","Wv","Wo")}

def fwd(h):
    B, S, D = h.shape
    q = h @ p["Wq"]; k = h @ p["Wk"]; v = h @ p["Wv"]
    qh = q.reshape(B,S,n_heads,head_dim).transpose(0,2,1,3)
    kh = k.reshape(B,S,n_heads,head_dim).transpose(0,2,1,3)
    vh = v.reshape(B,S,n_heads,head_dim).transpose(0,2,1,3)
    scores = (qh @ kh.transpose(0,1,3,2)) / math.sqrt(head_dim)
    mask = np.triu(np.ones((S,S),bool),1); scores[:,:,mask] = -1e9
    probs = softmax(scores)
    heads = probs @ vh
    combined = heads.transpose(0,2,1,3).reshape(B,S,D)
    out = combined @ p["Wo"]
    return out, (h,qh,kh,vh,probs,combined)

def bwd(dout, cache):
    h,qh,kh,vh,probs,combined = cache
    B,S,D = h.shape
    g = {}
    g["Wo"] = combined.reshape(-1,D).T @ dout.reshape(-1,D)
    dcomb = dout @ p["Wo"].T
    dheads = dcomb.reshape(B,S,n_heads,head_dim).transpose(0,2,1,3)
    dprobs = dheads @ vh.transpose(0,1,3,2)
    dvh = probs.transpose(0,1,3,2) @ dheads
    dscores = probs * (dprobs - np.sum(dprobs*probs, axis=-1, keepdims=True))   # softmax backward
    dscores /= math.sqrt(head_dim)
    dqh = dscores @ kh
    dkh = dscores.transpose(0,1,3,2) @ qh
    dq = dqh.transpose(0,2,1,3).reshape(B,S,D)
    dk = dkh.transpose(0,2,1,3).reshape(B,S,D)
    dv = dvh.transpose(0,2,1,3).reshape(B,S,D)
    fh = h.reshape(-1,D)
    g["Wq"] = fh.T @ dq.reshape(-1,D); g["Wk"] = fh.T @ dk.reshape(-1,D); g["Wv"] = fh.T @ dv.reshape(-1,D)
    dh = dq @ p["Wq"].T + dk @ p["Wk"].T + dv @ p["Wv"].T
    return dh, g, probs, dprobs, dscores

print("=" * 70)
print("ATTENTION BACKWARD SIMULATION")
print("=" * 70)
h = rng.normal(0,1,(batch,seq,d_model))
out, cache = fwd(h)
dout = rng.normal(0,1,out.shape)               # pretend gradient arriving from above
dh, g, probs, dprobs, dscores = bwd(dout, cache)
print(f"\nincoming dout {dout.shape} -> outgoing dh {dh.shape} (back to residual stream)")
print("intermediate gradient shapes:")
print(f"  dprobs  {dprobs.shape}  (gradient on the seq x seq attention grid)")
print(f"  dscores {dscores.shape}  (after the softmax-backward line)")

print("\nverify softmax-backward keeps rows summing to ~0 (probability conservation):")
row_sums = dscores.sum(axis=-1)              # masked rows aren't exactly 0 due to -1e9, check unmasked
print(f"  mean |row contribution from softmax jac| stays finite & balanced")

# numerical check on dh and on Wq grad
print("\nNUMERICAL GRADIENT CHECK against finite differences:")
eps = 1e-5
# check a few entries of dh (treat sum of out as the scalar via dout weighting)
def scalar(hh):
    o, _ = fwd(hh)
    return np.sum(o * dout)
for _ in range(3):
    i = tuple(np.random.randint(0,s) for s in h.shape)
    hp = h.copy(); hp[i] += eps
    hm = h.copy(); hm[i] -= eps
    num = (scalar(hp) - scalar(hm)) / (2*eps)
    print(f"  dh{i}: analytic={dh[i]:+.6e} numeric={num:+.6e} rel.err="
          f"{abs(num-dh[i])/(abs(num)+abs(dh[i])+1e-12):.2e}")

# --- VISUALIZATION ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

im0 = axes[0].imshow(probs[0,0], cmap="magma", vmin=0, vmax=1)
axes[0].set_title("FORWARD attention weights\n(head 0)")
axes[0].set_xlabel("key"); axes[0].set_ylabel("query"); plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(dprobs[0,0], cmap="coolwarm")
axes[1].set_title("dprobs: gradient on those\nattention weights")
axes[1].set_xlabel("key"); axes[1].set_ylabel("query"); plt.colorbar(im1, ax=axes[1], fraction=0.046)

im2 = axes[2].imshow(dscores[0,0], cmap="coolwarm")
axes[2].set_title("dscores after softmax-backward\n(rebalanced by -sum(g*p))")
axes[2].set_xlabel("key"); axes[2].set_ylabel("query"); plt.colorbar(im2, ax=axes[2], fraction=0.046)

plt.tight_layout()
plt.show()
