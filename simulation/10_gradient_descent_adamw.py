"""
STAGE 10 — GRADIENT DESCENT & AdamW
===================================

Source: `adamw_update()` in training.py

WHAT THIS STAGE DOES
--------------------
Backprop told us, for every weight, the direction that INCREASES the loss (the
gradient). To LEARN we step the opposite way. The simplest rule is plain
gradient descent:
        w <- w - lr * grad
AdamW is a smarter version of that same idea, and it is what training.py uses.

PLAIN GRADIENT DESCENT, AND WHY IT STRUGGLES
--------------------------------------------
With a single global learning rate, you must pick one step size for every
weight. On a stretched, valley-shaped loss surface that means you either step
too big along the steep direction (and bounce/diverge) or too small along the
flat direction (and crawl). Adam fixes this by giving every weight its OWN,
adaptive step size.

ADAM = MOMENTUM + PER-WEIGHT SCALING
------------------------------------
It keeps two running averages per weight (the "optimizer state"):
    m = beta1*m + (1-beta1)*grad          smoothed gradient   (momentum / direction)
    v = beta2*v + (1-beta2)*grad^2        smoothed squared grad (how big/noisy it's been)
Then:
    m_hat = m / (1 - beta1^step)          bias correction (m,v start at 0, so early
    v_hat = v / (1 - beta2^step)          steps are biased toward 0 — this undoes it)
    update = m_hat / (sqrt(v_hat) + eps)  divide by typical size -> every weight
                                          gets a ~unit-scale step regardless of grad size
    w <- w - lr * update

THE 'W' IN AdamW: DECOUPLED WEIGHT DECAY
----------------------------------------
    w <- w - lr * (update + weight_decay * w)
The extra `weight_decay * w` term gently pulls every weight toward 0 each step
(regularisation: discourages large weights -> better generalisation). "Decoupled"
means it is applied directly to w, NOT folded into the gradient/Adam statistics.

SHAPES
------
m and v have EXACTLY the same shape as each parameter (one running stat per
weight). The update has that shape too, so `w -= lr*...` is elementwise.

WHERE IT GOES AFTERWARDS
------------------------
This mutates `p` in place — the new weights are used by the next forward pass.
Loop: forward -> loss -> backward -> adamw_update -> repeat (the training loop).
"""

import numpy as np
import matplotlib.pyplot as plt

# A 2-D loss surface so we can SEE optimizers move: a stretched bowl (ill-conditioned).
def loss(w):      # w = [x, y]
    return 0.5 * (w[0]**2 * 0.1 + w[1]**2 * 5.0)   # very flat in x, steep in y
def grad(w):
    return np.array([0.1 * w[0], 5.0 * w[1]])

def run_sgd(w0, lr, steps):
    w = w0.copy(); path = [w.copy()]
    for _ in range(steps):
        w = w - lr * grad(w); path.append(w.copy())
    return np.array(path)

def run_adamw(w0, lr, steps, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    w = w0.copy(); m = np.zeros_like(w); v = np.zeros_like(w); path=[w.copy()]
    for t in range(1, steps+1):
        g = grad(w)
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*(g*g)
        mh = m / (1 - b1**t)
        vh = v / (1 - b2**t)
        update = mh / (np.sqrt(vh) + eps)
        w = w - lr * (update + wd * w)
        path.append(w.copy())
    return np.array(path)

print("=" * 70)
print("GRADIENT DESCENT & AdamW SIMULATION")
print("=" * 70)
w0 = np.array([9.0, 9.0]); steps = 60
sgd_small = run_sgd(w0, lr=0.05, steps=steps)
sgd_big   = run_sgd(w0, lr=0.45, steps=steps)   # too big for the steep direction
adam_path = run_adamw(w0, lr=0.8, steps=steps)

print(f"\nstart at {w0}, loss={loss(w0):.3f}")
print(f"  plain SGD (lr=0.05, safe) final loss = {loss(sgd_small[-1]):.4f}  (slow crawl in flat x)")
print(f"  plain SGD (lr=0.45, big ) final loss = {loss(sgd_big[-1]):.4f}  (oscillates in steep y)")
print(f"  AdamW    (lr=0.80)       final loss = {loss(adam_path[-1]):.4f}  (per-axis step -> fast)")

print("\nSTEP-BY-STEP of AdamW on one weight (the bias-correction matters early):")
w = np.array([9.0, 9.0]); m=np.zeros(2); v=np.zeros(2)
for t in range(1, 4):
    g = grad(w)
    m = 0.9*m + 0.1*g; v = 0.999*v + 0.001*(g*g)
    mh = m/(1-0.9**t); vh = v/(1-0.999**t)
    upd = mh/(np.sqrt(vh)+1e-8)
    print(f"  step {t}: grad={np.round(g,3)} m={np.round(m,3)} v={np.round(v,4)} "
          f"-> update={np.round(upd,3)}  (note: update ~unit-scale on BOTH axes)")
    w = w - 0.8*(upd + 0.01*w)

# --- VISUALIZATION ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

gx, gy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
Z = 0.5 * (gx**2 * 0.1 + gy**2 * 5.0)
cs = axes[0].contour(gx, gy, Z, levels=25, cmap="Greys", linewidths=0.6)
axes[0].plot(sgd_small[:,0], sgd_small[:,1], "o-", ms=3, color="#4C72B0", label="SGD lr=0.05 (slow)")
axes[0].plot(sgd_big[:,0],   sgd_big[:,1],   "o-", ms=3, color="#C44E52", label="SGD lr=0.45 (oscillates)")
axes[0].plot(adam_path[:,0], adam_path[:,1], "o-", ms=3, color="#55A868", label="AdamW (adaptive)")
axes[0].scatter([0],[0], marker="*", s=200, color="gold", edgecolor="k", zorder=5, label="minimum")
axes[0].set_title("Descent paths on a stretched bowl\n(steep in y, flat in x)")
axes[0].set_xlabel("weight x"); axes[0].set_ylabel("weight y"); axes[0].legend(fontsize=8)

axes[1].plot([loss(w) for w in sgd_small], color="#4C72B0", label="SGD lr=0.05")
axes[1].plot([loss(w) for w in sgd_big],   color="#C44E52", label="SGD lr=0.45")
axes[1].plot([loss(w) for w in adam_path], color="#55A868", label="AdamW")
axes[1].set_yscale("log")
axes[1].set_title("Loss vs step\n(this is the curve printed during training)")
axes[1].set_xlabel("step"); axes[1].set_ylabel("loss (log)"); axes[1].legend()

plt.tight_layout()
plt.show()
