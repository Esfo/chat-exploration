"""
GRADIENT DESCENT SIMULATION
===========================

a real simulation, in the dice-roll sense: it iterates, it moves toward a goal,
and the meaning emerges from repetition rather than from a single snapshot.

what it simulates
-----------------
the inner loop of ../training.py: a parameter is nudged downhill by its gradient,
over and over, to MINIMIZE a loss. in the real model the loss is cross-entropy
over millions of parameters and you can't see it move. here we shrink the
parameter space to 2D so you can literally watch the point roll into the goal.

the optimizers are faithful to training.py:
    - "sgd"   : x -= lr * grad                      (plain gradient descent)
    - "adamw" : the exact m / v / bias-correction / weight-decay update from
                training.py:adamw_update, just applied to a 2D parameter.

why this is a simulation and the attention heatmap was not
---------------------------------------------------------
ITERATION : it runs N_STEPS, the same nudge repeated.
GOAL      : the minimum of the loss (marked * in the plots). the run ARRIVES there
            (or fails to).
EMERGENCE : run it from many random starts and a distribution appears that MEANS
            something -- how many steps it takes to reach the goal, and how often
            the learning rate is so big the run diverges instead. that histogram
            is the dice-roll: outcomes from repeated trials.

HOW TO USE
----------
`# %%` cell script. the knobs in CONFIG are the experiment. the lesson lives in
turning LEARNING_RATE up until descent turns into oscillation and then explosion,
and in comparing OPTIMIZER = "sgd" vs "adamw" on the same surface.
"""

# %%
# ---------------------------------------------------------------------------
# CONFIG  --  this is the experiment. change it and re-run.
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

#the loss surface: a tilted bowl  f(x) = 0.5 * (CURVATURE . x^2)
#its minimum (the GOAL) is at the origin, where f = 0.
#the two curvatures differ on purpose: the bowl is steep one way and shallow the
#other (an "ill-conditioned" valley). that gap is what makes plain SGD oscillate
#and what AdamW's per-direction scaling is built to handle -- a real, meaningful
#difference, not decoration.
CURVATURE = np.array([1.0, 12.0])   # [gentle direction, steep direction]

OPTIMIZER = "sgd"        # "sgd" or "adamw"
LEARNING_RATE = 0.05     # the knob. turn it up: crawl -> converge -> oscillate -> explode
N_STEPS = 80             # iterations per run
START = np.array([4.5, 3.0])   # where the single traced run begins

#"converged" means we got this close to the goal
TOLERANCE = 1e-2
#a run is "diverged" if it ever gets this far out (the loss exploded)
DIVERGED = 1e3

SEED = 1


def loss(x):
    """scalar loss at point x. the thing we are trying to minimize."""
    return 0.5 * np.sum(CURVATURE * x * x)


def grad(x):
    """gradient of loss at x. points uphill; we step against it."""
    return CURVATURE * x


# %%
# ---------------------------------------------------------------------------
# THE OPTIMIZERS  --  one step each. adamw mirrors training.py:adamw_update.
# ---------------------------------------------------------------------------
def sgd_step(x, g, lr, state, step):
    #plain descent: move straight downhill.
    return x - lr * g


def adamw_step(x, g, lr, state, step,
               beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
    #faithful to training.py:adamw_update.
    #weight_decay defaults to 0 here so it doesn't drag the goal off the origin;
    #set it to 0.01 to match training.py exactly and watch the min shift slightly.
    if not state:
        state["m"] = np.zeros_like(x)
        state["v"] = np.zeros_like(x)

    state["m"] = beta1 * state["m"] + (1 - beta1) * g
    state["v"] = beta2 * state["v"] + (1 - beta2) * (g * g)

    #bias correction: early averages are biased toward zero
    m_hat = state["m"] / (1 - beta1 ** step)
    v_hat = state["v"] / (1 - beta2 ** step)

    update = m_hat / (np.sqrt(v_hat) + eps)
    return x - lr * (update + weight_decay * x)


STEPPERS = {"sgd": sgd_step, "adamw": adamw_step}


def run(start, optimizer, lr, n_steps):
    """
    one full descent. returns the trajectory and how it ended.
    this is a single 'trial' -- the thing we will repeat many times below.
    """
    step_fn = STEPPERS[optimizer]
    x = np.array(start, dtype=float)
    state = {}
    path = [x.copy()]

    outcome = "ran out of steps"
    steps_taken = n_steps

    for step in range(1, n_steps + 1):
        x = step_fn(x, grad(x), lr, state, step)
        path.append(x.copy())

        if np.linalg.norm(x) > DIVERGED or not np.all(np.isfinite(x)):
            outcome = "DIVERGED"
            steps_taken = step
            break
        if np.linalg.norm(x) < TOLERANCE:
            outcome = "converged"
            steps_taken = step
            break

    return np.array(path), outcome, steps_taken


# %%
# ---------------------------------------------------------------------------
# RUN 1: a single descent, watched step by step.
# the loss falling toward 0 is the goal being arrived at over iterations.
# ---------------------------------------------------------------------------
path, outcome, steps_taken = run(START, OPTIMIZER, LEARNING_RATE, N_STEPS)

print(f"optimizer={OPTIMIZER}  lr={LEARNING_RATE}")
print(f"start={START}  ->  {outcome} after {steps_taken} steps")
for s in range(0, len(path), max(1, len(path) // 10)):
    print(f"  step {s:3d}: x={np.round(path[s], 3)}  loss={loss(path[s]):.4f}")

#draw the trajectory rolling across the loss surface toward the goal (*)
gx = np.linspace(-5, 5, 200)
gy = np.linspace(-5, 5, 200)
GX, GY = np.meshgrid(gx, gy)
GZ = 0.5 * (CURVATURE[0] * GX**2 + CURVATURE[1] * GY**2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.contour(GX, GY, GZ, levels=30, cmap="viridis", alpha=0.6)
ax1.plot(path[:, 0], path[:, 1], "o-", color="crimson", ms=3, lw=1, label="path")
ax1.plot(0, 0, "*", color="gold", ms=20, mec="black", label="goal (minimum)")
ax1.plot(*START, "s", color="black", ms=8, label="start")
ax1.set_title(f"{OPTIMIZER}, lr={LEARNING_RATE}: descent on the loss surface")
ax1.legend(); ax1.set_xlabel("x0 (gentle)"); ax1.set_ylabel("x1 (steep)")

ax2.plot([loss(p) for p in path], "o-", color="crimson", ms=3)
ax2.set_title("loss per iteration  (the goal = 0)")
ax2.set_xlabel("step"); ax2.set_ylabel("loss"); ax2.set_yscale("symlog")
plt.tight_layout(); plt.show()


# %%
# ---------------------------------------------------------------------------
# RUN MANY: the dice-roll. repeat the descent from many random starts and let a
# MEANINGFUL distribution emerge -- how many steps to reach the goal, and how
# often this learning rate diverges instead of converging.
# ---------------------------------------------------------------------------
N_TRIALS = 500
rng = np.random.default_rng(SEED)

steps_to_converge = []
n_diverged = 0
n_unfinished = 0

for _ in range(N_TRIALS):
    start = rng.uniform(-4.5, 4.5, size=2)
    _, outcome, steps_taken = run(start, OPTIMIZER, LEARNING_RATE, N_STEPS)
    if outcome == "converged":
        steps_to_converge.append(steps_taken)
    elif outcome == "DIVERGED":
        n_diverged += 1
    else:
        n_unfinished += 1

print(f"\n{N_TRIALS} trials  |  optimizer={OPTIMIZER}  lr={LEARNING_RATE}")
print(f"  converged   : {len(steps_to_converge)}")
print(f"  diverged    : {n_diverged}")
print(f"  not finished: {n_unfinished}  (needed more than {N_STEPS} steps)")

fig, ax = plt.subplots(figsize=(8, 4.5))
if steps_to_converge:
    ax.hist(steps_to_converge, bins=range(0, N_STEPS + 2), color="steelblue",
            edgecolor="black")
ax.set_title(f"steps to reach the goal over {N_TRIALS} random starts\n"
             f"({n_diverged} diverged, {n_unfinished} didn't finish)")
ax.set_xlabel("steps until converged"); ax.set_ylabel("number of runs")
plt.tight_layout(); plt.show()

#the takeaway the histogram MEANS: at a good lr almost every run lands in the goal
#in a tight band of steps. crank LEARNING_RATE up and re-run -- the converged bar
#empties out and 'diverged' takes over. that shift IS the lesson.
