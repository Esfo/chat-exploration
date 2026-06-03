#one step of training, in slow motion: take the model's weights and the
#gradient backprop produced, and nudge the weights to make the loss smaller.

# %%
#===simulated input (what the training loop hands this step)===
import numpy as np
import matplotlib.pyplot as plt

#weights: the model's trainable numbers (p in training.py). a real model has
#millions; we use two so we can actually watch them move.
weights = np.array([4.5, 3.0])

#sensitivity: how much the loss reacts to each weight. lopsided on purpose,
#the loss cares 12x more about the second weight than the first. real losses
#are uneven like this, and it's what makes the plain method struggle.
sensitivity = np.array([1.0, 12.0])

#loss(weights): how wrong the model currently is. lower is better, 0 is perfect.
#in training.py this is the cross-entropy; here it's a simple bowl so we can see it.
def loss_of(w):
    return 0.5 * np.sum(sensitivity * w**2)

#learning_rate (lr in training.py): how big each nudge is. the main knob.
learning_rate = 0.05

#"plain" nudges straight downhill. "adam" is training.py's adamw_update.
method = "plain"

#how many steps we allow, and what counts as "done" / "blown up"
max_steps = 80
done_when_closer_than = 0.01
blown_up_when_further_than = 1000.0


# %%
#===the update, repeated step by step===

#m and v are adam's memory (m and v in training.py). they stay unused for "plain".
m = np.zeros(2)  #running average of the gradient
v = np.zeros(2)  #running average of the squared gradient

path = [weights.copy()]
print(f"method={method}  learning_rate={learning_rate}  starting loss={loss_of(weights):.3f}")

for step in range(1, max_steps + 1):
    #gradient: which way increases the loss, and how steeply. backprop produces
    #this in training.py; our loss is simple enough to differentiate directly.
    gradient = sensitivity * weights

    if method == "plain":
        #step straight downhill
        weights = weights - learning_rate * gradient
    else:
        #blend this gradient into the running averages
        m = 0.9 * m + 0.1 * gradient
        v = 0.999 * v + 0.001 * gradient**2
        #the averages start near zero, so scale them up early on (bias correction)
        m_corrected = m / (1 - 0.9**step)
        v_corrected = v / (1 - 0.999**step)
        #big steps where the loss is gentle, small steps where it's steep
        weights = weights - learning_rate * m_corrected / (np.sqrt(v_corrected) + 1e-8)

    path.append(weights.copy())

    #stop early if we've basically arrived, or if the step size made it explode
    distance = np.sqrt(np.sum(weights**2))
    if step == 1 or step % 8 == 0:
        print(f"  step {step:3d}: weights={np.round(weights, 3)}  loss={loss_of(weights):.4f}")
    if distance < done_when_closer_than:
        print(f"  reached the minimum in {step} steps")
        break
    if distance > blown_up_when_further_than:
        print(f"  blew up after {step} steps (learning_rate too large)")
        break

path = np.array(path)


# %%
#===see what happened (plots stacked top to bottom)===

#a grid of the loss surface so we can draw the weights' path across it
grid = np.linspace(-5, 5, 200)
weight1, weight2 = np.meshgrid(grid, grid)
loss_surface = 0.5 * (sensitivity[0] * weight1**2 + sensitivity[1] * weight2**2)

figure, (top, bottom) = plt.subplots(2, 1, figsize=(7, 11))

#top: the path the weights took toward the lowest-loss point
top.contour(weight1, weight2, loss_surface, levels=30, cmap="viridis", alpha=0.6)
top.plot(path[:, 0], path[:, 1], "o-", color="crimson", ms=3, lw=1, label="weights over time")
top.plot(0, 0, "*", color="gold", ms=20, mec="black", label="lowest loss")
top.plot(path[0, 0], path[0, 1], "s", color="black", ms=8, label="started here")
top.set_title(f"how the two weights moved ({method}, learning_rate={learning_rate})")
top.set_xlabel("weight 1")
top.set_ylabel("weight 2")
top.legend()

#bottom: the loss itself falling as the steps go on
bottom.plot([loss_of(w) for w in path], "o-", color="crimson", ms=3)
bottom.set_title("loss going down, step by step")
bottom.set_xlabel("step")
bottom.set_ylabel("loss")
bottom.set_yscale("symlog")

plt.tight_layout()
plt.show()


# %%
#===output (what gets handed back to the training loop)===
#these updated weights get written back into the model and used in the next
#forward pass. that hand-back is the entire job of this step.
print("trained weights:", np.round(path[-1], 4))
print("final loss     :", round(float(loss_of(path[-1])), 4))


# %%
#===zoom out: run it from 500 random starts===
#one run is one story. real training starts from random weights (init_params with
#different seeds), so try many and watch the pattern: how many steps it usually
#needs, and how often this learning_rate just blows up instead.
random_starts = np.random.default_rng(1).uniform(-4.5, 4.5, size=(500, 2))

steps_needed = []
blew_up = 0
too_slow = 0

for start in random_starts:
    weights = start.copy()
    m = np.zeros(2)
    v = np.zeros(2)
    outcome = "too slow"

    for step in range(1, max_steps + 1):
        gradient = sensitivity * weights
        if method == "plain":
            weights = weights - learning_rate * gradient
        else:
            m = 0.9 * m + 0.1 * gradient
            v = 0.999 * v + 0.001 * gradient**2
            weights = weights - learning_rate * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)

        distance = np.sqrt(np.sum(weights**2))
        if distance > blown_up_when_further_than:
            outcome = "blew up"
            break
        if distance < done_when_closer_than:
            outcome = "done"
            steps_needed.append(step)
            break

    if outcome == "blew up":
        blew_up += 1
    elif outcome == "too slow":
        too_slow += 1

print(f"out of 500 runs (method={method}, learning_rate={learning_rate}):")
print(f"  reached the minimum: {len(steps_needed)}")
print(f"  blew up            : {blew_up}")
print(f"  too slow to finish : {too_slow}  (needed more than {max_steps} steps)")

figure, axis = plt.subplots(figsize=(7, 4.5))
if steps_needed:
    axis.hist(steps_needed, bins=range(0, max_steps + 2), color="steelblue", edgecolor="black")
axis.set_title(f"steps needed to reach the minimum\n({blew_up} blew up, {too_slow} too slow)")
axis.set_xlabel("steps")
axis.set_ylabel("number of runs")
plt.tight_layout()
plt.show()

#try it: set learning_rate=0.18 with method="plain" and almost every run blows up.
#switch to method="adam" and it survives much bigger steps. that's why training
#uses adam.
