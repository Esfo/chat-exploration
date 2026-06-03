# rolling downhill: how a model learns, in slow motion.
#
# a model "learns" by changing its numbers a tiny bit at a time to make its
# mistakes smaller. here we shrink that down to a single point trying to find
# the bottom of a valley. you can watch it roll, and play with how big its
# steps are.

# %%
# the valley we're rolling down
# -----------------------------
# picture a bowl. the bottom (0, 0) is the goal: that's where mistakes are
# smallest. this bowl is lopsided on purpose: much steeper one way than the
# other. that lopsidedness is what makes the simple method struggle and the
# smarter method worth having.
import numpy as np
import matplotlib.pyplot as plt

steepness = np.array([1.0, 12.0])   # [gentle direction, steep direction]

# how big a step to take each time. this is the main thing to play with:
# too small and it crawls, too big and it overshoots and flies off.
step_size = 0.05

# "simple" steps straight downhill. "smarter" is the method from training.py
# (called adam) that adjusts its step in each direction as it goes.
method = "simple"

# where the watched roll starts, and the most steps we'll let it take
start = np.array([4.5, 3.0])
step_limit = 80

# how close to the bottom counts as "made it", and how far out counts as "flew off"
made_it_distance = 0.01
flew_off_distance = 1000.0


# %%
# roll down once, watching every step
# -----------------------------------
position = start.astype(float)
trail = [position.copy()]

# the smarter method keeps two memories as it goes (both start empty and stay
# unused for the simple method):
recent_direction = np.zeros(2)   # the general downhill direction lately
recent_steepness = np.zeros(2)   # how steep each direction has been lately

print(f"method = {method},  step_size = {step_size}")
for step in range(1, step_limit + 1):
    # which way is uphill right now, and how steeply
    uphill = steepness * position

    if method == "simple":
        # move straight downhill, scaled by the step size
        position = position - step_size * uphill
    else:
        # fold this step's uphill into the running memories
        recent_direction = 0.9 * recent_direction + 0.1 * uphill
        recent_steepness = 0.999 * recent_steepness + 0.001 * uphill**2
        # those memories start out too small, so nudge them back up early on
        direction = recent_direction / (1 - 0.9**step)
        size = recent_steepness / (1 - 0.999**step)
        # step downhill, but shrink the step in directions that are steep and
        # grow it in directions that are gentle. that evens out the lopsided bowl.
        position = position - step_size * direction / (np.sqrt(size) + 1e-8)

    trail.append(position.copy())

    # how high up the valley wall we still are (0 at the bottom)
    height = 0.5 * np.sum(steepness * position**2)
    if step == 1 or step % 8 == 0:
        print(f"  step {step:3d}: at {np.round(position, 3)}, height {height:.4f}")

    distance_to_bottom = np.sqrt(np.sum(position**2))
    if distance_to_bottom < made_it_distance:
        print(f"  made it to the bottom in {step} steps")
        break
    if distance_to_bottom > flew_off_distance:
        print(f"  flew off after {step} steps (step_size is too big)")
        break

trail = np.array(trail)


# %%
# draw the path it took, and how its height fell
# ----------------------------------------------
spread = np.linspace(-5, 5, 200)
across, along = np.meshgrid(spread, spread)
valley_height = 0.5 * (steepness[0] * across**2 + steepness[1] * along**2)

figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5))

left.contour(across, along, valley_height, levels=30, cmap="viridis", alpha=0.6)
left.plot(trail[:, 0], trail[:, 1], "o-", color="crimson", ms=3, lw=1, label="path it took")
left.plot(0, 0, "*", color="gold", ms=20, mec="black", label="the bottom (goal)")
left.plot(*start, "s", color="black", ms=8, label="started here")
left.set_title(f"{method} steps, step_size = {step_size}")
left.legend()
left.set_xlabel("gentle direction")
left.set_ylabel("steep direction")

right.plot([0.5 * np.sum(steepness * spot**2) for spot in trail], "o-", color="crimson", ms=3)
right.set_title("how far from the goal, step by step")
right.set_xlabel("step")
right.set_ylabel("height above the bottom")
right.set_yscale("symlog")

plt.tight_layout()
plt.show()


# %%
# now do it from 500 random starting spots
# ----------------------------------------
# one roll is just one story. drop the point from hundreds of random places and
# a real pattern shows up: how many steps it usually needs to reach the bottom,
# and how often this step_size just makes it fly off instead. that pattern is
# the actual lesson, the same way rolling dice many times shows you the odds.
random_starts = np.random.default_rng(1).uniform(-4.5, 4.5, size=(500, 2))

steps_until_made_it = []
flew_off_count = 0
too_slow_count = 0

for spot in random_starts:
    position = spot.copy()
    recent_direction = np.zeros(2)
    recent_steepness = np.zeros(2)
    result = "too slow"

    for step in range(1, step_limit + 1):
        uphill = steepness * position
        if method == "simple":
            position = position - step_size * uphill
        else:
            recent_direction = 0.9 * recent_direction + 0.1 * uphill
            recent_steepness = 0.999 * recent_steepness + 0.001 * uphill**2
            direction = recent_direction / (1 - 0.9**step)
            size = recent_steepness / (1 - 0.999**step)
            position = position - step_size * direction / (np.sqrt(size) + 1e-8)

        distance_to_bottom = np.sqrt(np.sum(position**2))
        if distance_to_bottom > flew_off_distance:
            result = "flew off"
            break
        if distance_to_bottom < made_it_distance:
            result = "made it"
            steps_until_made_it.append(step)
            break

    if result == "flew off":
        flew_off_count += 1
    elif result == "too slow":
        too_slow_count += 1

print(f"\nout of 500 rolls (method = {method}, step_size = {step_size}):")
print(f"  reached the bottom : {len(steps_until_made_it)}")
print(f"  flew off           : {flew_off_count}")
print(f"  too slow to finish : {too_slow_count}  (needed more than {step_limit} steps)")

figure, axis = plt.subplots(figsize=(8, 4.5))
if steps_until_made_it:
    axis.hist(steps_until_made_it, bins=range(0, step_limit + 2), color="steelblue", edgecolor="black")
axis.set_title(f"how many steps it took to reach the bottom\n"
               f"({flew_off_count} flew off, {too_slow_count} were too slow)")
axis.set_xlabel("steps needed")
axis.set_ylabel("number of rolls")
plt.tight_layout()
plt.show()

# try this: bump step_size up to 0.18 with method = "simple" and almost every
# roll flies off. switch to method = "smarter" and it can take much bigger
# steps without flying off. that difference is why real training uses adam.
