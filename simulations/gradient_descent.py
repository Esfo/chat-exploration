import numpy as np
import matplotlib.pyplot as plt

method = "plain"           #"plain" steps straight downhill; "adam" adapts the step per weight
learning_rate = 0.05       #step size: too big and it blows up, too small and it crawls
start_weights = [4.5, 3.0] #the model's two weights before this step of training
sensitivity = [1.0, 12.0]  #how hard the loss reacts to each weight (uneven on purpose)
max_steps = 80             #how many update steps to take

weights = np.array(start_weights, dtype=float)
sensitivity = np.array(sensitivity, dtype=float)

m = np.zeros(2)            #adam's running average of the gradient
v = np.zeros(2)            #adam's running average of the squared gradient
path = [weights.copy()]

for step in range(1, max_steps + 1):
    gradient = sensitivity * weights  #in a real model backprop produces this
    if method == "plain":
        weights = weights - learning_rate * gradient
    else:
        m = 0.9 * m + 0.1 * gradient
        v = 0.999 * v + 0.001 * gradient ** 2
        m = m / (1 - 0.9 ** step)     #the averages start at zero, so scale them up early
        v = v / (1 - 0.999 ** step)
        weights = weights - learning_rate * m / (np.sqrt(v) + 1e-8)
    path.append(weights.copy())

path = np.array(path)
loss = 0.5 * (sensitivity * path ** 2).sum(axis=1)  #how wrong the model is at each step

print("weights in :", path[0])
print("weights out:", path[-1].round(4))  #these get written back into the model
print("loss       :", loss[0].round(4), "->", loss[-1].round(4))

# %%
grid = np.linspace(-5, 5, 200)
w1, w2 = np.meshgrid(grid, grid)
surface = 0.5 * (sensitivity[0] * w1 ** 2 + sensitivity[1] * w2 ** 2)

fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 11))
top.contour(w1, w2, surface, levels=30, cmap="viridis", alpha=0.6)
top.plot(path[:, 0], path[:, 1], "o-", color="crimson", ms=3, lw=1)
top.plot(0, 0, "*", color="gold", ms=20, mec="black")  #lowest loss
top.set_xlabel("weight 1")
top.set_ylabel("weight 2")
top.set_title(f"{method}, learning_rate={learning_rate}")

bottom.plot(loss, "o-", color="crimson", ms=3)
bottom.set_xlabel("step")
bottom.set_ylabel("loss")

plt.tight_layout()
plt.show()
