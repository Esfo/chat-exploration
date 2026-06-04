import numpy as np
import matplotlib.pyplot as plt

#a smooth underlying pattern the model should ideally discover
true_fn = lambda x: np.sin(2 * np.pi * x)

rng = np.random.default_rng(0)
train_x = np.linspace(0, 1, 9)                              #only a handful of training points...
train_y = true_fn(train_x) + rng.normal(0, 0.3, train_x.shape)  #...and they are noisy
test_x = np.linspace(0, 1, 200)                            #many clean points to judge generalisation
test_y = true_fn(test_x)

hidden_size = 60      #lots of capacity: more than enough to memorise 9 points
learning_rate = 0.02
rounds = 120000       #train for a long time so overfitting has room to set in

#one wide hidden layer with tanh, fitting a 1-D curve
w1 = rng.normal(0, 1, (1, hidden_size)) * 0.5
b1 = np.zeros(hidden_size)
w2 = rng.normal(0, 1, (hidden_size, 1)) * 0.5
b2 = 0.0

X = train_x[:, None]   #shape the inputs as a column so the matrix multiplies line up
Y = train_y[:, None]

def predict(x):
    hidden = np.tanh(x[:, None] @ w1 + b1)
    return (hidden @ w2 + b2)[:, 0]

train_losses, test_losses = [], []
best_test = np.inf
best_round = 0
best_weights = None

for r in range(rounds):
    #forward: input -> hidden layer -> single output value
    pre = X @ w1 + b1
    hidden = np.tanh(pre)
    out = hidden @ w2 + b2

    train_loss = ((out - Y) ** 2).mean()                #mean squared error on the training points
    test_loss = ((predict(test_x) - test_y) ** 2).mean()  #...and on unseen points
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    if test_loss < best_test:                           #remember the model that generalises best
        best_test, best_round = test_loss, r
        best_weights = (w1.copy(), b1.copy(), w2.copy(), b2)

    #backward: send the error back through the layer, written out
    d_out = 2 * (out - Y) / len(Y)
    d_w2 = hidden.T @ d_out
    d_b2 = d_out.sum()
    d_hidden = d_out @ w2.T
    d_pre = d_hidden * (1 - hidden ** 2)                #tanh derivative
    d_w1 = X.T @ d_pre
    d_b1 = d_pre.sum(axis=0)

    w2 -= learning_rate * d_w2
    b2 -= learning_rate * d_b2
    w1 -= learning_rate * d_w1
    b1 -= learning_rate * d_b1

print(f"training loss kept falling to {train_losses[-1]:.4f}")
print(f"test loss bottomed out at {best_test:.4f} (round {best_round}), then rose to {test_losses[-1]:.4f}")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(train_losses, label="training loss (points it sees)")
ax1.plot(test_losses, label="test loss (unseen points)")
ax1.axvline(best_round, color="gray", ls="--", label=f"best test (round {best_round})")
ax1.set_yscale("log")
ax1.set_xlabel("round")
ax1.set_ylabel("loss")
ax1.set_title("training loss falls forever, test loss turns back up")
ax1.legend()

#the overfit curve (final) versus the curve that generalised best (early stop)
w1f, b1f, w2f, b2f = w1, b1, w2, b2
overfit = np.tanh(test_x[:, None] @ w1f + b1f) @ w2f + b2f
w1, b1, w2, b2 = best_weights
best = predict(test_x)
w1, b1, w2, b2 = w1f, b1f, w2f, b2f

ax2.plot(test_x, test_y, color="black", label="true pattern")
ax2.plot(test_x, overfit[:, 0], color="crimson", label="final model (overfit)")
ax2.plot(test_x, best, color="green", label="best test model (early stop)")
ax2.scatter(train_x, train_y, color="black", zorder=5, label="noisy training points")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("the overfit model wiggles to hit the noise")
ax2.legend()

plt.tight_layout()
plt.show()
