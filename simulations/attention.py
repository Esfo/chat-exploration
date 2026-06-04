import numpy as np
import matplotlib.pyplot as plt

#attention as a soft dictionary lookup: each item carries a key and a value, a
#query asks for one key, and the model must learn to look at the matching item
#and read back its value. this is the mechanism at the heart of training.py.

n = 5                 #number of items in each sequence
d = 16                #size of the query/key space the model projects into
learning_rate = 0.1
max_rounds = 20000
batch_size = 32

rng = np.random.default_rng(0)

def make_example():
    keys = rng.permutation(n)            #every item gets a distinct key id (0..n-1)
    values = rng.integers(0, n, n)       #...and some value id (may repeat)
    ask = rng.integers(0, n)             #we will ask for the key held at this position
    x = np.zeros((n, 2 * n))
    x[np.arange(n), keys] = 1            #first n columns: the key, one-hot
    x[np.arange(n), n + values] = 1      #next n columns: the value, one-hot
    query = np.zeros(2 * n)
    query[keys[ask]] = 1                 #the query is just the key we're asking for
    return x, query, values[ask], ask    #target is the value at the matching item

#the only trainable weights: how to project items into keys, the query into a
#query vector, and items into the values that get read out
wk = rng.normal(0, 1 / np.sqrt(2 * n), (2 * n, d))
wq = rng.normal(0, 1 / np.sqrt(2 * n), (2 * n, d))
wv = rng.normal(0, 1 / np.sqrt(2 * n), (2 * n, n))

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

losses, accuracies = [], []
rounds = 0
streak = 0      #how many rounds in a row it has gotten a whole batch right

while True:
    g_wk = np.zeros_like(wk)
    g_wq = np.zeros_like(wq)
    g_wv = np.zeros_like(wv)
    batch_loss = 0.0
    batch_correct = 0

    for _ in range(batch_size):
        x, query, target, _ = make_example()

        #forward: project, score every item against the query, soften, read values
        k = x @ wk                       #(n, d) a key vector per item
        q = query @ wq                   #(d,)   the query vector
        scores = k @ q / np.sqrt(d)      #(n,)   how well each item matches the query
        attn = softmax(scores)           #(n,)   attention: where the query looks
        v = x @ wv                       #(n, n) a value vector per item
        out = attn @ v                   #(n,)   the read-out, used as logits
        probs = softmax(out)

        batch_loss += -np.log(probs[target] + 1e-12)
        batch_correct += int(probs.argmax() == target)

        #backward: send the error back through read-out, attention, and projections
        d_out = probs.copy()
        d_out[target] -= 1               #cross-entropy gradient on the logits
        d_attn = v @ d_out               #(n,) blame flowing to the attention weights
        g_wv += x.T @ np.outer(attn, d_out)
        d_scores = attn * (d_attn - d_attn @ attn)   #softmax derivative for attention
        d_scores /= np.sqrt(d)
        d_k = np.outer(d_scores, q)      #(n, d)
        d_q = k.T @ d_scores             #(d,)
        g_wk += x.T @ d_k
        g_wq += np.outer(query, d_q)

    wk -= learning_rate * g_wk / batch_size
    wq -= learning_rate * g_wq / batch_size
    wv -= learning_rate * g_wv / batch_size

    losses.append(batch_loss / batch_size)
    accuracies.append(batch_correct / batch_size)
    rounds += 1
    streak = streak + 1 if accuracies[-1] == 1.0 else 0
    if streak >= 500:        #keep going past the first success to sharpen the attention
        break
    if rounds >= max_rounds:
        break

print("solved" if streak >= 500 else "gave up", "after", rounds, "rounds")

#show where the trained model looks on one fresh example
x, query, target, ask = make_example()
attn = softmax((x @ wk) @ (query @ wq) / np.sqrt(d))
print(f"asked for the key at item {ask}; attention there = {attn[ask]:.2f}")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(losses, color="crimson", label="loss")
ax1.plot(accuracies, color="green", label="accuracy")
ax1.set_xlabel("round")
ax1.set_title(f"learned to look up the right item in {rounds} rounds")
ax1.legend()

ax2.bar(range(n), attn, color=["green" if i == ask else "steelblue" for i in range(n)])
ax2.set_xlabel("item position")
ax2.set_ylabel("attention weight")
ax2.set_title(f"the query asked for item {ask}; attention spikes there")
plt.tight_layout()
plt.show()
