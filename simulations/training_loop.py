import numpy as np
import matplotlib.pyplot as plt

text = "the cat sat on the mat "  #the tiny corpus the model learns to predict
learning_rate = 0.5               #how hard each round nudges the weights
rounds = 300                      #how many times we train over the text

chars = sorted(set(text))
char_id = {c: i for i, c in enumerate(chars)}
ids = np.array([char_id[c] for c in text])
vocab = len(chars)

current = ids[:-1]  #each character in the text...
following = ids[1:] #...paired with the character that actually comes next

weights = np.zeros((vocab, vocab))  #weights[c] are the model's scores for what follows c
losses = []

for _ in range(rounds):
    logits = weights[current]                          #score every possible next character
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)   #softmax turns scores into probabilities

    loss = -np.log(probs[np.arange(len(following)), following] + 1e-12).mean()  #how surprised it is
    losses.append(loss)

    grad = probs.copy()
    grad[np.arange(len(following)), following] -= 1     #cross-entropy gradient, as in training.py
    grad /= len(following)
    np.add.at(weights, current, -learning_rate * grad)  #push each seen character's scores downhill

#let the trained model write text by sampling one character at a time
rng = np.random.default_rng(0)
c = char_id[text[0]]
generated = [chars[c]]
for _ in range(40):
    p = np.exp(weights[c] - weights[c].max())
    p = p / p.sum()
    c = rng.choice(vocab, p=p)  #the next character feeds back in to pick the one after it
    generated.append(chars[c])

print("loss     :", round(losses[0], 3), "->", round(losses[-1], 3))
print("generated:", "".join(generated))

# %%
plt.figure(figsize=(7, 4.5))
plt.plot(losses, color="crimson")
plt.xlabel("round")
plt.ylabel("loss")
plt.title(f"learning_rate={learning_rate}")
plt.tight_layout()
plt.show()
