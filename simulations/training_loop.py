import numpy as np
import matplotlib.pyplot as plt

text = "banana"           #the basic example the model must learn to predict
learning_rate = 0.5       #how hard each round nudges the weights
hidden_size = 8           #width of the middle layer the error backpropagates through
max_rounds = 10000        #safety cap so the loop can't run forever

chars = sorted(set(text))
char_id = {c: i for i, c in enumerate(chars)}
ids = np.array([char_id[c] for c in text])
vocab = len(chars)

current = np.eye(vocab)[ids[:-1]]  #each character the model sees, as a one-hot row
following = ids[1:]                #the character id that should come next

rng = np.random.default_rng(0)
w1 = rng.normal(0, 1 / np.sqrt(vocab), (vocab, hidden_size))        #input -> hidden weights
w2 = rng.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, vocab))  #hidden -> output weights
losses = []
rounds = 0

while True:
    #forward: input through the hidden layer, then out to next-character scores
    pre = current @ w1
    hidden = np.maximum(pre, 0)                         #relu: keep positives, zero the rest
    logits = hidden @ w2
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)    #softmax turns scores into probabilities

    loss = -np.log(probs[np.arange(len(following)), following] + 1e-12).mean()  #how surprised it is
    losses.append(loss)

    #backward: send the error back through each layer, one step at a time
    d_logits = probs.copy()
    d_logits[np.arange(len(following)), following] -= 1  #cross-entropy gradient, as in training.py
    d_logits /= len(following)
    d_w2 = hidden.T @ d_logits                           #blame the output weights
    d_hidden = d_logits @ w2.T                           #pass the blame back to the hidden layer
    d_pre = d_hidden * (pre > 0)                         #relu only passes blame where it was active
    d_w1 = current.T @ d_pre                             #blame the input weights

    #nudge every weight against its gradient
    w2 -= learning_rate * d_w2
    w1 -= learning_rate * d_w1

    rounds += 1
    guesses = probs.argmax(axis=1)                       #the model's best guess for each next character
    if (guesses == following).all():                     #stop once every guess is correct
        break
    if rounds >= max_rounds:                             #or give up if it never gets there
        break

solved = bool((guesses == following).all())

#let the trained model write the text out, one character at a time
c = char_id[text[0]]
generated = [chars[c]]
for _ in range(len(text) - 1):
    hidden = np.maximum(np.eye(vocab)[c] @ w1, 0)
    c = int((hidden @ w2).argmax())  #the next character feeds back in to pick the one after it
    generated.append(chars[c])

print("got it right" if solved else "gave up", "after", rounds, "rounds")
print("generated:", "".join(generated))

# %%
plt.figure(figsize=(7, 4.5))
plt.plot(losses, color="crimson")
plt.xlabel("round")
plt.ylabel("loss")
plt.title(f"learned '{text}' in {rounds} rounds")
plt.tight_layout()
plt.show()
