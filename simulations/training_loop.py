import numpy as np
import matplotlib.pyplot as plt

text = "banana"           #the basic example the model must learn to predict
learning_rate = 0.5       #how hard each round nudges the weights
max_rounds = 10000        #safety cap so the loop can't run forever

chars = sorted(set(text))
char_id = {c: i for i, c in enumerate(chars)}
ids = np.array([char_id[c] for c in text])
vocab = len(chars)

current = ids[:-1]   #each character in the text...
following = ids[1:]  #...paired with the character that actually comes next

weights = np.zeros((vocab, vocab))  #weights[c] are the model's scores for what follows c
losses = []
rounds = 0

while True:
    logits = weights[current]                          #score every possible next character
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)   #softmax turns scores into probabilities

    loss = -np.log(probs[np.arange(len(following)), following] + 1e-12).mean()  #how surprised it is
    losses.append(loss)

    grad = probs.copy()
    grad[np.arange(len(following)), following] -= 1     #cross-entropy gradient, as in training.py
    grad /= len(following)
    np.add.at(weights, current, -learning_rate * grad)  #backpropagate the error into the weights

    rounds += 1
    guesses = probs.argmax(axis=1)                      #the model's best guess for each next character
    if (guesses == following).all():                   #stop once every guess is correct
        break
    if rounds >= max_rounds:                            #or give up if it never gets there
        break

solved = bool((guesses == following).all())

#let the trained model write the text out, one character at a time
c = char_id[text[0]]
generated = [chars[c]]
for _ in range(len(text) - 1):
    c = int(weights[c].argmax())  #the next character feeds back in to pick the one after it
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
