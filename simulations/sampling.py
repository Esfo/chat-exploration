import numpy as np
import matplotlib.pyplot as plt

temperature = 1.0          #<1 sharpens toward the top word, >1 flattens toward random
samples = 5000             #how many times to draw a next token (the dice rolls)

words = ["the", "cat", "sat", "on", "mat"]    #the candidate next tokens
logits = np.array([3.0, 1.0, 0.5, 0.2, 2.5])  #raw scores the model gives each one

scaled = logits / temperature          #temperature reshapes the scores before softmax
scaled = scaled - scaled.max()          #subtract the max for stability, as in training.py
probs = np.exp(scaled) / np.exp(scaled).sum()  #softmax: scores become probabilities

rng = np.random.default_rng(0)
draws = rng.choice(len(words), size=samples, p=probs)  #sample a next token, over and over
counts = np.bincount(draws, minlength=len(words))

for i, word in enumerate(words):
    print(f"{word:>4}: prob {probs[i]:.3f}   drawn {counts[i]} times")
print("picked:", words[draws[0]])  #one draw is the token the model actually emits

# %%
fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 9))
top.bar(words, probs, color="steelblue")
top.set_ylabel("probability")
top.set_title(f"temperature = {temperature}")
bottom.bar(words, counts / samples, color="crimson")
bottom.set_ylabel(f"fraction drawn over {samples} samples")
plt.tight_layout()
plt.show()
