import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from training import Config, init_params, forward, backward, adamw_update  #the real model

text = "banana"        #the basic example the model must learn to predict
learning_rate = 0.05   #how hard each round nudges the weights
max_rounds = 5000      #safety cap so the loop can't run forever

chars = sorted(set(text))
char_id = {c: i for i, c in enumerate(chars)}
ids = np.array([char_id[c] for c in text])

x = ids[:-1][None, :]  #each character the model sees (one batch row)
y = ids[1:][None, :]   #the character that should come next

cfg = Config(
    context_length=x.shape[1],
    d_model=16,
    n_layers=1,
    head_dim=4,
    mlp_multiplier=2.0,
    tokenpath="", textsource="", batch_size=1,  #unused here; we feed x/y directly
    learning_rate=learning_rate, train_steps=0, log_every=0,
)
cfg.vocab_size = len(chars)

weights = init_params(cfg)  #the model's trainable weights, randomly initialised
state = {}                  #adamw's memory across rounds
losses = []
rounds = 0

while True:
    logits, h, caches = forward(x, weights, cfg)                 #predict the next character
    loss, grads = backward(x, y, logits, h, caches, weights, cfg)  #backpropagate the error
    rounds += 1
    adamw_update(weights, grads, state, lr=learning_rate, step=rounds)  #nudge every weight

    losses.append(loss)
    guesses = logits.argmax(axis=-1)   #the model's best guess for each next character
    if (guesses == y).all():           #stop once every guess is correct
        break
    if rounds >= max_rounds:            #or give up if it never gets there
        break

solved = bool((guesses == y).all())
generated = text[0] + "".join(chars[i] for i in guesses[0])

print("got it right" if solved else "gave up", "after", rounds, "rounds")
print("generated:", generated)

# %%
plt.figure(figsize=(7, 4.5))
plt.plot(losses, color="crimson")
plt.xlabel("round")
plt.ylabel("loss")
plt.title(f"learned '{text}' in {rounds} rounds")
plt.tight_layout()
plt.show()
