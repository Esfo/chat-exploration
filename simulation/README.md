# Training Stage Simulations

Standalone, runnable breakdowns of every relevant training stage in
[`../training.py`](../training.py). Each script isolates one stage, walks through
its math, prints the **shapes** of every intermediate (where each came from, why
it has that shape, and where it goes next), and saves a relevant visualization to
`visualizations/`.

Tokenization and text-processing are intentionally **excluded** (as requested) —
these focus purely on the neural-network learning math.

## The stages

| # | Simulation | Source in `training.py` | What it teaches |
|---|------------|-------------------------|-----------------|
| 01 | `01_weight_initialization.py` | `init_params()` | Why weights start random and scaled by `1/sqrt(fan_in)` |
| 02 | `02_embeddings.py` | top of `forward()` | Token + position lookup tables; how IDs become vectors |
| 03 | `03_softmax.py` | `softmax()` | Scores → probabilities, and the max-subtraction stability trick |
| 04 | `04_attention.py` | `attention_forward()` | Q/K/V, heads, scaled scores, causal mask, weighted values |
| 05 | `05_mlp_feedforward.py` | MLP block in `forward()` | Expand → ReLU → shrink, per-token processing |
| 06 | `06_residual_connections.py` | `h = h + ...` lines | The gradient highway; vanishing-gradient demo |
| 07 | `07_cross_entropy_loss.py` | `cross_entropy()` | The loss being minimized and its clean `probs - onehot` gradient |
| 08 | `08_backpropagation.py` | `backward()` | **The full backprop chain**, verified by a numerical gradient check |
| 09 | `09_attention_backward.py` | `attention_backward()` | The softmax-Jacobian line and attention's backward shapes |
| 10 | `10_gradient_descent_adamw.py` | `adamw_update()` | **Gradient descent**, momentum, per-weight scaling, weight decay |
| 11 | `11_training_loop.py` | `train()` | Capstone: wires it all together and trains a tiny transformer |

Stages 08 (backpropagation) and 10 (gradient descent) are the two the request
called out specifically and are the most heavily annotated. Stages 08, 09 and 11
build a small but **real** transformer and prove the hand-derived gradients are
correct against finite differences.

## Running

Each file is self-contained (only `numpy` + `matplotlib`):

```bash
pip install numpy matplotlib
python3 08_backpropagation.py        # run one stage
python3 run_all.py                   # run every stage in order
```

Every script prints an annotated trace to stdout and writes a `.png` into
`visualizations/`.
