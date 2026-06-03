"""
shared kit for the training simulations.

every simulation in this directory is meant to be a faithful, smaller-scale
re-implementation of the matching piece of ../training.py. the math here must
match training.py exactly; only the dimensions are shrunk so the numbers are
small enough to see.

this module holds the pieces more than one simulation needs:
    - softmax (copied verbatim from training.py so a sim can never drift from it)
    - heatmap / matrix rendering helpers built on matplotlib

nothing here is interactive on its own. the simulations import it.
"""

import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    """
    convert raw scores aka logits into probabilities.

    this is copied verbatim from training.py:softmax so the simulations
    reuse the exact behaviour, including the max-subtraction stability trick.
    """

    #subtract the largest score for numerical stability
    #this does not change the final probabilities
    #it prevents np.exp() from overflowing on very large numbers
    x = x - np.max(x, axis=-1, keepdims=True)

    #turn scores into positive numbers
    exp_x = np.exp(x)

    #divide each exp score by the sum so the values are probabilities summing to 1
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def show_matrix(matrix, title="", row_labels=None, col_labels=None,
                ax=None, cmap="viridis", annotate=True, fmt="{:.2f}"):
    """
    draw a 2D array as a heatmap with optional cell numbers and axis labels.

    this is the workhorse renderer the simulations lean on: attention scores,
    attention weights, projections, masks, etc. are all just 2D arrays.
    """

    matrix = np.asarray(matrix)
    assert matrix.ndim == 2, "show_matrix expects a 2D array"

    own_figure = ax is None
    if own_figure:
        _, ax = plt.subplots(figsize=(0.9 * matrix.shape[1] + 2,
                                       0.7 * matrix.shape[0] + 1.5))

    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    ax.set_title(title)

    if col_labels is not None:
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
    if row_labels is not None:
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_yticklabels(row_labels)

    #write the value into each cell so the sim is readable as numbers, not just colour
    if annotate:
        #pick black or white text per cell based on background brightness
        threshold = (matrix.max() + matrix.min()) / 2.0
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                value = matrix[r, c]
                colour = "white" if value < threshold else "black"
                ax.text(c, r, fmt.format(value), ha="center", va="center",
                        color=colour, fontsize=8)

    if own_figure:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

    return ax
