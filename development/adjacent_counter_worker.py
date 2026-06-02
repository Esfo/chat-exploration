
from collections import Counter

def count_adjacent_batch(args):
    texts, minsize, maxsize = args

    counts = Counter()

    for text in texts:
        for size in range(minsize, maxsize + 1):
            counts.update(
                text[start:start + size]
                for start in range(0, len(text) - size + 1)
            )

    return counts
