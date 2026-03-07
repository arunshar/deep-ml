import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:

    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    # compute fold sizes
    fold_sizes = [n_samples // k] * k
    for i in range(n_samples % k):
        fold_sizes[i] += 1

    folds = []
    current = 0

    for fold_size in fold_sizes:
        start = current
        end = current + fold_size

        test_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))

        folds.append((train_indices.tolist(), test_indices.tolist()))

        current = end

    return folds