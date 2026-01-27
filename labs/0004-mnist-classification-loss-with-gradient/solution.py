import numpy as np

def loss_function(
    preds: np.ndarray,
    target: np.ndarray,
    reduction: str = "mean",
    eps: float = 1e-12,
):
    """
    preds:  [N, C] probabilities (after softmax)
    target: [N]    class indices (int)
    reduction: 'mean' | 'sum' | 'none'
    Returns:
      loss:  scalar if reduction in {'mean','sum'} else shape (N,)
      grad:  same shape as preds
    """
    # Basic validation
    preds = np.asarray(preds)
    target = np.asarray(target)

    if preds.ndim != 2:
        raise ValueError(f"preds must be [N, C], got shape={preds.shape}")
    if target.ndim != 1:
        raise ValueError(f"target must be [N], got shape={target.shape}")

    N, C = preds.shape
    if target.shape[0] != N:
        raise ValueError(f"target length {target.shape[0]} must match preds batch size {N}")

    # Ensure integer indices
    if not np.issubdtype(target.dtype, np.integer):
        target = target.astype(np.int64)

    # Bounds check
    if target.min() < 0 or target.max() >= C:
        raise ValueError(f"target values must be in [0, {C-1}], got min={target.min()}, max={target.max()}")

    # Numerical stability: avoid log(0)
    p = np.clip(preds, eps, 1.0)

    idx = np.arange(N)
    p_true = p[idx, target]                 # (N,)
    per_example_loss = -np.log(p_true)      # (N,)

    # Reduction
    if reduction == "none":
        loss = per_example_loss             # (N,)
        scale = 1.0                         # grad matches per-example loss (no averaging)
    elif reduction == "sum":
        loss = np.sum(per_example_loss)     # scalar
        scale = 1.0                         # sum => no 1/N scaling
    elif reduction == "mean":
        loss = np.mean(per_example_loss)    # scalar
        scale = 1.0 / N                     # mean => include 1/N scaling
    else:
        raise ValueError("reduction must be one of: 'none', 'sum', 'mean'")

    # Gradient w.r.t. probabilities preds:
    # d/dp_true [-log(p_true)] = -1/p_true
    grad = np.zeros_like(preds, dtype=preds.dtype)
    grad[idx, target] = (-1.0 / p_true) * scale

    return loss, grad