import numpy as np

def route(x, W_gate):
    """
    Route inputs to experts in a Mixture of Experts layer.
    """

    # Linear gate
    logits = x @ W_gate  # (batch_size, num_experts)

    # Stable softmax
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    weights = exp / np.sum(exp, axis=1, keepdims=True)

    return weights