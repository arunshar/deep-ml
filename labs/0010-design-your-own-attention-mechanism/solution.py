import numpy as np

def attention(Q, K, V):
    """
    Scaled dot-product attention with stable softmax.

    Args:
        Q: (batch_size, query_len, dim)
        K: (batch_size, key_len, dim)
        V: (batch_size, key_len, dim)

    Returns:
        output: (batch_size, query_len, dim)
    """
    # Dimensions
    dim = Q.shape[-1]

    # 1. Compute scaled dot-product scores
    # (B, Q_len, D) @ (B, D, K_len) -> (B, Q_len, K_len)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(dim)

    # 2. Stable softmax along key dimension
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 3. Weighted sum of values
    # (B, Q_len, K_len) @ (B, K_len, D) -> (B, Q_len, D)
    output = np.matmul(weights, V)

    return output
