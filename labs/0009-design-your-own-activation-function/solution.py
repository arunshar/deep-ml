import numpy as np

def activation(x):
    """
    Mish activation: x * tanh(softplus(x))
    Stable softplus: max(x, 0) + log1p(exp(-abs(x)))
    """
    x = np.asarray(x)

    # Numerically stable softplus
    softplus = np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))

    # Mish
    return x * np.tanh(softplus)
