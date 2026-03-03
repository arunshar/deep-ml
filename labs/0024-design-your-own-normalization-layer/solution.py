import numpy as np

def normalize(x):
    x = np.asarray(x)
    eps = 1e-5
    mu = np.mean(x, axis=1, keepdims=True)
    var = np.mean((x - mu) ** 2, axis=1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)