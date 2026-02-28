import numpy as np

def softmax_derivative(x: list[float]) -> list[list[float]]:
    # 1. Compute Softmax: s_i = exp(x_i) / sum(exp(x_j))
    # We subtract np.max(x) for numerical stability (prevents overflow)
    x = np.array(x)
    exp_x = np.exp(x - np.max(x))
    s = exp_x / np.sum(exp_x)
    
    # 2. Compute Jacobian matrix
    # Initialize an empty matrix of size n x n
    n = len(s)
    jacobian = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal: s_i * (1 - s_i)
                jacobian[i][j] = s[i] * (1 - s[i])
            else:
                # Off-diagonal: -s_i * s_j
                jacobian[i][j] = -s[i] * s[j]
                
    return jacobian.tolist()