import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    # Ensure float computations
    A = A.astype(float)
    b = b.astype(float)

    m = b.shape[0]
    x = np.zeros(m, dtype=float)

    for _ in range(n):
        x_old = x.copy()
        x_new = np.zeros_like(x_old)

        for i in range(m):
            # sum of A[i, j] * x_old[j] for j != i
            sigma = np.dot(A[i, :], x_old) - A[i, i] * x_old[i]
            x_new[i] = (b[i] - sigma) / A[i, i]

        # Round the entire intermediate solution to 4 decimals
        x = np.round(x_new, 4)

    return x.tolist()