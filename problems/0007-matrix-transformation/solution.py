import numpy as np

def transform_matrix(
    A: list[list[int | float]],
    T: list[list[int | float]],
    S: list[list[int | float]]
) -> list[list[float]] | int:
    try:
        A_np = np.array(A, dtype=float)
        T_np = np.array(T, dtype=float)
        S_np = np.array(S, dtype=float)

        # Check squareness of T and S
        if T_np.ndim != 2 or S_np.ndim != 2 or A_np.ndim != 2:
            return -1
        if T_np.shape[0] != T_np.shape[1]:
            return -1
        if S_np.shape[0] != S_np.shape[1]:
            return -1

        n = T_np.shape[0]
        m = S_np.shape[0]

        # Check shape compatibility: T^{-1} (n×n) * A (n×m) * S (m×m)
        if A_np.shape != (n, m):
            return -1

        # Invertibility check with tolerance
        tol = 1e-12
        if abs(np.linalg.det(T_np)) < tol or abs(np.linalg.det(S_np)) < tol:
            return -1

        T_inv = np.linalg.inv(T_np)
        transformed = T_inv @ A_np @ S_np
        return transformed.