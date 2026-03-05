import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute approximate SVD of a real 2x2 matrix using one Jacobi rotation.

    Returns (U, S, Vt) such that A ≈ U @ diag(S) @ Vt
    """
    A = np.asarray(A, dtype=float)
    if A.shape != (2, 2):
        raise ValueError("A must be shape (2, 2)")

    # 1) Form B = A^T A (symmetric)
    B = A.T @ A
    b11, b12, b22 = B[0, 0], B[0, 1], B[1, 1]

    # 2) One Jacobi rotation angle
    # theta = 0.5 * atan2(2*b12, b11 - b22)
    theta = 0.5 * np.arctan2(2.0 * b12, b11 - b22)
    c = np.cos(theta)
    s = np.sin(theta)

    # 3) Right singular vectors V = R(theta)
    V = np.array([[c, -s],
                  [s,  c]], dtype=float)

    # 4) Diagonalize: D = V^T B V (should be diagonal for 2x2 with one Jacobi step)
    D = V.T @ B @ V
    lam1 = D[0, 0]
    lam2 = D[1, 1]

    # Singular values (non-negative)
    eps = 1e-12
    s1 = np.sqrt(max(lam1, 0.0))
    s2 = np.sqrt(max(lam2, 0.0))
    S = np.array(