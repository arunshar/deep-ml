import numpy as np

def newton_schulz5(G: np.ndarray, steps: int = 5, eps: float = 1e-7) -> np.ndarray:
    """
    Implement the Newton-Schulz iteration for matrix orthogonalization/preconditioning.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    
    # A. Frobenius Normalization
    norm = np.linalg.norm(G, ord='fro')
    X = G / (norm + eps)
    
    # B. Transpose Handling
    # Ensure the iteration operates on the smaller dimension to remain efficient
    rows, cols = X.shape
    transposed = False
    if rows > cols:
        X = X.T
        transposed = True
        
    # C. Quintic Newton-Schulz Iteration
    for _ in range(steps):
        A = X @ X.T
        A2 = A @ A
        # X_{k+1} = a*X_k + (b*A + c*A^2)*X_k
        X = a * X + b * (A @ X) + c * (A2 @ X)
        
    # D. Transpose Back if we flipped it earlier
    if transposed:
        X = X.T
        
    return X

def muon_step(theta: np.ndarray, B: np.ndarray, grad: np.ndarray, 
              eta: float, mu: float