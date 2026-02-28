import numpy as np
from typing import Callable

def newtons_method_optimization(
    gradient_func: Callable[[list[float]], list[float]],
    hessian_func: Callable[[list[float]], list[list[float]]],
    x0: list[float],
    tol: float = 1e-6,
    max_iter: int = 100
) -> list[float]:
    x = np.array(x0, dtype=float)
    
    for _ in range(max_iter):
        # 1. Compute Gradient (g) and Hessian (H) at current point
        g = np.array(gradient_func(x.tolist()))
        H = np.array(hessian_func(x.tolist()))
        
        # 2. Check for convergence: if the gradient norm is small, we are at the minimum
        if np.linalg.norm(g) < tol:
            break
            
        # 3. Calculate the Newton step: delta = H^-1 * g
        # Using np.linalg.solve(H, g) is more stable than inverting the matrix
        step = np.linalg.solve(H, g)
        
        # 4. Update x: x = x - step
        x = x - step
        
    return x.tolist()