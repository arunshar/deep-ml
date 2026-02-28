import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
    # Convert x to a numpy array for easier manipulation
    x = np.array(x, dtype=float)
    n = len(x)
    
    # Evaluate the function at the original point to get output dimension m
    f_x = np.array(f(x.tolist()))
    m = len(f_x)
    
    # Initialize an empty Jacobian matrix of size m x n
    jacobian = np.zeros((m, n))
    
    for i in range(n):
        # Create a copy of x to perturb the i-th component
        x_plus_h = x.copy()
        x_plus_h[i] += h
        
        # Evaluate function at the perturbed point
        f_x_plus_h = np.array(f(x_plus_h.tolist()))
        
        # Calculate partial derivative: [f(x + h_i) - f(x)] / h
        # This fills the i-th column of the Jacobian matrix
        jacobian[:, i] = (f_x_plus_h - f_x) / h
        
    return jacobian.tolist()