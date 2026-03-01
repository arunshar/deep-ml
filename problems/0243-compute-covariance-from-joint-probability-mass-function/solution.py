import numpy as np

def covariance_from_joint_pmf(x_values: list, y_values: list, joint_pmf: np.ndarray) -> float:
    """
    Compute the covariance of X and Y from their joint PMF.
    """
    x = np.array(x_values)
    y = np.array(y_values)
    
    # 1. Marginal Probabilities and Expected Value of X (E[X])
    # Sum across the columns (axis=1) to get the probability of each X regardless of Y
    p_x = np.sum(joint_pmf, axis=1)
    e_x = np.sum(x * p_x)
    
    # 2. Marginal Probabilities and Expected Value of Y (E[Y])
    # Sum across the rows (axis=0) to get the probability of each Y regardless of X
    p_y = np.sum(joint_pmf, axis=0)
    e_y = np.sum(y * p_y)
    
    # 3. Expected Value of the Product (E[XY])
    # Iterate through the grid to calculate the double summation
    e_xy = 0.0
    for i in range(len(x)):
        for j in range(len(y)):
            e_xy += x[i] * y[j] * joint_pmf[i][j]
            
    # 4. Calculate Covariance
    # Cov(X, Y) = E[XY] - E[X]*E[Y]
   