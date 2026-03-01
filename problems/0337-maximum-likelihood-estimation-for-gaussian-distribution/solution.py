import numpy as np

def gaussian_mle(data: np.ndarray) -> tuple:
    """
    Compute Maximum Likelihood Estimates for Gaussian distribution parameters.
    """
    # 1. Edge Case: If the data array is empty, return a default tuple
    if data.size == 0:
        return 0.0, 0.0
        
    # 2. Calculate the MLE for the mean
    # Formula: sum(x) / n
    mean_mle = np.mean(data)
    
    # 3. Calculate the MLE for the variance
    # Formula: sum((x - mean)^2) / n
    # NumPy's np.var defaults to population variance (ddof=0), which divides by n perfectly!
    variance_mle = np.var(data)
    
    # 4. Return the results as a tuple of standard Python floats
    return float(mean_mle), float(variance_mle)