import numpy as np
from scipy import stats

def confidence_interval(data: list[float], confidence_level: float = 0.95) -> dict:
    """
    Calculate confidence interval for population mean.
    """
    # 1. Convert list to a numpy array for vectorized math
    arr = np.array(data)
    n = len(arr)
    
    # 2. Calculate the sample mean
    mean = np.mean(arr)
    
    # 3. Calculate sample standard deviation and Standard Error (SE)
    # Important: ddof=1 ensures we calculate the sample standard deviation (dividing by n-1)
    std_dev = np.std(arr, ddof=1)
    se = std_dev / np.sqrt(n)
    
    # 4. Determine degrees of freedom
    df = n - 1
    
    # 5. Find the critical t-value
    # alpha is the probability that the true mean is outside the confidence interval.
    # We divide by 2 because it's a two-tailed test (split between lower and upper bounds).
    # stats.t.ppf gives us the exact t-value for the cumulative probability.
    alpha = 1.0 - confidence_level
    t_critical = 