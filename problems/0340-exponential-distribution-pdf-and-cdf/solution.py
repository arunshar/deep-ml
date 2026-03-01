import numpy as np

def exponential_distribution(x: list, lam: float) -> dict:
    """
    Compute exponential distribution properties.
    """
    # 1. Edge Case: If lambda is not positive, return all None
    if lam <= 0:
        return {
            'pdf': None,
            'cdf': None,
            'mean': None,
            'variance': None
        }
        
    # 2. Calculate Theoretical Mean and Variance
    # Mean = 1 / lambda
    # Variance = 1 / lambda^2
    mean = 1.0 / lam
    variance = 1.0 / (lam ** 2)
    
    pdf_list = []
    cdf_list = []
    
    # 3. Calculate PDF and CDF for each point in x
    for val in x:
        # Edge Case: For x < 0, PDF and CDF are exactly 0
        if val < 0:
            pdf_list.append(0.0)
            cdf_list.append(0.0)
        else:
            # PDF = lambda * e^(-lambda * x)
            pdf_val = lam * np.exp(-lam * val)
            
            # CDF = 1 - e^(-lambda * x)
            cdf_val = 1.0 - np.exp(-lam * val)
            
 