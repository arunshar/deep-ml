import math

def normal_pdf(x: float, mean: float, std_dev: float) -> float:
    """
    Calculate the probability density function (PDF) of the normal distribution.
    """
    # Edge case protection: Standard deviation cannot be zero or negative
    if std_dev <= 0:
        return 0.0
        
    # 1. Calculate the Variance (sigma squared)
    variance = std_dev ** 2
    
    # 2. Calculate the coefficient (the left side of the formula)
    # 1 / sqrt(2 * pi * sigma^2)
    # Note: math.sqrt(2 * math.pi * variance) is mathematically the same as (std_dev * math.sqrt(2 * math.pi))
    coefficient = 1.0 / math.sqrt(2 * math.pi * variance)
    
    # 3. Calculate the exponent (the power to which 'e' is raised)
    # -(x - mu)^2 / (2 * sigma^2)
    exponent = -((x - mean) ** 2) / (2 * variance)
    
    # 4. Combine them using math.exp() for e^exponent
    pdf_value = coefficient * math.exp(exponent)
    
    # 5. Return the result rounded to 5 decimal places as requested
    return round