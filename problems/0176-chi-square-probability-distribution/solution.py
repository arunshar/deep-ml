import math

def chi_square_probability(x: float, k: int) -> float:
    """
    Calculate the probability density of x in a Chi-square distribution
    with k degrees of freedom.
    """
    # 1. Edge Case: Chi-square is only defined for x >= 0
    if x < 0 or k <= 0:
        return 0.0
        
    k_half = k / 2.0
    
    # 2. Handle x=0 edge cases safely to avoid math domain errors
    if x == 0 and k < 2:
        return 0.0  # Mathematically undefined/diverges, typical programmatic fallback
        
    # 3. Calculate the Denominator: 2^(k/2) * Gamma(k/2)
    denominator = (2 ** k_half) * math.gamma(k_half)
    
    # 4. Calculate the x term: x^((k/2) - 1)
    # Note: If x=0 and k=2, this naturally becomes 0^0 which Python evaluates to 1.0
    x_term = x ** (k_half - 1)
    
    # 5. Calculate the exponential term: e^(-x/2)
    exp_term = math.exp(-x / 2.0)
    
    # 6. Combine all parts together
    probability = (1.0 / denominator) * x_term * exp_term
    
    # 7. Return round