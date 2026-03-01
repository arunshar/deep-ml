import math

def poisson_probability(k: int, lam: float) -> float:
    """
    Calculate the probability of observing exactly k events in a fixed interval,
    given the mean rate of events lam, using the Poisson distribution formula.
    """
    # Edge case: The number of events (k) cannot be negative.
    # A mean rate (lam) of 0 means events never happen.
    if k < 0 or lam < 0:
        return 0.0
    if lam == 0:
        return 1.0 if k == 0 else 0.0
        
    # 1. Calculate the numerator: (lam^k) * e^(-lam)
    numerator = (lam ** k) * math.exp(-lam)
    
    # 2. Calculate the denominator: k! (k factorial)
    denominator = math.factorial(k)
    
    # 3. Calculate the final probability
    probability = numerator / denominator
    
    # 4. Return the result rounded to 5 decimal places
    return round(probability, 5)