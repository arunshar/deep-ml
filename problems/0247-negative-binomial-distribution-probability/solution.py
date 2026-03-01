import math

def negative_binomial_pmf(k: int, r: int, p: float) -> float:
    """
    Calculate the probability of observing exactly k failures
    before achieving r successes in independent Bernoulli trials.
    """
    # 1. Handle Edge Cases
    # k must be non-negative, r must be positive, p must be in (0, 1]
    if k < 0 or r <= 0 or p <= 0.0 or p > 1.0:
        return 0.0
        
    # 2. Calculate the binomial coefficient: (k + r - 1) choose k
    # This represents the number of valid sequences of successes and failures
    combinations = math.comb(k + r - 1, k)
    
    # 3. Calculate the probability of the successes
    success_prob = p ** r
    
    # 4. Calculate the probability of the failures
    failure_prob = (1.0 - p) ** k
    
    # 5. Multiply everything together according to the formula
    probability = combinations * success_prob * failure_prob
    
    # 6. Return the result rounded to 5 decimal places as requested
    return round(float(probability), 5)