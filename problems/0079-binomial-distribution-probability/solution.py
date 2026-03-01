import math

def binomial_probability(n: int, k: int, p: float) -> float:
    """
    Calculate the probability of exactly k successes in n Bernoulli trials.
    """
    # Edge case: You cannot have more successes than trials, or negative successes
    if k > n or k < 0:
        return 0.0
        
    # Edge case: Probability must be between 0 and 1
    if p < 0.0 or p > 1.0:
        return 0.0
        
    # 1. Calculate the binomial coefficient "n choose k"
    # This represents the number of possible ways k successes can occur in n trials
    coeff = math.comb(n, k)
    
    # 2. Calculate the probability of the successes
    success_term = p ** k
    
    # 3. Calculate the probability of the failures
    failure_term = (1.0 - p) ** (n - k)
    
    # 4. Multiply them together to get the final probability
    probability = coeff * success_term * failure_term
    
    # The built-in test cases handle rounding to 5 decimal places for you!
    return float(probability)