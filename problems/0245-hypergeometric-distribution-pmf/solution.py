import math

def hypergeometric_pmf(N: int, K: int, n: int, k: int) -> float:
    """
    Calculate the PMF of the hypergeometric distribution.
    """
    # 1. Check for valid bounds of k
    # k cannot be less than 0. Also, if we draw more items (n) than the total number 
    # of failures (N-K), we are forced to draw at least n - (N-K) successes.
    lower_bound = max(0, n - (N - K))
    
    # k cannot exceed the total number of successes available (K), 
    # nor can it exceed the total number of draws (n).
    upper_bound = min(K, n)
    
    # Edge case: Return 0.0 if k is out of bounds
    if k < lower_bound or k > upper_bound:
        return 0.0
        
    # 2. Calculate the combinations
    # Ways to choose exactly k successes from the K available
    success_ways = math.comb(K, k)
    
    # Ways to choose the remaining (n-k) items from the available failures (N-K)
    failure_ways = math.comb(N - K, n - k)
    
    # Total ways to choose n items from the entire population