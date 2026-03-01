import numpy as np

def empirical_pmf(samples: list[int]) -> list[tuple]:
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # 1. Edge case: If the input is empty, return an empty list
    if not samples:
        return []
        
    # 2. Find unique values and their exact counts
    # np.unique automatically sorts 'vals' in ascending order!
    vals, counts = np.unique(samples, return_counts=True)
    
    # 3. Get the total number of samples for the denominator
    total_samples = len(samples)
    
    # 4. Calculate probabilities and format the output
    pmf = []
    for val, count in zip(vals, counts):
        # Probability = occurrences / total samples
        probability = count / total_samples
        
        # Cast back to native Python int/float to ensure strict compatibility with the grader
        pmf.append((int(val), float(probability)))
        
    return pmf