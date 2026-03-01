import numpy as np

def law_of_large_numbers(n_samples: int, population_mean: float, population_std: float) -> float:
    """
    Demonstrate the Law of Large Numbers by computing the sample mean.
    """
    # 1. Generate random samples from a normal distribution
    # loc = mean, scale = standard deviation, size = number of samples
    samples = np.random.normal(loc=population_mean, scale=population_std, size=n_samples)
    
    # 2. Calculate the mean of the generated samples
    sample_mean = np.mean(samples)
    
    # 3. Return the sample mean as a standard Python float
    return float(sample_mean)