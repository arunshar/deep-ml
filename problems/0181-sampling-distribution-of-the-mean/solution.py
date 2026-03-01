import numpy as np

def simulate_clt(num_samples: int, sample_size: int, distribution: str = 'uniform') -> float:
    """
    Compute the mean of sample means to demonstrate the sampling distribution.
    """
    # 1. Generate the random samples based on the selected distribution
    # We generate a 2D array where each row is a sample, and each column is an observation
    if distribution == 'uniform':
        samples = np.random.uniform(0, 1, size=(num_samples, sample_size))
    elif distribution == 'exponential':
        samples = np.random.exponential(1, size=(num_samples, sample_size))
    else:
        raise ValueError("Unsupported distribution type. Use 'uniform' or 'exponential'.")
        
    # 2. Calculate the mean of each individual sample
    # axis=1 tells NumPy to calculate the mean across the columns (horizontally for each row)
    sample_means = np.mean(samples, axis=1)
    
    # 3. Calculate the overall mean of those sample means
    mean_of_means = np.mean(sample_mea