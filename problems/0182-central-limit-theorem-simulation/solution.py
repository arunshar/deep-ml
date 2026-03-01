import numpy as np

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    Simulate the Central Limit Theorem.
    """
    # 1. Set the random seed for reproducibility
    np.random.seed(seed)
    
    # 2. Generate samples and define theoretical population parameters
    if distribution == 'uniform':
        # Uniform(0, 1) -> mean = 0.5, variance = 1/12
        samples = np.random.uniform(0, 1, size=(runs, n))
        pop_mean = 0.5
        pop_std = np.sqrt(1 / 12)
        
    elif distribution == 'exponential':
        # Exponential(scale=1.0) -> mean = 1.0, variance = 1.0^2
        samples = np.random.exponential(1.0, size=(runs, n))
        pop_mean = 1.0
        pop_std = 1.0
        
    elif distribution == 'bernoulli':
    # Generate random floats between 0 and 1, and check if they are < 0.3
    # astype(int) converts the True/False boolean array into 1s and 0s
        samples = (np.random.rand(runs, n) < 0.3).astype(int)
        