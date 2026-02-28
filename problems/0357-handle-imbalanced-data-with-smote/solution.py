import numpy as np

def smote(X_minority: np.ndarray, n_synthetic: int, k: int = 5, random_seed: int = 42) -> np.ndarray:
    """
    Generate synthetic samples using SMOTE algorithm.
    """
    # 0. Set the random seed to ensure reproducibility
    np.random.seed(random_seed)
    
    X_minority = np.array(X_minority)
    n_samples, n_features = X_minority.shape
    
    # THE FIX FOR TEST CASE 5:
    # If 0 synthetic samples are requested, return an empty 2D array with correct feature dimension
    if n_synthetic == 0:
        return np.empty((0, n_features))
    
    # Edge case: If there are fewer samples than k+1, cap the number of neighbors
    actual_k = min(k, n_samples - 1)
    
    synthetic_samples = []
    
    for _ in range(n_synthetic):
        # 1. Randomly select a base sample from the minority class
        idx = np.random.randint(0, n_samples)
        base_sample = X_minority[idx]
        
        # 2. Find its k nearest neighbors
        distances = np.linalg.norm(