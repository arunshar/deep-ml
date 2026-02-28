import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    """
    # 1. Convert to numpy array for vectorized operations
    arr = np.array(data)
    
    # 2. Mean and Median
    mean_val = np.mean(arr)
    median_val = np.median(arr)
    
    # 3. Mode
    # np.unique returns the unique values and their frequencies.
    # We find the index of the highest frequency to get the mode.
    vals, counts = np.unique(arr, return_counts=True)
    mode_val = vals[np.argmax(counts)]
    
    # 4. Variance and Standard Deviation
    # The problem specifies "Population variance (divide by N)".
    # NumPy's default for np.var and np.std is population (ddof=0).
    var_val = np.var(arr)
    std_val = np.std(arr)
    
    # 5. Percentiles
    q25 = np.percentile(arr, 25)
    q50 = np.percentile(arr, 50)  # This is functionally identical to the median
    q75 = np.percentile(arr, 75)
    
    