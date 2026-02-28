import numpy as np

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.
    """
    # 1. Create a copy of the data so we don't modify the original array in place
    imputed_data = np.copy(data)
    
    # 2. Iterate over each column independently
    for col_idx in range(imputed_data.shape[1]):
        col = imputed_data[:, col_idx]
        
        # 3. Create a boolean mask of valid (non-NaN) values
        valid_mask = ~np.isnan(col)
        
        # 4. If the column has all missing values, leave it unchanged
        if not np.any(valid_mask):
            continue
            
        # Extract just the valid numbers to compute our statistic
        valid_values = col[valid_mask]
        
        # 5. Compute the imputation statistic based on the strategy
        if strategy == 'mean':
            stat = np.mean(valid_values)
            
        elif strategy == 'median':
 