import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    '''
    Fill in missing values (NaN) in the input array.
    
    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)
    
    Returns:
        X_clean: Array with no NaN values, same shape as X
    '''
    X_clean = X.copy()
    
    for j in range(X_clean.shape[1]):
        col = X_clean[:, j]

        mean_val = np.nanmean(col)

        if np.isnan(mean_val):
            mean_val = 0.0

        col[np.isnan(col)] = mean_val

    return X_clean
