import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data using least squares.
    Returns (W, b).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n = X.shape[0]
    ones = np.ones((n, 1), dtype=float)
    X_aug = np.hstack([X, ones])  # shape (n, d+1)

    theta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)

    W_out = theta[:-1]
    b_out = float(theta[-1])

    return W_out, b_out