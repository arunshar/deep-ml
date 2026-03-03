import numpy as np

def calculate_portfolio_variance(cov_matrix: list[list[float]], weights: list[float]) -> float:
    """
    Calculate the variance of a portfolio.

    Args:
        cov_matrix (list[list[float]]): Covariance matrix of asset returns.
        weights (list[float]): Portfolio weights.

    Returns:
        float: Portfolio variance.
    """

    # Convert to numpy arrays
    cov = np.array(cov_matrix, dtype=float)
    w = np.array(weights, dtype=float)

    # Validation: covariance matrix must be square
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be square.")

    # Validation: dimensions must match
    if cov.shape[0] != w.shape[0]:
        raise ValueError("Weights must match covariance matrix dimensions.")

    # Compute portfolio variance
    variance = w.T @ cov @ w

    return float(variance)