import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.

    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return

    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Standardize the data
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized = (data - mean) / std

    # Covariance matrix
    cov_matrix = np.cov(standardized, rowvar=False)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Take top k components
    components = eigenvectors[:, :k]

    # Fix sign convention
    for j in range(components.shape[1]):
        for i 