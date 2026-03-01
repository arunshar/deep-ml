import numpy as np

def multivariate_kl_divergence(mu_p: np.ndarray, Cov_p: np.ndarray, mu_q: np.ndarray, Cov_q: np.ndarray) -> float:
    """
    Computes the KL divergence between two multivariate Gaussian distributions.
    """
    # 1. Determine the dimensionality (p) of the distributions
    p = mu_p.shape[0]
    
    # 2. Calculate the determinants of both covariance matrices
    det_p = np.linalg.det(Cov_p)
    det_q = np.linalg.det(Cov_q)
    
    # Calculate the first term: log(|Sigma_q| / |Sigma_p|)
    log_det_term = np.log(det_q / det_p)
    
    # 3. Calculate the inverse of the second covariance matrix (Sigma_q^-1)
    inv_Cov_q = np.linalg.inv(Cov_q)
    
    # 4. Calculate the Mahalanobis distance term: (mu_p - mu_q)^T * Sigma_q^-1 * (mu_p - mu_q)
    # The '@' operator in Python performs matrix multiplication
    mean_diff = mu_p - mu_q
    mean_term = mean_diff.T @ inv_Cov_q @ mean_diff
    
    # 5. Calculate the Trace term: tr(Sigma_q^-1 * Sigma_p)
    trace_term = 