import numpy as np

def compute_elbo(x: list[float], q_mean: float, q_std: float, 
                 prior_mean: float, prior_std: float, 
                 likelihood_std: float, n_samples: int = 1000) -> float:
    """
    Compute the Evidence Lower Bound (ELBO) for variational inference.
    """
    
    # Helper function to compute the Gaussian Log PDF (vectorized)
    def gaussian_log_pdf(val, mean, std):
        variance = std ** 2
        return -0.5 * np.log(2 * np.pi * variance) - ((val - mean) ** 2) / (2 * variance)

    # 1. Sample 'z' from our variational distribution q(z)
    z_samples = np.random.normal(loc=q_mean, scale=q_std, size=n_samples)
    
    # 2. Expected Log-Likelihood: E_q[log p(x|z)]
    # p(x_i | z) is Normal with mean=z and std=likelihood_std
    log_likelihoods = np.zeros(n_samples)
    for x_i in x:
        # Sum the log likelihood for every data point x_i
        log_likelihoods += gaussian_log_pdf(x_i, z_samples, likelihood_std)
        
    expected_log