import numpy as np

def bayesian_inference_beta_binomial(prior_alpha: float, prior_beta: float,
                                     successes: int, trials: int) -> tuple[float, float, float]:
    """
    Perform Bayesian inference for Beta-Binomial model.
    """
    # 1. Calculate the number of failures
    failures = trials - successes
    
    # 2. Update the posterior parameters
    # posterior_alpha = prior_alpha + successes
    posterior_alpha = float(prior_alpha + successes)
    
    # posterior_beta = prior_beta + failures
    posterior_beta = float(prior_beta + failures)
    
    # 3. Calculate the posterior mean
    # Mean of Beta(a, b) is a / (a + b)
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    
    # 4. Return the tuple of the updated parameters and the mean
    return posterior_alpha, posterior_beta, float(posterior_mean)