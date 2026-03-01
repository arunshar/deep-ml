def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
    """
    Calculate posterior probabilities using Bayes' Theorem.
    """
    # 1. Calculate the numerator for each hypothesis: P(E | H_i) * P(H_i)
    # We use zip() to pair each prior with its corresponding likelihood
    unnormalized_posteriors = [prior * likelihood for prior, likelihood in zip(priors, likelihoods)]
    
    # 2. Calculate the denominator: The Law of Total Probability P(E)
    # This is simply the sum of all our unnormalized values
    marginal_likelihood = sum(unnormalized_posteriors)
    
    # Edge case protection: Avoid division by zero if the evidence is entirely impossible
    if marginal_likelihood == 0.0:
        return [0.0 for _ in priors]
        
    # 3. Calculate the final posterior probabilities by dividing by the marginal likelihood
    posteriors = [unnormalized / marginal_likelihood for unnormalized in unnormalized_posteriors]
    
    # 4. Return the list of posteri