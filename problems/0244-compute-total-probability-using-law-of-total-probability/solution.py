def law_of_total_probability(priors: dict, conditionals: dict) -> float:
    """
    Compute P(A) using the Law of Total Probability.
    
    Args:
        priors: Dictionary mapping partition event names to P(Bi)
        conditionals: Dictionary mapping partition event names to P(A|Bi)
        
    Returns:
        float: The total probability P(A), rounded to 4 decimal places
    """
    # 1. Calculate the sum of P(Bi) * P(A|Bi) for all events
    # We iterate over the keys in the priors dictionary.
    total_prob = sum(priors[event] * conditionals[event] for event in priors)
    
    # 2. Return the result rounded to 4 decimal places
    return round(total_prob, 4)