def conditional_probability(joint_distribution: dict) -> float:
    """
    Compute conditional probability P(A|B) from a joint probability distribution.
    """
    # 1. Extract P(A and B)
    # The probability where both A and B occur
    p_a_and_b = joint_distribution.get(('A', 'B'), 0.0)
    
    # 2. Extract P(not A and B)
    # The probability where A does not occur, but B still occurs
    p_not_a_and_b = joint_distribution.get(('`A', 'B'), 0.0)
    
    # 3. Calculate Marginal Probability P(B)
    # P(B) is the sum of all joint probabilities where B is true
    p_b = p_a_and_b + p_not_a_and_b
    
    # 4. Edge Case: Prevent division by zero
    # Conditional probability is only defined if P(B) > 0
    if p_b == 0.0:
        return 0.0
        
    # 5. Calculate the Conditional Probability P(A|B)
    p_a_given_b = p_a_and_b / p_b
    
    # Return the raw float (the built-in test case will handle the rounding)
    return float(p_a_given_b)