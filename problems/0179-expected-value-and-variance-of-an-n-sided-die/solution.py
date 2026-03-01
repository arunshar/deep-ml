def dice_statistics(n: int) -> tuple[float, float]:
    """
    Compute the expected value and variance of a fair n-sided die roll.
    """
    # 1. Calculate the Expected Value
    # Formula: (n + 1) / 2
    expected_value = (n + 1) / 2.0
    
    # 2. Calculate the Variance
    # Formula: (n^2 - 1) / 12
    variance = ((n ** 2) - 1) / 12.0
    
    # 3. Return the results as a tuple of floats
    return float(expected_value), float(variance)