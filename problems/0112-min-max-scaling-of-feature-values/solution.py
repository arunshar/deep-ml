def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    """
    # Edge case: empty list
    if not x:
        return []
        
    # 1. Find the minimum and maximum values in the list
    x_min = min(x)
    x_max = max(x)
    
    # 2. Special Case: If all elements are identical, avoid division by zero
    if x_min == x_max:
        return [0.0] * len(x)
        
    # 3. Apply the Min-Max formula to each element
    normalized_list = []
    for val in x:
        # X' = (X - X_min) / (X_max - X_min)
        scaled_val = (val - x_min) / (x_max - x_min)
        
        # Round the result to 4 decimal places as requested
        normalized_list.append(round(scaled_val, 4))
        
    return normalized_list