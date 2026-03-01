def conditional_probability(data: list[tuple[str, str]], x: str, y: str) -> float:
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    """
    # 1. Initialize counters
    count_x = 0
    count_x_and_y = 0
    
    # 2. Iterate through the dataset to count occurrences
    for item_x, item_y in data:
        # Check if the condition X=x is met
        if item_x == x:
            count_x += 1
            
            # If X=x is met, also check if Y=y is met
            if item_y == y:
                count_x_and_y += 1
                
    # 3. Edge Case: Prevent division by zero if X=x never appears in the data
    if count_x == 0:
        return 0.0
        
    # 4. Calculate the probability
    probability = count_x_and_y / count_x
    
    # 5. Return the result rounded to 4 decimal places
    return round(probability, 4)