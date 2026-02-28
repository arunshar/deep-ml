import math

def phi_corr(x: list[int], y: list[int]) -> float:
    """
    Calculate the Phi coefficient between two binary variables.
    """
    # Edge case: lists must be of the same length and not empty
    if not x or not y or len(x) != len(y):
        return 0.0
        
    # 1. Initialize counters for the four possible combinations
    n00 = 0  # x = 0, y = 0
    n01 = 0  # x = 0, y = 1
    n10 = 0  # x = 1, y = 0
    n11 = 0  # x = 1, y = 1
    
    # 2. Iterate through the lists and count the occurrences
    for val_x, val_y in zip(x, y):
        if val_x == 0 and val_y == 0:
            n00 += 1
        elif val_x == 0 and val_y == 1:
            n01 += 1
        elif val_x == 1 and val_y == 0:
            n10 += 1
        elif val_x == 1 and val_y == 1:
            n11 += 1
            
    # 3. Calculate the numerator
    numerator = (n00 * n11) - (n01 * n10)
    
    # 4. Calculate the denominator (inside the square root first)
    denominator_squared = (n00 + n01) * (n1