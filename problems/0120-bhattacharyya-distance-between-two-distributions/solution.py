import numpy as np

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    """
    Calculate the Bhattacharyya distance between two probability distributions.
    """
    # 1. Edge Cases: Check if lists are empty or of different lengths
    if not p or not q or len(p) != len(q):
        return 0.0
        
    # Convert lists to NumPy arrays for element-wise math
    p_arr = np.array(p)
    q_arr = np.array(q)
    
    # 2. Calculate the Bhattacharyya Coefficient (BC)
    # BC = sum(sqrt(P(X) * Q(X)))
    bc = np.sum(np.sqrt(p_arr * q_arr))
    
    # Edge case: If the distributions are completely disjoint, BC will be 0.
    # We must prevent a logarithm of 0 error (which mathematically means infinite distance).
    if bc <= 0:
        return float('inf')
        
    # 3. Calculate the Bhattacharyya Distance (BD)
    # BD = -ln(BC)
    bd = -np.log(bc)
    
    # 4. Return the result rounded to 4 decimal places
    return round(float(bd), 4)