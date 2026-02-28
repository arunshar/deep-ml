import numpy as np

def calculate_contrast(img) -> int:
    """
    Calculate the contrast of a grayscale image.
    Contrast = Max Pixel Value - Min Pixel Value
    """
    # 1. Use numpy's max and min functions to find the extremes
    max_val = np.max(img)
    min_val = np.min(img)
    
    # 2. Calculate the difference
    contrast = max_val - min_val
    
    # 3. Return as an integer
    return int(contrast)