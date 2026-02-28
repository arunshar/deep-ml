import numpy as np

def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminosity method.
    
    Args:
        image: RGB image as list or numpy array of shape (H, W, 3)
               with values in range [0, 255]
               
    Returns:
        Grayscale image as 2D list with integer values,
        or -1 if input is invalid
    """
    # 1. Edge Case: Validate format and convert to numpy array
    # A try-except block safely catches jagged/irregular lists 
    # that cannot be converted to a clean multi-dimensional array
    try:
        img = np.array(image, dtype=np.float32)
    except Exception:
        return -1
        
    # 2. Edge Case: Check for valid 3D shape and 3 color channels
    if img.ndim != 3 or img.shape[2] != 3:
        return -1
        
    # Edge Case: Check for empty dimensions (H or W is 0)
    if img.shape[0] == 0 or img.shape[1] == 0:
        return -1
        
    # 3. Edge Case: Check if pixel values are outside 0-255