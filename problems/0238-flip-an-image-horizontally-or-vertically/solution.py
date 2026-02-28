import numpy as np

def flip_image(image, direction):
    """
    Flip an image horizontally or vertically.
    """
    # 1. Edge Case: Invalid direction string
    if direction not in ['horizontal', 'vertical']:
        return -1
        
    # 2. Edge Case: Invalid input format (catch jagged lists)
    try:
        img = np.array(image)
    except Exception:
        return -1
        
    # 3. Edge Case: Check if it's a valid 2D or 3D array
    if img.ndim not in [2, 3]:
        return -1
        
    # 4. Edge Case: Check for empty dimensions
    if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        return -1
        
    # 5. Perform the flip using NumPy slicing
    if direction == 'horizontal':
        # Reverse the columns (left-right)
        # img[:, ::-1] means "take all rows, reverse the columns"
        flipped_img = img[:, ::-1]
        
    elif direction == 'vertical':
        # Reverse the rows (top-bottom)
        # img[::-1, :] means "reverse the rows, ta