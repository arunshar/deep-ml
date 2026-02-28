import numpy as np

def zero_pad_image(img, pad_width):
    """
    Add zero padding around a grayscale image.
    """
    # 1. Edge Case: Check if pad_width is a non-negative integer
    if not isinstance(pad_width, int) or isinstance(pad_width, bool) or pad_width < 0:
        return -1
        
    # 2. Edge Case: Validate format (catch jagged lists or invalid types)
    try:
        img_arr = np.array(img)
    except Exception:
        return -1
        
    # 3. Edge Case: Check if it's strictly a 2D array
    if img_arr.ndim != 2:
        return -1
        
    # 4. Edge Case: Check for empty dimensions (e.g., [], [[]])
    if img_arr.size == 0 or img_arr.shape[0] == 0 or img_arr.shape[1] == 0:
        return -1
        
    # 5. Apply Zero Padding using NumPy
    # When pad_width is a single integer, np.pad adds that many 
    # elements to the beginning and end of every dimension.
    padded_img = np.pad(img_arr, pad_width=pad_width, mode='constant', constant_values=0)
    
    