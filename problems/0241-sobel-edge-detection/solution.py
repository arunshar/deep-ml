import numpy as np

def sobel_edge_detection(image):
    # 1. Edge Cases & Validation
    # Catch empty lists, jagged arrays, or non-numeric types
    try:
        img = np.array(image, dtype=np.float32)
    except Exception:
        return -1
        
    # Check if the array is strictly 2D
    if img.ndim != 2:
        return -1
        
    # Check if dimensions are at least 3x3
    H, W = img.shape
    if H < 3 or W < 3:
        return -1
        
    # Check if any pixel values are out of bounds
    if np.any(img < 0) or np.any(img > 255):
        return -1

    # 2. Vectorized Convolution using array slicing
    # Instead of looping, we shift the entire image grid to multiply with the kernel weights
    
    # Horizontal Gradient (Gx)
    # [[-1,  0,  1], 
    #  [-2,  0,  2], 
    #  [-1,  0,  1]]
    gx = (
        -1 * img[0:-2, 0:-2] + 1 * img[0:-2, 2:] +
        -2 * img[1:-1, 0:-2] + 2 * img[1:-1, 2:] +
        -1 * img[2:, 0:-2]   + 1 * img[2:, 2:]
    )
    
    # Vertica