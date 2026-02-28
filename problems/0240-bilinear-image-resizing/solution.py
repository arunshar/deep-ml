import numpy as np

def bilinear_resize(image, new_height: int, new_width: int) -> list:
    # 1. Convert to numpy array for matrix operations
    img = np.array(image, dtype=np.float32)
    old_height, old_width = img.shape[0], img.shape[1]
    
    # 2. Calculate scale factors
    sy = old_height / new_height
    sx = old_width / new_width
    
    # 3. Create a grid of target coordinates (i, j)
    # i goes from 0 to new_height-1, j goes from 0 to new_width-1
    i = np.arange(new_height)
    j = np.arange(new_width)
    
    # Map target coordinates back to source coordinates (y, x)
    x = j * sx
    y = i * sy
    
    # Create 2D grids for x and y to calculate all pixels simultaneously
    xv, yv = np.meshgrid(x, y)
    
    # 4. Find the top-left (x0, y0) and bottom-right (x1, y1) neighbors
    x0 = np.floor(xv).astype(int)
    y0 = np.floor(yv).astype(int)
    
    # Clamp x1 and y1 to the image boundaries to prevent out-of-bounds errors
    x1 = np.clip(x0 + 1, 0, old_width -