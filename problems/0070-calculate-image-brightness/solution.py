def calculate_brightness(img):
    # 1. Edge Case: Check if the image matrix is empty
    if not img:
        return -1
    
    # 2. Edge Case: Check for inconsistent row lengths
    # We use the length of the first row as the standard
    first_row_length = len(img[0])
    
    total_sum = 0
    total_pixels = 0
    
    for row in img:
        if len(row) != first_row_length:
            return -1
        
        for pixel in row:
            # 3. Edge Case: Check if pixel values are outside 0-255
            if pixel < 0 or pixel > 255:
                return -1
            
            total_sum += pixel
            total_pixels += 1
            
    # 4. Calculate average and round to two decimal places
    if total_pixels == 0:
        return -1
        
    average_brightness = total_sum / total_pixels
    return round(float(average_brightness), 2)