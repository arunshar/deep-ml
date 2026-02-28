import numpy as np

def detect_outliers_iqr(data: list[float], k: float = 1.5) -> dict:
    """
    Detect and remove outliers using the IQR method.
    """
    # Edge case for empty data
    if not data:
        return {
            'cleaned_data': [],
            'outlier_indices': [],
            'lower_bound': 0.0,
            'upper_bound': 0.0
        }
        
    # 1. Calculate Q1 (25th percentile) and Q3 (75th percentile)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    
    # 2. Calculate IQR and the threshold boundaries
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    # 3. Iterate through the data to separate valid values and outlier indices
    cleaned_data = []
    outlier_indices = []
    
    for i, val in enumerate(data):
        if val < lower_bound or val > upper_bound:
            # It's an outlier, record its index
            outlier_indices.append(i)
        else:
            # It's valid, round it to 4