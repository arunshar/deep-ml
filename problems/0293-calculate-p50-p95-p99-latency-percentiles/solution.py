import numpy as np

def calculate_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    """
    Calculate P50, P95, and P99 latency percentiles.
    """
    # 1. Edge Case: If the input list is empty, return 0.0 for all percentiles
    if not latencies:
        return {'P50': 0.0, 'P95': 0.0, 'P99': 0.0}
        
    # 2. Calculate the percentiles using NumPy
    # np.percentile uses linear interpolation by default
    percentiles = np.percentile(latencies, [50, 95, 99])
    
    # 3. Round to 4 decimal places and construct the result dictionary
    return {
        'P50': round(float(percentiles[0]), 4),
        'P95': round(float(percentiles[1]), 4),
        'P99': round(float(percentiles[2]), 4)
    }