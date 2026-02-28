def compute_arithmetic_intensity(flops: float, bytes_accessed: float, peak_performance: float, peak_bandwidth: float) -> dict:
    # 1. Calculate Arithmetic Intensity (I)
    arithmetic_intensity = flops / bytes_accessed
    
    # 2. Calculate the Ridge Point
    ridge_point = peak_performance / peak_bandwidth
    
    # 3. Determine the bottleneck and achieved performance
    if arithmetic_intensity < ridge_point:
        bottleneck = 'memory-bound'
        # Performance is limited by memory: Performance = Intensity * Bandwidth
        achieved_performance = arithmetic_intensity * peak_bandwidth
    else:
        bottleneck = 'compute-bound'
        # Performance is limited by peak hardware capability
        achieved_performance = peak_performance
        
    # 4. Calculate Utilization Percentage
    utilization_percent = (achieved_performance / peak_performance) * 100
    
    return {
        'arithmetic_intensity': round(arithmetic_intensity, 4),
        'ridge_point': round(rid