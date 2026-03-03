import math
from typing import List

def clip_gradients_by_global_norm(gradients: List[List[float]], max_norm: float) -> List[List[float]]:
    # Compute global L2 norm
    sum_sq = 0.0
    for arr in gradients:
        for g in arr:
            sum_sq += g * g

    global_norm = math.sqrt(sum_sq)

    # If nothing to clip (or all zeros), return as-is (copy for safety/consistency)
    if global_norm == 0.0 or global_norm <= max_norm:
        return [arr[:] for arr in gradients]

    # If max_norm is 0 (or negative), scale everything to 0
    if max_norm <= 0.0:
        return [[0.0 for _ in arr] for arr in gradients]

    # Scale down so that global norm becomes max_norm
    scale = max_norm / global_norm
    return [[g * scale for g in arr] for arr in gradients]