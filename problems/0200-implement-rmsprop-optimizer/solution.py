import math
from typing import List, Tuple

def rmsprop_update(
    params: List[float],
    grads: List[float],
    cache: List[float],
    lr: float = 0.01,
    beta: float = 0.9,
    epsilon: float = 1e-8
) -> Tuple[List[float], List[float]]:
    """
    Perform RMSProp optimization update.
    """
    # Make fresh lists (don’t mutate inputs unexpectedly)
    new_cache = []
    new_params = []

    for p, g, v in zip(params, grads, cache):
        v_new = beta * v + (1.0 - beta) * (g * g)
        p_new = p - lr * (g / (math.sqrt(v_new) + epsilon))
        new_cache.append(v_new)
        new_params.append(p_new)

    return new_params, new_cache