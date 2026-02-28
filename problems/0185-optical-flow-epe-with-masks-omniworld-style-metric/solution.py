import numpy as np

def flow_epe(pred, gt, mask=None, max_flow=None):
    # 1. Convert inputs to numpy arrays for vector math
    # Arrays have shape (H, W, 2)
    pred = np.array(pred, dtype=np.float32)
    gt = np.array(gt, dtype=np.float32)
    
    # 2. Calculate EPE for every pixel: sqrt(sum of squared differences)
    # axis=-1 refers to the [u, v] dimension
    epe = np.linalg.norm(pred - gt, axis=-1)
    
    # 3. Handle invalid values (NaN, Inf) in the flow itself
    # We create a validity mask: True if value is a finite number
    valid = np.isfinite(epe)
    
    # 4. Incorporate the provided mask if it exists
    if mask is not None:
        mask = np.array(mask, dtype=bool)
        # Combine with our validity mask
        valid = np.logical_and(valid, mask)
    
    # 5. Check if we have any valid pixels left to calculate an average
    if not np.any(valid):
        return -1.0
        
    # 6. Extract only the valid EPE values
    valid_epe = epe[valid]
    
    # 7. Ap