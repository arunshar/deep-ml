import math

def warmup_cosine_schedule(T: int, W: int, lr_max: float, lr_min: float) -> list[float]:
    lrs = []
    
    for t in range(T):
        if t < W:
            # Linear warmup
            lr = (t / W) * lr_max
        else:
            # Cosine decay
            decay_progress = (t - W) / (T - W)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
            lr = lr_min + (lr_max - lr_min) * cosine_decay
        
        lrs.append(lr)
    
    return lrs