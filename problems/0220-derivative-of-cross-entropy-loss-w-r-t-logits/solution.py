import numpy as np

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
    # 1. Convert logits to numpy array for vector operations
    logits_array = np.array(logits)
    
    # 2. Compute Softmax probabilities (P)
    # Subtracting max(logits) provides numerical stability
    exp_logits = np.exp(logits_array - np.max(logits_array))
    probabilities = exp_logits / np.sum(exp_logits)
    
    # 3. Create the one-hot target vector (Y)
    # y = [0, 0, ..., 1, ..., 0] where the 1 is at the target index
    y = np.zeros_like(probabilities)
    y[target] = 1.0
    
    # 4. The gradient is simply P - Y
    gradient = probabilities - y
    
    return gradient.tolist()