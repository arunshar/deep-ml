def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]],
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
    """
    Apply weight decay (L2 regularization) to parameters.
    """
    updated_parameters = []
    
    # 1. Iterate through the parameter groups using zip
    for params, grads, apply_decay in zip(parameters, gradients, apply_to_all):
        updated_group = []
        
        # 2. Iterate through each individual parameter and its gradient
        for w, g in zip(params, grads):
            
            # 3. Apply the base gradient descent step: w = w - lr * grad
            w_new = w - lr * g
            
            # 4. If this parameter group requires weight decay, apply the penalty
            if apply_decay:
                # w = w - lr * weight_decay * w
                w_new -= lr * weight_decay * w
                
            updated_group.append(w_new)
            
        updated_p