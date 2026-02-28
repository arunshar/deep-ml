import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    # 1. Define derivatives for each function
    # 'square': x^2 -> 2x
    # 'sin': sin(x) -> cos(x)
    # 'exp': e^x -> e^x
    # 'log': ln(x) -> 1/x
    
    def get_val_and_grad(func_name, val):
        if func_name == 'square':
            return val**2, 2 * val
        elif func_name == 'sin':
            return np.sin(val), np.cos(val)
        elif func_name == 'exp':
            return np.exp(val), np.exp(val)
        elif func_name == 'log':
            return np.log(val), 1 / val
        return val, 1.0

    # 2. To apply chain rule: f(g(x))' = f'(g(x)) * g'(x)
    # We evaluate from right to left to keep track of the inner values
    current_x = x
    total_derivative = 1.0
    
    # functions are applied right to left, so reverse the list to process inside-out
    for func in reversed(functions):
        val, grad = get_val_and_grad(func, current_x)
        # Multiply the derivati