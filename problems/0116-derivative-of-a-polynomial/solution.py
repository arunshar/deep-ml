def poly_term_derivative(c: float, x: float, n: float) -> float:
    # If the exponent is 0, the derivative of a constant is 0
    if n == 0:
        return 0.0
    
    # Apply the power rule: c * n * x^(n-1)
    derivative_value = c * n * (x ** (n - 1))
    
    return float(derivative_value)