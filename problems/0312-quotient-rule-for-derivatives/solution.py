import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    # 1. Create polynomial objects from the coefficients
    g = np.poly1d(g_coeffs)
    h = np.poly1d(h_coeffs)
    
    # 2. Get the derivatives of the polynomials
    g_prime = g.deriv()
    h_prime = h.deriv()
    
    # 3. Evaluate the polynomials and their derivatives at point x
    gx = g(x)
    hx = h(x)
    gpx = g_prime(x)
    hpx = h_prime(x)
    
    # 4. Apply the quotient rule formula
    # f'(x) = (g'(x)h(x) - g(x)h'(x)) / (h(x)^2)
    numerator = (gpx * hx) - (gx * hpx)
    denominator = hx ** 2
    
    derivative_value = numerator / denominator
    
    return float(derivative_value)