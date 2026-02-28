import numpy as np

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
    if func_name == 'poly2d':
        # f(x,y) = x²y + xy²
        # ∂f/∂x = 2xy + y²
        # ∂f/∂y = x² + 2xy
        x, y = point
        df_dx = 2 * x * y + y**2
        df_dy = x**2 + 2 * x * y
        return (df_dx, df_dy)

    elif func_name == 'exp_sum':
        # f(x,y) = e^(x+y)
        # ∂f/∂x = e^(x+y)
        # ∂f/∂y = e^(x+y)
        x, y = point
        df_dx = np.exp(x + y)
        df_dy = np.exp(x + y)
        return (float(df_dx), float(df_dy))

    elif func_name == 'product_sin':
        # f(x,y) = x·sin(y)
        # ∂f/∂x = sin(y)
        # ∂f/∂y = x·cos(y)
        x, y = point
        df_dx = np.sin(y)
        df_dy = x * np.cos(y)
        return (float(df_dx), float(df_dy))

    elif func_name == 'poly3d':
        # f(x,y,z) = x²y + yz²
        # ∂f/∂x = 2xy
        # ∂f/∂y = x² + z²
        # ∂f/∂z = 2yz
        x, y, z = point
        df_dx = 2 *