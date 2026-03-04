import math

def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    a, b = matrix[0][0], matrix[0][1]
    c, d = matrix[1][0], matrix[1][1]

    t = a + d               # trace
    det = a * d - b * c     # determinant

    disc = t * t - 4 * det  # discriminant

    # Clamp tiny negative due to floating point error
    if disc < 0 and disc > -1e-12:
        disc = 0.0

    sqrt_disc = math.sqrt(disc)

    lam1 = (t + sqrt_disc) / 2.0
    lam2 = (t - sqrt_disc) / 2.0

    return sorted([float(lam1), float(lam2)], reverse=True)