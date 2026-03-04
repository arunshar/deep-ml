def scalar_multiply(matrix: list[list[int | float]], scalar: int | float) -> list[list[int | float]]:
    # Return a new matrix where each element is multiplied by the scalar
    return [[x * scalar for x in row] for row in matrix]