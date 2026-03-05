from typing import List, Union

Number = Union[int, float]

def _minor(matrix: List[List[Number]], skip_row: int, skip_col: int) -> List[List[Number]]:
    """Return the minor matrix after removing skip_row and skip_col (0-based)."""
    return [
        [val for j, val in enumerate(row) if j != skip_col]
        for i, row in enumerate(matrix) if i != skip_row
    ]

def _determinant(matrix: List[List[Number]]) -> float:
    """Recursive determinant via Laplace expansion (cofactor expansion)."""
    n = len(matrix)
    if n == 0:
        raise ValueError("Empty matrix has no determinant.")
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square.")

    # Base cases
    if n == 1:
        return float(matrix[0][0])
    if n == 2:
        a, b = matrix[0]
        c, d = matrix[1]
        return float(a * d - b * c)

    # Expand along the first row (row 0)
    det = 0.0
    for j in range(n):
        sign = -1.0 if (j % 2 == 1) else 1.0  # (-1)^(0+j)