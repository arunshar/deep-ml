def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if not matrix:
        return []

    # Optional: handle empty first row
    if not matrix[0]:
        return []

    m = len(matrix)
    n = len(matrix[0])

    # Optional (defensive) ragged-check:
    if any(len(row) != n for row in matrix):
        raise ValueError("Input must be a rectangular matrix")

    if mode == "row":
        return [sum(row) / len(row) for row in matrix]

    if mode == "column":
        return [sum(matrix[i][j] for i in range(m)) / m for j in range(n)]

    raise ValueError("mode must be 'row' or 'column'")