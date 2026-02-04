def matrix_dot_vector(a: list[list[int | float]], b: list[int | float]) -> list[int | float] | int:
    if not a or len(a[0]) != len(b):
        return -1

    result = []
    for row in a:
        dot = 0
        for x, y in zip(row, b):
            dot += x * y
        result.append(dot)

    return result
