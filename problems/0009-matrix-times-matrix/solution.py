def matrixmul(a: list[list[int | float]],
              b: list[list[int | float]]) -> list[list[int | float]] | int:

    # Check if matrices are empty
    if not a or not b or not a[0] or not b[0]:
        return -1

    m = len(a)
    n = len(a[0])
    p = len(b)
    q = len(b[0])

    # Shape compatibility check
    if n != p:
        return -1

    # Initialize result matrix
    c = [[0 for _ in range(q)] for _ in range(m)]

    # Matrix multiplication
    for i in range(m):
        for j in range(q):
            for k in range(n):
                c[i][j] += a[i][k] * b[k][j]

    return c