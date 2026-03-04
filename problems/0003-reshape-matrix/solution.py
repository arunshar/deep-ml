import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	r, c = new_shape

    # Basic validation of target shape
    if r < 0 or c < 0:
        return []

    # Handle empty matrix cases
    if not a:
        return [] if r * c != 0 else []

    # If there are rows but first row is empty, treat cols as 0
    if not a[0]:
        return [] if r * c != 0 else []

    m, n = len(a), len(a[0])

    # Optional: reject ragged matrices
    if any(len(row) != n for row in a):
        return []

    total = m * n
    if total != r * c:
        return []

    # Flatten in row-major order
    flat = [x for row in a for x in row]

    # Build reshaped matrix
    out = []
    idx = 0
    for _ in range(r):
        out.append(flat[idx: idx + c])
        idx += c

    return out