def find_treasure(start_x: float) -> float:
    # f(x) = x^4 - 3x^3 + 2
    # f'(x) = x^2(4x - 9) => stationary points at x=0 and x=9/4
    # x=0 is not a minimum; x=9/4 is the (global) minimum.
    return 9.0 / 4.0