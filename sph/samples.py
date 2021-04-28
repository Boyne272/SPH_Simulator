"""Sample initial system funcitons for common simulations."""


def step_wave(x, y) -> int:
    """The example probelm we were given to solve in the brief."""
    if 0 <= y <= 2 or (0 <= x <= 3 and 0 <= y <= 5):
        return 1
    else:
        return 0
