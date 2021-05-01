"""Sample initial system funcitons for common simulations."""

import numpy as np


def step_wave(x, y) -> int:
    """The example probelm we were given to solve in the brief."""
    return int(
        0 <= y <= 2 or
        (0 <= x <= 3 and 0 <= y <= 5)
    )


def center_step_wave(x, y) -> int:
    """Same as the example problem but centerised."""
    return int(
        0 <= y <= 2 or
        (8 <= x <= 12 and 0 <= y <= 5)
    )


def gaussian_wave(x, y, cent=10., mag=3., std=1., base=2.) -> int:
    """A guassian wave."""
    num = (x - cent)**2
    den = 2*std**2
    boundary = base + mag * np.exp(-num/den)
    return y <= boundary
