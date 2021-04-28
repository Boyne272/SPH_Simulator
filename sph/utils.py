"""Vairous utilities."""

# plotting ------------------------------------------------

import matplotlib.pyplot as plt

def mpl_settings():
    """Nice Matplotlib settings."""
    plt.rc('axes', titlesize=20, labelsize=20)
    plt.rc('axes.formatter', limits=[-4, 4])
    plt.rc('ytick', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('lines', linewidth=1.5, markersize=7)
    plt.rc('figure', figsize=(9, 9))
    plt.rc('legend', fontsize=15)
mpl_settings()

# saving --------------------------------------------------


def csv_header(time: float, iteration: int = None):
    """Create a header for the particle data csv file."""
    return f'''# Created by team Southern on {datetime.now().strftime("%Y-%m-%d-%Hhr-%Mm")}
    # Simulation time {time}s {"(iteration %i)" % iteration if iteration else ""}
    # [#], [m], [m], [m/s], [m/s], [Pa], [Kg/(m^3)], [bool]
    ID,R_x,R_y,V_x,V_y,Pressure,Density,Boundary
    '''


# maths ---------------------------------------------------

import numpy as np

def is_divisible_by(arr: np.ndarray, scalar: float) -> bool:
    """Check the given array is divisible by the scalar in all dimensions."""
    tmp = (arr / scalar) - (arr / scalar).astype(int)
    return np.isclose(np.linalg.norm(tmp), 0)


# geometry ------------------------------------------------

def rectangle(x0: float, y0: float, x1: float, y1: float, gap: float):
    """Iterate a rectangle's cordinates perimiter with the given cordinates."""
    for x in np.arange(x0, x1 + gap/1000, gap):
        # gap/1000 includes corners if [x0, x1] is multiple of gap
        yield x, y0
        yield x, y1
    for y in np.arange(y0 + gap, y1 - gap + gap/1000, gap):
        # +- gap excludes the corners that were added above
        yield x0, y
        yield x1, y


# logic ---------------------------------------------------


def has_duplicates(values):
    """Check the given array is unique."""
    return len(np.unique(values, axis=0)) != len(values)


def neighbours(i: int, j: int, i_max: int, j_max: int):
    """Iterate all neighbouring cords."""
    left = max(0, i)
    right = min(i, i_max)
    lower = max(0, j)
    upper = min(j, j_max)
    for x in range(left, right):
        for y in range(lower, upper):
            yield x, y
