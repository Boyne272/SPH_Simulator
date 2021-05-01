"""Unit tests for grid.py."""

from numpy import ceil
import matplotlib.pyplot as plt
from pytest import mark

from sph.objects.system import System
from sph.objects.grid import Grid
from sph.utils import mpl_settings
from sph.functions import (
    center_step_wave,
    gaussian_wave,
    step_wave
)



def test_grid_populate():
    """Ensure the gird populates the expected number of particles."""
    system = System()
    grid = Grid(system, lambda x, y: y<0.5)
    assert len(grid.search_dict) == 22 * 22
    assert len(grid.particle_list) == 1886
    assert len([p for p in grid.particle_list if not p.bound]) == (0.5 * 1) / 0.02**2
    # assert len([p for p in grid.particle_list if p.bound]) == 636  # TODO find a way to calculate this


def test_grid_bins():
    """Ensure the grid bins as exected (dependent on `test_grid_populate`)."""
    system = System()
    grid = Grid(system, lambda x, y: y<0.5)
    grid.update_grid()
    grid_sizes = [len(list_) for list_ in grid.search_dict.values()]
    assert sum(grid_sizes) == len(grid.particle_list), 'all particles should live have been binned'
    assert min(grid_sizes) == 0, 'some grids should be empty'
    assert max(grid_sizes) == ceil(system.d_srch/system.dx)**2, 'there should be at least one maxed out grid'


@mark.parametrize('func', (
    center_step_wave,
    gaussian_wave,
    step_wave
))
def test_grid_functions(func):
    """Ensure a range of functions and intial setups all run and can be plotted."""
    system = System(min_x=(0, 0), max_x=[20, 10], dx=0.2)
    grid = Grid(system, gaussian_wave)
    mpl_settings()
    grid.plot()
    assert plt.gcf().get_axes(), 'a figure should have been plotted'
    print(f'{grid!r} successfully plotted')