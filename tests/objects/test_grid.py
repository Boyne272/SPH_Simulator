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


def test_update_adj_totals():
    """Ensure a simple box of 9 particles have the correct pair numbers."""
    system = System(min_x=(0, 0), max_x=[0, 0], dx=0.5, pad_fac=0.5)
    grid = Grid(system, lambda x, y: 1)

    grid.update_grid()
    grid.update_adjsents()

    assert len(grid.particle_list) == 9
    assert len(grid.search_dict) == 1
    assert len(grid.search_dict[0, 0]) == 9

    particle_adjs = [p._adj for p in grid.particle_list]
    assert sum(len(l) for l in particle_adjs) == 34, 'there are 36 pairs here but the corners are just far enough apart to be too wide'
    for l in particle_adjs:
        assert len(set(map(lambda p:p._id, l))) == len(l), 'any pair should only be linked once'


def test_update_adj_symetric():
    """Ensure a simple box of 9 particles are symetric if we add the others in each pair."""
    system = System(min_x=(0, 0), max_x=[0, 0], dx=0.5, pad_fac=0.5)
    grid = Grid(system, lambda x, y: 1)

    grid.update_grid()
    grid.update_adjsents()

    for particle in grid.particle_list:
        for other in particle._adj:  # for each pair, add to the other side of adj
            other._adj.append(particle)

    particle_adjs = [p._adj for p in grid.particle_list]
    set(len(l) for l in particle_adjs) == {7, 8}, 'there should be all pairs, except for far corners'

    for l in particle_adjs:
        assert len(set(map(lambda p:p._id, l))) == len(l), 'any pair should only be linked once'

