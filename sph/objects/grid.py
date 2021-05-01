"""Grid class for inital setup and binning/pairing paricles."""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from sph.objects.system import System
from sph.objects.particle import Particle
from sph.utils import is_divisible_by, has_duplicates, neighbours, rectangle


def boundary_wrapper(func: Callable, system: System) -> Callable:
    """Padd the return value so that areas outside the inner grid are boundaries."""
    def inner(x: float, y: float):
        if any((
            x > system.x_inner_max[0],
            y > system.x_inner_max[1],
            x < system.x_inner_min[0],
            y < system.x_inner_min[1]
        )):
            return -1
        return func(x, y)
    return inner


class Grid:
    """A grid that inialises, stores and sorts Particles."""

    def __init__(self, system: System, func: callable):
        """."""
        self.sys = system           # parent system
        self.particle_list = []     # list of all particles present
        self._populate(func)

        x_range = system.x_max - system.x_min
        self.grid_lims = np.ceil(x_range / system.d_srch)
        self.search_dict = {
            (i, j): []
            for i in np.arange(self.grid_lims[0], dtype=int)
            for j in np.arange(self.grid_lims[1], dtype=int)
        }
        # TODO distinct particles, bound and unbound lists

    @property
    def x_array(self) -> np.ndarray:
        """A 2d matrix of all current particle positions."""
        # TODO find an elegant way to cache me (but note when I change)
        return np.vstack(tuple(p.x for p in self.particle_list))

    def __repr__(self) -> str:
        """Print out the assoicated system and the number of particles."""
        return f'Grid ({self.grid_lims}) with {len(self.particle_list)} particles in system {self.sys.parameter_hash[:5]}'

    def _place_point(self, x: float, y: float, bound: int = 0):
        """Place particle at point given and assigns the particle attribute boundary.

        parameters:
            x: location of particle (+ve to the right)
            y: location of particle (+ve upwards)
            bound: 1 if position is fixed

        """
        self.particle_list.append(Particle(
            np.array([x, y]),
            rho=self.sys.rho0,
            m=self.sys.dx**2*self.sys.rho0,
            P=0.,
            bound=bound
        ))

    def _populate(self, func):
        """Setup particles in the grid.

        parameters:
            func: f(x: np.ndarray) -> bool, where bool is position in fluid

        """
        assert callable(func), 'func must be a function'
        func = boundary_wrapper(func, self.sys)

        for x in np.arange(*self.sys.x_range, self.sys.dx):
            for y in np.arange(*self.sys.y_range, self.sys.dx):
                f_val = func(x, y)
                if f_val == 1:
                    self._place_point(x, y, bound=0)
                elif f_val == -1:
                    self._place_point(x, y, bound=-1)

        assert not has_duplicates([p.x for p in self.particle_list]), 'ensure correct placement'

    def update_grid(self):
        """Allocate all the points to their bin in the grid and clear the adj lists."""
        for list_ in self.search_dict.values():
            list_.clear()  # reset the grid

        for particle in self.particle_list:
            i, j = (particle.x - self.sys.x_min) / self.sys.d_srch
            self.search_dict[int(i), int(j)].append(particle)  # index error means particle leaked
            particle._adj.clear()  # forget previous neighbours

    def plot(self, **kwargs):
        """Plots the current positions of all particles."""
        kwargs = {
            'c': [p.bound for p in self.particle_list],
            'cmap': 'Dark2_r',
            **kwargs
        }
        plt.scatter(self.x_array[:, 0], self.x_array[:, 1], **kwargs)
        plt.gca().set(xlabel='x', ylabel='y', title=f'System {self.sys.parameter_hash[:5]} @{self.sys.t_curr}s')

    def find_neighbours():
        """."""
        # TODO make me super fancy optimised


    # def neighbour_iterate(self, part):
    #     """Find all the particles within 2h of the specified particle"""
    #     part.adj = []  # needs to be reseted every time it's called
    #     # TODO find a better place to put me
    #     # TODO find a better way to do the following loop
    #     for i in range(max(0, part.grid_cord[0] - 1),
    #                    min(part.grid_cord[0] + 2, self.grid_max[0])):
    #         for j in range(max(0, part.grid_cord[1] - 1),
    #                        min(part.grid_cord[1] + 2, self.grid_max[1])):
    #             for other_part in self.search_grid[i, j]:
    #                 if part is not other_part:
    #                     dn = part.x - other_part.x  # ####### use this later
    #                     dist = np.sqrt(np.sum(dn ** 2))
    #                     if dist < 2.0 * self.sys.h:
    #                         part.adj.append(other_part)
    #     return None

    # def neighbour_iterate_half(self, part):
    #     """Find upper only particles within 2h of the specified particle
    #     part: class object
    #         particle from particles class
    #     """
    #     part.adj = []  # needs to be reseted every time it's called

    #     # pick the correct sencil points
    #     for i in range(max(0, part.grid_cord[0] - 1),
    #                    min(part.grid_cord[0] + 2, self.grid_max[0])):
    #         for j in range(max(0, part.grid_cord[1] - 1),
    #                        min(part.grid_cord[1] + 2, self.grid_max[1])):
    #             # in the row above
    #             if (j == part.grid_cord[1] + 1):
    #                 self.non_central_gridpoint(part, i, j)
    #             # if in the current row
    #             elif (j == part.grid_cord[1]):
    #                 # left point
    #                 if (i == part.grid_cord[0] - 1):
    #                     self.non_central_gridpoint(part, i, j)
    #                 # center point
    #                 elif (i == part.grid_cord[0]):
    #                     self.central_gridpoint(part, i, j)
    #     return None

    # def non_central_gridpoint(self, part, i, j):
    #     """Find neighbouring grids of particle (excluding its own)
    #     part: class object
    #         particle from particles class
    #     i: index
    #         x-grid coordinate
    #     j: index
    #         y-grid coordinate
    #     """
    #     # for all particles in the grid
    #     for other_part in self.search_grid[i, j]:
    #         # add it to the adjasent list
    #         dn = part.x - other_part.x
    #         dist = np.sqrt(np.sum(dn ** 2))
    #         if dist < 2.0 * self.sys.h:
    #             part.adj.append(other_part)

    # def central_gridpoint(self, part, i, j):
    #     """Find neighbouring grids of particle (excluding its own)
    #     part: class object
    #         particle from particles class
    #     i: index
    #         x-grid coordinate
    #     j: index
    #         y-grid coordinate
    #     """
    #     # for all particles in the grid
    #     for other_part in self.search_grid[i, j]:

    #         # if not the particle
    #         if part is not other_part:
    #             # if not below current particle
    #             if (other_part.id < part.id):
    #                 # add it to the adjasent list
    #                 dn = part.x - other_part.x
    #                 dist = np.sqrt(np.sum(dn ** 2))
    #                 if dist < 2.0 * self.sys.h:
    #                     part.adj.append(other_part)
