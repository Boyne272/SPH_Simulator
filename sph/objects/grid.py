"""Grid class for inital setup and binning/pairing paricles."""

from datetime import datetime
from typing import Callable, Iterable, List, Tuple

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

    def __init__(self, system: System, func: callable = None, init_csv: str = None):
        """."""  # TODO add me
        self.sys = system

        # setup particles
        self.particle_list: List[Particle] = []
        self.particle_list_bound: List[Particle] = []
        self.particle_list_free: List[Particle] = []
        if func:
            self._populate_from_func(func)
        elif csv:
            self._populate_from_csv(init_csv)
        else:
            raise ValueError('Either a function or a initial configuration is need to populate')

        # setup bin grid
        x_range = system.x_max - system.x_min
        self.grid_lims = np.ceil(x_range / system.d_srch)
        self.search_dict = {
            (i, j): []
            for i in np.arange(self.grid_lims[0], dtype=int)
            for j in np.arange(self.grid_lims[1], dtype=int)
        }
        self.walls = self._calcualate_walls()

    def __repr__(self) -> str:
        """Print out the assoicated system and the number of particles."""
        return f'Grid ({self.grid_lims}) with {len(self.particle_list)} particles in system {self.sys.parameter_hash[:5]}'

    # particle populating ---------------------------------

    def _place_point(self, x: float, y: float, bound: int = 0):
        """Place particle at point given and assigns the particle attribute boundary.

        parameters:
            x: location of particle (+ve to the right)
            y: location of particle (+ve upwards)
            bound: 1 if position is fixed

        """
        particle = Particle(
            np.array([x, y]),
            # v zero (start stationary)
            # a and D zero (start unchanging)
            P=0.,  # P zero (as rho = sys.rho0 -> uniform density)
            m=self.sys.dx**2*self.sys.rho0,
            rho=self.sys.rho0,
            bound=bound
        )
        if not bound:
            self.particle_list_free.append(particle)
        else:
            self.particle_list_bound.append(particle)
        self.particle_list.append(particle)

    def _populate_from_func(self, func):
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

    def _populate_from_csv(self, init_csv: str):
        """."""
        # TODO implement me
        raise NotImplementedError()

    # particle positioning --------------------------------

    def update_grid(self):
        """Allocate all the points to their bin in the grid and clear the adj lists."""
        for list_ in self.search_dict.values():
            list_.clear()  # reset the grid

        for particle in self.particle_list:
            i, j = (particle.x - self.sys.x_min) / self.sys.d_srch
            self.search_dict[int(i), int(j)].append(particle)  # index error means particle leaked
            particle._adj.clear()  # forget previous neighbours

    def update_adjs(self):
        """."""  # TODO make me
        for i, j in self.search_dict:
            # get all particles in this and adjasent grids
            relevant = self.search_dict[i, j].copy()
            for x, y in neighbours(i, j, *self.grid_lims):
                relevant.extend(self.search_dict[x, y])  # pragma: no cover

            for particle in self.search_dict[i, j]:
                for other in relevant:
                    if (particle._id > other._id) and np.linalg.norm(particle.x - other.x) < self.sys.d_srch:
                        particle._adj.append(other)

    # def update_adjs(self):
    #     # TODO make me super fancy optimised

    @property
    def x_array(self) -> np.ndarray:
        """A 2d matrix of all current particle positions."""
        # TODO find an elegant way to cache me (but note when I change)
        # TODO use me more
        return np.vstack(tuple(p.x for p in self.particle_list))

    def _get_grid_corners(self, i, j) -> Iterable[np.ndarray]:
        """For the given bin return the cordinates of all four corners."""
        lower_left = self.sys.x_min + self.sys.d_srch * np.array([i, j])
        yield lower_left
        yield lower_left + self.sys.d_srch*np.array([0., 1.])
        yield lower_left + self.sys.d_srch*np.array([1., 1.])
        yield lower_left + self.sys.d_srch*np.array([1., 0.])

    def _calcualate_walls(self) -> List[Tuple[int, int]]:
        """Return the index's of every bins within d_srch of each wall."""
        walls = {
            'left': set(),
            'upper': set(),
            'lower': set(),
            'right': set(),
        }
        x_left, y_lower = self.sys.x_inner_min
        x_right, y_upper = self.sys.x_inner_max
        for i, j in self.search_dict:
            for x, y in self._get_grid_corners(i, j):
                if np.abs(x - x_left) < self.sys.d_srch:
                    walls['left'].add((i, j))
                if np.abs(x - x_right) < self.sys.d_srch:
                    walls['right'].add((i, j))
                if np.abs(y - y_lower) < self.sys.d_srch:
                    walls['lower'].add((i, j))
                if np.abs(y - y_upper) < self.sys.d_srch:
                    walls['upper'].add((i, j))
        return walls

    # visulisation ----------------------------------------

    def plot(self, **kwargs):
        """Plots the current positions of all particles."""
        kwargs = {
            'c': [p.bound for p in self.particle_list],
            'cmap': 'Dark2_r',
            **kwargs
        }
        plt.scatter(self.x_array[:, 0], self.x_array[:, 1], **kwargs)
        plt.gca().set(xlabel='x', ylabel='y', title=f'System {self.sys.parameter_hash[:5]} @{self.sys.t_curr}s')

    # saving ----------------------------------------------

    @property
    def particle_csv(self) -> str:
        """Return the current particle configurations as a csv."""
        lines = [
            f'# Created @{datetime.now().strftime("%c")} by team Southern',
            f'# {self.sys.summary}',
            f'# {self!r}',
            Particle.csv_header
        ]
        for particle in self.particle_list:
            lines.append(particle.csv)
        return '\n'.join(lines)
