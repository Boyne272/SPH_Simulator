"""SPH simulation objects."""

from collections import Counter
from itertools import count
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from sph.utils import is_divisible_by, has_duplicates, neighbours, rectangle


@dataclass(eq=False)
class Particle(object):
    """SPH Simulation Particle."""

    sys: 'SysVals' = None  # TODO remove defualt value

    x: np.ndarray = np.zeros(2)  # current position (m)
    v: np.ndarray = np.zeros(2)  # current velocity (m/s)
    a: np.ndarray = np.zeros(2)  # current acceleration (m/s^2)

    D: float = 0.  # current rate of change of density (kg/m^3s)
    P: float = 0.  # current pressure of particle (Pa)
    m: float = 0.  # mass of the particle (kg)  # TODO validate me
    rho: float = 0. # density at particles position # TODO validate me

    bound: bool = False  # is a boundary (i.e. fixed) particle
    adj: list = None #field(default_factory=lambda:list())  # list of adjasent particles
    list_num: np.ndarray = None  # TODO change me to something more sensible

    id: int = field(default_factory=lambda:next(Particle.n_particles))  # identifier for this particle
    n_particles: ClassVar = count(0)  # counter for all particles

    def calc_index(self) -> np.ndarray:
        """Calculates particle's location in the search grid."""
        self.grid_cord = np.array(
            (self.x - self.sys.min_x) / (2.0 * self.sys.h),
            int
        )
        return self.grid_cord

    def __post_init__(self):
        """Validate the inputs."""
        assert self.x.shape == self.v.shape == self.a.shape == (2,), 'only 2d supported currently'
        self.calc_index()

    @property
    def csv_str(self) -> str:
        """Csv string for loading this particle: ID,R_x,R_y,V_x,V_y,a_x,a_y,m,D,P,Rho,Bound"""
        return ','.join(map(str, (
            self.id,
            self.x[0],
            self.x[1],
            self.v[0],
            self.v[1],
            self.a[0],
            self.a[1],
            self.m,
            self.D,
            self.P,
            self.rho,
            self.bound
        )))

    @classmethod
    def from_csv(string: str) -> "Particle":
        """The inverse of csv_str operation to make a particle from the csv string."""
        id_, x0, x1, v0, v1, a0, a1, m, d, p, rho, bound = string.split(',')
        return Particle(
            id=id_,
            x=np.array((x0, x1)),
            v=np.array((v0, v1)),
            a=np.array((a0, a1)),
            m=m,
            D=d,
            P=p,
            rho=rho,
            bound=bound
        )

@dataclass(eq=False)
class SysVals:
    """All the constants that create this system."""

    # system main parameters
    x_min: tuple = (0., 0.)        # lower left corner
    x_max: tuple = (1., 1.)        # upper right corner
    dx: float = 0.02               # initial particle spacing
    h_fac: float = 1.3             # bin half size constant (unitless)

    # normal system parameters
    t_curr: float = 0.0                   # current time of the system (s)
    mu: float = 0.001                     # viscosity (Pa s)
    rho0: float = 1000                    # initial particle density (kg/m^3)
    c0: float = 20                        # speed of sound in water (m/s)
    gamma: float = 7                      # stiffness value (dimensionless)
    interval_smooth: float = 15           # timesteps to smooth rho (dimensionless)
    interval_save: float = 15             # timesteps to save the state (dimensionless)
    CFL: float = 0.2                      # CFL constant (dimensionless)
    g: np.ndarray = 9.81 * np.array([0, -1])   # gravity value (m/s^2)
    P_fac: float = 1.05                   # scale for LJ reference pressure
    x_ref: float = 0.9                    # scale for LJ reference distance

    # random seed
    # seed: float = np.rand() # TODO add me

    # derived attributes in __post_init__
    h: float = 0.0
    lil_bit: float = 0.0
    B: float = 0.0
    w_fac1: float = 0.0
    w_fac2: float = 0.0
    P_ref: float = 0.0
    d_ref: float = 0.0
    min_x: np.array = None
    max_x: np.array = None

    def __post_init__(self):
        """Set all the dervied constants."""
        # cast values
        self.min_x = np.array(self.x_min, float)
        self.max_x = np.array(self.x_max, float)

        # determine_values
        self.h = self.dx*self.h_fac                   # bin half-size
        self.sr = 2 * self.h                           # search radius
        self.lil_bit = self.dx*0.01                   # to include upper limits
        self.B = self.rho0 * self.c0**2 / self.gamma  # pressure constant (Pa)
        self.w_fac1 = 10 / (7 * np.pi * self.h ** 2)  # constant often used
        self.w_fac2 = 10 / (7 * np.pi * self.h ** 3)  # constant often used
        self.P_ref = self.B*(self.P_fac**self.gamma - 1)  # boundary pressure to prevent leakages (Pa).
        self.d_ref = self.x_ref * self.dx             # distance boundary pressure (m)

        # validate TODO add me in
        # assert is_divisible_by(self.max_x - self.min_x, self.dx), 'dx must fit into range without remainder'

    def __eq__(self, other):
        """Relying on __repr__ is safer for array comparison."""
        return str(self) == str(other)

    def to_dict(self) -> dict:
        """Write this system as a dict that can be jsonified."""
        raise NotImplementedError()

    @classmethod
    def from_dict(dict) -> "SysVals":
        """Write this system as a dict that can be jsonified."""
        raise NotImplementedError()


class Grid:
    """A grid object that holds, sorts and finds particles efficently."""

    def __init__(self, sys: SysVals):
        """."""
        self.sys = sys
        self.particle_list = []

        self.grid_max = None
        self.search_grid = None

    def initialise_grid(self, func):
        """Initalise simulation grid.

        parameters:
            func: f(x: np.ndarray) -> bool, where bool is position in fluid

        """
        assert callable(func), 'func must be a function'

        # set internal points
        x_points = np.arange(self.sys.min_x[0], self.sys.max_x[0] + self.sys.lil_bit, self.sys.dx)
        y_points = np.arange(self.sys.min_x[1], self.sys.max_x[1] + self.sys.lil_bit, self.sys.dx)
        for x in x_points:
            for y in y_points:
                if func(x, y):
                    self.place_point(x, y, bound=False)

        self.add_boundaries()  # create the boundary points

        assert not has_duplicates([p.x for p in self.particle_list])

        # setup the search array (find size then create array)
        self.grid_max = np.array((self.sys.max_x-self.sys.min_x)/(2.0*self.sys.h)+1, int)
        self._grid_points = [
            (i, j)
            for i in range(self.grid_max[0])
            for j in range(self.grid_max[1])
        ]
        self.search_grid = np.empty(self.grid_max, object)

    def add_boundary_layer(self):
        """Add a single layer of boundary particles around the perimeter.

        Note: will also expand x_max and x_min to incorporate the new limits of the
        system.

        """
        self.sys.min_x -= self.sys.dx
        self.sys.max_x += self.sys.dx

        for x, y in rectangle(*self.sys.min_x, *self.sys.max_x, self.sys.dx):
            self.place_point(x, y, bound=1)
        # TODO use a 1, 0, -1 input funciton instead

    # TODO optimise everything below me

    def add_boundaries(self):
        """Add the boundary points to pad at least 2h around the edge."""
        # create the boundary points
        tmp_diff = 0
        print(f'# points {len(self.particle_list)}')

        # while tmp_diff < 2.0*self.sys.h:
        #     tmp_diff += self.sys.dx
        #     self.add_boundary_layer()

        while tmp_diff < 2.0*self.sys.h:
            tmp_diff += self.sys.dx
            tmp_min = self.sys.min_x - tmp_diff
            tmp_max = self.sys.max_x + tmp_diff

            # upper and lower rows
            for x in np.arange(tmp_min[0], tmp_max[0] + self.sys.lil_bit, self.sys.dx):
                self.place_point(x, tmp_min[1], bound=1)
                self.place_point(x, tmp_max[1], bound=1)

            # left and right (removing corners)
            tmp = np.arange(tmp_min[1], tmp_max[1] + self.sys.lil_bit, self.sys.dx)
            for i, y in enumerate(tmp):
                if i != 0 and i != len(tmp)-1:
                    self.place_point(tmp_min[0], y, bound=1)
                    self.place_point(tmp_max[0], y, bound=1)

        # # account for the boundary particle changing limits
        self.sys.min_x -= tmp_diff
        self.sys.max_x += tmp_diff

    def place_point(self, x: float, y: float, bound: bool = False):
        """Place particle at point given and assigns the particle attribute boundary.

        parameters:
            x: location of particle assuming positive to the right and negative to the left
            y: location of particle assuming positive up and negative down

        """
        self.particle_list.append(Particle(
            self.sys,
            np.array([x, y]),
            rho=self.sys.rho0,
            P=0.,
            bound=(1. if bound else 0.)
        ))

    def allocate_to_grid(self):
        """Allocate all the points to a grid to aid neighbour searching"""
        for i, j in self._grid_points:
            # reset the grid
            self.search_grid[i, j] = []

        for particle in self.particle_list:
            # calculate all particles positions in the grid
            i, j = particle.calc_index()
            self.search_grid[i, j].append(particle)


    def neighbour_iterate(self, part):
        """Find all the particles within 2h of the specified particle"""
        part.adj = []  # needs to be reseted every time it's called
        # TODO find a better place to put me
        # TODO find a better way to do the following loop
        for i in range(max(0, part.grid_cord[0] - 1),
                       min(part.grid_cord[0] + 2, self.grid_max[0])):
            for j in range(max(0, part.grid_cord[1] - 1),
                           min(part.grid_cord[1] + 2, self.grid_max[1])):
                for other_part in self.search_grid[i, j]:
                    if part is not other_part:
                        dn = part.x - other_part.x  # ####### use this later
                        dist = np.sqrt(np.sum(dn ** 2))
                        if dist < 2.0 * self.sys.h:
                            part.adj.append(other_part)
        return None

    def neighbour_iterate_half(self, part):
        """Find upper only particles within 2h of the specified particle
        part: class object
            particle from particles class
        """
        part.adj = []  # needs to be reseted every time it's called

        # pick the correct sencil points
        for i in range(max(0, part.grid_cord[0] - 1),
                       min(part.grid_cord[0] + 2, self.grid_max[0])):
            for j in range(max(0, part.grid_cord[1] - 1),
                           min(part.grid_cord[1] + 2, self.grid_max[1])):
                # in the row above
                if (j == part.grid_cord[1] + 1):
                    self.non_central_gridpoint(part, i, j)
                # if in the current row
                elif (j == part.grid_cord[1]):
                    # left point
                    if (i == part.grid_cord[0] - 1):
                        self.non_central_gridpoint(part, i, j)
                    # center point
                    elif (i == part.grid_cord[0]):
                        self.central_gridpoint(part, i, j)
        return None

    def non_central_gridpoint(self, part, i, j):
        """Find neighbouring grids of particle (excluding its own)
        part: class object
            particle from particles class
        i: index
            x-grid coordinate
        j: index
            y-grid coordinate
        """
        # for all particles in the grid
        for other_part in self.search_grid[i, j]:
            # add it to the adjasent list
            dn = part.x - other_part.x
            dist = np.sqrt(np.sum(dn ** 2))
            if dist < 2.0 * self.sys.h:
                part.adj.append(other_part)

    def central_gridpoint(self, part, i, j):
        """Find neighbouring grids of particle (excluding its own)
        part: class object
            particle from particles class
        i: index
            x-grid coordinate
        j: index
            y-grid coordinate
        """
        # for all particles in the grid
        for other_part in self.search_grid[i, j]:

            # if not the particle
            if part is not other_part:
                # if not below current particle
                if (other_part.id < part.id):
                    # add it to the adjasent list
                    dn = part.x - other_part.x
                    dist = np.sqrt(np.sum(dn ** 2))
                    if dist < 2.0 * self.sys.h:
                        part.adj.append(other_part)

    def plot_current_state(self):
        """
        Plots the current state of the system (i.e. where every particle is)
        in space.
        """
        x = np.array([p.x for p in self.grid.particle_list])
        bs = [p.bound for p in self.grid.particle_list]
        plt.scatter(x[:, 0], x[:, 1], c=bs)
        plt.gca().set(xlabel='x', ylabel='y', title='Current State')
