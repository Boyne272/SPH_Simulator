"""SPH simulation objects."""

from collections import Counter
from dataclasses import dataclass, field
from itertools import count
import json
from typing import Callable

import numpy as np

from sph.utils import is_divisible_by, has_duplicates, neighbours, rectangle
from sph.objects.particle import Particle


# def boundary_wrapper(func: Callable, system: System) -> Callable:
#     """Padd the return value so that areas outside the inner grid are boundaries."""
#     def inner(x: float, y: float):
#         if any((
#             x > system.x_inner_max[0],
#             y > system.x_inner_max[1],
#             x < system.x_inner_min[0],
#             y < system.x_inner_min[1]
#         )):
#             return -1
#         return func(x, y)
#     return inner


# class Grid:
#     """A grid object that holds, sorts and finds particles efficently."""

#     def __init__(self, sys: SysVals):
#         """."""
#         self.sys = sys
#         self.particle_list = []

#         self.grid_max = None
#         self.search_grid = None

#     def initialise_grid(self, func: Callable):
#         """Initalise simulation grid.

#         notes:
#             - the function will be wrapped such that values outside system.x_inner_min/max
#               are boundary particles

#         parameters:
#             func: f(x: float, y: float) -> int:
#                 return 1 indicates particle at that point
#                 return 0 indicates no particle at that point
#                 return -1 indicates boundary particle at that point

#         """
#         assert callable(func), 'func must be a function'
#         func = boundary_wrapper(func, self.system)

#         # set internal points
#         x_points = np.arange(self.sys.min_x[0], self.sys.max_x[0] + self.sys.lil_bit, self.sys.dx)
#         y_points = np.arange(self.sys.min_x[1], self.sys.max_x[1] + self.sys.lil_bit, self.sys.dx)
#         for x in x_points:
#             for y in y_points:
#                 if func(x, y):
#                     self.place_point(x, y, bound=False)

#         self.add_boundaries()  # create the boundary points

#         assert not has_duplicates([p.x for p in self.particle_list])

#         # setup the search array (find size then create array)
#         self.grid_max = np.array((self.sys.max_x-self.sys.min_x)/(2.0*self.sys.h)+1, int)
#         self._grid_points = [
#             (i, j)
#             for i in range(self.grid_max[0])
#             for j in range(self.grid_max[1])
#         ]
#         self.search_grid = np.empty(self.grid_max, object)

#     def add_boundary_layer(self):
#         """Add a single layer of boundary particles around the perimeter.

#         Note: will also expand x_max and x_min to incorporate the new limits of the
#         system.

#         """
#         self.sys.min_x -= self.sys.dx
#         self.sys.max_x += self.sys.dx

#         for x, y in rectangle(*self.sys.min_x, *self.sys.max_x, self.sys.dx):
#             self.place_point(x, y, bound=1)
#         # TODO use a 1, 0, -1 input funciton instead

#     # TODO optimise everything below me

#     def add_boundaries(self):
#         """Add the boundary points to pad at least 2h around the edge."""
#         # create the boundary points
#         tmp_diff = 0
#         print(f'# points {len(self.particle_list)}')

#         # while tmp_diff < 2.0*self.sys.h:
#         #     tmp_diff += self.sys.dx
#         #     self.add_boundary_layer()

#         while tmp_diff < 2.0*self.sys.h:
#             tmp_diff += self.sys.dx
#             tmp_min = self.sys.min_x - tmp_diff
#             tmp_max = self.sys.max_x + tmp_diff

#             # upper and lower rows
#             for x in np.arange(tmp_min[0], tmp_max[0] + self.sys.lil_bit, self.sys.dx):
#                 self.place_point(x, tmp_min[1], bound=1)
#                 self.place_point(x, tmp_max[1], bound=1)

#             # left and right (removing corners)
#             tmp = np.arange(tmp_min[1], tmp_max[1] + self.sys.lil_bit, self.sys.dx)
#             for i, y in enumerate(tmp):
#                 if i != 0 and i != len(tmp)-1:
#                     self.place_point(tmp_min[0], y, bound=1)
#                     self.place_point(tmp_max[0], y, bound=1)

#         # # account for the boundary particle changing limits
#         self.sys.min_x -= tmp_diff
#         self.sys.max_x += tmp_diff

#     def place_point(self, x: float, y: float, bound: bool = False):
#         """Place particle at point given and assigns the particle attribute boundary.

#         parameters:
#             x: location of particle assuming positive to the right and negative to the left
#             y: location of particle assuming positive up and negative down

#         """
#         self.particle_list.append(Particle(
#             self.sys,
#             np.array([x, y]),
#             rho=self.sys.rho0,
#             P=0.,
#             bound=(1. if bound else 0.)
#         ))

#     def allocate_to_grid(self):
#         """Allocate all the points to a grid."""
#         for i, j in self._grid_points:
#             self.search_grid[i, j] = []  # TODO make dict of lists # reset the grid

#         for particle in self.particle_list:
#             i, j = np.array(
#                 (particle.x - particle.sys.min_x) / (2.0 * particle.sys.h),
#                 int
#             )  # TODO if you change to particle.sys -> self.sys the world collapses ..... why
#             self.search_grid[i, j].append(particle)


#     def neighbour_iterate(self, part):
#         """Find all the particles within 2h of the specified particle"""
#         part.adj = []  # needs to be reseted every time it's called
#         # TODO find a better place to put me
#         # TODO find a better way to do the following loop
#         for i in range(max(0, part.grid_cord[0] - 1),
#                        min(part.grid_cord[0] + 2, self.grid_max[0])):
#             for j in range(max(0, part.grid_cord[1] - 1),
#                            min(part.grid_cord[1] + 2, self.grid_max[1])):
#                 for other_part in self.search_grid[i, j]:
#                     if part is not other_part:
#                         dn = part.x - other_part.x  # ####### use this later
#                         dist = np.sqrt(np.sum(dn ** 2))
#                         if dist < 2.0 * self.sys.h:
#                             part.adj.append(other_part)
#         return None

#     def neighbour_iterate_half(self, part):
#         """Find upper only particles within 2h of the specified particle
#         part: class object
#             particle from particles class
#         """
#         part.adj = []  # needs to be reseted every time it's called

#         # pick the correct sencil points
#         for i in range(max(0, part.grid_cord[0] - 1),
#                        min(part.grid_cord[0] + 2, self.grid_max[0])):
#             for j in range(max(0, part.grid_cord[1] - 1),
#                            min(part.grid_cord[1] + 2, self.grid_max[1])):
#                 # in the row above
#                 if (j == part.grid_cord[1] + 1):
#                     self.non_central_gridpoint(part, i, j)
#                 # if in the current row
#                 elif (j == part.grid_cord[1]):
#                     # left point
#                     if (i == part.grid_cord[0] - 1):
#                         self.non_central_gridpoint(part, i, j)
#                     # center point
#                     elif (i == part.grid_cord[0]):
#                         self.central_gridpoint(part, i, j)
#         return None

#     def non_central_gridpoint(self, part, i, j):
#         """Find neighbouring grids of particle (excluding its own)
#         part: class object
#             particle from particles class
#         i: index
#             x-grid coordinate
#         j: index
#             y-grid coordinate
#         """
#         # for all particles in the grid
#         for other_part in self.search_grid[i, j]:
#             # add it to the adjasent list
#             dn = part.x - other_part.x
#             dist = np.sqrt(np.sum(dn ** 2))
#             if dist < 2.0 * self.sys.h:
#                 part.adj.append(other_part)

#     def central_gridpoint(self, part, i, j):
#         """Find neighbouring grids of particle (excluding its own)
#         part: class object
#             particle from particles class
#         i: index
#             x-grid coordinate
#         j: index
#             y-grid coordinate
#         """
#         # for all particles in the grid
#         for other_part in self.search_grid[i, j]:

#             # if not the particle
#             if part is not other_part:
#                 # if not below current particle
#                 if (other_part.id < part.id):
#                     # add it to the adjasent list
#                     dn = part.x - other_part.x
#                     dist = np.sqrt(np.sum(dn ** 2))
#                     if dist < 2.0 * self.sys.h:
#                         part.adj.append(other_part)

#     def plot_current_state(self):
#         """
#         Plots the current state of the system (i.e. where every particle is)
#         in space.
#         """
#         x = np.array([p.x for p in self.grid.particle_list])
#         bs = [p.bound for p in self.grid.particle_list]
#         plt.scatter(x[:, 0], x[:, 1], c=bs)
#         plt.gca().set(xlabel='x', ylabel='y', title='Current State')
