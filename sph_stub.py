"""SPH class to find nearest neighbours..."""

from itertools import count

import numpy as np
import matplotlib.pyplot as plt


class SPH_main(object):
    """Primary SPH object"""

    def __init__(self):
        self.h = 0.0
        self.h_fac = 0.0
        self.dx = 0.0

        self.min_x = np.zeros(2)
        self.max_x = np.zeros(2)
        self.max_list = np.zeros(2, int)

        self.particle_list = []
        self.search_grid = np.empty((0, 0), object)

    def set_values(self):
        """Set simulation parameters."""

        self.min_x[:] = (0.0, 0.0)
        self.max_x[:] = (1.0, 1.0)
        self.dx = 0.02
        self.h_fac = 1.3
        self.h = self.dx*self.h_fac

    def initialise_grid(self):
        """Initalise simulation grid."""

        """Increases the minimum and maximum to account for the virtual particle padding that is required at boundaries"""
        self.min_x -= 2.0*self.h
        self.max_x += 2.0*self.h
        
        """Calculates the size of the array required to store the search array"""
        self.max_list = np.array((self.max_x-self.min_x)/(2.0*self.h)+1,
                                 int)
                                 
        self.search_grid = np.empty(self.max_list, object)

    def place_points(self, xmin, xmax):
        """Place points in a rectangle with a square spacing of size dx"""

        x = np.array(xmin)

        while x[0] <= xmax[0]:
            x[1] = xmin[1]
            while x[1] <= xmax[1]:
                particle = SPH_particle(self, x)
                particle.calc_index()
                self.particle_list.append(particle)
                x[1] += self.dx
            x[0] += self.dx

    def allocate_to_grid(self):
        """Allocate all the points to a grid in order to aid neighbour searching"""
        for i in range(self.max_list[0]):
            for j in range(self.max_list[1]):
                self.search_grid[i, j] = []

        for cnt in self.particle_list:
            self.search_grid[cnt.list_num[0], cnt.list_num[1]].append(cnt)

    def neighbour_iterate(self, part):
        """Find all the particles within 2h of the specified particle"""
        for i in range(max(0, part.list_num[0]-1),
                       min(part.list_num[0]+2, self.max_list[0])):
            for j in range(max(0, part.list_num[1]-1),
                           min(part.list_num[1]+2, self.max_list[1])):
                for other_part in self.search_grid[i, j]:
                    if part is not other_part:
                        dn = part.x-other_part.x
                        dist = np.sqrt(np.sum(dn**2))
                        if dist < 2.0*self.h:
                            """This is only for demonstration - Your code will need to do all the particle to particle calculations at this point rather than simply displaying the vector to the neighbour"""
                            print("id:", other_part.id, "dn:", dn)
                 
    def plot_current_state(self):
        """
        Plots the current state of the system (i.e. where every particle is)
        in space.
        """
        x = np.array([p.x for p in self.particle_list])
        plt.scatter(x[:,0], x[:, 1])
        plt.gca().set(xlabel='x', ylabel='y', title='Current State')


class SPH_particle(object):
    """Object containing all the properties for a single particle"""

    _ids = count(0)

    def __init__(self, main_data=None, x=np.zeros(2)):
        self.id = next(self._ids)
        self.main_data = main_data
        self.x = np.array(x)
        self.v = np.zeros(2)
        self.a = np.zeros(2)
        self.D = 0
        self.rho = 0.0
        self.P = 0.0
        self.m = 0.0

    def calc_index(self):
        """Calculates the 2D integer index for the particle's location in the search grid"""
        self.list_num = np.array((self.x-self.main_data.min_x) /
                                 (2.0*self.main_data.h), int)

if __name__ == '__main__':
    """Create a single object of the main SPH type"""
    domain = SPH_main()
    
    """Calls the function that sets the simulation parameters"""
    domain.set_values()
    """Initialises the search grid""" 
    domain.initialise_grid()
    
    """Places particles in a grid over the entire domain - In your code you will need to place the fluid particles in only the appropriate locations"""
    domain.place_points(domain.min_x, domain.max_x)
    
    """This is only for demonstration only - In your code these functions will need to be inside the simulation loop"""
    """This function needs to be called at each time step (or twice a time step if a second order time-stepping scheme is used)"""
    domain.allocate_to_grid()
    """This example is only finding the neighbours for a single partle - this will need to be inside the simulation loop and will need to be called for every particle"""
    domain.neighbour_iterate(domain.particle_list[100])
