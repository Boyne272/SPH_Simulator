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
        self.mu = 0.0
        self.rho0 = 0.0
        self.c0 = 0.0
        self.t_curr = 0.0
        self.gamma = 0.0
        self.interval_smooth = 0
        self.interval_save = 0
        self.CFL = 0
        self.B = 0.0
        self.g = 0.0
        self.file = None

        self.min_x = np.zeros(2)
        self.max_x = np.zeros(2)
        self.max_list = np.zeros(2, int)

        self.particle_list = []
        self.search_grid = np.empty((0, 0), object)


    def set_values(self):
        """Set simulation parameters."""

        self.min_x[:] = (0.0, 0.0)
        self.max_x[:] = (1.0, 1.0)                                 # insert units
        self.dx = 0.02                                             # insert units
        self.h_fac = 1.3
        self.h = self.dx*self.h_fac                                # bin half-size
        self.mu = 0.001                                            # viscosity (Pa s)
        self.rho0 = 1000                                           # initial particle density (kg/m^3)
        self.c0 = 20                                               # speed of sound in water (m/s)
        self.t_curr = 0.0                                          # current time of the system (s)
        self.gamma = 7                                             # stiffness value, dimensionless
        self.interval_smooth = 15                                  # number of timesteps to which smooth rho
        self.interval_save = 15                                    # number of timesteps to which save the current state
        self.CFL = 0.2                                             # CFL constant, dimensionless
        self.B = self.rho0 * self.c0**2 / self.gamma               # pressure constant (Pa)
        self.g = 9.81 * np.array([0, -1])                          # gravity value (m/s^2)


    def initialise_grid(self):
        """Initalise simulation grid."""
        # account for the virtual particle padding at boundaries
        self.min_x -= 2.0*self.h
        self.max_x += 2.0*self.h

        # Calculates the size of search array
        self.max_list = np.array((self.max_x-self.min_x)/(2.0*self.h)+1, int)
        # Create the search array
        self.search_grid = np.empty(self.max_list, object)


    def place_points(self, xmin, xmax):
        """Place points in a rectangle with a square spacing of size dx"""

        x = np.array(xmin)

        while x[0] <= xmax[0]:
            x[1] = xmin[1]
            while x[1] <= xmax[1]:
                # create particle object and assign index
                particle = SPH_particle(self, x)
                particle.calc_index()

                # intiialise physical paramteres of particles
                particle.rho = self.rho0
                particle.m = self.dx**2 * self.rho0
                particle.P = self.B

                # append particle object to list of particles
                self.particle_list.append(particle)
                x[1] += self.dx
            x[0] += self.dx


    def allocate_to_grid(self):
        """Allocate all the points to a grid to aid neighbour searching"""
        for i in range(self.max_list[0]):
            for j in range(self.max_list[1]):
                self.search_grid[i, j] = []

        for cnt in self.particle_list:
            self.search_grid[cnt.list_num[0], cnt.list_num[1]].append(cnt)


    def neighbour_iterate(self, part):
        """Find all the particles within 2h of the specified particle"""
        part.ajc = []
        for i in range(max(0, part.list_num[0]-1),
                       min(part.list_num[0]+2, self.max_list[0])):
            for j in range(max(0, part.list_num[1]-1),
                           min(part.list_num[1]+2, self.max_list[1])):
                for other_part in self.search_grid[i, j]:
                    if part is not other_part:
                        dn = part.x-other_part.x  # ########### use this later
                        dist = np.sqrt(np.sum(dn**2))
                        if dist < 2.0*self.h:
                            part.ajc.append(other_part)

        return None


    def plot_current_state(self):
        """
        Plots the current state of the system (i.e. where every particle is)
        in space.
        """
        x = np.array([p.x for p in self.particle_list])
        plt.scatter(x[:, 0], x[:, 1])
        plt.gca().set(xlabel='x', ylabel='y', title='Current State')


    def W(self, p_i, p_j_list):
        """
        :param p_i: (object) position of particle where calculations are being performed
        :param p_j_list: (list of objects) position of particles influencing particle i
        :return: (np array) smoothing factor for particle i being affected by particles j
        """
        xi = p_i.x
        xj = np.array([p.x for p in p_j_list])
        r = xi - xj
        j_list = np.sqrt(np.sum(r**2, axis=1))/ self.h
        assert ((j_list >= 0).all()), "q must be a positive value"

        w_fac = 10 / (7 * np.pi * self.h ** 2)
        for i, q in enumerate(j_list):
            if 0 <= q < 1:
                j_list[i] = w_fac*(1 - 1.5*q**2 + 0.75*q**3)
            elif 1 <= q <= 2:
                j_list[i] = w_fac*(0.25 * (2 - q)**3)
            else:
                j_list[i] = 0
        return np.array(j_list)


    def dW(self, p_i, p_j_list):
        """
        :param p_i: (object) position of particle where calculations are being performed
        :param p_j_list: (list of objects) position of particles influencing particle i
        :return: (np array) derivative of smoothing factor for particle i being affected by particles j
        """
        xi = p_i.x
        xj = np.array([p.x for p in p_j_list])
        r = xi - xj
        j_list = np.sqrt(np.sum(r**2, axis=1))/ self.h
        assert ((j_list >= 0).all()), "q must be a positive value"

        w_fac = 10 / (7 * np.pi * self.h ** 3)
        for i, q in enumerate(j_list):
            if 0 <= q < 1:
                j_list[i] = w_fac * (-3*q + (9/4)*q**2)
            elif 1 <= q <= 2:
                j_list[i] = w_fac * (-(3/4)*(2-q)**2)
            else:
                j_list[i] = 0
        return np.array(j_list)


    def rho_smoothing(self, p_i, p_j_list):
        """
        :param p_i: (object) position of particle where calculations are being performed
        :param p_j_list: (list of objects) position of particles influencing particle i
        :return: (np array) smoothed density of particle i
        """
        assert(p_i in p_j_list) , "must include particle i in this calculation"
        w_list = self.W(p_i, p_j_list)
        p_j_rho = np.array([p.rho for p in p_j_list])
        assert((p_j_rho > 0).all()), "density must be always positive"
        rho = np.sum(w_list) / np.sum(w_list / p_j_rho)
        return rho


    def timestepping(self, tf):
        """Timesteps the physical problem with a set dt until user-specified time is reached"""
        dt = 0.1 * self.h / self.c0
        t = 0
        assert (tf >= dt), "time to short to resolve problem"

        count = 0
        while t <= tf:
            print("Iteration %g..."%(count + 1))

            # find all the derivatives for each particle
            for i, p_i in enumerate(self.particle_list):
                # create list of neighbours for particle i
                self.neighbour_iterate(p_i)

                # calculate smoothing contribution from all neighbouring particles
                dW_i = self.dW(p_i, p_i.ajc)

                # calculate acceleration and rate of change of density
                for j, p_j in enumerate(p_i.ajc):
                    r_vec = p_i.x - p_j.x
                    r_mod = np.sqrt(np.sum(r_vec ** 2))
                    e_ij = r_vec / r_mod
                    v_ij = p_i.v - p_j.v

                    a = self.g
                    a -= p_j.m * (p_i.P / p_i.rho ** 2 + p_j.P / p_j.rho ** 2) * dW_i[j] * e_ij
                    a += self.mu * p_j.m * (1/p_i.rho**2 + 1/p_j.rho**2)*dW_i[j]*v_ij/ r_mod
                    p_i.a = a

                    p_i.D = p_j.m * dW_i[j] * (v_ij[0]*e_ij[0] + v_ij[1]*e_ij[1])


            # updating each particles values
            for i, p_i in enumerate(self.particle_list):
                # update position -- needs to be updated before new velocity is computed
                p_i.x = p_i.x + dt * p_i.v

                # update velocity
                p_i.v = p_i.v + dt * p_i.a

                # update density, smooths if count is a multiple of smoothing
                p_i.rho = p_i.rho + dt * p_i.D
                if count%self.interval_smooth == 0:
                    p_j_list = p_i.ajc[:]
                    p_j_list.append(p_i)
                    p_i.rho = self.rho_smoothing(p_i, p_j_list)

                # update pressure
                p_i.P = self.B * ((p_i.rho/self.rho0)**self.gamma - 1)

                # update particle indices
                p_i.calc_index()

            # re-allocate particles to grid
            self.allocate_to_grid()

            count += 1
            t += dt

        return None


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
        self.adj = []

    def calc_index(self):
        """
        Calculates the 2D integer index for the particle's
        location in the search grid
        """
        self.list_num = np.array((self.x-self.main_data.min_x) /
                                 (2.0*self.main_data.h), int)



if __name__ == '__main__':
    """Create a single object of the main SPH type"""
    domain = SPH_main()

    """Calls the function that sets the simulation parameters"""
    domain.set_values()
    """Initialises the search grid"""
    domain.initialise_grid()

    """
    Places particles in a grid over the entire domain - In your code you
    will need to place the fluid particles in only the appropriate locations
    """
    domain.place_points(domain.min_x, domain.max_x)

    """This is only for demonstration only - In your code these functions
    will need to be inside the simulation loop"""
    """This function needs to be called at each time step
    (or twice a time step if a second order time-stepping scheme is used)"""
    domain.allocate_to_grid()
    """This example is only finding the neighbours for a single partle
    - this will need to be inside the simulation loop and will need to be
    called for every particle"""
    # domain.neighbour_iterate(domain.particle_list[100])

    domain.timestepping(tf=2e-4)
