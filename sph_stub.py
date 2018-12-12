from itertools import count

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import pickle as pi


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
        self.max_x[:] = (1.0, 1.0)                        # insert units
        self.dx = 0.02                                    # insert units
        self.h_fac = 1.3
        self.h = self.dx*self.h_fac                       # bin half-size
        self.mu = 0.001                                   # viscosity (Pa s)
        self.rho0 = 1000                                  # initial particle density (kg/m^3)
        self.c0 = 20                                      # speed of sound in water (m/s)
        self.t_curr = 0.0                                 # current time of the system (s)
        self.gamma = 7                                    # stiffness value, dimensionless
        self.interval_smooth = 15                         # number of timesteps to which smooth rho
        self.interval_save = 15                           # number of timesteps to which save the current state
        self.CFL = 0.2                                    # CFL constant, dimensionless
        self.B = self.rho0 * self.c0**2 / self.gamma      # pressure constant (Pa)
        self.g = 9.81 * np.array([0, -1])                 # gravity value (m/s^2)
        
        self.w_fac1 = 10 / (7 * np.pi * self.h ** 2)
        self.w_fac2 = 10 / (7 * np.pi * self.h ** 3)

    def initialise_grid(self):
        """Initalise simulation grid."""
        # account for the virtual particle padding at boundaries
        self.min_x -= 2.0*self.h
        self.max_x += 2.0*self.h

        # Calculates the size of search array
        self.max_list = np.array((self.max_x-self.min_x)/(2.0*self.h)+1, int)
        # Create the search array
        self.search_grid = np.empty(self.max_list, object)

    def place_points(self, xmin, xmax, bound=0):
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
                particle.P = 0.
                particle.bound = bound

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
        part.adj = []          # needs to be reseted every time it's called
        for i in range(max(0, part.list_num[0]-1),
                       min(part.list_num[0]+2, self.max_list[0])):
            for j in range(max(0, part.list_num[1]-1),
                           min(part.list_num[1]+2, self.max_list[1])):
                for other_part in self.search_grid[i, j]:
                    if part is not other_part:
                        dn = part.x-other_part.x  # ########### use this later
                        dist = np.sqrt(np.sum(dn**2))
                        if dist < 2.0*self.h:
                            part.adj.append(other_part)

        return None

    def plot_current_state(self):
        """
        Plots the current state of the system (i.e. where every particle is)
        in space.
        """
        x = np.array([p.x for p in self.particle_list])
        bs = [p.bound for p in self.particle_list]
        plt.scatter(x[:, 0], x[:, 1], c=bs)
        plt.gca().set(xlabel='x', ylabel='y', title='Current State')

    def W(self, p_i, p_j_list):
        """
        :param p_i: (object) position of particle where
                    calculations are being performed
        :param p_j_list: (list of objects) position of particles
                          influencing particle i
        :return: (np array) smoothing factor for particle i being
                  affected by particles j
        """
        xi = p_i.x
        xj = np.array([p.x for p in p_j_list])
        r = xi - xj
        j_list = np.sqrt(np.sum(r**2, axis=1)) / self.h
        assert ((j_list >= 0).all()), "q must be a positive value"

        for i, q in enumerate(j_list):
            if 0 <= q < 1:
                j_list[i] = self.w_fac1*(1 - 1.5*q**2 + 0.75*q**3)
            elif 1 <= q <= 2:
                j_list[i] = self.w_fac1*(0.25 * (2 - q)**3)
            else:
                j_list[i] = 0
        return np.array(j_list)

    def dW(self, p_i, p_j_list):
        """
        :param p_i: (object) position of particle where calculations are
                    being performed
        :param p_j_list: (list of objects) position of particles influencing
                         particle i
        :return: (np array) derivative of smoothing factor for particle i
                    being affected by particles j
        """
        xi = p_i.x
        xj = np.array([p.x for p in p_j_list])
        r = xi - xj
        j_list = np.sqrt(np.sum(r**2, axis=1)) / self.h
        assert ((j_list >= 0).all()), "q must be a positive value"

        for i, q in enumerate(j_list):
            if 0 <= q < 1:
                j_list[i] = self.w_fac2 * (-3*q + (9/4)*q**2)
            elif 1 <= q <= 2:
                j_list[i] = self.w_fac2 * (-(3/4)*(2-q)**2)
            else:
                j_list[i] = 0
        return np.array(j_list)

    def rho_smoothing(self, p_i, p_j_list):
        """
        :param p_i: (object) position of particle where calculations are
                    being performed
        :param p_j_list: (list of objects) position of particles influencing
                          particle i
        :return: (np array) smoothed density of particle i
        """
        assert(p_i in p_j_list), "must include particle i in this calculation"
        w_list = self.W(p_i, p_j_list)
        p_j_rho = np.array([p.rho for p in p_j_list])
        #assert((p_j_rho > 0).all()), "density must be always positive"
        rho = np.sum(w_list) / np.sum(w_list / p_j_rho)
        return rho

    def timestepping(self, tf):
        """
        Timesteps the physical problem with a set dt until user-specified
        time is reached
        """
        dt = 0.1 * self.h / self.c0
        v_ij_max = 0
#        a_max = 0
#        rho_max_condition = 0
        assert (tf >= dt), "time to short to resolve problem, dt=%.4f" % dt

        count = 0
        while self.t_curr <= tf:
            sys.stdout.write('\rTime: %.3f' % self.t_curr)
            sys.stdout.flush()

            # find all the derivatives for each particle
            for i, p_i in enumerate(self.particle_list.copy()):
                # create list of neighbours for particle i
                self.neighbour_iterate(p_i)

                # calculate smoothing from all neighbouring particles
                dW_i = self.dW(p_i, p_i.adj)

                # calculate rate of change
                p_i.a = self.g
                p_i.D = 0
                for j, p_j in enumerate(p_i.adj.copy()):
                    r_vec = p_i.x - p_j.x
                    r_mod = np.sqrt(np.sum(r_vec ** 2))
                    e_ij = r_vec / r_mod
                    v_ij = p_i.v - p_j.v
                    p_i.a = p_i.a - (p_j.m * (p_i.P / p_i.rho ** 2 +
                                     p_j.P / p_j.rho ** 2) * dW_i[j] * e_ij)
                    p_i.a = p_i.a + (self.mu * p_j.m *(1 / p_i.rho**2 +
                                     1 / p_j.rho**2) * dW_i[j] * v_ij / r_mod)

                    # if p_i.id == 10:
                    #     print(p_i.D, p_j.m * dW_i[j], v_ij, e_ij)
                    p_i.D = p_i.D + p_j.m * dW_i[j] * (v_ij[0]*e_ij[0] + v_ij[1]*e_ij[1])  ######## make this a dot prod!!!
                    # if p_i.id == 10:
                    #     print(p_i.D)
                    if dW_i[j] > 0:
                        print(p_i.id, p_i.D)

                    v_ij_max = np.amax((np.linalg.norm(v_ij), v_ij_max))
                
                # find the update timestep params
#                a_max = np.amax((np.linalg.norm(p_i.a), a_max))
#                rho_condition = np.sqrt((p_i.rho/self.rho0)**(self.gamma-1))
#                rho_max_condition = np.amax((rho_max_condition, rho_condition))

            # Updating the time step
            if count > 0:
                # cfl_dt = self.h / v_ij_max
                # f_dt = np.sqr/t(self.h / a_max)
                # a_dt = np.amin(self.h / (self.c0 * rho_max_condition))
                # dt = self.CFL * np.amin([cfl_dt, f_dt, a_dt])
                pass
            
            # updating each particles values
            for i, p_i in enumerate(self.particle_list):
                # if particle is not at the boundary
                if not p_i.bound:
                    p_i.x = p_i.x + dt * p_i.v  # update position
                                                # needs to be before velocity
                    p_i.v = p_i.v + dt * p_i.a  # update velocity

                p_i.rho = p_i.rho + dt * p_i.D  # update density
                # smooths if count is a multiple of smoothing
                if count % self.interval_smooth == 0:
                    p_j_list = p_i.adj[:]
                    p_j_list.append(p_i)
                    p_i.rho = self.rho_smoothing(p_i, p_j_list)

                # update pressure
                p_i.P = self.B * ((p_i.rho/self.rho0)**self.gamma - 1)

                # update particle indices
                p_i.calc_index()

            # re-allocate particles to grid
            self.allocate_to_grid()

            # save state
            # if count % self.interval_save:
            #     self.save_state()
            # update counts

            count += 1
            self.t_curr += dt

        print('\nDone!')
        # self.file.close()  # close the save file when done
        return None

    def update_dt(self): #, a, v_ij, rho):
        """
        To be deleted if v_ij or a are not used as attributes of particles
        :return: chosen time step for the iteration
        """

        v_ij = [p.v_ij for p in self.particle_list]
        v_ij_max = np.amax(v_ij)
        cfl_dt = self.h / v_ij_max

        #a = [p.a for p in self.particle_list]
        a_max = np.amax(a)
        f_dt = np.sqrt(self.h / a_max)

        #rho = [p.rho for p in self.particle_list]
        a_dt = np.amin(self.h / (self.c0 * np.sqrt((rho / self.rho0) ** (self.gamma - 1))))

        chosen_dt = self.CFL * np.amin([cfl_dt, f_dt, a_dt])
        return chosen_dt

    def set_up_save(self, name=None, path='raw_data/'):
        """
        Saves the initial setup of the system and creates the csv file to
        store ongoing results as solution runs.
        Files are stored with name in file path (defaults to raw_data folder
        with name given by the time of the simulation).
        """

        # pick a defualt name if none given
        time = datetime.now().strftime('%Y-%m-%d-%Hhr-%Mm')
        if name is None:
            name = time
        assert type(name) is str, 'Name must be a string'
        # assert os.path.isdir(path), path + ' directory does not exist'
        assert self.file is None, "can't run twice as pickling an open file"

        # save the config file
        file = open(path + name + '_config.pkl', 'wb')
        to_save = vars(self).copy()
        [to_save.pop(key) for key in ('search_grid', 'particle_list')]
        pi.dump(to_save, file, pi.HIGHEST_PROTOCOL)
        file.close()

        # set up the csv file
        # replace any previous file with same name
        self.file = open(path + name + '.csv', 'wb').close()
        # open the new file in append mode
        self.file = open(path + name + '.csv', 'a')
        # header comments
        self.file.write('# Created by team Southern on ' + time + '\n')
        # set add in the column titles
        self.file.write("# [s], , [m], [m], [m/s], [m/s], [m/s^2], [m/s^2]," +
                        " [Pa], [Kg/(m^3)], [bool]\n")
        self.file.write("Time,ID,R_x,R_y,V_x,V_y,a_x,a_y,Pressure," +
                        "Density,Boundary\n")
        print('saving to ' + path + name + '.csv')
        # save initial state
        self.save_state()

    def save_state(self):
        """
        Append the current state of every particle in the system to the
        end of the csv file.
        """
        assert self.file is not None, 'set_up_save() has not been run'

        for p in self.particle_list:
            string = ''.join([str(v) + ',' for v in (self.t_curr, p.id, p.x[0],
                              p.x[1], p.v[0], p.v[1], p.a[0], p.a[1],
                              p.P, p.rho, p.bound)]) + '\n'
            self.file.write(string)


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
        self.bound = None
        self.adj = []

    def calc_index(self):
        """
        Calculates the 2D integer index for the particle's
        location in the search grid
        """
        self.list_num = np.array((self.x-self.main_data.min_x) /
                                 (2.0*self.main_data.h), int)


if __name__ == '__main__':

    def init_grid_better():
        """
        Create the intial system given in the documentation.
        Note the x, y axis are scaled to be 1, 2 respectivley

        This function operates by removing particles from a full grid,
        not ideal for user friendlyness
        """
        # set up the system with no particles
        system = SPH_main()
        system.set_values()
        system.max_x[:] = (20., 10.)  # set the grid to correct dimensions
        system.dx = 1

        system.h = system.dx * system.h_fac  # ############## caution here
        system.initialise_grid()

        # set up a full grid the grid
        system.place_points(system.min_x, system.max_x)

        for p in system.particle_list:
            p.v = np.array([1, 0])

        # remove the unwanted points
        for p in system.particle_list.copy():
            if 20 > p.x[0] > 0 and 10 > p.x[1] > 0:  # not boundary node
                if p.x[1] > 5 or (p.x[0] > 3 and p.x[1] > 2):
                    system.particle_list.remove(p)

        # set the boundary nodes
        for p in system.particle_list.copy():
            if p.x[0] > 20 or p.x[0] < 0 or p.x[1] > 10 or p.x[1] < 0:
                p.bound = 1
                p.v = np.array([0, 0])

        system.allocate_to_grid()
        # system.set_up_save()
        xs = np.array([p.x for p in system.particle_list])
        bs = [p.bound for p in system.particle_list]
        plt.scatter(xs[:, 0], xs[:, 1], c=bs)

        return system

    # """Create a single object of the main SPH type"""
    # domain = SPH_main()
    #
    # """Calls the function that sets the simulation parameters"""
    # domain.set_values()
    # """Initialises the search grid"""
    # domain.initialise_grid()
    #
    # """
    # Places particles in a grid over the entire domain - In your code you
    # will need to place the fluid particles in only the appropriate locations
    # """
    # domain.place_points(domain.min_x, domain.max_x)
    #
    # """This is only for demonstration only - In your code these functions
    # will need to be inside the simulation loop"""
    # """This function needs to be called at each time step
    # (or twice a time step if a second order time-stepping scheme is used)"""
    # domain.allocate_to_grid()
    # """This example is only finding the neighbours for a single partle
    # - this will need to be inside the simulation loop and will need to be
    # called for every particle"""
    # # domain.neighbour_iterate(domain.particle_list[100])

    domain = init_grid_better()
    plt.close()
    # domain.timestepping(tf=15e-3)
    # domain.plot_current_state()

    # domain = init_grid_better()
    domain.timestepping(tf=1)
    domain.plot_current_state()

    plt.show()