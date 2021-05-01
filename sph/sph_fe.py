from itertools import count
import numpy as np
import matplotlib.pyplot as plt
import os
from sys import stdout
from datetime import datetime
import pickle as pi
import warnings
import time

from sph.animate_results import load_and_set
from sph.sph import Grid, SysVals
from sph.objects import Particle
from sph.samples import step_wave


class SPH_main(object):
    """
    A class for Smoothed Particle Hydrodynamics (SPH), a meshless method
    used for solving the Navier-Stokes equation to simulate a wave
    generation problem.

    ....

    Attributes
    ----------

    h : float -- determined attribute
        bin half size (meters). [Deafult value = 1.3]
    h_fac : float -- set attribute
        bin half size constant (unitless).
    dx : float -- set attribute
        particle spacing (meters).
    mu : float -- set attribute
        viscosity (Pa s) [Deafult value = 0.001]
    rho0 : integer -- set attribute
        initial particle density (kg/m^3). [Deafult value = 1000]
    c0 : integer -- set attribute
        fit-for-purpose speed of sound in water (m/s). [Deafult value = 20]
    t_curr : float -- set attribute
        current time of the system (s).
    gamma : constant -- set attribute
        stiffness value (unitless). [Deafult value = 7]
    interval_smooth : integer -- set attribute
        number of timesteps to which smooth rho (unitless). [Deafult value = 15]
    interval_save : integer -- set attribute
        number of timesteps at which the current states are saved (unitless).
        [Deafult value = 15]
    CFL : float -- set attribute
        Scale factor for Courant–Friedrichs–Lewy condition (unitless). [Deafult value = 0.2]
    B : float -- determined attribute
        pressure constant (Pa).
    g : 1D array -- set attribute
        body force based 2D vector [gravity value (m/s^2)]. [Deafult value = [0, -9.81] ]
    file : file -- determined attribute
        a file of results for post processing and visulaization.
    min_x : list
        lower-left boundary for the domain.
    max_x : list
        upper-right boundaries for the domain.
    grid_max : list -- determined attribute
        binning grid dimensions.
    search_grid : array
        binning grid.
    t_curr : float -- set attribute
        Dyanimc. Stores time at which simulation is being run. Starts at 0.
    w_fac1 : float -- set attribute
        A constant for the smoothing function W (m^-2).
    w_fac2 : float -- set attribute
        A constant for the derivative smoothing function dW (m^-3.)
    particle_list : list -- determined attribute
        A list of particles to be simulated.
    search_grid : array -- determined attribute
        An array of searched neighbouring particles.
    lil_bit : float-- determined attribute
        An upper limit to get np arrange working as desired.
    P_ref : float -- determined attribute
        Boundary reference pressure to prevent leakages (Pa).
    d_ref : float -- determined attribute
        Reference distance for enforcing boundary pressure (m).
    func : list -- set attribute
        A list containing O and 1 to distinguish fluid particles from
        boundaries.
    interval_smooth : int -- set attribute
        interval of timesteps at which density smoothing function is called
    interval_save : int -- set attribute
        interval of timesteps at which data is appended to file


    Methods
    -------
    determine_values():
        Aids to determine initial simulation parameters.
    initialise_grid():
        Intializes the domain for simulation.
    add_boundaries():
        Adds the boundary points of at least 2h around the edges.
    place_points(xmin, xmax):
        Place points in a rectangle with a square spacing of specific value.
    allocate_to_grid():
        Allocates all the points to a grid to aid neighbours' searching.
    neighbour_iterate(part):
        Finds all the particles within the search range of a specific particle.
    plot_current_state():
        Plots the current state of the system (i.e. where every particles are)
        in space.
    W(p_i, p_j_list):
        Calculates Smoothing factor for a particle being affected by
        neighbouring particles within the specified neighbourhood.
    dW(p_i, p_j_list):
        Calculates the derivative Smoothing factor for a particle being
        affected by neighbouring particles within the specified neighbourhood.
    LJ_boundary_force(p):
        Enforces boundary force to prevent fluid particles' leaking.
    rho_smoothing(p_i, p_j_list):
        determines the smoothed density of a particle interest.
    timestepping(tf):
        Timesteps the physical problem with a set dt until
        user-specified time is reached.
    set_up_save(name, path):
        Saves the initial setup of the system and creates the csv file to
        store ongoing results as solution runs.
    save_state():
        Append the current state of every particle in the system to the
        end of the csv file.
    R_artificial_pressure(p_i, p_j_list, step) -- sph_ap only:
        Determines the R component of the artificial pressure.
    dW_artificial_pressure(p_i, p_j_list, step) -- sph_ap only:
        Calculates the derivative Smoothing factor component of the artificial
        pressure for a particle being affected by neighbouring particles
        within the specified neighbourhood.
    """

    def __init__(self, system: SysVals = None, grid: Grid = None):
        """Store or create the system and grid objects."""
        self.sys = system or SysVals(x_min=(0.0, 0.0), x_max=(1.0, 1.0), dx=0.02)
        self.grid = grid or Grid(self.sys)
        self.file = None

    def W(self, p_i, p_j_list):
        """
        Computes the smoothing parameter for a particle i and all its influencing neighbours
        Parameters
        ----------
        p_i: (object)
            particle where calculations are being performed
        p_j_list: (list of objects)
            particles influencing particle i

        Returns
        --------
        j_list:(np array)
            smoothing factor for particle i being affected by particles j
        """
        xi = p_i.x
        xj = np.array([p.x for p in p_j_list])
        r = xi - xj
        j_list = np.sqrt(np.sum(r ** 2, axis=1)) / self.sys.h
        assert ((j_list >= 0).all()), "q must be a positive value"

        for i, q in enumerate(j_list):
            if 0 <= q < 1:
                j_list[i] = self.sys.w_fac1 * (1 - 1.5 * q ** 2 + 0.75 * q ** 3)
            elif 1 <= q <= 2:
                j_list[i] = self.sys.w_fac1 * (0.25 * (2 - q) ** 3)
            else:
                j_list[i] = 0
        return np.array(j_list)

    def dW(self, p_i, p_j_list):
        """
        Computes the derivative of the smoothing parameter for a particle i and all
        its influencing neighbours
        Parameters
        ----------
        p_i: (object)
            particle where calculations are being performed
        p_j_list: (list of objects)
            list of particles influencing particle i

        Returns
        --------
        j_list:(np array)
            derivative of smoothing factor for particle i being affected by particles j
        """
        xi = p_i.x
        xj = np.array([p.x for p in p_j_list])
        r = xi - xj
        j_list = np.sqrt(np.sum(r ** 2, axis=1)) / self.sys.h
        assert ((j_list >= 0).all()), "q must be a positive value"

        for i, q in enumerate(j_list):
            if 0 <= q < 1:
                j_list[i] = self.sys.w_fac2 * (-3 * q + (9 / 4) * q ** 2)
            elif 1 <= q <= 2:
                j_list[i] = self.sys.w_fac2 * (-(3 / 4) * (2 - q) ** 2)
            else:
                j_list[i] = 0
        return np.array(j_list)

    def rho_smoothing(self, p_i, p_j_list):
        """
        Computes the smoothed density of a particle i and
        Parameters
        ----------
        p_i: (object)
            particle where calculations are being performed
        p_j_list: (list of objects)
            list of particles influencing particle i

        Returns
        --------
        rho:(float)
            particle i smoothed density
        """
        assert (p_i in p_j_list), "must include particle i in this calculation"
        w_list = self.W(p_i, p_j_list)
        p_j_rho = np.array([p.rho for p in p_j_list])
        assert ((p_j_rho > 0).all()), "density must be always positive"
        rho = np.sum(w_list) / np.sum(w_list / p_j_rho)
        return rho

    def LJ_boundary_force(self, p):
        """
        Adds acceleration to a particle p using a Lennard-Jones potential proportional
        to its distance to the outermost boundary wall
        Parameters
        ----------
        p: (object)
            particle where calculations are being performed

        """
        r_wall_left = abs(p.x[0] - self.sys.min_x[0])
        if r_wall_left != 0:
            q_ref_left = self.sys.d_ref / r_wall_left
            if q_ref_left > 1:
                p.a[0] = p.a[0] + (self.sys.P_ref * (q_ref_left ** 4 -
                                   q_ref_left ** 2) / (r_wall_left * p.rho))

        r_wall_bottom = abs(p.x[1] - self.sys.min_x[1])
        if r_wall_bottom != 0:
            q_ref_bottom = self.sys.d_ref / r_wall_bottom
            if q_ref_bottom > 1:
                p.a[1] = p.a[1] + (self.sys.P_ref * (q_ref_bottom ** 4 -
                                   q_ref_bottom ** 2) / (r_wall_bottom*p.rho))

        r_wall_right = abs(p.x[0] - self.sys.max_x[0])
        if r_wall_right != 0:
            q_ref_right = self.sys.d_ref / r_wall_right
            if q_ref_right > 1:
                p.a[0] = p.a[0] - (self.sys.P_ref * (q_ref_right ** 4 -
                                   q_ref_right ** 2) / (r_wall_right * p.rho))

        r_wall_top = abs(p.x[1] - self.sys.max_x[1])
        if r_wall_top != 0:
            q_ref_top = self.sys.d_ref / r_wall_top
            if q_ref_top > 1:
                p.a[1] = p.a[1] - (self.sys.P_ref * (q_ref_top ** 4 -
                                   q_ref_top ** 2) / (r_wall_top * p.rho))
        return None

    def timestepping(self, tf):
        """
        Timesteps the physical problem with a set dt
        until user-specified time is reached.
        Uses Forward Euler timestepping
        """

        # initialise vairables
        dt = 0.1 * self.sys.h / self.sys.c0
        print('DT ---------------- ', dt)
        v_ij_max = 0
        a_max = 0
        rho_max_condition = 0
        assert (tf >= dt), "time to short to resolve problem, dt=%.4f" % (dt)

        count = 1
        while self.sys.t_curr <= tf:
            stdout.write('\rTime: %.3f' % self.sys.t_curr)
            stdout.flush()

            # find all the derivatives for each particle
            for i, p_i in enumerate(self.grid.particle_list):
                # create list of neighbours for particle i
                self.grid.neighbour_iterate_half(p_i)

                if p_i.adj != []:
                    # calculate smoothing from all neighbouring particles
                    dW_i = self.dW(p_i, p_i.adj)

                    # calculate acceleration and rate of change of density,
                    # find maximum relative velocity amongst all particles and
                    # their neighbours and the maximum acceleration amongst
                    # particles
                    for j, p_j in enumerate(p_i.adj.copy()):
                        r_vec = p_i.x - p_j.x
                        r_mod = np.sqrt(np.sum(r_vec ** 2))
                        e_ij = r_vec / r_mod
                        v_ij = p_i.v - p_j.v

                        # update acceleration and density for p_i
                        tmp_a1 = (p_j.m * (p_i.P / p_i.rho ** 2 +
                                  p_j.P / p_j.rho ** 2) * dW_i[j] * e_ij)
                        p_i.a = p_i.a - tmp_a1
                        tmp_a2 = (self.sys.mu * p_j.m * (1 / p_i.rho**2 +
                                  1 / p_j.rho**2) * dW_i[j] * v_ij / r_mod)
                        p_i.a = p_i.a + tmp_a2

                        tmp_d = p_j.m * dW_i[j] * v_ij.dot(e_ij)
                        p_i.D = p_i.D + tmp_d

                        # do the same for particle j
                        p_j.a = p_j.a + tmp_a1
                        p_j.a = p_j.a - tmp_a2
                        p_j.D = p_j.D + tmp_d
                        v_ij_max = np.amax((np.linalg.norm(v_ij), v_ij_max))

                    # implementing boundary repulsion
                    self.LJ_boundary_force(p_i)

                    # Max values to calculate the time step
                    a_max = np.amax((np.linalg.norm(p_i.a), a_max))
                    rho_condition = np.sqrt((p_i.rho / self.sys.rho0) **
                                            (self.sys.gamma-1))
                    rho_max_condition = np.amax((rho_max_condition,
                                                 rho_condition))

                elif ((p_i.x < self.sys.min_x).any() or
                      (p_i.x > self.sys.max_x).any()):
                    # remove leaked particles
                    warnings.warn("Particle %g has leaked" % (p_i.id))
                    self.grid.particle_list.remove(p_i)

            # Updating the time step
            if count > 1:
                cfl_dt = self.sys.h / v_ij_max
                f_dt = np.sqrt(self.sys.h / a_max)
                a_dt = np.amin(self.sys.h / (self.sys.c0 * rho_max_condition))
                dt = self.sys.CFL * np.amin([cfl_dt, f_dt, a_dt])

            # if smoothing find all adjasent particles
            if count % self.sys.interval_smooth == 0:
                for p_i in self.grid.particle_list:
                    self.grid.neighbour_iterate(p_i)

            # updating each particles values
            assert all(i == p_i.id for i, p_i in enumerate(self.grid.particle_list))
            for i, p_i in enumerate(self.grid.particle_list.copy()):
                # if particle is not at the boundary
                if not p_i.bound:
                    p_i.x = p_i.x + dt * p_i.v  # update position
                    # positions needs to be before velocity
                    p_i.v = p_i.v + dt * p_i.a  # update velocity

                # for all particles: update density,
                # smooths if count is a multiple of smoothing
                p_i.rho = p_i.rho + dt * p_i.D
                if count % self.sys.interval_smooth == 0:
                    p_j_list = p_i.adj[:]
                    p_j_list.append(p_i)
                    p_i.rho = self.rho_smoothing(p_i, p_j_list)

                # update pressure
                p_i.P = self.sys.B * ((p_i.rho / self.sys.rho0) ** self.sys.gamma - 1)

                # reset the acceleration and D values
                p_i.a = self.sys.g
                p_i.D = 0

            # re-allocate particles to grid
            self.grid.allocate_to_grid()

            # append the state to file
            if count % self.sys.interval_save == 0:
                self.save_state()

            # update count and t
            count += 1
            self.sys.t_curr += dt

        # close file
        self.file.close()
        return None

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
        assert os.path.isdir(path), path + ' directory does not exist'
        assert self.file is None, "can't run twice as pickling an open file"

        # save the config file
        file = open(path + name + '_config.pkl', 'wb')
        to_save = vars(self).copy()
        [to_save.pop(key) for key in ('grid',)]
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
        self.file.write("# [s], , [m], [m], [m/s], [m/s]," +
                        " [Pa], [Kg/(m^3)], [bool]\n")
        self.file.write("Time,ID,R_x,R_y,V_x,V_y,Pressure," +
                        "Density,Boundary\n")
        print('saving to ' + path + name + '.csv ...')
        # save initial state
        # self.save_state()

    def save_state(self):
        """
        Append the current state of every particle in the system to the
        end of the csv file.
        """
        assert self.file is not None, 'set_up_save() has not been run'

        for p in self.grid.particle_list:
            string = ''.join([str(v) + ','
                              for v in (self.sys.t_curr, p.id, p.x[0], p.x[1],
                                        p.v[0], p.v[1], p.P,
                                        p.rho, p.bound)]) + '\n'
            self.file.write(string)


def sph_simulation(
    x_min,
    x_max,
    t_final,
    dx,
    func,
    path_name='./',
    ani=True,
    file_name=None,
    ani_step=None,
    ani_key=None,
    **kwargs
):
    """
    Simulates fluid flow from user-specified initial state and timeframe using
    smoothed particle hydrodynamics method.

    Parameters
    ----------
    x_min : list-like
        List with [x,y] coordinates of bottom left boundary, assumed rectangular
    x_max : list-like
        List with [x,y] coordinates of upper right boundary, assumed rectangular
    t_final : float
        Timeframe of simulation.
    dx : float
        Initial distance between particles in domain. Particles assumed to be
        equally spaced on domain specified by func
    func : function
        A function that specifies the initial distribution of particles in the
        domain with a boolean output.
    path name : string
        Path where files are to be saved.
    ani : boolean
        "True" if animation is to be displayed and "False" if otherwise.

    Other Parameters
    ----------------
    ani_step: int
        frame skipping
    ani_key: string
        header for colorplot. Choose beteween: ID, Pressure, Density, V_x, and V_y
    h_fac : float -- set attribute
        bin half size constant (unitless).
    mu : float -- set attribute
        viscosity (Pa s) [Deafult value = 0.001]
    rho0 : integer -- set attribute
        initial particle density (kg/m^3). [Deafult value = 1000]
    c0 : integer -- set attribute
        fit-for-purpose speed of sound in water (m/s). [Deafult value = 20]
    gamma : constant -- set attribute
        stiffness value (unitless). [Deafult value = 7]
    interval_smooth : integer -- set attribute
        number of timesteps to which smooth rho (unitless). [Deafult value = 15]
    interval_save : integer -- set attribute
        number of timesteps at which the current states are saved (unitless).
        [Deafult value = 15]
    CFL : float -- set attribute
        Scale factor for Courant–Friedrichs–Lewy condition (unitless). [Deafult value = 0.2]
    g : 1D array -- set attribute
        body force based 2D vector [gravity value (m/s^2)]. [Deafult value = [0, -9.81] ]
    tf : float
        Total real time to run simulation.
    P_ref : float (only in the Forward Euler Module)
        Boundary reference pressure to prevent leakages (Pa).
    d_ref : float (only in the Forward Euler Module)
        Reference distance for enforcing boundary pressure (m).
    file_name : string
        Name of file to be saved. If None saves with current time[Default = None]

    """
    # set the system
    system = SysVals(x_min, x_max, dx=dx, **kwargs)
    grid = Grid(system)
    grid.initialise_grid(func)
    grid.allocate_to_grid()

    solver = SPH_main(system, grid)
    solver.set_up_save(name=file_name, path=path_name)

    # solve the system
    t = time.time()
    solver.timestepping(tf=t_final)
    print('\nTime taken :', time.time()-t)

    # animate result
    if ani:
        ani = load_and_set(solver.file.name, ani_key=ani_key or 'Density')
        ani.animate(ani_step=ani_step or 1)
        plt.show()

    return solver


if __name__ == '__main__':
    sph_simulation(
        x_min=[0, 0],
        x_max=[20, 10],
        t_final=0.5,
        dx=1,
        func=step_wave,
        path_name='./raw_data/',
        ani_step=1,
        ani_key="Pressure",
        file_name="final_sim",
        x_ref=0.9
    )
