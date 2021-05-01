"""Functions for solving sph navier stokes equation."""

import logging
from typing import Callable, Tuple

import numpy as np

from sph.objects.particle import Particle
from sph.objects.system import System
from sph.objects.grid import Grid


logger = logging.getLogger(__name__)


# smoothing kernal funtions -------------------------------


def W(sys: System, p_i: Particle, p_j: Particle) -> float :
    """Computes the smoothing parameter for a particle pair i, j.

    parameters:
        sys: System holding relevant constants
        p_i: Particle one
        p_j: Particle two

    """
    q = np.linalg.norm(p_i.x - p_j.x) / sys.h
    if 0 <= q < 1:
        return sys.w_fac1 * (1 - 1.5*(q**2) + 0.75*(q**3))
    if 1 <= q <= 2:
        return sys.w_fac1 * 0.25*((2 - q)**3)
    return 0.


def dW(sys: System, p_i: Particle, p_j: Particle) -> float:
    """Computes the derivative of the smoothing parameter for a particle pair i, j.

    parameters:
        sys: System holding relevant constants
        p_i: Particle one
        p_j: Particle two

    """
    q = np.linalg.norm(p_i.x - p_j.x) / sys.h
    if 0 <= q < 1:
        return sys.w_fac2 * (-3*q + 2.25*(q**2))
    if 1 <= q <= 2:
        return sys.w_fac2 * -0.75*((2-q)**2)
    return 0.


# stability equatins --------------------------------------


def compute_pressure(sys: System, p_i: Particle) -> float:
    """Compute the pressure of particle i via the Tait equation."""
    return sys.B * (-1 + (p_i.rho/sys.rho0)**sys.gamma)


def rho_smoothing(sys: System, p_i: Particle) -> float:
    """Compute the smoothed density of a particle pair i for all adj particles

    parameters:
        sys: System holding relevant constants
        p_i: Particle must have up to date _adj list

    """
    p_j_list = p_i._adj + p_i  # must include particle i in this calculation
    num, den = 0., 0.  # TODO check this does not give a zero by zero error
    for p_j in p_j_list:
        w_j = W(p_i, p_j)
        num += w_j
        den += w_j / p_j.rho
    return num / den


def lj_boundary_force(sys: System, p_i: Particle, side: str):
    """Adds acceleration to a particle near system bounaries using a Lennard-Jones potential.

    parameters:
        sys: System holding relevant constants
        p_i: Particle to be have the potential added
        loc: the location of the boundary
        side: which side of the system the boundary is on

    """
    # TODO find a better place for me as I do redundent work
    if side == 'left':
        axis = 0
        direction = +1
        loc = sys.x_inner_min[0]
    elif side == 'right':
        axis = 0
        direction = -1
        loc = sys.x_inner_max[0]
    elif side == 'lower':
        axis = 1
        direction = +1
        loc = sys.x_inner_min[1]
    elif side == 'upper':
        axis = 1
        direction = -1
        loc = sys.x_inner_max[1]

    dist = np.abs(p_i.x[axis] - loc)
    q_ref = sys.d_ref / dist
    if q_ref > 1:  # if within sys.d_ref of the boundary
        p_i.a[axis] += sys.P_ref * direction * (q_ref**4 - q_ref**2) / (dist*p_i.rho)


# navier stokes equations ---------------------------------


def compute_differentials(sys: System, p_i: Particle, p_j: Particle) -> float:
    """Compute the differential equations for the interaction between particle pair i, j only.

    Notes:
        - g is not considered in the equation here as all p_k.a = g after each iteration
        - values are stored on the Particle.a and Particle.D attributes

    parameters:
        sys: System holding relevant constants
        p_i: Particle one
        p_j: Particle two

    returns:
        v_ij_mag: magnitdue of the relative velocity (for dynamic dt calculations)

    """
    # compute vectors
    r_ij = p_i.x - p_j.x
    r_mod = np.linalg.norm(r_ij)  # TODO this is the 3rd time we find this dist, find a better way
    e_ij = r_ij / r_mod
    v_ij = p_i.v - p_j.v

    # compute common values
    dw = dW(sys, p_i, p_j)
    sq_rho_i = p_i.rho**2
    sq_rho_j = p_j.rho**2

    # compute differential equations
    tmp_a1 = p_j.m * dw * e_ij * (p_i.P/sq_rho_i + p_j.P/sq_rho_j)
    tmp_a2 = sys.mu * p_j.m * dw * v_ij * (1/sq_rho_i + 1/sq_rho_j) / r_mod
    tmp_d = p_j.m * dw * v_ij.dot(e_ij)

    # update acceleration and density for p_i
    p_i.a -= tmp_a1  # TODO ensure I am safe to use
    p_i.a += tmp_a2
    p_i.D += tmp_d

    # update acceleration and density for p_i
    p_j.a += tmp_a1
    p_j.a -= tmp_a2
    p_j.D += tmp_d

    return np.linalg.norm(v_ij)


# solution logic ------------------------------------------


def sph_iterate(sys: System, grid: Grid, step_meth: Callable, smooth_rho: bool = False):
    """Iterate the sph simulation forwards in time by one iteration.

    notes:
        - grid management is handled here
        - grid reassinment is at the start of the iteration (so bins will be invalid on return)
        - differential values (e.g. a, D) are left as their computed selves after calling
        - differential values are reset at the start of the iteration

    parameters:
        sys: the system we are solving in
        grid: the grid obj with all particles present
        step_meth: the method for time stepping, can be one of:
            - euler
        smooth: whether to smooth density this iteration

    """
    logger.info('Updating Grid...')
    grid.update_grid()  # bin all particles
    grid.update_adjs()  # populate _adj lists

    if smooth_rho:
        logger.info('Applying density smoothing...')
        for p_i in grid.particle_list:
            p_i.rho = rho_smoothing(sys, p_i)

    logger.info('Resetting differentials...')
    for p_i in grid.particle_list:
        p_i.a = sys.g  # rather than zero, then g not in `compute_differentials`
        p_i.D = 0

    logger.info('Computing differentials...')
    for p_i in grid.particle_list:
        for p_j in p_i._adj:
            v_ij_mag = compute_differentials(sys, p_i, p_j)  # calculates dA and dD
            # TODO update v_ij_max

    logger.info('Applying boundary forces...')
    for side in grid.walls:  # add boundary force to prevent leakages
        for i, j in grid.walls[side]:
            for particle in grid.search_dict[i, j]:
                if not particle.bound:
                    lj_boundary_force(sys, particle, side)

        # TODO update a_max, rho_max

    # TODO compute new CFL dt

    logger.info('Advancing particles...')
    for p_i in grid.particle_list: # iterate values forwards in time
        step_meth(p_i, sys.dt)
        p_i.P = compute_pressure(sys, p_i)
    sys.t_curr += sys.dt

    logger.info('Iteration finished!')


def euler_step(p_i: Particle, dt:float) -> None:
    """Update position, velocity, density by euler's method.

    Notes:
        - only update position and velocity for non-fixed particles
        - position must be updated before velocity # TODO find why

    """
    if not p_i.bound:
        p_i.x += dt*p_i.v  # position must be updated first
        p_i.v += dt*p_i.a
    p_i.rho += dt*p_i.D
