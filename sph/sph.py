"""SPH simulation objects."""

from itertools import count
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np


@dataclass(eq=False)
class Particle(object):
    """SPH Simulation Particle."""

    main_data: bool = None  # parent obj TODO change me to something more sensible

    x: np.ndarray = np.zeros(2)  # current position (m)
    v: np.ndarray = np.zeros(2)  # current velocity (m/s)
    a: np.ndarray = np.zeros(2)  # current acceleration (m/s^2)

    D: float = 0.  # current rate of change of density (kg/m^3s)
    P: float = 0.  # current pressure of particle (Pa)
    m: float = 0.  # mass of the particle (kg)  # TODO validate me

    bound: bool = False  # is a boundary (i.e. fixed) particle
    adj: list = None #field(default_factory=lambda:list())  # list of adjasent particles
    list_num: np.ndarray = None  # TODO change me to something more sensible

    id: int = field(default_factory=lambda:next(Particle.n_particles))  # identifier for this particle
    n_particles: ClassVar = count(0)  # counter for all particles

    def calc_index(self) -> np.ndarray:
        """Calculates particle's location in the search grid."""
        self.list_num = np.array((self.x - self.main_data.sys.min_x) /
                                 (2.0 * self.main_data.sys.h), int)

    def __post_init__(self):
        """Validate the inputs."""
        assert self.x.shape == self.v.shape == self.a.shape, 'must have consistent dimension'



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

    grid_max = np.zeros(2, int)  # TODO move to grid obj

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
        # TODO fix me
        self.min_x = np.array(self.x_min, float)
        self.max_x = np.array(self.x_max, float)

        # determine_values
        self.h = self.dx*self.h_fac                   # bin half-size
        self.lil_bit = self.dx*0.01                   # to include upper limits
        self.B = self.rho0 * self.c0**2 / self.gamma  # pressure constant (Pa)
        self.w_fac1 = 10 / (7 * np.pi * self.h ** 2)  # constant often used
        self.w_fac2 = 10 / (7 * np.pi * self.h ** 3)  # constant often used
        self.P_ref = self.B*(self.P_fac**self.gamma - 1)  # boundary pressure to prevent leakages (Pa).
        self.d_ref = self.x_ref * self.dx             # distance boundary pressure (m)

    def __eq__(self, other):
        """Relying on __repr__ is safer for array comparison."""
        return str(self) == str(other)


class Grid:
    """A grid object for sorting particles and finding neighbours efficently."""

    def __init__(self, sys: SysVals):
        """."""
        self.sys = sys
