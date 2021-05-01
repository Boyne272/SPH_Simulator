"""SPH simulation particle class."""

from dataclasses import dataclass
from itertools import count
from typing import ClassVar

import numpy as np


@dataclass(eq=False)
class Particle(object):
    """SPH Simulation Particle."""

    x: np.ndarray = np.zeros(2)  # current position (m)
    v: np.ndarray = np.zeros(2)  # current velocity (m/s)
    a: np.ndarray = np.zeros(2)  # current acceleration (m/s^2)

    D: float = 0.  # current rate of change of density (kg/m^3s)
    P: float = 0.  # current pressure of particle (Pa)
    m: float = 0.  # mass of the particle (kg)
    rho: float = 0. # density at particles position (kg/m^2s)
    bound: int = 0  # is boundary particle (binary)

    _id: int = None  # identifier for this particle
    _adj: list = None # list of adjasent particles

    particle_counter: ClassVar = count(1)  # counter for all particles
    csv_header: ClassVar = 'ID,x_x,x_y,v_x,v_y,a_x,a_y,m,D,P,rho,bound'

    def __post_init__(self):
        """Set up-defualtable attributes and validate the inputs."""
        self._adj: list["Partile"] = []
        self._id = self._id or next(self.particle_counter)
        assert self.x.shape == self.v.shape == self.a.shape == (2,), 'only 2d supported'

    def __eq__(self, other):
        """Relying on defualt __repr__ is allows for array comparison."""
        return str(self) == str(other)

    @property
    def csv(self) -> str:
        """Csv string for loading this particle: ID,R_x,R_y,V_x,V_y,a_x,a_y,m,D,P,Rho,Bound"""
        return ','.join([
            str(self._id),
            str(self.x[0]),
            str(self.x[1]),
            str(self.v[0]),
            str(self.v[1]),
            str(self.a[0]),
            str(self.a[1]),
            str(self.m),
            str(self.D),
            str(self.P),
            str(self.rho),
            str(self.bound)
        ])

    @staticmethod
    def from_str(string: str) -> "Particle":
        """The inverse of csv_str operation to make a particle from the csv string."""
        _id, x_x, x_y, v_x, v_y, a_x, a_y, m, d, p, rho, bound = string.split(',')
        return Particle(
            _id=int(_id),
            x=np.array([x_x, x_y], float),
            v=np.array([v_x, v_y], float),
            a=np.array([a_x, a_y], float),
            m=float(m),
            D=float(d),
            P=float(p),
            rho=float(rho),
            bound=int(bound)
        )
