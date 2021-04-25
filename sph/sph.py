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
        self.list_num = np.array((self.x - self.main_data.min_x) /
                                 (2.0 * self.main_data.h), int)

    def __post_init__(self):
        """Validate the inputs."""
        assert self.x.shape == self.v.shape == self.a.shape, 'must have consistent dimension'
