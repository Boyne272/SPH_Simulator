"""Physical system class."""

from dataclasses import dataclass
import json
from typing import get_type_hints

import numpy as np

from sph.objects.random import RngState
from sph.utils import md5_hash


@dataclass(eq=False)
class System:
    """All the constants that create this system."""

    # system main parameters
    min_x: tuple = (0., 0.)    # lower left corner
    max_x: tuple = (1., 1.)    # upper right corner
    dx: float = 0.02            # initial particle spacing
    h_fac: float = 1.3          # bin half size constant (unitless)

    # normal system parameters
    mu: float = 0.001           # viscosity (Pa s)
    rho0: float = 1000.         # initial particle density (kg/m^3)
    c0: float = 20.             # speed of sound in water (m/s)
    gamma: float = 7.           # stiffness value (dimensionless)
    CFL: float = 0.2            # CFL constant (dimensionless)
    grav: tuple = (0., -9.81)   # gravity mangintude (m/s^2)
    P_fac: float = 1.05         # scale for LJ reference pressure
    x_ref: float = 0.9          # scale for LJ reference distance
    pad_fac: float = 1.         # scale for padding boundaries
    smooth_steps: int = 15      # timesteps to smooth rho (dimensionless)
    save_steps: int = 15        # timesteps to save the state (dimensionless)

    # vairable system parameters
    t_curr: float = 0.0         # current time of the system (s)

    # random seed
    seed: int = None            # int32 seed for the random initalisation
    rand: RngState = None       # random generator

    # derived attributes in __post_init__
    h: float = 0.0
    d_srch: float = 0.0
    dt: float = 0.0
    B: float = 0.0
    w_fac1: float = 0.0
    w_fac2: float = 0.0
    P_ref: float = 0.0
    d_ref: float = 0.0

    def __post_init__(self):
        """Set all the dervied constants."""
        # set the seed
        self.seed = self.seed or np.random.randint(0, 2147483647)
        self.rand = self.rand or RngState(self.seed)        # random state for the system

        # calculate derived attributes
        self.h = self.dx*self.h_fac                         # bin half-size
        self.dt = 0.1 * self.h / self.c0                    # reasonable time step TODO add update method
        self.d_srch = 2 * self.h                            # search radius
        self.B = self.rho0 * self.c0**2 / self.gamma        # pressure constant (Pa)
        self.w_fac1 = 10 / (7 * np.pi * self.h ** 2)        # constant often used
        self.w_fac2 = 10 / (7 * np.pi * self.h ** 3)        # constant often used
        self.P_ref = self.B*(self.P_fac**self.gamma - 1)    # boundary pressure to prevent leakages (Pa).
        self.d_ref = self.x_ref * self.dx                   # distance boundary pressure (m)
        self.g = np.array(self.grav, float)

        # expand the range for boundaries
        self.x_inner_min = np.array(self.min_x, float)
        self.x_inner_max = np.array(self.max_x, float)
        self.x_min = self.x_inner_min - (self.d_srch * self.pad_fac)
        self.x_max = self.x_inner_max + (self.d_srch * self.pad_fac)
        self.x_range = np.array([self.x_min[0], self.x_max[0]])
        self.y_range = np.array([self.x_min[1], self.x_max[1]])

    @property
    def as_dict(self) -> dict:
        """Write this system as a dict, replacing the rand object with its string representation."""
        props = {
            k: v for k, v in vars(self).items()
            if k not in ['x_max', 'x_min', 'x_inner_min', 'x_inner_max', 'x_range', 'y_range', 'g']
        }  # arrays bad for json/hash
        return {**props, 'rand': self.rand.as_string()}

    def as_json(self, rand: bool = True) -> str:
        """Write this system as a json object.

        parameters:
            rand: include the full random state, if false the random seed will be used

        """
        to_dump = self.as_dict
        to_dump.pop('' if rand else 'rand', '')
        return json.dumps(to_dump, sort_keys=True, indent=4)

    @property
    def parameter_hash(self) -> str:
        """This is a hash of the static system parameters."""
        params = {**self.as_dict, 't_curr': None, 'rand': None, 'seed': None}
        return md5_hash(json.dumps(params, sort_keys=True))

    @property
    def summary(self) -> str:
        """Summary of this system at this point in time."""
        # TODO add service version in here for non-backwards verbosity
        return f'@{self.t_curr}s \t {self.rand!r} \t system:{self.parameter_hash} \t seed:{self.seed}'

    def __eq__(self, other):
        """Use the hash property to compare the initial state of two systems."""
        return self.summary == other.summary

    @staticmethod
    def from_dict(sys_dict) -> "System":
        """Load a system from a dict obj.

        Notes:
            - RngState will either be set from seed or from more explict state if present
            - this will try cast all types

        """
        if 'rand' in sys_dict:
            rand = RngState.from_string(sys_dict.pop('rand'))
        else:
            rand = None

        for key, type_ in get_type_hints(System).items():
            try:
                if key in sys_dict:
                    sys_dict[key] = type_(sys_dict[key])
            except Exception as err:
                raise ValueError(f"Failed to convert property {key} to type {type_}") from err

        return System(**sys_dict, rand=rand)
