"""Random generator object that can be easily saved as a string."""
import json
from base64 import b64encode, b64decode

import numpy as np
from numpy.random.mtrand import RandomState

from sph.utils import md5_hash


class RngState(RandomState):
    """A random state with todict and tostr methods."""

    def as_string(self) -> str:
        """Convert the state of self into a json string."""
        name, ints, pos, has_guass, cached_guassian = self.get_state()
        params = {
            'name': name,
            'int_bytes': b64encode(ints.tobytes()).decode('ascii'),
            'int_type': str(ints.dtype),
            'pos': pos,
            'has_guass': has_guass,
            'cached_guassian': cached_guassian
        }
        return json.dumps(params, sort_keys=True)

    @staticmethod
    def from_string(string: str) -> "RngState":
        """Create RandomState object from `as_string` output."""
        params = json.loads(string)
        rand = RngState()
        rand.set_state((
            params['name'],
            np.frombuffer(b64decode(params['int_bytes']), params['int_type']),
            params['pos'],
            params['has_guass'],
            params['cached_guassian']
        ))
        return rand

    def __repr__(self) -> str:
        """Display this random state with its hash."""
        return f'Random State ({md5_hash(self.as_string())})'
