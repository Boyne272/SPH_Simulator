"""Common testing fixtures and data."""

from unittest.mock import Mock

import numpy as np
from pytest import fixture


@fixture
def mock_SysVals():
    """Mock sys object for unit tests."""
    return Mock(
        h=0.25,
        min_x=np.array([0., 0.])
    )
