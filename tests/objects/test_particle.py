"""Unit tests for particle.py."""

from csv import DictReader
from io import StringIO

import numpy as np

from sph.objects.particle import Particle


def test_from_str():
    """Ensure a particle can be created and saved to a csv string."""
    p1 = Particle()
    p1.x[:] = (1, 2)  # set some dummy values
    p1.a[:] = (-1, 0)  # set some dummy values

    p2 = Particle.from_str(p1.csv)

    assert p2.csv == p1.csv
    assert p2 == p1


def test_making_csv(mock_SysVals):
    """Ensure the particles csv_header aligns with the string property."""
    p1 = Particle()
    p1.x[:] = (1, 2)  # set some dummy values
    p1.a[:] = (-1, 0)  # set some dummy values

    csv_str = Particle.csv_header + '\n' + p1.csv
    reader = DictReader(StringIO(csv_str))

    p1_from_csv = next(reader)

    assert p1_from_csv['x_x'] == str(p1.x[0])
    assert p1_from_csv['x_y'] == str(p1.x[1])
    assert p1_from_csv['v_x'] == str(p1.v[0])
    assert p1_from_csv['v_y'] == str(p1.v[1])
    assert p1_from_csv['a_x'] == str(p1.a[0])
    assert p1_from_csv['a_y'] == str(p1.a[1])
    assert p1_from_csv['D'] == str(p1.D)
    assert p1_from_csv['m'] == str(p1.m)
    assert p1_from_csv['bound'] == str(p1.bound)
