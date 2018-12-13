#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
<Description>

Created on Thu Dec 13 16:06:51 2018
@author: Richard Boyne rmb115@ic.ac.uk
"""

import numpy as np
import sph_stub as sph
import pandas as pd

def test_speedofsound():
    "test the speed never exceeds speed of sound (20)"
    v = np.sqrt(data['V_x']**2+data['V_y']**2)
    assert np.all(v < 20)


def test_density():
    dens = data['Density']
    rho0 = dens.loc[0].mean()

    # find the data lims
    N = len(times)-1  # ignore the first time
    minimums = np.empty(N)
    maximums = np.empty(N)
    stds = np.empty(N)
    for i, t in enumerate(times[1:]):
        minimums[i] = dens.loc[t].min()
        maximums[i] = dens.loc[t].max()
        stds[i] = dens.loc[t].std()

    # assert these values
    assert np.all(minimums > 0)  # check that density is positive
    assert np.all(minimums > rho0/1.5)  # check density is in bounds
    assert np.all(maximums < rho0*1.5)  # check density is in bounds
    assert np.all(stds > 0)  # check that density changes


def test_overlap():
    for _ in range(2):
        t = times[np.random.randint(len(times))]
        current = data.loc[t]
        positions = np.array(list(zip(current['R_x'], current['R_y'])))

        for i, p in enumerate(positions):
            for j, other in enumerate(positions):
                if i != j:
                    diff = sum((p-other)**2)
                    assert not np.isclose(diff, 0)
