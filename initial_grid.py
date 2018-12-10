#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
<Description>

Created on Mon Dec 10 16:33:23 2018
@author: Richard Boyne rmb115@ic.ac.uk
"""

from sph_stub import SPH_main, SPH_particle
import numpy as np


def init_grid():
    """
    Create the intial system given in the documentation.
    Note the x, y axis are scaled to be 1, 2 respectivley

    This function operates by removing particles from a full grid, not ideal
    for user friendlyness
    """
    # set up the system with no particles
    system = SPH_main()
    system.set_values()
    system.max_x[:] = (2., 1.)  # set the grid to be the correct dimensions
    system.initialise_grid()

    # set up a full grid the grid
    system.place_points(system.min_x, system.max_x)
    # remove the unwanted points
    for p in system.particle_list.copy():
        if 2 > p.x[0] > 0 and 1 > p.x[1] > 0:  # not boundary node
            if p.x[1] > 0.5:
                system.particle_list.remove(p)
            elif p.x[0] > 0.3 and p.x[1] > 0.3:
                system.particle_list.remove(p)
    system.plot_current_state()

    return system