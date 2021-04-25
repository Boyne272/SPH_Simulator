"""Vairous utilities."""

# plotting ------------------------------------------------

import matplotlib.pyplot as plt

def mpl_settings():
    """Nice Matplotlib settings."""
    plt.rc('axes', titlesize=20, labelsize=20)
    plt.rc('axes.formatter', limits=[-4, 4])
    plt.rc('ytick', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('lines', linewidth=1.5, markersize=7)
    plt.rc('figure', figsize=(9, 9))
    plt.rc('legend', fontsize=15)
mpl_settings()

# saving --------------------------------------------------


def csv_header():
    """Create a header for the particle data csv file."""
    return f'''# Created by team Southern on {datetime.now().strftime("%Y-%m-%d-%Hhr-%Mm")}
    # [s], [#], [m], [m], [m/s], [m/s], [Pa], [Kg/(m^3)], [bool]
    Time,ID,R_x,R_y,V_x,V_y,Pressure,Density,Boundary
    '''
