import numpy as np
import sph_stub as sph

if __name__ == '__main__':
    """Create a single object of the main SPH type"""
    domain = sph.SPH_main()

    """Calls the function that sets the simulation parameters"""
    domain.set_values()
    """Initialises the search grid"""
    domain.initialise_grid()

    """
    Places particles in a grid over the entire domain - In your code you
    will need to place the fluid particles in only the appropriate locations
    """
    domain.place_points(domain.min_x, domain.max_x)

    """This is only for demonstration only - In your code these functions
    will need to be inside the simulation loop"""
    """This function needs to be called at each time step
    (or twice a time step if a second order time-stepping scheme is used)"""
    domain.allocate_to_grid()
    """This example is only finding the neighbours for a single partle
    - this will need to be inside the simulation loop and will need to be
    called for every particle"""
    domain.neighbour_iterate(domain.particle_list[100])

    domain.timestepping(tf=2e-4)


def test_speedofsound():
    save_file = open('raw_data/2018-12-12-14hr-15m.csv', 'r')

    for line in save_file:
        if line[0] == "#":
            pass
        else:
            data = line.split()


    y = 1
    assert x == y

def test_density():
    file_path = 3

    # np.load(file_path, 'r')

    y = 3
    assert file_path == y