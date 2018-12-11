import sph_stub as sph
import numpy as np
import pickle as pi
import os


def test_W_dW():
    domain = sph.SPH_main()
    domain.set_values()
    a = sph.SPH_particle(x=np.array([5, 6]))
    b = sph.SPH_particle(x=np.array([5.02, 6.04]))
    c = sph.SPH_particle(x=np.array([5.0003, 6.0003]))
    d = sph.SPH_particle(x=np.array([9, 15]))
    p_j_list = [b, c, d]

    assert(np.isclose(domain.W(a, p_j_list),
                      [3.6895734125, 672.40868104, 0.]).all())
    assert (np.isclose(domain.dW(a, p_j_list),
                       [-1520.71259924, -1251.03179429, 0.]).all())
    return None


def test_rho_smoothing():
    domain = sph.SPH_main()
    domain.set_values()

    a = sph.SPH_particle(x=np.array([5, 6]))
    a.rho = 1000  # kg m^-3

    b = sph.SPH_particle(x=np.array([5.02, 6.04]))
    b.rho = 1200  # kg m^-3

    c = sph.SPH_particle(x=np.array([5.0003, 6.0003]))
    c.rho = 900  # kg m^-3

    d = sph.SPH_particle(x=np.array([9, 15]))
    d.rho = 100  # kg m^-3

    p_j_list = [a, b, c, d]
    assert (np.isclose(domain.rho_smoothing(a, p_j_list), 947.9241831713144))
    return None


# def test_setup_save():
#     # setup the system and save
#     domain = sph.SPH_main()
#     domain.set_values()
#     domain.initialise_grid()
#     domain.place_points(domain.min_x, domain.max_x)
#     domain.set_up_save(name='test', path='./')
#     domain.file.close()
#
#     # check the files exist
#     assert os.path.exists('test_config.pkl')
#     assert os.path.exists('test.csv')
#
#     # load files up again
#     load_dict = pi.load(open('./test_config.pkl', 'rb'))
#     load_csv = open('./test.csv', 'rb')
#
#     # check pickle saves correctly
#     for key in load_dict.keys():
#         if key != 'file':
#             assert np.all(load_dict[key] == vars(domain)[key])
#
#     for i, line in enumerate(load_csv):
#         if i < 2:
#             assert str(line)[2] == '#'  # check first lines two lead with #
#     assert i == 2  # check only 3 lines saved
