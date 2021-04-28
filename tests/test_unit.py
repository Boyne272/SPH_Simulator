"""Small collection of old unit tests."""

import numpy as np
import pickle as pi
import os

# import sph.sph_ap as ap
import sph.sph_fe as fe
# import sph.sph_ie as ie


def test_W_dW():
    domain1 = fe.SPH_main()
    system = domain1.sys
    # domain1.determine_values()
    # domain2 = ie.SPH_main()
    # domain2.determine_values()
    # domain3 = ap.SPH_main()
    # domain3.determine_values()

    # Particle class in fe, ie, ap are identical and unchanged
    a = fe.Particle(system, x=np.array([5, 6]))
    b = fe.Particle(system, x=np.array([5.02, 6.04]))
    c = fe.Particle(system, x=np.array([5.0003, 6.0003]))
    d = fe.Particle(system, x=np.array([9, 15]))
    p_j_list = [b, c, d]

    assert(np.isclose(domain1.W(a, p_j_list),
                      [3.6895734125, 672.40868104, 0.]).all())
    assert (np.isclose(domain1.dW(a, p_j_list),
                       [-1520.71259924, -1251.03179429, 0.]).all())
    # assert (np.isclose(domain2.W(a, p_j_list),
    #                    [3.6895734125, 672.40868104, 0.]).all())
    # assert (np.isclose(domain2.dW(a, p_j_list, 1),
    #                    [-1520.71259924, -1251.03179429, 0.]).all())
    # assert (np.isclose(domain3.W(a, p_j_list),
    #                    [3.6895734125, 672.40868104, 0.]).all())
    # assert (np.isclose(domain3.dW(a, p_j_list, 1),
    #                    [-1520.71259924, -1251.03179429, 0.]).all())


def test_rho_smoothing():
    domain1 = fe.SPH_main()
    system = domain1.sys
    # domain1.determine_values()
    # domain2 = ie.SPH_main()
    # domain2.determine_values()
    # domain3 = ap.SPH_main()
    # domain3.determine_values()

    a = fe.Particle(system, x=np.array([5, 6]))
    a.rho = 1000  # kg m^-3

    b = fe.Particle(system, x=np.array([5.02, 6.04]))
    b.rho = 1200  # kg m^-3

    c = fe.Particle(system, x=np.array([5.0003, 6.0003]))
    c.rho = 900  # kg m^-3

    d = fe.Particle(system, x=np.array([9, 15]))
    d.rho = 100  # kg m^-3

    p_j_list = [a, b, c, d]
    assert (np.isclose(domain1.rho_smoothing(a, p_j_list), 947.9241831713144))
    # assert (np.isclose(domain2.rho_smoothing(a, p_j_list), 947.9241831713144))
    # assert (np.isclose(domain3.rho_smoothing(a, p_j_list), 947.9241831713144))


# def test_setup_save():
#     # setup the system and save
#     def f(x, y):
#         if 0 <= y <= 2 or (0 <= x <= 3 and 0 <= y <= 5):
#             return 1
#         else:
#             return 0

#     # Save function is identical in all 3
#     domain1 = fe.SPH_main(x_min=[0, 0], x_max=[20, 20], dx=1)
#     # domain1.determine_values()
#     domain1.initialise_grid(f)
#     domain1.grid.allocate_to_grid()
#     domain1.set_up_save(name='test', path='./')
#     domain1.file.close()

#     # check the files exist
#     assert os.path.exists('test_config.pkl')
#     assert os.path.exists('test.csv')

#     # load files up again
#     load_dict = pi.load(open('./test_config.pkl', 'rb'))
#     load_csv = open('./test.csv', 'rb')

#     # check pickle saves correctly
#     for key in load_dict.keys():
#         if key != 'file':
#             assert np.all(load_dict[key] == vars(domain1)[key])

#     for i, line in enumerate(load_csv):
#         if i < 2:
#             assert str(line)[2] == '#'  # check first lines two lead with #




# def test_R_dW_artificial_pressure():
#     x_min = [0,0]
#     x_max = [10, 20]
#     dx = 1
#     def f(x, y):
#         if 0 <= y <= 2 or (0 <= x <= 3 and 0 <= y <= 5):
#             return 1
#         else:
#             return 0
#     domain = ap.SPH_main(x_min, x_max, dx=dx)
#     domain.determine_values()  # TODO fix me to have system rather than domain
#     domain.initialise_grid(f)
#     domain.allocate_to_grid()

#     p_i = ap.Particle(domain, x=np.array([5, 6]))
#     p_i.P = 5000
#     p_i.rho = 1200

#     p_j1 = ap.Particle(domain, x=np.array([5.02, 5.92]))
#     p_j1.P = 2000
#     p_j1.rho = 1205

#     p_j2 = ap.Particle(domain, x=np.array([5.04, 5.97]))
#     p_j2.P = 2000
#     p_j2.rho = 1205

#     p_j3 = ap.Particle(domain, x=np.array([5.08, 6.04]))
#     p_j3.P = 2000
#     p_j3.rho = 1205

#     pj_list = [p_j1, p_j2, p_j3]

#     assert (np.isclose(domain.R_artificial_pressure(p_i, pj_list, 1), [0.00009699, 0.00009699, 0.00009699]).all())
#     assert (np.isclose(domain.dW_artificial_pressure(p_i, pj_list, 1), [-0.03870062, -0.02372638, -0.04184862]).all())
