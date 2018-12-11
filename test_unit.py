import sph_stub as sph
import numpy as np


def test_W_dW():
    domain = sph.SPH_main()
    domain.set_values()
    a = sph.SPH_particle(x=np.array([5, 6]))
    b = sph.SPH_particle(x=np.array([5.02, 6.04]))
    c = sph.SPH_particle(x=np.array([5.0003, 6.0003]))
    d = sph.SPH_particle(x=np.array([9, 15]))
    p_j_list = [b, c, d]

    assert(np.isclose(domain.W(a, p_j_list), [3.6895734125, 672.40868104, 0.]).all())
    assert (np.isclose(domain.dW(a, p_j_list), [-1520.71259924, -1251.03179429, 0.]).all())
    return None


def test_rho_smoothing():
    domain = sph.SPH_main()
    domain.set_values()

    a = sph.SPH_particle(x=np.array([5, 6]))
    a.rho = 1000 #kg m^-3

    b = sph.SPH_particle(x=np.array([5.02, 6.04]))
    b.rho = 1200  # kg m^-3

    c = sph.SPH_particle(x=np.array([5.0003, 6.0003]))
    c.rho = 900  # kg m^-3

    d = sph.SPH_particle(x=np.array([9, 15]))
    d.rho = 100  # kg m^-3

    p_j_list = [a, b, c, d]
    assert (np.isclose(domain.rho_smoothing(a, p_j_list), 947.9241831713144))
    return None


def test_update_dt():
    domain = sph.SPH_main()
    domain.set_values()

    a = [10, 10]
    rho = [1000]
    v_ij = [20, 20]

    cfl_dt_check = (0.02*1.3)/20
    f_dt_check = np.sqrt((0.02*1.3)/10)
    a_dt_check = np.amin((0.02*1.3)/(20*np.sqrt((rho/1000)**(7-1))))

    dt_check = 0.2*np.amin([cfl_dt_check, f_dt_check, a_dt_check])
    print("check", dt_check)

    assert(np.isclose(domain.update_dt(a, v_ij, rho), dt_check))