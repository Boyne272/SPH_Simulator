# import numpy as np
# import sph_stub as sph
# import pandas as pd
#
# x_min = [0, 0]
# x_max = [20, 10]
# t_final = 2
# dx = 0.8
#
#
# def f(x, y):
#     if 0 <= y <= 2 or (0 <= x <= 3 and 0 <= y <= 5):
#         return 1
#     else:
#         return 0
#
#
# # sph.sph_simulation(x_min, x_max, t_final, dx, f, path_name= "./raw_data")
# # self.file.write("Time,ID,R_x,R_y,V_x,V_y,a_x,a_y,Pressure," +
# #                         "Density,Boundary\n")
#
# # file_name = 'raw_data/2018-12-12-20hr-16m.csv'
# def test_speedofsound():
#
#     data = pd.read_csv(file_name, skiprows=2, index_col=False)
#     data = data.set_index('Time')
#
#     v = np.sqrt(data['V_x']**2+data['V_y']**2)
#
#     c0 = 20
#     assert np.all(v < c0)
#
#
# def test_density():
#     data = pd.read_csv(file_name, skiprows=2, index_col=False)
#     #data = data.set_index('Time')
#
#     rho = data['Density']
#
#     bound = data['Boundary']
#     #data
#
#     # check that density is positive
#     assert np.all(rho > 0)
#     #check that density changes