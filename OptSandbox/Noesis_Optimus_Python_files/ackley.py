import numpy as np

a = [[0, 0]]
def ackley(x, a=20, b=0.2, c=2 * np.pi):
    function_name = 'Ackley'  # to output name of function
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    if x.ndim == 1:
        x = np.reshape(x, (-1, 2))  # reshape into 2d array
    #
    n_points, n_features = np.shape(x)
    y = np.empty((n_points, 1))
    #
    for ii in range(n_points):
        s1 = sum(x[ii, :] ** 2)
        s2 = sum(np.cos(c * x[ii, :]))
        y[ii] = -a * np.exp(-b * np.sqrt(s1 / n_features)) - np.exp(s2 / n_features) + a + np.exp(1)
    return np.atleast_1d(y), function_name
b = ackley(a)[0][0][0]

open('ackley_output.txt', 'w').write(str(b))

# in_1 = np.linspace(-10, 10, 100).reshape(-1, 1)
# in_2 = np.linspace(-10, 10, 100).reshape(-1, 1)
# in_mesh = np.meshgrid(in_1, in_2)
# in_arr = np.array([x.reshape(-1, 1) for x in in_mesh]).squeeze().T
#
# out_surf = ackley(in_arr)[0].reshape(np.shape(in_mesh[0]))

# from matplotlib import pyplot as plt
# fig = plt.figure()
# ax1 = fig.add_subplot(111, projection='3d')
# ax1.plot_surface(in_mesh[0], in_mesh[1], out_surf, cmap='viridis')
# plt.show()