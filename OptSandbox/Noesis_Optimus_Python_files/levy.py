import numpy as np

a = [[0, 0]]
def levy(x):
    function_name = 'Levy'  # to output name of function
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    if x.ndim == 1:
        x = np.reshape(x, (-1, 2))  # reshape into 2d array
    #
    n_points, n_features = np.shape(x)
    y = np.empty((n_points, 1))
    #
    for ii in range(n_points):
        z = 1 + (x[ii, :] - 1) / 4
        y[ii] = (np.sin(np.pi * z[0]) ** 2
                 + sum((z[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * z[:-1] + 1) ** 2))
                 + (z[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * z[-1]) ** 2))
    return np.atleast_1d(y), function_name
b = levy(a)[0][0][0]

open('levy_output.txt', 'w').write(str(b))