import numpy as np

a = [[0, 0]]
def rosenbrock(x):  # rosen.m
    function_name = 'Rosenbrock'  # to output name of function
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    if x.ndim == 1:
        x = np.reshape(x, (-1, 2))  # reshape into 2d array
    #
    n_points, n_features = np.shape(x)
    y = np.empty((n_points, 1))
    #
    for ii in range(n_points):
        x0 = x[ii, :-1]
        x1 = x[ii, 1:]
        y[ii] = (sum((1 - x0) ** 2)
                 + 100 * sum((x1 - x0 ** 2) ** 2))
    return np.atleast_1d(y), function_name
b = rosenbrock(a)[0][0][0]

open('rosenbrock_output.txt', 'w').write(str(b))