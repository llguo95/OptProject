import numpy as np

a = [[0, 0]]
def schwefel(x):  # schw.m
    function_name = 'Schwefel'  # to output name of function
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    if x.ndim == 1:
        x = np.reshape(x, (-1, 2))  # reshape into 2d array
    #
    n_points, n_features = np.shape(x)
    y = np.empty((n_points, 1))
    #
    for ii in range(n_points):
        y[ii] = 418.9829 * n_features - sum(x[ii, :] * np.sin(np.sqrt(abs(x[ii, :]))))
    return np.atleast_1d(y), function_name
b = schwefel(a)[0][0][0]

open('schwefel_output.txt', 'w').write(str(b))