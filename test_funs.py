import numpy as np

# a = [[0, 0]]
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
# b = ackley(a)[0][0][0]

# open('ackley_output.txt', 'w').write(str(b))

# a = [[0, 0]]
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
# b = levy(a)[0][0][0]

# open('levy_output.txt', 'w').write(str(b))

# a = [[0, 0]]
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
# b = rosenbrock(a)[0][0][0]

# open('rosenbrock_output.txt', 'w').write(str(b))

# a = [[0, 0]]
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
# b = schwefel(a)[0][0][0]

# open('schwefel_output.txt', 'w').write(str(b))