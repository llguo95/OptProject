import numpy as np
import matplotlib.pyplot as plt

# from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from scipy.optimize import rosen

from scipy.stats import norm

# x = np.linspace(0, 1, 10)
# y = np.sin(2 * np.pi * x)
# zero_array = np.zeros(np.shape(y))
#
# z = np.array([max(a) for a in zip(y, zero_array)])
# z1 = np.array([np.argmax(a) for a in zip(y, zero_array)])
# print(z1)
#
# plt.plot(x, z)
# plt.show()

# print(10 ** np.linspace(-10, 4, 50))

# print(rosen([1, 1]))

a = [1]
b = a
a = [2]
print(b)