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

# a = [1]
# b = a
# a = [2]
# print(b)

# x = np.arange(1, 10000)
# y = np.cumsum(1 / np.tan(x))
#
# plt.plot(x, y, '.-')
# plt.grid()
#
# plt.figure()
# plt.plot(x, np.sin(x))
# plt.plot(x, np.cos(x))
#
# plt.show()

# exec(open("RegSandbox/GPyTest.py").read())

# import itertools
# import sympy
#
# b = sympy.primerange(3, 18)
# a = itertools.permutations(b, 2)
# for i in a:
#     print(i)

a = np.loadtxt('OptSandbox/ProgTxt/in_iter_1.csv')