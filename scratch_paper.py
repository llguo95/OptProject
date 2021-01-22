import numpy as np
import matplotlib.pyplot as plt

# from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from scipy.optimize import rosen

from scipy.stats import norm

import GPy.models

import pandas as pd

# in_1 = np.linspace(50, 200, 10)
# in_2 = np.linspace(30, 34.18, 10)
# in_mesh = np.meshgrid(in_1, in_2)
# in_arr = np.array([layer.reshape(-1, 1) for layer in in_mesh]).squeeze().T
#
# output = np.loadtxt('Data/c_Resp_Surface_case6.txt').reshape((100,))
# df = pd.DataFrame()
# df.insert(0, 'h0', in_arr[:, 1])
# df.insert(1, 'rs', in_arr[:, 0])
# df.insert(2, 'gap', output)
#
# print(df)

# df.to_csv('Data/resp_data_LF5.csv', index=False)

df = pd.read_csv('test.csv')
print(df.loc[22, :])

# print(np.array(df[['h0', 'rs']]).reshape((100, 2)))

##############################################################################

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

# a = np.loadtxt('OptSandbox/ProgTxt/in_iter_1.csv')

# a = np.array([float('1')])
# print(np.shape(a))

# a = np.array([1, 2, 3])
# print(a)
# a = np.array(a)
# print(a)