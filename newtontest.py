import numpy as np
import matplotlib.pyplot as plt

# from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from scipy.optimize import rosen

from scipy.stats import norm

def f(x):
    return x * np.cos(x)

def df(x):
    return np.cos(x) - x * np.sin(x)

def d2f(x):
    return -2 * np.sin(x) - x * np.cos(x)

x = 0.5

X = np.array(x)
y = np.array(f(x))
k = 0
for i in range(k):
    x = x - df(x) / d2f(x)
    X = np.append(X, x); y = np.append(y, f(x))

des_grid = np.linspace(-2, 5, 100).reshape(-1, 1)

noise = np.random.rand(100).reshape((100, 1))

plt.plot(des_grid, f(des_grid) + noise, '--')
# plt.scatter(X, y, color='r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('x*cos(x) (noisy)')

plt.grid()

plt.savefig('noisy.png')

plt.show()