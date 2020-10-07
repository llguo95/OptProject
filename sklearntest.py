import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *

def f(x):
    return x * np.cos(x)

X = np.array([-1, 0, 0.5, 3]).reshape(-1, 1)
y = f(X)

D = np.linspace(-2, 4, 100).reshape(-1, 1)

kernel = RBF()
gpr = GaussianProcessRegressor(kernel=kernel,
                               random_state=0).fit(X, y)
y_pred, sigma = gpr.predict(D, return_std=True)
print(X)
print(y)

plt.plot(D, f(D))
plt.plot(D, y_pred)
plt.scatter(X, y, color='r')
plt.grid()
plt.show()