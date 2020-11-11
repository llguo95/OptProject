import numpy as np
import matplotlib.pyplot as plt
import GPy.models

from sklearn.gaussian_process import GaussianProcessRegressor

np.random.seed(123)

def f(x):
    return np.sin(2 * np.pi * x)

X = np.linspace(-1, 1, 10).reshape(-1, 1)
Y = f(X)

m = GPy.models.GPRegression(X, Y)

# m.optimize_restarts(restarts=4)

# m.parameters[0][0]
# m.parameters[1][0] = 0
print(m.parameters[0])

# print(m)

x = np.linspace(-1, 1, 100).reshape(-1, 1)
mu, sigma = m.predict(x)

plt.plot(x, mu)
plt.scatter(X, Y, color='red')

plt.show()