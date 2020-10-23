import numpy as np
import matplotlib.pyplot as plt
import GPy.models


################################################################################
# Expensive Function
def fe(x):
    return (6.0 * x - 2.) ** 2 * np.sin(12 * x - 4)


# Cheap Function
def fc(x):
    A = 0.5
    B = 10
    C = 5
    return A * fe(x) + B * (x - 0.5) - C

# Expensive Function
def ge(x):
    return np.array(x[:, 0] ** 2 + x[:, 1] ** 2).reshape(-1, 1)

# Cheap Function
def gc(x):
    A = 0.5
    B = 10
    C = 5
    return A * ge(x) + B * (x[:, 0].reshape(-1, 1) - 0.5) - C

################################################################################

# x = np.linspace(0, 1, 100).reshape(-1, 1)
#
# Xl = np.linspace(0, 1, 11).reshape(-1, 1)
# Xh = np.array([0, 0.4, 0.6, 0.8, 1]).reshape(-1, 1)
#
# X = [Xl, Xh]
#
# Yl = fc(Xl)
# Yh = fe(Xh)
#
# Y = [Yl, Yh]

des_grid_x = np.linspace(-2, 2, 100)
des_grid_y = np.linspace(-2, 2, 100)
des_grid_xx, des_grid_yy = np.meshgrid(des_grid_x, des_grid_y)
des_grid = np.array([des_grid_xx.reshape(-1, 1), des_grid_yy.reshape(-1, 1)]).squeeze().T

low_grid_x = np.linspace(-1.5, 1.5, 5)
low_grid_y = np.linspace(-1.5, 1.5, 5)
low_grid_xx, low_grid_yy = np.meshgrid(low_grid_x, low_grid_y)
low_grid = np.array([low_grid_xx.reshape(-1, 1), low_grid_yy.reshape(-1, 1)]).squeeze().T

x = des_grid

Xl = low_grid
Xh = np.array([[0.1, 0.3], [-1.5, 1], [0, 0.5], [1.2, -0.3]])

X = [Xl, Xh]

Yl = gc(Xl)
Yh = ge(Xh)

Y = [Yl, Yh]

# print(X)
# print(Y)

m = GPy.models.multiGPRegression(X, Y)

m.optimize_restarts(restarts=4)
m.models[1]['Gaussian_noise.variance'] = 0.

print(m.predict(x))

### Visualization

# plt.plot(x, m.predict(x)[0][0])
# plt.plot(x, m.predict(x)[0][1])
# plt.plot(x, fc(x))
# plt.plot(x, fe(x))

# plt.scatter(Xl, Yl)
# plt.scatter(Xh, Yh)

# plt.grid()

# m.plot()
# plt.show()
