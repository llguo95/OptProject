import numpy as np
import matplotlib.pyplot as plt
import GPy_MF.models


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
    return A * ge(x) + B * (x[:, 1].reshape(-1, 1) - 0.5) - C

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

# high_grid_x = np.linspace(-1.5, 1.5, 3)
# high_grid_y = np.linspace(-1.5, 1.5, 3)
# high_grid_xx, high_grid_yy = np.meshgrid(high_grid_x, high_grid_y)
# high_grid = np.array([high_grid_xx.reshape(-1, 1), high_grid_yy.reshape(-1, 1)]).squeeze().T

x = des_grid

Xl = low_grid
# Xh = high_grid
Xh = np.array([[0.1, 0.3], [-1.5, 1], [0, 0.5], [1.2, -0.3], [-1, -1]])

X = [Xl, Xh]

Yl = gc(Xl)
Yh = ge(Xh)

Y = [Yl, Yh]

# print(X)
# print(Y)

m = GPy_MF.models.multiGPRegression(X, Y)

m.optimize_restarts(restarts=4, verbose=False)
m.models[1]['Gaussian_noise.variance'] = 0.

mu, sigma = m.predict(x)

mu = [a.reshape(np.shape(des_grid_xx)) for a in mu]
sigma = [a.reshape(np.shape(des_grid_xx)) for a in sigma]

### Visualization

fig1, axs1 = plt.subplots(2, 3, figsize=(8, 5))

# print(gc(des_grid))

axs1[0, 0].contourf(des_grid_xx, des_grid_yy, gc(des_grid).reshape(np.shape(des_grid_xx)))
axs1[1, 0].contourf(des_grid_xx, des_grid_yy, ge(des_grid).reshape(np.shape(des_grid_xx)))
axs1[0, 1].contourf(des_grid_xx, des_grid_yy, mu[0])
axs1[1, 1].contourf(des_grid_xx, des_grid_yy, mu[1])
axs1[0, 2].contourf(des_grid_xx, des_grid_yy, sigma[0])
axs1[1, 2].contourf(des_grid_xx, des_grid_yy, sigma[1])

plt.tight_layout()

# plt.plot(x, m.predict(x)[0][0])
# plt.plot(x, m.predict(x)[0][1])
# plt.plot(x, fc(x))
# plt.plot(x, fe(x))

# plt.scatter(Xl, Yl)
# plt.scatter(Xh, Yh)

# plt.grid()

# m.plot()
plt.show()