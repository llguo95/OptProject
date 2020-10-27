import numpy as np
import matplotlib.pyplot as plt
import GPy.models

from sklearn.gaussian_process import GaussianProcessRegressor

np.random.seed(123)

################################################################################
# Expensive Function
def fe(x):
    return (6.0 * x - 2.) ** 2 * np.sin(12 * x - 4)


# Cheap Function
def fc(x):
    # return x * np.cos(x)
    A = 0.5
    B = 10
    C = 5
    return A * fe(x) + B * (x - 0.5) - C


################################################################################

x = np.linspace(0, 1, 100).reshape(-1, 1)

# Xl = np.linspace(0, 1, 11).reshape(-1, 1)
Xl = np.linspace(0.04, 0.96, 11).reshape(-1, 1)
# Xl = np.array([0.4]).reshape(-1, 1)
Xh = np.array([0, 0.4, 0.6, 0.8, 1]).reshape(-1, 1)
# Xh = np.array([0.4, 0.8]).reshape(-1, 1)

X = [Xl, Xh]

Yl = fc(Xl)
Yh = fe(Xh)

Y = [Yl, Yh]

# print(X)
# print(Y)

m = GPy.models.multiGPRegression(X, Y)

# gpr = GaussianProcessRegressor().fit(Xl, Yl)
# mu_par, sigma_par = gpr.predict(np.array(x), return_std=True)

m.optimize_restarts(restarts=4, verbose=False)
# m.models[0]['Gaussian_noise.variance'] = 0
m.models[1]['Gaussian_noise.variance'] = 0.

print(m)

### Prediction (MAKE SURE ALL HYPERPARAMETERS ARE SET CORRECTLY)
mu, sigma = m.predict(x)

# print(sigma[0])

### Visualization

plt.plot(x, mu[0], color='r', label='MF cheap GPR (regular GPR)')
plt.plot(x, mu[0] + 2 * sigma[0], color='k')
plt.plot(x, mu[0] - 2 * sigma[0], color='k')
plt.fill_between(x.flatten(), mu[0].flatten() - 2 * sigma[0].flatten(), mu[0].flatten() + 2 * sigma[0].flatten(), alpha=0.2)

# plt.plot(x, mu_par, color='r', label='Regular GPR', alpha=0.3)
# plt.plot(x, mu_par + 2 * sigma_par, color='k')
# plt.plot(x, mu_par - 2 * sigma_par, color='k')
# plt.fill_between(x.flatten(), mu_par.flatten() - 2 * sigma_par.flatten(), mu_par.flatten() + 2 * sigma_par.flatten(), alpha=0.2)

plt.plot(x, mu[1], color='b', label='MF expensive GPR')
plt.plot(x, mu[1] + 2 * sigma[1], color='k')
plt.plot(x, mu[1] - 2 * sigma[1], color='k')
plt.fill_between(x.flatten(), mu[1].flatten() - 2 * sigma[1].flatten(), mu[1].flatten() + 2 * sigma[1].flatten(), alpha=0.2)

plt.plot(x, fc(x), '--', color='orange', label='Exact cheap function')
plt.plot(x, fe(x), '--', color='g', label='Exact expensive function')

plt.legend()

plt.scatter(Xl, Yl, color='red')
# plt.scatter(Xh, Yh, color='blue')

plt.grid()

# m.plot()
plt.show()
