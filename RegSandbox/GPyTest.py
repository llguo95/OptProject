import numpy as np
import matplotlib.pyplot as plt
import GPy.models

from sklearn.gaussian_process import GaussianProcessRegressor

np.random.seed(123)

################################################################################
# Expensive Function
def fe(x):
    return (6.0 * x - 2.) ** 2 * np.sin(12 * x - 4) + 0.1


# Cheap Function
def fc(x):
    # return x * np.cos(x)
    A = 0.5
    B = 10
    C = 5
    return A * fe(x) + B * (x - 0.5) - C + 0.2


################################################################################

optimizer_bool = True
optimizer_string = 'simplex'
num_of_restarts = 10

###########

x = np.linspace(0, 1, 100).reshape(-1, 1)

Xl = np.linspace(0, 1, 6).reshape(-1, 1)
# Xl = np.random.random(7).reshape(-1, 1)
# Xl = np.array([0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).reshape(-1, 1)
# Xl = np.linspace(0.04, 0.96, 11).reshape(-1, 1)

Xh = np.array([0, 0.4, 1]).reshape(-1, 1)
# Xh = np.array([0, 0.4, 0.6, 0.8, 1]).reshape(-1, 1)
# Xh = np.array([0.4, 0.8]).reshape(-1, 1)

X = [Xl, Xh]

Yl = fc(Xl)
Yh = fe(Xh)

Y = [Yl, Yh]

m = GPy.models.multiGPRegression(X, Y)

# m = GPy.models.GPRegression(Xl, Yl)

# gpr = GaussianProcessRegressor().fit(Xl, Yl)
# mu_par, sigma_par = gpr.predict(np.array(x), return_std=True)

m.models[0].preferred_optimizer = optimizer_string
m.models[1].preferred_optimizer = optimizer_string

if optimizer_bool:
    m.optimize_restarts(restarts=num_of_restarts, verbose=False)

# m.models[0]['Gaussian_noise.variance'] = 0
# m.models[0]['rbf.variance'] = 1.5
# m.models[0]['rbf.lengthscale'].fix(0.1)

# m.models[1]['Gaussian_noise.variance'] = 0.001
# m.models[1]['Gaussian_noise.variance'].fix(0)
# m.models[1]['rbf.variance'] = 1.5
# m.models[1]['rbf.lengthscale'].fix(0.1)

print(m.models[1].log_likelihood())

print(m)

### Prediction (MAKE SURE ALL HYPERPARAMETERS ARE SET CORRECTLY)
mu, sigma = m.predict(x)

### Visualization

plt.plot(x, mu[0], color='b', label='MF cheap GPR (regular GPR)')
plt.plot(x, mu[0] + 2 * sigma[0], color='k', lw=.5)
plt.plot(x, mu[0] - 2 * sigma[0], color='k', lw=.5)
plt.fill_between(x.flatten(), mu[0].flatten() - 2 * sigma[0].flatten(), mu[0].flatten() + 2 * sigma[0].flatten(), alpha=0.2, color='b')

# plt.plot(x, mu_par, color='r', label='Regular GPR', alpha=0.3)
# plt.plot(x, mu_par + 2 * sigma_par, color='k')
# plt.plot(x, mu_par - 2 * sigma_par, color='k')
# plt.fill_between(x.flatten(), mu_par.flatten() - 2 * sigma_par.flatten(), mu_par.flatten() + 2 * sigma_par.flatten(), alpha=0.2)

plt.plot(x, mu[1], color='orange', label='MF expensive GPR')
plt.plot(x, mu[1] + 2 * sigma[1], color='k', lw=.5)
plt.plot(x, mu[1] - 2 * sigma[1], color='k', lw=.5)
plt.fill_between(x.flatten(), mu[1].flatten() - 2 * sigma[1].flatten(), mu[1].flatten() + 2 * sigma[1].flatten(), alpha=0.2, color='orange')

plt.plot(x, fc(x), '--', color='b', label='Exact cheap function')
plt.plot(x, fe(x), '--', color='orange', label='Exact expensive function')

plt.legend()

plt.scatter(Xl, Yl, color='b')
plt.scatter(Xh, Yh, color='orange')

plt.grid()

# m.plot()
plt.show()
