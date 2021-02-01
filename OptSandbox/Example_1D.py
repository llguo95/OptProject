import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *

from scipy.stats import norm

def acqEI(x_par, gpr, X_train, xi=0):
    mu_par, sigma_par = gpr.predict(np.array(x_par), return_std=True)
    mu_par = mu_par.flatten()
    f_max_X_train = max(f(X_train))

    z = (mu_par - f_max_X_train - xi) / sigma_par

    res_0 = (mu_par - f_max_X_train - xi) * norm.cdf(z) + sigma_par * norm.pdf(z)
    zero_array = np.zeros(np.shape(res_0))

    res = np.multiply(res_0, [np.argmax(a) for a in zip(zero_array, sigma_par)])

    return res

def acqUCB(x_par, gpr, X_train, kappa=2):
    mu_par, sigma_par = gpr.predict(np.array(x_par).reshape(-1, 1), return_std=True)
    return mu_par.flatten() + kappa * sigma_par

def f(x):
    # x = 5 * x
    return - 10 * x * np.cos(x * 11)

### 1D
X = np.array([[0.5]])
y = f(X)

des_grid = np.linspace(0, 1, 100).reshape(-1, 1)

### Loop

x = X[0]

n_features = 1
k = 10
for i in range(k):
    # gpr_step = GaussianProcessRegressor().fit(X, y)
    gpr_step = GaussianProcessRegressor(kernel=RBF(length_scale=.1) + WhiteKernel(noise_level=.02)).fit(X, y)
    # gpr_step = GaussianProcessRegressor(kernel=RBF(length_scale=0.2)).fit(X, y)
    mu_par, sigma_par = gpr_step.predict(np.array(x).reshape((1, n_features)), return_std=True)

    x = des_grid[np.argmax(acqUCB(des_grid, gpr_step, X))]
    y_step = f(x)
    X = np.append(X, x).reshape(-1, n_features)
    y = np.append(y, y_step).reshape(-1, 1)

y_pred, sigma_pred = gpr_step.predict(des_grid, return_std=True)

### Visualization ###

fig1, axs1 = plt.subplots(2, 1, figsize=(4, 6))

axs1[0].plot(des_grid, -f(des_grid), '--', color='r')
axs1[0].plot(des_grid, -y_pred, 'r', lw=2)
axs1[0].plot(des_grid, -y_pred.flatten() - 2 * sigma_pred, 'k', lw=.5)
axs1[0].plot(des_grid, -y_pred.flatten() + 2 * sigma_pred, 'k', lw=.5)

# print(des_grid.flatten())
# print(-y_pred.flatten() - 2 * sigma_pred.flatten())

axs1[0].fill_between(des_grid.flatten(), -y_pred.flatten() - 2 * sigma_pred, -y_pred.flatten() + 2 * sigma_pred, alpha=0.2, color='r')
axs1[0].scatter(X[:-1], -y[:-1], color='r')
# axs1[0].set_xlabel('x')
axs1[0].set_ylabel('y')
# axs1[0].set_title('x*cos(x) BO iteration step %d' % k)
# axs1[0].set_ylim([-3.5, 2.75])

acq = acqUCB(des_grid, gpr_step, X)
axs1[1].plot(des_grid, (acq - min(acq)) / (max(acq) - min(acq)), color='r')
axs1[1].scatter(des_grid[np.argmax(acq)], 1, color='k')
axs1[1].set_xlabel('x')
axs1[1].set_ylabel('Acquisition (normalized)')
# axs1[1].set_title('UCB')

plt.tight_layout()
for ax in axs1: ax.grid()

# print(X)
# print(X[np.argmin(-y)])
# print(-y)
# print(min(-y))

plt.savefig('BO_step_%d.png' % (k - 1))
plt.show()