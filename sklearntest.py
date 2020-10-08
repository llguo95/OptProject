import numpy as np
import matplotlib.pyplot as plt

# from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from scipy.optimize import rosen

from scipy.stats import norm

def acqEI(x_par, gpr, X_train, xi=0):
    mu_par, sigma_par = gpr.predict(np.array(x_par), return_std=True)
    mu_par = mu_par.flatten()
    f_max_X_train = max(f(X_train))
    z = (mu_par - f_max_X_train - xi) / sigma_par
    # if sigma_par > 0:
    #     res = (mu_par - f_min_X_train - xi) * norm.cdf(z) + sigma_par * norm.pdf(z)
    # else:
    #     res = 0
    return (mu_par - f_max_X_train - xi) * norm.cdf(z) + sigma_par * norm.pdf(z)

# def acqUCB(x_par, X_train, kappa):
#     # mu_par, sigma_par = gpr.predict(np.array(x_par).reshape(-1, 1), return_std=True)
#     return mu_par.flatten() + kappa * sigma_par

def f(x):
    return -x * np.cos(x)

def g(x):
    return np.array(rosen(x.T)).reshape(-1, 1)

X = np.array([[0.5]])
y = f(X)

des_grid = np.linspace(-2, 5, 100).reshape(-1, 1)

# X = np.array([[0.5, 0.4]])
# y = g(X)
#
# des_grid_x = np.linspace(-2, 2, 50)
# des_grid_y = np.linspace(-2, 2, 50)
# des_grid_xx, des_grid_yy = np.meshgrid(des_grid_x, des_grid_y)
# des_grid = np.array([des_grid_xx.reshape(-1, 1), des_grid_yy.reshape(-1, 1)]).squeeze().T

x = X[0]

n_features = 1
k = 6
for i in range(k):
    gpr_step = GaussianProcessRegressor().fit(X, y)
    mu_par, sigma_par = gpr_step.predict(np.array(x).reshape((1, n_features)), return_std=True)

    x = des_grid[np.argmax(acqEI(des_grid, gpr_step, X))]; y_step = f(x)
    X = np.append(X, x).reshape(-1, 1); y = np.append(y, y_step).reshape(-1, 1)

y_pred, sigma_pred = gpr_step.predict(des_grid, return_std=True)

### Visualization ###

fig1, axs1 = plt.subplots(2, 1, figsize=(5, 8))

axs1[0].plot(des_grid, -f(des_grid), '--')
axs1[0].plot(des_grid, -y_pred, 'r', lw=2)
axs1[0].plot(des_grid, -y_pred.flatten() - 2 * sigma_pred, 'k', lw=.5)
axs1[0].plot(des_grid, -y_pred.flatten() + 2 * sigma_pred, 'k', lw=.5)
axs1[0].fill_between(des_grid.flatten(), -y_pred.flatten() - 2 * sigma_pred, -y_pred.flatten() + 2 * sigma_pred, alpha=0.2, color='r')
axs1[0].scatter(X[:-1], -y[:-1], color='r')
axs1[0].set_xlabel('x')
axs1[0].set_ylabel('y')
axs1[0].set_title('x*cos(x) BO iteration step %d' % k)
axs1[0].set_ylim([-3.5, 2.75])

# gpr_step = GaussianProcessRegressor().fit(X, y)
# mu_par, sigma_par = gpr_step.predict(np.array(x).reshape((1, 1)), return_std=True)
#
# x = des_grid[np.argmax(acqEI(des_grid, gpr_step, X))]; y_step = f(x)
# X = np.append(X, x).reshape(-1, 1); y = np.append(y, y_step).reshape(-1, 1)

acq = acqEI(des_grid, gpr_step, X)
axs1[1].plot(des_grid, acqEI(des_grid, gpr_step, X) / max(acqEI(des_grid, gpr_step, X)))
axs1[1].scatter(des_grid[np.argmax(acq)], max(acq) / max(acqEI(des_grid, gpr_step, X)), color='k')
axs1[1].set_xlabel('x')
axs1[1].set_ylabel('Acquisition (normalized)')
axs1[1].set_title('Expected improvement')

plt.tight_layout()
for ax in axs1: ax.grid()

plt.savefig('step_%d.png' % k)
plt.show()