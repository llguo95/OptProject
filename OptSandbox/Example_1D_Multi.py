import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *

from scipy.stats import norm

import GPy.models

def acqEI(x_par, gpr, X_train, xi=0):
    mu_par, sigma_par = gpr.predict(np.array(x_par))

    mu_par = mu_par[1]
    sigma_par = sigma_par[1]

    mu_par = mu_par.flatten()
    # print(X_train)
    f_max_X_train = max(fe(X_train))

    z = (mu_par - f_max_X_train - xi) / sigma_par

    res_0 = (mu_par - f_max_X_train - xi) * norm.cdf(z) + sigma_par * norm.pdf(z)
    zero_array = np.zeros(np.shape(res_0))

    res = np.multiply(res_0, [np.argmax(a) for a in zip(zero_array, sigma_par)])

    return res

def acqUCB(x_par, gpr, X_train, kappa=.1):
    mu_par, sigma_par = gpr.predict(np.array(x_par).reshape(-1, 1))

    mu_par = mu_par[1]
    sigma_par = sigma_par[1]

    return mu_par + kappa * sigma_par

# def f(x):
#     return -x * np.cos(x)

# Expensive Function
def fe(x):
    return - (6.0 * x - 2.) ** 2 * np.sin(12 * x - 4)

# Cheap Function
def fc(x):
    A = 0.5
    B = 10
    C = 5
    return A * fe(x) + B * (x - 0.5) - C

### 1D
# X = np.array([[0.5]])
# y = f(X)

des_grid = np.linspace(0, 1, 100).reshape(-1, 1)

Xl = np.linspace(0, 1, 11).reshape(-1, 1)
Xh = np.array([0, 0.4, 0.6, 0.8, 1]).reshape(-1, 1)

X = [Xl, Xh]

Yl = fc(Xl)
Yh = fe(Xh)

Y = [Yl, Yh]

### Loop

x = X[0][5]

mfDoE = np.array([x])
mfDoE_evals = np.array([fe(x)])

n_features = 1
k = 2
for i in range(k):
    # gpr_step = GaussianProcessRegressor().fit(Xh, Yh)
    mfgpr_step = GPy.models.multiGPRegression(mfDoE, mfDoE_evals)

    # mu_par, sigma_par = gpr_step.predict(np.array(x).reshape((1, n_features)), return_std=True)
    mu_mf, sigma_mf = mfgpr_step.predict(np.array(x).reshape((1, n_features)))

    # x = des_grid[np.argmax(acqEI(des_grid, gpr_step, X))]
    x = des_grid[np.argmax(acqUCB(des_grid, mfgpr_step, X))]

    # y_step = f(x)
    y_step = fe(x)

    mfDoE = np.append(mfDoE, x).reshape(-1, n_features)
    mfDoE_evals = np.append(mfDoE_evals, y_step).reshape(-1, 1)

y_pred, sigma_pred = mfgpr_step.predict(des_grid)
y_pred = y_pred[1]
sigma_pred = sigma_pred[1]

print(mfDoE)
print(mfDoE_evals)

### Visualization ###

fig1, axs1 = plt.subplots(2, 1, figsize=(5, 8))

axs1[0].plot(des_grid, -fe(des_grid), '--')
axs1[0].plot(des_grid, -fc(des_grid), '--')
axs1[0].plot(des_grid, -y_pred, 'r', lw=2)
# axs1[0].plot(des_grid, -y_pred.flatten() - 2 * sigma_pred, 'k', lw=.5)
# axs1[0].plot(des_grid, -y_pred.flatten() + 2 * sigma_pred, 'k', lw=.5)
# axs1[0].fill_between(des_grid.flatten(), -y_pred.flatten() - 2 * sigma_pred, -y_pred.flatten() + 2 * sigma_pred, alpha=0.2, color='r')
axs1[0].scatter(mfDoE[:-1], -mfDoE_evals[:-1], color='r')
axs1[0].set_xlabel('x')
axs1[0].set_ylabel('y')
axs1[0].set_title('BO iteration step %d' % k)
# axs1[0].set_ylim([-3.5, 2.75])

acq = acqUCB(des_grid, mfgpr_step, X)
print(acq)
# axs1[1].plot(des_grid, acqEI(des_grid, gpr_step, X) / max(acqEI(des_grid, gpr_step, X)))
# axs1[1].scatter(des_grid[np.argmax(acq)], max(acq) / max(acqEI(des_grid, gpr_step, X)), color='k')
# axs1[1].set_xlabel('x')
# axs1[1].set_ylabel('Acquisition (normalized)')
# axs1[1].set_title('Expected improvement')
#
# plt.tight_layout()
# for ax in axs1: ax.grid()
#
# print(X)
# print(X[np.argmin(-y)])
# print(-y)
# print(min(-y))
#
# # plt.savefig('step_%d.png' % (k - 1))
plt.show()