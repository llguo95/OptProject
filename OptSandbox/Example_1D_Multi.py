import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *

from scipy.stats import norm

import GPy_MF.models

np.random.seed(123)

def acqEI(x_par, gpr, X_train, xi=0):
    mu_par, sigma_par = gpr.predict(np.array(x_par))

    mu_par = mu_par[1]
    sigma_par = sigma_par[1]

    mu_par = mu_par.flatten()
    f_max_X_train = max(fe(X_train))

    z = (mu_par - f_max_X_train - xi) / sigma_par

    res_0 = (mu_par - f_max_X_train - xi) * norm.cdf(z) + sigma_par * norm.pdf(z)
    zero_array = np.zeros(np.shape(res_0))

    res = np.multiply(res_0, [np.argmax(a) for a in zip(zero_array, sigma_par)])

    return res

def acqUCB(x_par, gpr, X_train, kappa=2):
    print(x_par)
    mu_par, sigma_par = gpr.predict(np.array(x_par).reshape(-1, 1))

    # mu_par = mu_par[1]
    # sigma_par = sigma_par[1]

    acq_lf = mu_par[0] + kappa * sigma_par[0]
    acq_hf = mu_par[1] + kappa * sigma_par[1]

    return acq_lf, acq_hf
    # return acq_hf

# def f(x):
#     return -x * np.cos(x)

# Expensive Function
def fe(x):
    return - 10 * x * np.cos(x * 11)
    # return - (6.0 * x - 2.) ** 2 * np.sin(12 * x - 4)

# Cheap Function
def fc(x):
    A = 0.5
    B = 10
    # B = 2
    C = 5
    return A * fe(x) + B * (x - 0.5) - C
    # return fe(x)

### 1D
# X = np.array([[0.5]])
# y = f(X)

des_grid = np.linspace(0, 1, 100).reshape(-1, 1)

# Xl = np.linspace(0, 1, 11).reshape(-1, 1)
# Xh = np.array([0, 0.4, 0.6, 0.8, 1]).reshape(-1, 1)
#
# X = [Xl, Xh]
#
# Yl = fc(Xl)
# Yh = fe(Xh)
#
# Y = [Yl, Yh]

### Loop

x = np.array([0.5])

mfDoE = [np.array([x]), np.array([x])]
mfDoE_hist = [np.array([x]), np.array([x])]

mfDoE_evals = [np.array([fc(x)]), np.array([fe(x)])]
mfDoE_evals_hist = [np.array([fc(x)]), np.array([fe(x)])]

n_features = 1
k = 16 # number of iterations
for i in range(k): # optimization loop
    p = np.random.random()

    if i > 0:
        mfDoE_new = mfDoE[m][-1]
        mfDoE_new_eval = mfDoE_evals[m][-1]

    # if p <= 1/4 or i % 5 == 0:
    if i % 5 == 0:
        g = fe
        m = 1
    else:
        g = fc
        m = 0

    # print(mfDoE_evals)
    mfgpr_step = GPy_MF.models.multiGPRegression(mfDoE, mfDoE_evals)

    # mfgpr_step.optimize_restarts(restarts=2)

    mfgpr_step.models[0]['Gaussian_noise.variance'] = 0
    # mfgpr_step.models[0]['rbf.variance'] = 1.5
    mfgpr_step.models[0]['rbf.lengthscale'] = 0.1
    # # #
    mfgpr_step.models[1]['Gaussian_noise.variance'] = 0
    # # mfgpr_step.models[1]['rbf.variance'] = 1.5
    mfgpr_step.models[1]['rbf.lengthscale'] = 0.1

    # print('step', i)
    # print(mfgpr_step)

    mu_mf, sigma_mf = mfgpr_step.predict(np.array(x).reshape((1, n_features)))

    x = des_grid[np.argmax(acqUCB(des_grid, mfgpr_step, mfDoE)[m])]

    y_step = g(x)

    mfDoE[m] = np.append(mfDoE[m], x).reshape(-1, n_features)
    mfDoE_evals[m] = np.append(mfDoE_evals[m], y_step).reshape(-1, 1)

    # print(m)
    # print(mfDoE[m])

mfDoE_hist[m] = mfDoE[m][:-1]
mfDoE_evals_hist[m] = mfDoE_evals[m][:-1]

mfDoE_hist[1 - m] = mfDoE[1 - m]
mfDoE_evals_hist[1 - m] = mfDoE_evals[1 - m]

y_pred, sigma_pred = mfgpr_step.predict(des_grid)
# y_pred = y_pred[1]
# sigma_pred = sigma_pred[1]

# print(mfDoE[0])
# print(-mfDoE_evals[0])

### Visualization ###

fig1, axs1 = plt.subplots(2, 1, figsize=(5, 8))

axs1[0].plot(des_grid, -fe(des_grid), '--', color='orange', label='Expensive function (exact)')
axs1[0].plot(des_grid, -fc(des_grid), '--', color='blue', label='Cheap function (exact)')
axs1[0].plot(des_grid, -y_pred[1], 'orange', lw=2)
axs1[0].plot(des_grid, -y_pred[0], 'blue', lw=2)
axs1[0].plot(des_grid, -y_pred[0].flatten() - 2 * sigma_pred[0].flatten(), 'k', lw=.5)
axs1[0].plot(des_grid, -y_pred[0].flatten() + 2 * sigma_pred[0].flatten(), 'k', lw=.5)
axs1[0].plot(des_grid, -y_pred[1].flatten() - 2 * sigma_pred[1].flatten(), 'k', lw=.5)
axs1[0].plot(des_grid, -y_pred[1].flatten() + 2 * sigma_pred[1].flatten(), 'k', lw=.5)

# print(max(sigma_pred[1]))

axs1[0].fill_between(des_grid.flatten(), -y_pred[1].flatten() + 2 * sigma_pred[1].flatten(), -y_pred[1].flatten() - 2 * sigma_pred[1].flatten(), alpha=0.2, color='orange')
axs1[0].fill_between(des_grid.flatten(), -y_pred[0].flatten() + 2 * sigma_pred[0].flatten(), -y_pred[0].flatten() - 2 * sigma_pred[0].flatten(), alpha=0.2, color='blue')

# axs1[0].fill_between(des_grid.flatten(), -y_pred[0].flatten() - 10000 * sigma_pred[0].flatten(), -y_pred[0].flatten() + 10000 * sigma_pred[0].flatten(), alpha=0.2, color='r')

axs1[0].scatter(mfDoE_hist[1], -mfDoE_evals_hist[1], color='orange')
axs1[0].scatter(mfDoE_hist[0], -mfDoE_evals_hist[0], color='blue')

# print(m_pre)
# print(mfDoE[0])
# print(-mfDoE_evals[0])
# axs1[0].scatter(mfDoE_new, -mfDoE_new_eval, color='red')

# axs1[0].set_xlabel('x')
# axs1[0].set_ylabel('y')
# axs1[0].set_title('BO iteration step %d' % k)
# axs1[0].set_ylim([-3.5, 2.75])
axs1[0].legend()

acq = acqUCB(des_grid, mfgpr_step, mfDoE)
# print(acq[0])
axs1[1].plot(des_grid, (acq[0] - min(acq[0])) / (max(acq[0]) - min(acq[0])), color='blue')
axs1[1].scatter(des_grid[np.argmax(acq[0])], 1, color='k')
axs1[1].plot(des_grid, (acq[1] - min(acq[1])) / (max(acq[1]) - min(acq[1])), color='orange')
axs1[1].scatter(des_grid[np.argmax(acq[1])], 1, color='k')
axs1[1].set_xlabel('x')
axs1[1].set_ylabel('Acquisition (normalized)')
# axs1[1].set_title('Expected improvement')
#
plt.tight_layout()

for ax in axs1: ax.grid()
#
# print(X)
# print(X[np.argmin(-y)])
# print(-y)
# print(min(-y))
#
# plt.savefig('MFBO_step_%d.png' % (k - 1))
plt.show()