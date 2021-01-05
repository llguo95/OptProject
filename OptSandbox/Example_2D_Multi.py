import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import GPy.models
from sklearn.preprocessing import StandardScaler

import os
folder_path = os.getcwd()

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

def acqUCB(x_par, gpr, X_train=None, kappa=4):
    # print(x_par)
    mu_par, sigma_par = gpr.predict(x_par)

    # mu_par = mu_par[1]
    # sigma_par = sigma_par[1]

    acq_lf = mu_par[0] + kappa * sigma_par[0]
    acq_hf = mu_par[1] + kappa * sigma_par[1]

    return acq_lf, acq_hf

# Expensive Function
def fe(x):
    return 10 * np.cos(np.pi / 2 * x[:, 0]) * np.cos(np.pi / 4 * x[:, 1])

# Cheap Function
def fc(x):
    A = 0.5
    # B = 10
    B = 5
    C = 5
    return A * fe(x) + B * (x[:, 0] - 0.5) - C

### 1D
# X = np.array([[0.5]])
# y = f(X)

# des_grid = np.linspace(0, 1, 100).reshape(-1, 1)

# Xl = np.linspace(0, 1, 11).reshape(-1, 1)
# Xh = np.array([0, 0.4, 0.6, 0.8, 1]).reshape(-1, 1)
#
# X = [Xl, Xh]
#
# Yl = fc(Xl)
# Yh = fe(Xh)
#
# Y = [Yl, Yh]

### 2D
X = np.array([[0.5, 0.4]])
Y = fe(X).reshape(-1, 1)

in_1 = np.linspace(-2, 2, 100)
in_2 = np.linspace(-2, 2, 100)
in_mesh = np.meshgrid(in_1, in_2)
in_arr = np.array([layer.reshape(-1, 1) for layer in in_mesh]).squeeze().T

scaler = StandardScaler()
scaler.fit(in_arr)

X_scaled = scaler.transform(X)
des_grid_scaled = scaler.transform(in_arr)

### Loop

x = np.array([[0.5, 0.5]])

mfDoE = [np.array(x), np.array(x)]
mfDoE_hist = [np.array(x), np.array(x)]

mfDoE_evals = [np.array([fc(x)]), np.array([fe(x)])]
mfDoE_evals_hist = [np.array([fc(x)]), np.array([fe(x)])]

n_features = 2
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
    mfgpr_step = GPy.models.multiGPRegression(mfDoE, mfDoE_evals)

    # mfgpr_step.optimize_restarts(restarts=2)

    mfgpr_step.models[0]['Gaussian_noise.variance'] = 0
    # mfgpr_step.models[0]['rbf.variance'] = 1.5
    # mfgpr_step.models[0]['rbf.lengthscale'] = 0.1
    # # #
    mfgpr_step.models[1]['Gaussian_noise.variance'] = 0
    # # mfgpr_step.models[1]['rbf.variance'] = 1.5
    # mfgpr_step.models[1]['rbf.lengthscale'] = 0.1

    # print('step', i)
    # print(mfgpr_step)

    mu_mf, sigma_mf = mfgpr_step.predict(np.array(x).reshape((1, n_features)))

    x = in_arr[np.argmax(acqUCB(in_arr, mfgpr_step, mfDoE)[m])]

    y_step = g(np.array([x]))

    mfDoE[m] = np.append(mfDoE[m], x).reshape(-1, n_features)
    mfDoE_evals[m] = np.append(mfDoE_evals[m], y_step).reshape(-1, 1)

    # print(m)
    # print(mfDoE[m])

mfDoE_hist[m] = mfDoE[m][:-1]
mfDoE_evals_hist[m] = mfDoE_evals[m][:-1]

mfDoE_hist[1 - m] = mfDoE[1 - m]
mfDoE_evals_hist[1 - m] = mfDoE_evals[1 - m]

# test = mfgpr_step.predict(in_arr)
# print(test)
y_pred_arr, sigma_pred_arr = mfgpr_step.predict(in_arr)
y_pred_grid = [y_fid.reshape(np.shape(in_mesh[0])) for y_fid in y_pred_arr]
sigma_pred_grid = [sigma_fid.reshape(np.shape(in_mesh[0])) for sigma_fid in sigma_pred_arr]

fig1 = plt.figure(figsize=(11, 14))

ax1 = fig1.add_subplot(321, projection='3d')
ax1.plot_surface(in_mesh[0], in_mesh[1], y_pred_grid[0], cmap='viridis')

ax1 = fig1.add_subplot(322, projection='3d')
ax1.plot_surface(in_mesh[0], in_mesh[1], y_pred_grid[1], cmap='viridis')

ax1 = fig1.add_subplot(323, projection='3d')
ax1.plot_surface(in_mesh[0], in_mesh[1], sigma_pred_grid[0], cmap='viridis')

ax1 = fig1.add_subplot(324, projection='3d')
ax1.plot_surface(in_mesh[0], in_mesh[1], sigma_pred_grid[1], cmap='viridis')

ax1 = fig1.add_subplot(325, projection='3d')
ax1.plot_surface(in_mesh[0], in_mesh[1], acqUCB(in_arr, mfgpr_step)[0].reshape(np.shape(in_mesh[0])), cmap='viridis')

ax1 = fig1.add_subplot(326, projection='3d')
ax1.plot_surface(in_mesh[0], in_mesh[1], acqUCB(in_arr, mfgpr_step)[1].reshape(np.shape(in_mesh[0])), cmap='viridis')

fig2 = plt.figure()

ax = fig2.add_subplot(111, projection='3d')
ax.plot_surface(in_mesh[0], in_mesh[1], fe(in_arr).reshape(np.shape(in_mesh[0])), cmap='viridis')

fig3, axs = plt.subplots(2, 2)
axs[0, 0].contourf(in_mesh[0], in_mesh[1], y_pred_grid[0], cmap='viridis')
axs[0, 1].contourf(in_mesh[0], in_mesh[1], y_pred_grid[1], cmap='viridis')
axs[1, 0].contourf(in_mesh[0], in_mesh[1], sigma_pred_grid[0], cmap='viridis')
axs[1, 1].contourf(in_mesh[0], in_mesh[1], sigma_pred_grid[1], cmap='viridis')

print(mfDoE)
print(mfDoE_evals)
axs[0, 0].scatter(mfDoE_hist[0][:, 0], mfDoE_hist[0][:, 1], color='red')
axs[0, 1].scatter(mfDoE_hist[1][:, 0], mfDoE_hist[1][:, 1], color='red')
# axs[0, 1].scatter(mfDoE_hist[1], mfDoE_evals_hist[1])

# y_pred = y_pred[1]
# sigma_pred = sigma_pred[1]

# print(mfDoE[0])
# print(-mfDoE_evals[0])

### Visualization ###

# fig1, axs1 = plt.subplots(2, 1, figsize=(5, 8))
#
# axs1[0].plot(des_grid, -fe(des_grid), '--', color='orange', label='Expensive function (exact)')
# axs1[0].plot(des_grid, -fc(des_grid), '--', color='blue', label='Cheap function (exact)')
# axs1[0].plot(des_grid, -y_pred[1], 'orange', lw=2)
# axs1[0].plot(des_grid, -y_pred[0], 'blue', lw=2)
# axs1[0].plot(des_grid, -y_pred[0].flatten() - 2 * sigma_pred[0].flatten(), 'k', lw=.5)
# axs1[0].plot(des_grid, -y_pred[0].flatten() + 2 * sigma_pred[0].flatten(), 'k', lw=.5)
# axs1[0].plot(des_grid, -y_pred[1].flatten() - 2 * sigma_pred[1].flatten(), 'k', lw=.5)
# axs1[0].plot(des_grid, -y_pred[1].flatten() + 2 * sigma_pred[1].flatten(), 'k', lw=.5)
#
# # print(max(sigma_pred[1]))
#
# axs1[0].fill_between(des_grid.flatten(), -y_pred[1].flatten() + 2 * sigma_pred[1].flatten(), -y_pred[1].flatten() - 2 * sigma_pred[1].flatten(), alpha=0.2, color='orange')
# axs1[0].fill_between(des_grid.flatten(), -y_pred[0].flatten() + 2 * sigma_pred[0].flatten(), -y_pred[0].flatten() - 2 * sigma_pred[0].flatten(), alpha=0.2, color='blue')
#
# # axs1[0].fill_between(des_grid.flatten(), -y_pred[0].flatten() - 10000 * sigma_pred[0].flatten(), -y_pred[0].flatten() + 10000 * sigma_pred[0].flatten(), alpha=0.2, color='r')
#
# axs1[0].scatter(mfDoE_hist[1], -mfDoE_evals_hist[1], color='orange')
# axs1[0].scatter(mfDoE_hist[0], -mfDoE_evals_hist[0], color='blue')
#
# # print(m_pre)
# # print(mfDoE[0])
# # print(-mfDoE_evals[0])
# # axs1[0].scatter(mfDoE_new, -mfDoE_new_eval, color='red')
#
# # axs1[0].set_xlabel('x')
# # axs1[0].set_ylabel('y')
# # axs1[0].set_title('BO iteration step %d' % k)
# # axs1[0].set_ylim([-3.5, 2.75])
# axs1[0].legend()
#
# acq = acqUCB(des_grid, mfgpr_step, mfDoE)
# # print(acq[0])
# axs1[1].plot(des_grid, (acq[0] - min(acq[0])) / (max(acq[0]) - min(acq[0])), color='blue')
# axs1[1].scatter(des_grid[np.argmax(acq[0])], 1, color='k')
# axs1[1].plot(des_grid, (acq[1] - min(acq[1])) / (max(acq[1]) - min(acq[1])), color='orange')
# axs1[1].scatter(des_grid[np.argmax(acq[1])], 1, color='k')
# axs1[1].set_xlabel('x')
# axs1[1].set_ylabel('Acquisition (normalized)')
# # axs1[1].set_title('Expected improvement')
# #
# plt.tight_layout()
#
# for ax in axs1: ax.grid()
# #
# # print(X)
# # print(X[np.argmin(-y)])
# # print(-y)
# # print(min(-y))
# #
# # plt.savefig('MFBO_step_%d.png' % (k - 1))
plt.show()