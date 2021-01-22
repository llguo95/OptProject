import numpy as np
import matplotlib.pyplot as plt
import GPy.models

from scipy.stats import multivariate_normal

from test_funs import *

from sklearn.preprocessing import StandardScaler

np.random.seed(123)

def f(x):
    x = scaler.inverse_transform(x)

    # return -rosenbrock(x)[0]
    return -schwefel(x)[0]
    # return -ackley(x)[0]
    # return -levy(x)[0]

    # return np.array([[multivariate_normal.pdf(x, mean=[0, 0])]])
    # return np.array([10 * np.cos(np.pi / 2 * x[:, 0]) * np.cos(np.pi / 4 * x[:, 1])])

def acqUCB(x, gpr, kappa=2):
    mu, sigma = gpr.predict(x)
    return mu + kappa * sigma

# x1 = np.linspace(-2, 2, 100); x2 = np.linspace(-2, 2, 100)
# x1 = np.linspace(-5, 5, 100); x2 = np.linspace(-5, 5, 100)
# x1 = np.linspace(-10, 10, 100); x2 = np.linspace(-10, 10, 100)
x1 = np.linspace(-500, 500, 100); x2 = np.linspace(-500, 500, 100)

x_mesh = np.meshgrid(x1, x2)
x_arr = np.hstack([layer.reshape(-1, 1) for layer in x_mesh])
scaler = StandardScaler()
scaler.fit(x_arr)

x_arr_tf = scaler.transform(x_arr)
# x_arr = np.vstack(np.array(x_mesh).T)

n_it = 49
# X = np.array([[0, 0]])
X = np.array([[1, 1]])
X_tf = scaler.transform(X)
Y = f(X_tf)
n_features = 2
for i in range(n_it):
    gpr_step = GPy.models.GPRegression(X_tf, Y)

    ### HP fixture
    # gpr_step.parameters[0]['rbf.variance'].fix(1e6)
    gpr_step.parameters[0]['rbf.lengthscale'].fix(.2)
    # gpr_step.parameters[1]['Gaussian_noise.variance'].fix(.5)

    ### HPO
    # gpr_step.preferred_optimizer = 'lbfgsb'
    # if True:
    #     gpr_step.optimize_restarts(num_restarts=4, verbose=False)

    x_tf = x_arr_tf[np.argmax(acqUCB(x_arr_tf, gpr_step))]

    ### Random point selection
    # if i > 0 and i % 5 == 0:
    #     if np.any(np.all(np.isin(X_tf, x_tf, True), axis=1)):
    #         # print(x_tf); print(X_tf)
    #         print(str(i) + ', repeat')
    #         x_tf = x_arr_tf[np.random.randint(len(x_arr_tf))]

    y = f(x_tf)

    X = np.append(X, scaler.inverse_transform(x_tf)).reshape(-1, n_features)
    X_tf = np.append(X_tf, x_tf).reshape(-1, n_features)

    Y = np.append(Y, y).reshape(-1, 1)

mu_arr, sigma_arr = gpr_step.predict(x_arr_tf)

def lvs(arr):
    return np.linspace(min(arr), max(arr), 100).flatten()

fig1, axs = plt.subplots(2, 2)
cf = [None] * 4
cf[0] = axs[0, 0].contourf(x_mesh[0], x_mesh[1], -f(x_arr_tf).reshape(np.shape(x_mesh[0])), levels=lvs(-f(x_arr_tf)))
axs[0, 0].set_title('True objective')
fig1.colorbar(cf[0], ax=axs[0, 0])

cf[1] = axs[0, 1].contourf(x_mesh[0], x_mesh[1], -mu_arr.reshape(np.shape(x_mesh[0])), levels=lvs(-mu_arr))
axs[0, 1].set_title('GP mean')
fig1.colorbar(cf[1], ax=axs[0, 1])

cf[2] = axs[1, 0].contourf(x_mesh[0], x_mesh[1], sigma_arr.reshape(np.shape(x_mesh[0])), levels=lvs(sigma_arr))
axs[1, 0].set_title('GP std')
fig1.colorbar(cf[2], ax=axs[1, 0])

acq_norm = (acqUCB(x_arr_tf, gpr_step) - min(acqUCB(x_arr_tf, gpr_step)))/(max(acqUCB(x_arr_tf, gpr_step)) - min(acqUCB(x_arr_tf, gpr_step)))

cf[3] = axs[1, 1].contourf(x_mesh[0], x_mesh[1], acq_norm.reshape(np.shape(x_mesh[0])), levels=lvs(acq_norm))
axs[1, 1].set_title('Acquisition (UCB, normalized)')
fig1.colorbar(cf[3], ax=axs[1, 1])

k = 0
for i in range(2):
    for j in range(2):
        axs[i, j].scatter(X[:-1, 0], X[:-1, 1], c=np.linspace(-1, 1, n_it), cmap='Greys')
        axs[i, j].scatter(X[-1, 0], X[-1, 1], c='red')
        # axs[i, j].scatter(1, 1, c='green')
        axs[i, j].contour(cf[k], colors='k', linewidths=.1)
        # for m in range(n_it):
        #     axs[i, j].annotate(m + 1, (X[m, 0], X[m, 1]), xytext=(-3.5, 5), textcoords='offset pixels', color='red')
        k += 1

print(X)
print()
print(-Y)
print()
print('Evaluation no. ', np.argmin(-Y) + 1)
print('in:', X[np.argmin(-Y)])
print('out:', min(-Y))
print()
print(gpr_step)

# x1_c = x1[::19]; x2_c = x2[::19]
# x_mesh_c = np.meshgrid(x1_c, x2_c)
# x_arr_c = np.hstack([layer.reshape(-1, 1) for layer in x_mesh_c])
# scaler_c = StandardScaler()
# scaler_c.fit(x_arr_c)
# x_arr_c_tf = scaler_c.transform(x_arr_c)
#
# X_reg_data_tf = X_tf[:-1]
#
# gpr_c = GPy.models.GPRegression(X_reg_data_tf, f(X_reg_data_tf))
#
# gpr_c.parameters[1]['Gaussian_noise.variance'].fix(0)
# mu_arr_c, sigma_arr_c = gpr_c.predict(x_arr_tf)
#
# plt.figure()
# cf_gpr = plt.contourf(x_mesh[0], x_mesh[1], -mu_arr_c.reshape(np.shape(x_mesh[0])), levels=lvs(-mu_arr_c))
# plt.scatter(scaler.inverse_transform(X_reg_data_tf)[:, 0], scaler.inverse_transform(X_reg_data_tf)[:, 1])
# plt.contour(cf_gpr, colors='k', linewidths=.1)

# plt.savefig('test1.svg')

plt.show()

# # print(X);
# # print()
# # print(Y);
# # print()
# # print(np.argmin(Y));
# # print();
# # print(min(Y));
# # print()
# # print(gpr_step)
#
# mu, sigma = gpr_step.predict(des_grid)
# fig1, axs1 = plt.subplots(2, 1, figsize=(5, 8))
# axs1[0].plot(des_grid, -f(des_grid), '--', color='red', lw=.75, label='True function')
# axs1[0].plot(des_grid, -mu, color='red', label='GPR mean')
# axs1[0].plot(des_grid, -mu + 2 * sigma, color='k', lw=.5)
# axs1[0].plot(des_grid, -mu - 2 * sigma, color='k', lw=.5)
# axs1[0].fill_between(des_grid.flatten(), (-mu - 2 * sigma).flatten(), (-mu + 2 * sigma).flatten(),
#                      alpha=.2, color='red', label='GPR 95% Confidence bound')
# axs1[0].scatter(X[:-1], -Y[:-1], c='r')
# axs1[0].set_ylabel('y')
# axs1[0].set_ylim([-1.25, 1.25])
# for i in range(n_it):
#     axs1[0].annotate(i + 1, (X[i], -Y[i]), xytext=(-3.5, 5), textcoords='offset pixels')
# acq_des = acqUCB(des_grid, gpr_step)
# axs1[1].plot(des_grid, (acq_des - min(acq_des)) / (max(acq_des) - min(acq_des)), color='red')
# axs1[1].set_xlabel('x')
# axs1[1].set_ylabel('Acquisition (normalized)')
# plt.tight_layout()
# for ax in axs1: ax.grid()