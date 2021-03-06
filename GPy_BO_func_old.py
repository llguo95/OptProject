import numpy as np
import matplotlib.pyplot as plt
import GPy.models

from scipy.stats import multivariate_normal

from test_funs import *

from sklearn.preprocessing import StandardScaler

np.random.seed(123)

def acqUCB(x, gpr, kappa=2):
    mu, sigma = gpr.predict(x)
    return mu + kappa * sigma

def BO(fun, x0, n_it, acq=acqUCB, HP_args=None, no_repeats=False):
    def f(x):
        x = scaler.inverse_transform(x)

        return -fun(x)[0]
        # return -rosenbrock(x)[0]
        # return -ackley(x)[0]
        # return -levy(x)[0]
        # return -schwefel(x)[0]

        # return np.array([[multivariate_normal.pdf(x, mean=[0, 0])]])
        # return np.array([10 * np.cos(np.pi / 2 * x[:, 0]) * np.cos(np.pi / 4 * x[:, 1])])
    f_name = fun([[0]])[1].lower()
    if f_name == 'rosenbrock':
        x1 = np.linspace(-2, 2, 100); x2 = np.linspace(-2, 2, 100)
    elif f_name == 'ackley':
        x1 = np.linspace(-5, 5, 100); x2 = np.linspace(-5, 5, 100)
    elif f_name == 'levy':
        x1 = np.linspace(-10, 10, 100); x2 = np.linspace(-10, 10, 100)
    elif f_name == 'schwefel':
        x1 = np.linspace(-500, 500, 100); x2 = np.linspace(-500, 500, 100)
    else:
        x1 = np.linspace(-1, 1, 100); x2 = np.linspace(-1, 1, 100)

    x_mesh = np.meshgrid(x1, x2)
    x_arr = np.hstack([layer.reshape(-1, 1) for layer in x_mesh])
    scaler = StandardScaler()
    scaler.fit(x_arr)

    x_arr_tf = scaler.transform(x_arr)
    # x_arr = np.vstack(np.array(x_mesh).T)

    # X = np.array([[0, 0]])
    X = np.array([x0])

    X_tf = scaler.transform(X)
    Y = f(X_tf)

    n_features = 2

    for i in range(n_it - 1):
        gpr_step = GPy.models.GPRegression(X_tf, Y)

        if HP_args is not None:
            ### HP fixture
            if HP_args['HP_fix']:
                gpr_step.parameters[0]['rbf.variance'].fix(HP_args['rbf.variance'])
                gpr_step.parameters[0]['rbf.lengthscale'].fix(HP_args['rbf.lengthscale'])
                gpr_step.parameters[1]['Gaussian_noise.variance'].fix(HP_args['Gaussian_noise.variance'])

            ### HPO
            if HP_args['HPO']:
                gpr_step.preferred_optimizer = HP_args['HPO_opt']
                gpr_step.optimize_restarts(num_restarts=HP_args['HPO_n'], verbose=False)

        ### Random point selection
        if no_repeats:
            if i > 0 and i % 5 == 0:
                if np.any(np.all(np.isin(X_tf, x_tf, True), axis=1)):
                    print(str(i) + ', repeat')
                    x_tf = x_arr_tf[np.random.randint(len(x_arr_tf))]
        x_tf = x_arr_tf[np.argmax(acq(x_arr_tf, gpr_step))]
        y = f(x_tf)

        X = np.append(X, scaler.inverse_transform(x_tf)).reshape(-1, n_features)
        X_tf = np.append(X_tf, x_tf).reshape(-1, n_features)

        Y = np.append(Y, y).reshape(-1, 1)
    return X, -Y

# fun = ackley
# x0 = [1, 1]
# n_it = 49
#
# HP_args = {'HP_fix': False,
#            'rbf.variance': 1,
#            'rbf.lengthscale': 1,
#            'Gaussian_noise.variance': 1,
#            'HPO': False,
#            'HPO_opt': 'lbfgsb',
#            'HPO_n': 4}
#
# X, Y = BO(fun, x0, n_it, HP_args=HP_args)
#
# print(X); print(Y)

# mu_arr, sigma_arr = gpr_step.predict(x_arr_tf)

# def lvs(arr):
#     return np.linspace(min(arr), max(arr), 100).flatten()

# fig1, axs = plt.subplots(2, 2)
# cf = [None] * 4
# cf[0] = axs[0, 0].contourf(x_mesh[0], x_mesh[1], -f(x_arr_tf).reshape(np.shape(x_mesh[0])), levels=lvs(-f(x_arr_tf)))
# axs[0, 0].set_title('True objective')
# fig1.colorbar(cf[0], ax=axs[0, 0])
#
# cf[1] = axs[0, 1].contourf(x_mesh[0], x_mesh[1], -mu_arr.reshape(np.shape(x_mesh[0])), levels=lvs(-mu_arr))
# axs[0, 1].set_title('GP mean')
# fig1.colorbar(cf[1], ax=axs[0, 1])
#
# cf[2] = axs[1, 0].contourf(x_mesh[0], x_mesh[1], sigma_arr.reshape(np.shape(x_mesh[0])), levels=lvs(sigma_arr))
# axs[1, 0].set_title('GP std')
# fig1.colorbar(cf[2], ax=axs[1, 0])
#
# acq_norm = (acqUCB(x_arr_tf, gpr_step) - min(acqUCB(x_arr_tf, gpr_step)))/(max(acqUCB(x_arr_tf, gpr_step)) - min(acqUCB(x_arr_tf, gpr_step)))
#
# cf[3] = axs[1, 1].contourf(x_mesh[0], x_mesh[1], acq_norm.reshape(np.shape(x_mesh[0])), levels=lvs(acq_norm))
# axs[1, 1].set_title('Acquisition (UCB, normalized)')
# fig1.colorbar(cf[3], ax=axs[1, 1])
#
# k = 0
# for i in range(2):
#     for j in range(2):
#         axs[i, j].scatter(X[:-1, 0], X[:-1, 1], c=np.linspace(-1, 1, n_it), cmap='Greys')
#         axs[i, j].scatter(X[-1, 0], X[-1, 1], c='red')
#         # axs[i, j].scatter(1, 1, c='green')
#         axs[i, j].contour(cf[k], colors='k', linewidths=.1)
#         # for m in range(n_it):
#         #     axs[i, j].annotate(m + 1, (X[m, 0], X[m, 1]), xytext=(-3.5, 5), textcoords='offset pixels', color='red')
#         k += 1

# print(X)
# print()
# print(-Y)
# print()
# print('Evaluation no. ', np.argmin(-Y) + 1)
# print('in:', X[np.argmin(-Y)])
# print('out:', min(-Y))
# print()
# print(gpr_step)

# plt.show()