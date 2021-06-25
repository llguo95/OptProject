###############
### IMPORTS ###
###############

import numpy as np
import matplotlib.pyplot as plt
import time
import GPy

import emukit.multi_fidelity
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import make_non_linear_kernels, \
    NonLinearMultiFidelityModel

import test_funs

np.random.seed(123)

### Fidelity number ###
n_fid = 2

### Dimension number ###
x_dim = 1

### (No. of) input data points ###
n = [5 * (n_fid - i) for i in range(n_fid)]  # Include possibility to insert training data of choice.


def denvsrecmain(n_fid, x_dim, n):
    ##########################################
    ### Function definition (per fidelity) ###
    ##########################################
    f_exact = test_funs.ackley

    def f_name():
        return f_exact(0)[1]

    def f_m(x):
        return f_exact(x)[0]

    a = 4 * (np.random.rand(n_fid - 1) - .5)
    b = 4 * (np.random.rand(n_fid - 1) - .5)
    c = 4 * (np.random.rand(n_fid - 1) - .5)
    d = np.random.randint(-4, 5, size=(n_fid - 1, x_dim + 1))

    def f(x, fid):
        if fid == n_fid - 1:
            return f_m(x)
        else:
            x1_ptb = np.array([x_i[0] - d[fid][0] for x_i in x])[:, None]
            return a[fid] * f(x, fid + 1) + b[fid] * x1_ptb + c[fid]

    ###################
    ### Data supply ###
    ###################

    ### Plotting data ###
    x_min = [-5] * x_dim
    x_max = [5] * x_dim
    x_plot_grid = [np.linspace(x_min[j], x_max[j], 50)[:, None] for j in range(x_dim)]
    x_plot_mesh = np.meshgrid(*x_plot_grid)
    x_plot_list = np.hstack([layer.reshape(-1, 1) for layer in x_plot_mesh])

    X_plot_mf = convert_x_list_to_array([x_plot_list] * n_fid)
    X_plot_mf_list = X_plot_mf.reshape((n_fid, len(x_plot_list), x_dim + 1))

    ### Training data ###
    x_train = [x_plot_list[::len(x_plot_list) // n_i] for n_i in
               n]  # Include possibility to insert training data of choice.
    y_train = [f(x_train[j], j) for j in range(n_fid)]

    X_train_mf, Y_train_mf = convert_xy_lists_to_arrays(x_train, y_train)

    ### Training parameters ###
    n_opt_restarts = 1

    ############################
    ### MFGPR-AR CALCULATION ###
    ############################

    ### AR kernel ###
    kernels_fid_AR = []
    for k in range(n_fid): kernels_fid_AR.append(GPy.kern.RBF(input_dim=x_dim))

    kernel_AR = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels_fid_AR)

    ### GP training / predictive distribution calculation ###
    gpy_m_AR = GPyLinearMultiFidelityModel(X_train_mf, Y_train_mf, kernel_AR, n_fidelities=n_fid)

    ### Fixing AR kernel parameters ###
    gpy_m_AR.likelihood.fix([0, 0])

    m_AR = GPyMultiOutputWrapper(gpy_m_AR, n_fid, n_optimization_restarts=n_opt_restarts,
                                 verbose_optimization=False)

    ### AR HPO ###
    m_AR_pre_HPO = m_AR
    m_AR.optimize()

    ### AR Prediction ###
    mu_AR, sigma_AR = np.array([m_AR.predict(X_plot_mf_list[j]) for j in range(n_fid)]).swapaxes(0, 1)

    ############################
    ### MFGPR-RL CALCULATION ###
    ############################

    ### RL kernels ###
    kernels_RL = [GPy.kern.RBF(x_dim) for i in range(n_fid)]

    ### GP training / predictive distribution calculation ###
    m_RL = GPy.models.multiGPRegression(x_train, y_train, kernel=kernels_RL)  # Improve kernel selection...?

    ### Fixing RL kernel parameters ###
    for model in m_RL.models: model.Gaussian_noise.variance.fix(0)

    ### RL HPO ###
    m_RL_pre_HPO = m_RL
    m_RL.optimize_restarts(restarts=n_opt_restarts, verbose=False)

    ### RL Prediction ###
    mu_RL, sigma_RL = m_RL.predict(x_plot_list)

    #############################
    ### MFGPR-NRL CALCULATION ###
    #############################

    ### RNL kernels ###
    kernels_RNL = make_non_linear_kernels(base_kernel_class=GPy.kern.RBF, n_fidelities=2,
                                          n_input_dims=X_train_mf.shape[1] - 1)

    ### GP training / predictive distribution calculation ###
    m_RNL = NonLinearMultiFidelityModel(X_train_mf, Y_train_mf, n_fidelities=2, kernels=kernels_RNL,  # n_samples=1,
                                        verbose=False, optimization_restarts=5)

    ### Fixing RL kernel parameters ###
    for model in m_RNL.models: model.Gaussian_noise.variance.fix(0)

    ### RNL HPO ###
    m_RNL_pre_HPO = m_RNL
    m_RNL.optimize() # Always random restart

    ### RNL Prediction ###
    mu_RNL, sigma_RNL = np.array([m_RNL.predict(X_plot_mf_list[j]) for j in range(n_fid)]).swapaxes(0, 1)

    return


denvsrecmain(n_fid=n_fid, x_dim=x_dim, n=n)
