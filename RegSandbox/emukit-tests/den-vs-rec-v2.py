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

import test_funs

np.random.seed(123)

### Fidelity number ###
n_fid = 4

### Dimension number ###
x_dim = 3

#################
### Functions ###
#################

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

##########
### 1D ###
##########

# ### Plotting data ###
#
# x_min = -5; x_max = 5
# x_plot = np.linspace(x_min, x_max, 200)[:, None]
#
# y_plot = [f(x_plot, fid) for fid in range(n_fid)]
#
# X_plot_mf = convert_x_list_to_array([x_plot] * n_fid)
# X_plot_mf_list = X_plot_mf.reshape((n_fid, len(x_plot), x_dim + 1))
#
# ### Training data ###
#
# n = [32, 16, 8, 4]
#
# x_train = [np.linspace(x_min, x_max, n_i)[:, None] for n_i in n] # Non-nested DoE
# y_train = [f(x_train[j], j) for j in range(n_fid)]

##########
### 2D ###
##########

### Plotting data ###

# x1_min = -5; x1_max = 5
# x2_min = -5; x2_max = 5
# x1_plot = np.linspace(x1_min, x1_max, 150)[:, None]
# x2_plot = np.linspace(x2_min, x2_max, 150)[:, None]
#
# x_plot_mesh = np.meshgrid(x1_plot, x2_plot)
# x_plot = np.hstack([layer.reshape(-1, 1) for layer in x_plot_mesh])
#
# # y_plot = [f(x_plot, fid) for fid in range(m)]
#
# X_plot_mf = convert_x_list_to_array([x_plot] * n_fid)
# X_plot_mf_list = X_plot_mf.reshape((n_fid, len(x_plot), x_dim + 1))
#
# ### Training data ###
#
# n = [81, 64, 36, 16]
#
# x_train = [x_plot[::len(x_plot) // n_i] for n_i in n]
# y_train = [f(x_train[j], j) for j in range(n_fid)]
#
# # print(x_train[2])
# # print(f(x_train[0], 0))
#
# X_train_mf, Y_train_mf = convert_xy_lists_to_arrays(x_train, y_train)

##########
### dD ###
##########

### Plotting data ###

x_min = [-5] * x_dim
x_max = [5] * x_dim
x_plot_grid = [np.linspace(x_min[j], x_max[j], 50)[:, None] for j in range(x_dim)]
x_plot_mesh = np.meshgrid(*x_plot_grid)
x_plot_list = np.hstack([layer.reshape(-1, 1) for layer in x_plot_mesh])

X_plot_mf = convert_x_list_to_array([x_plot_list] * n_fid)
X_plot_mf_list = X_plot_mf.reshape((n_fid, len(x_plot_list), x_dim + 1))

### Training data ###

n = [5 * (n_fid - i) for i in range(n_fid)] # Include possibility to insert training data of choice.

x_train = [x_plot_list[::len(x_plot_list) // n_i] for n_i in n] # Include possibility to insert training data of choice.
y_train = [f(x_train[j], j) for j in range(n_fid)]

X_train_mf, Y_train_mf = convert_xy_lists_to_arrays(x_train, y_train)

############################
### DENSE GP CALCULATION ###
############################

n_opt_restarts = 3

kernels_mf = []
for k in range(n_fid): kernels_mf.append(GPy.kern.RBF(input_dim=x_dim))

lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels_mf)

start_den = time.time()

gpy_m_den_mf = GPyLinearMultiFidelityModel(X_train_mf, Y_train_mf, lin_mf_kernel, n_fidelities=n_fid)

### Fixing kernel parameters ###
for k in range(n_fid): gpy_m_den_mf.mixed_noise.likelihoods_list[k].fix(0)
# print(gpy_m_den_mf)

m_den_mf = GPyMultiOutputWrapper(gpy_m_den_mf, n_fid, n_optimization_restarts=n_opt_restarts, verbose_optimization=True)

end_den_1 = time.time()
print('Dense MFGPR construction', end_den_1 - start_den)

### Dense HPO ###
m_den_mf_pre_HPO = m_den_mf
m_den_mf.optimize()

end_den_2 = time.time()
print('Dense MFGPR construction + HPO', end_den_2 - start_den)
# print(gpy_m_den_mf)

### Prediction ###
mu_den_mf = [m_den_mf.predict(X_plot_mf_list[j])[0] for j in range(n_fid)]
sigma_den_mf = [m_den_mf.predict(X_plot_mf_list[j])[1] for j in range(n_fid)]

end_den_3 = time.time()
print('Dense MFGPR construction + HPO + prediction', end_den_3 - start_den)

################################
### RECURSIVE GP CALCULATION ###
################################

start_rec = time.time()

m_rec_mf = GPy.models.multiGPRegression(x_train, y_train, kernel=[GPy.kern.RBF(x_dim) for i in range(n_fid)]) # Improve kernel selection...?

end_rec_1 = time.time()
print('Recursive MFGPR construction', end_rec_1 - start_rec)

for k in range(n_fid): m_rec_mf.models[k]['Gaussian_noise.variance'].fix(0)

### Recursive HPO ###
m_rec_mf_pre_HPO = m_rec_mf
m_rec_mf.optimize_restarts(restarts=n_opt_restarts, verbose=True)

end_rec_2 = time.time()
print('Recursive MFGPR construction + HPO', end_rec_2 - start_rec)

# for k in range(m): print(m_rec_mf.models[k])

### Prediction ###
mu_rec_mf, sigma_rec_mf = m_rec_mf.predict(x_plot_list)

end_rec_3 = time.time()
print('Recursive MFGPR construction + HPO + prediction', end_rec_3 - start_rec)

#####################
### VISUALIZATION ###
#####################

# for k in range(n_fid):
#     plt.plot(x_plot, y_plot[k], 'r', alpha=(k + 1) / n_fid, linewidth=2 * (k + 1) / n_fid)
#     plt.scatter(x_train[k], y_train[k], c='r', alpha=(k + 1) / n_fid)
#     plt.plot(x_plot, mu_den_mf[k], 'g--', alpha=(k + 1) / n_fid, linewidth=2 * (k + 1) / n_fid)
#     plt.plot(x_plot, mu_rec_mf[k], 'b-.', alpha=(k + 1) / n_fid, linewidth=2 * (k + 1) / n_fid)
#
# plt.grid()
# plt.show()