# General imports

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import time

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
# %matplotlib inline

np.random.seed(20)

####

import GPy
import emukit.multi_fidelity
import emukit.test_functions
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel

#### DENSE GP CALCULATION WITH EMUKIT

## Generate samples from the Forrester / non-linear sine function

high_fidelity = emukit.test_functions.forrester.forrester
low_fidelity = emukit.test_functions.forrester.forrester_low

# high_fidelity = emukit.test_functions.non_linear_sin.nonlinear_sin_high
# low_fidelity = emukit.test_functions.non_linear_sin.nonlinear_sin_low

x_plot = np.linspace(0, 1, 200)[:, None]
y_plot_l = low_fidelity(x_plot)
y_plot_h = high_fidelity(x_plot)

x_train_l = np.atleast_2d(np.random.rand(12)).T
x_train_h = np.atleast_2d(np.random.permutation(x_train_l)[:6])
y_train_l = low_fidelity(x_train_l)
y_train_h = high_fidelity(x_train_h)

# x_plot = np.linspace(0, 1, 200)[:, None]
# y_plot_l = low_fidelity(x_plot)
# y_plot_h = high_fidelity(x_plot)
#
# n_low_fidelity_points = 50
# n_high_fidelity_points = 14
#
# x_train_l = np.linspace(0, 1, n_low_fidelity_points)[:, None]
# y_train_l = low_fidelity(x_train_l)
#
# x_train_h = x_train_l[::4, :]
# # print(len(x_train_h))
# y_train_h = high_fidelity(x_train_h)

####

## Convert lists of arrays to ndarrays augmented with fidelity indicators

from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays

X_train, Y_train = convert_xy_lists_to_arrays([x_train_l, x_train_h], [y_train_l, y_train_h])
# print(x_train_l)
# print(x_train_h)
# print(y_train_l)
# print(y_train_h)

## Plot the original functions

# plt.figure(figsize=(12, 8))
# plt.plot(x_plot, y_plot_l, 'b')
# plt.plot(x_plot, y_plot_h, 'r')
# plt.scatter(x_train_l, y_train_l, color='b', s=40)
# plt.scatter(x_train_h, y_train_h, color='r', s=40)
# plt.ylabel('f (x)')
# plt.xlabel('x')
# plt.legend(['Low fidelity', 'High fidelity'])
# plt.title('High and low fidelity Forrester functions')

####

## Construct a linear multi-fidelity model

kernels = [GPy.kern.RBF(1), GPy.kern.RBF(1)]
lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)

## Timing
start = time.time()
gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=2)
end = time.time()

print('dense', end - start)

##

gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)
gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(0)

## Wrap the model using the given 'GPyMultiOutputWrapper'

lin_mf_model = model = GPyMultiOutputWrapper(gpy_lin_mf_model, 2, n_optimization_restarts=15, verbose_optimization=False)

## Fit the model

lin_mf_model.optimize()
print(gpy_lin_mf_model)

####

## Convert x_plot to its ndarray representation

X_plot = convert_x_list_to_array([x_plot, x_plot])
X_plot_l = X_plot[:len(x_plot)]#; print(X_plot_l)
X_plot_h = X_plot[len(x_plot):]#; print(X_plot_h)

## Compute mean predictions and associated variance

lf_mean_lin_mf_model, lf_var_lin_mf_model = lin_mf_model.predict(X_plot_l)
lf_std_lin_mf_model = np.sqrt(lf_var_lin_mf_model)
hf_mean_lin_mf_model, hf_var_lin_mf_model = lin_mf_model.predict(X_plot_h)
hf_std_lin_mf_model = np.sqrt(hf_var_lin_mf_model)

## Plot the posterior mean and variance

plt.figure(figsize=(12, 8))
plt.fill_between(x_plot.flatten(), (lf_mean_lin_mf_model - 1.96 * lf_std_lin_mf_model).flatten(),
                 (lf_mean_lin_mf_model + 1.96 * lf_std_lin_mf_model).flatten(), facecolor='g', alpha=0.3)
plt.fill_between(x_plot.flatten(), (hf_mean_lin_mf_model - 1.96 * hf_std_lin_mf_model).flatten(),
                 (hf_mean_lin_mf_model + 1.96 * hf_std_lin_mf_model).flatten(), facecolor='y', alpha=0.3)

plt.plot(x_plot, y_plot_l, 'b')
plt.plot(x_plot, y_plot_h, 'r')
plt.plot(x_plot, lf_mean_lin_mf_model, '--', color='g')
plt.plot(x_plot, hf_mean_lin_mf_model, '--', color='y')
plt.scatter(x_train_l, y_train_l, color='b', s=40)
plt.scatter(x_train_h, y_train_h, color='r', s=40)
plt.ylabel('f (x)')
plt.xlabel('x')
plt.legend(['Low Fidelity', 'High Fidelity', 'Predicted Low Fidelity (dense)', 'Predicted High Fidelity (dense)'])
# plt.title('Linear multi-fidelity model fit to low and high fidelity Forrester function')

# ####
#
# ## Create standard GP model using only high-fidelity data
#
# kernel = GPy.kern.RBF(1)
# high_gp_model = GPy.models.GPRegression(x_train_h, y_train_h, kernel)
# high_gp_model.Gaussian_noise.fix(0)
#
# ## Fit the GP model
#
# high_gp_model.optimize_restarts(5, verbose=False)
#
# ## Compute mean predictions and associated variance
#
# hf_mean_high_gp_model, hf_var_high_gp_model = high_gp_model.predict(x_plot)
# hf_std_hf_gp_model = np.sqrt(hf_var_high_gp_model)

# ####
#
# ## Plot the posterior mean and variance for the high-fidelity GP model
#
# plt.figure(figsize=(12, 8))
#
# plt.fill_between(x_plot.flatten(), (hf_mean_lin_mf_model - 1.96*hf_std_lin_mf_model).flatten(),
#                  (hf_mean_lin_mf_model + 1.96*hf_std_lin_mf_model).flatten(), facecolor='y', alpha=0.3)
# plt.fill_between(x_plot.flatten(), (hf_mean_high_gp_model - 1.96*hf_std_hf_gp_model).flatten(),
#                  (hf_mean_high_gp_model + 1.96*hf_std_hf_gp_model).flatten(), facecolor='k', alpha=0.1)
#
# plt.plot(x_plot, y_plot_h, color='r')
# plt.plot(x_plot, hf_mean_lin_mf_model, '--', color='y')
# plt.plot(x_plot, hf_mean_high_gp_model, 'k--')
# plt.scatter(x_train_h, y_train_h, color='r')
# plt.xlabel('x')
# plt.ylabel('f (x)')
# plt.legend(['True Function', 'Linear Multi-fidelity GP', 'High fidelity GP'])
# plt.title('Comparison of linear multi-fidelity model and high fidelity GP')

#### RECURSIVE GP CALCULATION WITH MF-GPY

# print([x_train_l, x_train_h])
start = time.time()
m_rec = GPy.models.multiGPRegression([x_train_l, x_train_h], [y_train_l, y_train_h])
end = time.time()
print('recursive', end - start)

m_rec.models[0]['Gaussian_noise.variance'].fix(0)
m_rec.models[1]['Gaussian_noise.variance'].fix(0)

m_rec.optimize_restarts(restarts=4, verbose=False)
print(m_rec)

mu_rec, sigma_rec = m_rec.predict(x_plot)
lf_mu_rec = mu_rec[0]
lf_sigma_rec = sigma_rec[0]
hf_mu_rec = mu_rec[1]
hf_sigma_rec = sigma_rec[1]

plt.plot(x_plot, lf_mu_rec, '--')
plt.plot(x_plot, hf_mu_rec, '--')
plt.fill_between(x_plot.flatten(), (lf_mu_rec - 1.96 * lf_sigma_rec).flatten(),
                 (lf_mu_rec + 1.96 * lf_sigma_rec).flatten(), facecolor='cyan', alpha=0.3)
plt.fill_between(x_plot.flatten(), (hf_mu_rec - 1.96 * hf_sigma_rec).flatten(),
                 (hf_mu_rec + 1.96 * hf_sigma_rec).flatten(), facecolor='orange', alpha=0.3)
plt.legend(['Low Fidelity', 'High Fidelity', 'Predicted Low Fidelity (dense)', 'Predicted High Fidelity (dense)',
            'Predicted Low Fidelity (recursive)', 'Predicted High Fidelity (recursive)'])

plt.show()