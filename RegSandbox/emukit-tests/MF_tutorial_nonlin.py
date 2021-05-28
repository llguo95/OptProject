# General imports

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
# %matplotlib inline

####

import GPy
import emukit.multi_fidelity
import emukit.test_functions
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel

####

from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays

####

np.random.seed(20)

## Generate data for nonlinear example

high_fidelity = emukit.test_functions.non_linear_sin.nonlinear_sin_high
low_fidelity = emukit.test_functions.non_linear_sin.nonlinear_sin_low

####

x_plot = np.linspace(0, 1, 200)[:, None]
y_plot_l = low_fidelity(x_plot)
y_plot_h = high_fidelity(x_plot)

n_low_fidelity_points = 50
n_high_fidelity_points = 14

x_train_l = np.linspace(0, 1, n_low_fidelity_points)[:, None]
y_train_l = low_fidelity(x_train_l)

x_train_h = x_train_l[::4, :]
y_train_h = high_fidelity(x_train_h)

### Convert lists of arrays to ND-arrays augmented with fidelity indicators

X_train, Y_train = convert_xy_lists_to_arrays([x_train_l, x_train_h], [y_train_l, y_train_h])

# print(X_train)

plt.figure(figsize=(12, 8))
plt.plot(x_plot, y_plot_l, 'b')
plt.plot(x_plot, y_plot_h, 'r')
plt.scatter(x_train_l, y_train_l, color='b', s=40)
plt.scatter(x_train_h, y_train_h, color='r', s=40)
plt.xlabel('x')
plt.ylabel('f (x)')
plt.xlim([0, 1])
plt.legend(['Low fidelity', 'High fidelity'])
plt.title('High and low fidelity functions')

plt.figure(figsize=(12,8))
plt.ylabel('HF(x)')
plt.xlabel('LF(x)')
plt.plot(y_plot_l, y_plot_h, color=colors['purple'])
plt.title('Mapping from low fidelity to high fidelity')
plt.legend(['HF-LF Correlation'], loc='lower center')

####

## Construct a linear multi-fidelity model

kernels = [GPy.kern.RBF(1), GPy.kern.RBF(1)]
lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=2)
# print(gpy_lin_mf_model)
gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)
gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(0)

lin_mf_model = model = GPyMultiOutputWrapper(gpy_lin_mf_model, 2, n_optimization_restarts=5, verbose_optimization=False)

## Fit the model
lin_mf_model.optimize()

####

## Convert test points to appropriate representation

# print(x_plot)
X_plot = convert_x_list_to_array([x_plot, x_plot])
# print(X_plot)
X_plot_low = X_plot[:200]
X_plot_high = X_plot[200:]

## Compute mean and variance predictions

hf_mean_lin_mf_model, hf_var_lin_mf_model = lin_mf_model.predict(X_plot_high)
hf_std_lin_mf_model = np.sqrt(hf_var_lin_mf_model)

lf_mean_lin_mf_model, lf_var_lin_mf_model = lin_mf_model.predict(X_plot_low)
lf_std_lin_mf_model = np.sqrt(lf_var_lin_mf_model)

## Compare linear and nonlinear model fits

plt.figure(figsize=(12,8))
plt.plot(x_plot, y_plot_h, 'r')
plt.plot(x_plot, hf_mean_lin_mf_model, '--', color='y')
plt.scatter(x_train_h, y_train_h, color='r')
plt.fill_between(x_plot.flatten(), (hf_mean_lin_mf_model - 1.96*hf_std_lin_mf_model).flatten(),
                 (hf_mean_lin_mf_model + 1.96*hf_std_lin_mf_model).flatten(), color='y', alpha=0.3)
plt.xlim(0, 1)
plt.xlabel('x')
plt.ylabel('f (x)')
plt.legend(['True Function', 'Linear multi-fidelity GP'], loc='lower right')
plt.title('Linear multi-fidelity model fit to high fidelity function')

####

## Create nonlinear model

from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import make_non_linear_kernels, \
    NonLinearMultiFidelityModel

base_kernel = GPy.kern.RBF
kernels = make_non_linear_kernels(base_kernel, 2, X_train.shape[1] - 1)
# print(kernels)
nonlin_mf_model = NonLinearMultiFidelityModel(X_train, Y_train, n_fidelities=2, kernels=kernels, #n_samples=1,
                                              verbose=False, optimization_restarts=5)
# print(nonlin_mf_model.models[0])
# print(nonlin_mf_model.models[1])
for m in nonlin_mf_model.models:
    m.Gaussian_noise.variance.fix(0)

nonlin_mf_model.optimize()
print()
print(nonlin_mf_model.models[0])
print(nonlin_mf_model.models[1])

####

## Compute mean and variance predictions

hf_mean_nonlin_mf_model, hf_var_nonlin_mf_model = nonlin_mf_model.predict(X_plot_high)
hf_std_nonlin_mf_model = np.sqrt(hf_var_nonlin_mf_model)

lf_mean_nonlin_mf_model, lf_var_nonlin_mf_model = nonlin_mf_model.predict(X_plot_low)
lf_std_nonlin_mf_model = np.sqrt(lf_var_nonlin_mf_model)

## Plot posterior mean and variance of nonlinear multi-fidelity model

plt.figure(figsize=(12,8))
plt.fill_between(x_plot.flatten(), (lf_mean_nonlin_mf_model - 1.96*lf_std_nonlin_mf_model).flatten(),
                 (lf_mean_nonlin_mf_model + 1.96*lf_std_nonlin_mf_model).flatten(), color='g', alpha=0.3)
plt.fill_between(x_plot.flatten(), (hf_mean_nonlin_mf_model - 1.96*hf_std_nonlin_mf_model).flatten(),
                 (hf_mean_nonlin_mf_model + 1.96*hf_std_nonlin_mf_model).flatten(), color='y', alpha=0.3)
plt.plot(x_plot, y_plot_l, 'b')
plt.plot(x_plot, y_plot_h, 'r')
plt.plot(x_plot, lf_mean_nonlin_mf_model, '--', color='g')
plt.plot(x_plot, hf_mean_nonlin_mf_model, '--', color='y')
plt.scatter(x_train_h, y_train_h, color='r')
plt.xlabel('x')
plt.ylabel('f (x)')
plt.xlim(0, 1)
plt.legend(['Low Fidelity', 'High Fidelity', 'Predicted Low Fidelity', 'Predicted High Fidelity'])
plt.title('Nonlinear multi-fidelity model fit to low and high fidelity functions')

####

plt.figure(figsize=(12,8))
plt.ylabel('HF(x)')
plt.xlabel('LF(x)')
plt.plot(y_plot_l, y_plot_h, '-', color=colors['purple'])
plt.plot(lf_mean_nonlin_mf_model, hf_mean_nonlin_mf_model, 'k--')
plt.plot(lf_mean_lin_mf_model, hf_mean_lin_mf_model, 'r--')
plt.legend(['True HF-LF Correlation', 'Learned HF-LF Correlation (nonlinear)', 'Learned HF-LF Correlation (linear)'], loc='lower center')
plt.title('Mapping from low fidelity to high fidelity')

# plt.show()