import numpy as np
import matplotlib.pyplot as plt
import time
import GPy

import emukit
import emukit.multi_fidelity
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays

import test_funs

np.random.seed(123)

f = test_funs.ackley

def f_name():
    return f(0)[1]

def f_0(x): # Linear perturbation
    return 2 * f_1(x) + 5 * (x - 1) + 2

def f_1(x):
    return 3 * f_2(x) + 2 * (x + 2) - 1

def f_2(x):
    return f(x)[0]

x_min = -5; x_max = 5
x_plot = np.linspace(x_min, x_max, 200)[:, None]
y_plot_0 = f_0(x_plot)
y_plot_1 = f_1(x_plot)
y_plot_2 = f_2(x_plot)

n_0 = 20
n_1 = 10
n_2 = 5

## Training data

x_train_0 = np.linspace(x_min, x_max, n_0)[:, None] # Uniform initial DoE... would need to replace with proper opt-DoE
y_train_0 = f_0(x_train_0)

x_train_1 = np.random.permutation(x_train_0)[:n_1] # This is the nested DoE experiment.
y_train_1 = f_1(x_train_1)

x_train_2 = np.random.permutation(x_train_1)[:n_2] # This is the nested DoE experiment.
y_train_2 = f_2(x_train_2)

X_train, Y_train = convert_xy_lists_to_arrays([x_train_0, x_train_1, x_train_2], [y_train_0, y_train_1, y_train_2])

## DENSE GP CALCULATION WITH EMUKIT

kernels = [GPy.kern.RBF(1), GPy.kern.RBF(1), GPy.kern.RBF(1)]
lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)

start = time.time()

gpy_m_den = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=3)

end = time.time()
print('dense', end - start)

gpy_m_den.mixed_noise.Gaussian_noise.fix(0)
gpy_m_den.mixed_noise.Gaussian_noise_1.fix(0)
gpy_m_den.mixed_noise.Gaussian_noise_2.fix(0)

m_den = GPyMultiOutputWrapper(gpy_m_den, 3, n_optimization_restarts=5, verbose_optimization=False)

m_den.optimize()
print(gpy_m_den)

# Prediction
X_plot = convert_x_list_to_array([x_plot, x_plot, x_plot])
X_plot_0 = X_plot[:len(x_plot)]
X_plot_1 = X_plot[len(x_plot):2 * len(x_plot)]
X_plot_2 = X_plot[2 * len(x_plot):]

mu_den_0, sigma_den_0 = m_den.predict(X_plot_0)
mu_den_1, sigma_den_1 = m_den.predict(X_plot_1)
mu_den_2, sigma_den_2 = m_den.predict(X_plot_2)

## RECURSIVE GP CALCULATION WITH MF-GPY

start = time.time()

m_rec = GPy.models.multiGPRegression([x_train_0, x_train_1, x_train_2], [y_train_0, y_train_1, y_train_2])

end = time.time()
print('recursive', end - start)

m_rec.models[0]['Gaussian_noise.variance'].fix(0)
m_rec.models[1]['Gaussian_noise.variance'].fix(0)
m_rec.models[2]['Gaussian_noise.variance'].fix(0)

m_rec.optimize_restarts(restarts=5, verbose=False)
# print(m_rec)

mu_rec, sigma_rec = m_rec.predict(x_plot)
mu_rec_0 = mu_rec[0]; sigma_rec_0 = sigma_rec[0]
mu_rec_1 = mu_rec[1]; sigma_rec_1 = sigma_rec[1]
mu_rec_2 = mu_rec[2]; sigma_rec_2 = sigma_rec[2]

## VISUALIZATION
plt.plot(x_plot, y_plot_0, 'b')
plt.plot(x_plot, y_plot_1, 'r')
plt.plot(x_plot, y_plot_2, 'cyan')
plt.plot(x_plot, mu_den_0, '--', color='g')
plt.plot(x_plot, mu_den_1, '--', color='y')
plt.plot(x_plot, mu_den_2, '--', color='cyan')
plt.plot(x_plot, mu_rec_0, '--', color='g', alpha=.5)
plt.plot(x_plot, mu_rec_1, '--', color='y', alpha=.5)
plt.plot(x_plot, mu_rec_2, '--', color='cyan', alpha=.5)
plt.scatter(x_train_0, y_train_0, color='b', s=40)
plt.scatter(x_train_1, y_train_1, color='r', s=40)
plt.scatter(x_train_2, y_train_2, color='cyan', s=40)

plt.legend(['Low Fidelity', 'Middle Fidelity', 'High Fidelity',
            'LF_den', 'MF_den', 'HF_den',
            'LF_rec', 'MF_rec', 'HF_rec'])

plt.show()