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

f_exact = test_funs.ackley

def f_name():
    return f_exact(0)[1]

def f_m(x):
    return f_exact(x)[0]

m = 3
def f(x, fid):
    if fid == m:
        return f_m(x)
    else:
        a = 2 * np.random.rand() - 1
        b = 2 * np.random.rand() - 1
        c = 2 * np.random.rand() - 1
        d = np.random.randint(-4, 5)
        return a * f(x, fid + 1) + b * (x - d) + c

def f_0(x): # Linear perturbation
    return 1.5 * f_1(x) + 0.5 * (x - 1) + 2

def f_1(x):
    return -0.5 * f_2(x) + 1 * (x + 2) - 1

def f_2(x):
    return f_exact(x)[0]

x_min = -5; x_max = 5
x_plot = np.linspace(x_min, x_max, 200)[:, None]

y_plot = [f(x_plot, fid) for fid in range(m)]

y_plot_0 = f_0(x_plot)
y_plot_1 = f_1(x_plot)
y_plot_2 = f_2(x_plot)

n = [20, 10, 5]

n_0 = 20
n_1 = 10
n_2 = 5

## Training data

x_train = [np.linspace(x_min, x_max, n_i)[:, None] for n_i in n]
y_train = [f(x_train[j], j) for j in range(m)]

# x_train_0 = np.linspace(x_min, x_max, n_0)[:, None] # Uniform initial DoE... would need to replace with proper opt-DoE
# y_train_0 = f_0(x_train_0)
#
# x_train_1 = np.random.permutation(x_train_0)[:n_1] # This is the nested DoE experiment.
# y_train_1 = f_1(x_train_1)
#
# x_train_2 = np.random.permutation(x_train_1)[:n_2] # This is the nested DoE experiment.
# y_train_2 = f_2(x_train_2)

X_train_mf, Y_train_mf = convert_xy_lists_to_arrays(x_train, y_train)

# X_train, Y_train = convert_xy_lists_to_arrays([x_train_0, x_train_1, x_train_2], [y_train_0, y_train_1, y_train_2])

## DENSE GP CALCULATION WITH EMUKIT

kernels_mf = [GPy.kern.RBF(1)] * m

kernels = [GPy.kern.RBF(1), GPy.kern.RBF(1), GPy.kern.RBF(1)]
lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)

start = time.time()

gpy_m_den_mf = GPyLinearMultiFidelityModel(X_train_mf, Y_train_mf, lin_mf_kernel, n_fidelities=3)
print(gpy_m_den_mf)

# gpy_m_den = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=3)

end1 = time.time()
print('dense', end1 - start)

# gpy_m_den.mixed_noise.Gaussian_noise.fix(0)
# gpy_m_den.mixed_noise.Gaussian_noise_1.fix(0)
# gpy_m_den.mixed_noise.Gaussian_noise_2.fix(0)

m_den_mf = GPyMultiOutputWrapper(gpy_m_den_mf, m, n_optimization_restarts=4, verbose_optimization=False)
# m_den = GPyMultiOutputWrapper(gpy_m_den, 3, n_optimization_restarts=4, verbose_optimization=False)

m_den_mf.optimize()
# m_den.optimize()
end2 = time.time()
print('dense + HPO', end2 - start)
# print(gpy_m_den)

# Prediction
X_plot_mf = convert_x_list_to_array([x_plot] * m)
# X_plot = convert_x_list_to_array([x_plot, x_plot, x_plot])
# X_plot_0 = X_plot[:len(x_plot)]
# X_plot_1 = X_plot[len(x_plot):2 * len(x_plot)]
# X_plot_2 = X_plot[2 * len(x_plot):]

# mu_den_0, sigma_den_0 = m_den.predict(X_plot_0)
# mu_den_1, sigma_den_1 = m_den.predict(X_plot_1)
# mu_den_2, sigma_den_2 = m_den.predict(X_plot_2)

## RECURSIVE GP CALCULATION WITH MF-GPY

start = time.time()

m_rec_mf = GPy.models.multiGPRegression(x_train, y_train)
print(m_rec_mf.models[0])

# m_rec = GPy.models.multiGPRegression([x_train_0, x_train_1, x_train_2], [y_train_0, y_train_1, y_train_2])

end3 = time.time()
print('recursive', end3 - start)

for k in range(m): m_rec_mf.models[k]['Gaussian_noise.variance'].fix(0)

# m_rec.models[0]['Gaussian_noise.variance'].fix(0)
# m_rec.models[1]['Gaussian_noise.variance'].fix(0)
# m_rec.models[2]['Gaussian_noise.variance'].fix(0)

m_rec_mf.optimize_restarts(restarts=4, verbose=False)
# m_rec.optimize_restarts(restarts=4, verbose=False)
end4 = time.time()
print('recursive + HPO', end4 - start)
# print(m_rec)

# mu_rec, sigma_rec = m_rec.predict(x_plot)
# mu_rec_0 = mu_rec[0]; sigma_rec_0 = sigma_rec[0]
# mu_rec_1 = mu_rec[1]; sigma_rec_1 = sigma_rec[1]
# mu_rec_2 = mu_rec[2]; sigma_rec_2 = sigma_rec[2]

## VISUALIZATION
# plt.plot(x_plot, y_plot_0, 'b')
# plt.plot(x_plot, y_plot_1, 'r')
# plt.plot(x_plot, y_plot_2, 'cyan')
# plt.plot(x_plot, mu_den_0, '--', color='g')
# plt.plot(x_plot, mu_den_1, '--', color='y')
# plt.plot(x_plot, mu_den_2, '--', color='cyan')
# plt.plot(x_plot, mu_rec_0, '--', color='g', alpha=.5)
# plt.plot(x_plot, mu_rec_1, '--', color='y', alpha=.5)
# plt.plot(x_plot, mu_rec_2, '--', color='cyan', alpha=.5)
# plt.scatter(x_train_0, y_train_0, color='b', s=40)
# plt.scatter(x_train_1, y_train_1, color='r', s=40)
# plt.scatter(x_train_2, y_train_2, color='cyan', s=40)
#
# plt.legend(['Low Fidelity', 'Middle Fidelity', 'High Fidelity',
#             'LF_den', 'MF_den', 'HF_den',
#             'LF_rec', 'MF_rec', 'HF_rec'])
#
# plt.show()