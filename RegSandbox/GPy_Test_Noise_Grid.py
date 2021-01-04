import numpy as np
import matplotlib.pyplot as plt
import GPy_MF.models

np.random.seed(123)

################################################################################

# Expensive Function
def fe(x):
    return (6.0 * x - 2.) ** 2 * np.sin(12 * x - 4) + 0.1

# Cheap Function
def fc(x):
    A = 0.5
    B = 10
    C = 5
    return A * fe(x) + B * (x - 0.5) - C + 0.2

################################################################################
# Initializing parameters for experiment (HPO = Hyperparameter optimization)
HPO_bool = True
HPO_string = 'lbfgsb'
HPO_num_of_restarts = 10

# noise_var_lf = 1
# noise_var_hf = 1

################################################################################
# Inference grid
x = np.linspace(0, 1, 100).reshape(-1, 1)

# Low-fidelity DoE
Xl = np.linspace(0, 1, 6).reshape(-1, 1)
Yl = fc(Xl)

# High-fidelity DoE
# Xh = np.linspace(0, 1, 6).reshape(-1, 1)
Xh = np.array([0, 0.4, 1]).reshape(-1, 1)
# Xh = np.array([0, 0.4, 0.6, 0.8, 1]).reshape(-1, 1)
# Xh = np.array([0.4, 0.8]).reshape(-1, 1)
Yh = fe(Xh)

# Input and output
X = [Xl, Xh]
Y = [Yl, Yh]

# MFGP regression object
m = GPy_MF.models.multiGPRegression(X, Y)

# HPO procedure
if HPO_bool:
    # Select HP optimizer
    m.models[0].preferred_optimizer = HPO_string
    m.models[1].preferred_optimizer = HPO_string

    # Optimize
    m.optimize_restarts(restarts=HPO_num_of_restarts, verbose=False)

### HP GRID DEFINITIONS ###

noise_var_lf_list = np.linspace(0.5, 5, 10)
noise_var_hf_list = np.linspace(0, 1, 10)

a, b = np.meshgrid(noise_var_lf_list, noise_var_hf_list)
hp_list = np.array([a.reshape(-1, 1), b.reshape(-1, 1)]).squeeze().T

###

LML_list = np.zeros(len(hp_list))

count = 0
for noise_var_set in hp_list:
    m.models[0]['Gaussian_noise.variance'].fix(noise_var_set[0])
    m.models[1]['Gaussian_noise.variance'].fix(noise_var_set[1])

    LML_list[count] = m.models[1].log_likelihood()
    count += 1

# print(LML_list)

# print(m.models[1].log_likelihood())
#
# print(m)

### Visualization
vis = False
if vis:
    ### Prediction (MAKE SURE ALL HYPERPARAMETERS ARE SET CORRECTLY)
    mu, sigma = m.predict(x)

    plt.plot(x, mu[0], color='b', label='MF cheap GPR (regular GPR)')
    plt.plot(x, mu[0] + 2 * sigma[0], color='k', lw=.5)
    plt.plot(x, mu[0] - 2 * sigma[0], color='k', lw=.5)
    plt.fill_between(x.flatten(), mu[0].flatten() - 2 * sigma[0].flatten(), mu[0].flatten() + 2 * sigma[0].flatten(), alpha=0.2, color='b')

    # plt.plot(x, mu_par, color='r', label='Regular GPR', alpha=0.3)
    # plt.plot(x, mu_par + 2 * sigma_par, color='k')
    # plt.plot(x, mu_par - 2 * sigma_par, color='k')
    # plt.fill_between(x.flatten(), mu_par.flatten() - 2 * sigma_par.flatten(), mu_par.flatten() + 2 * sigma_par.flatten(), alpha=0.2)

    plt.plot(x, mu[1], color='orange', label='MF expensive GPR')
    plt.plot(x, mu[1] + 2 * sigma[1], color='k', lw=.5)
    plt.plot(x, mu[1] - 2 * sigma[1], color='k', lw=.5)
    plt.fill_between(x.flatten(), mu[1].flatten() - 2 * sigma[1].flatten(), mu[1].flatten() + 2 * sigma[1].flatten(), alpha=0.2, color='orange')

    plt.plot(x, fc(x), '--', color='b', label='Exact cheap function')
    plt.plot(x, fe(x), '--', color='orange', label='Exact expensive function')

    plt.legend()

    plt.scatter(Xl, Yl, color='b')
    plt.scatter(Xh, Yh, color='orange')

    plt.grid()

    # plt.savefig('noise_experiment_2_%s_%s.svg' % (noise_var_lf, noise_var_hf))
    # m.plot()
    plt.show()