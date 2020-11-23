import numpy as np
import matplotlib.pyplot as plt
import GPy.models

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
# Initializing parameters for experiment
HPO_bool = False
HPO_string = 'simplex'
HPO_num_of_restarts = 10

################################################################################
# Inference grid
x = np.linspace(0, 1, 100).reshape(-1, 1)

# Low-fidelity DoE
Xl = np.linspace(0, 1, 6).reshape(-1, 1)
Yl = fc(Xl)

# High-fidelity DoE
Xh = np.array([0, 0.4, 1]).reshape(-1, 1)
Yh = fe(Xh)

# Input and output
X = [Xl, Xh]
Y = [Yl, Yh]

# MFGP regression object
m = GPy.models.multiGPRegression(X, Y)

# HPO procedure
if HPO_bool:
    # Select HP optimizer
    m.models[0].preferred_optimizer = HPO_string
    m.models[1].preferred_optimizer = HPO_string

    # Optimize
    m.optimize_restarts(restarts=HPO_num_of_restarts, verbose=False)

### HP GRID DEFINITIONS ###

rbf_var_mag = 3
rbf_len_scale_mag = 4

noise_var_lf_list = np.linspace(0, 15, 10)
rbf_var_lf_list = np.linspace(0, 10 ** rbf_len_scale_mag, 10)
rbf_len_scale_lf_list = np.linspace(0, 10 ** rbf_var_mag, 10)

a, b, c = np.meshgrid(noise_var_lf_list, rbf_var_lf_list, rbf_len_scale_lf_list)
hp_lf_list = np.array([a.reshape(-1, 1), b.reshape(-1, 1), c.reshape(-1, 1)]).squeeze().T

noise_var_hf_list = np.linspace(0, 15, 10)
rbf_var_hf_list = np.linspace(0, 10 ** rbf_len_scale_mag, 10)
rbf_len_scale_hf_list = np.linspace(0, 10 ** rbf_var_mag, 10)

a, b, c = np.meshgrid(noise_var_hf_list, rbf_var_hf_list, rbf_len_scale_hf_list)
hp_hf_list = np.array([a.reshape(-1, 1), b.reshape(-1, 1), c.reshape(-1, 1)]).squeeze().T

###

LML_hf_list = np.zeros(len(hp_hf_list))

for hp_set in hp_lf_list:
    # Low-fidelity HP fix
    m.models[0]['Gaussian_noise.variance'].fix(hp_set[0])
    m.models[0]['rbf.variance'].fix(hp_set[1])
    m.models[0]['rbf.lengthscale'].fix(hp_set[2])

count = 0
for hp_set in hp_hf_list:
    # High-fidelity HP fix
    m.models[1]['Gaussian_noise.variance'].fix(hp_set[0])
    m.models[1]['rbf.variance'].fix(hp_set[1])
    m.models[1]['rbf.lengthscale'].fix(hp_set[2])

    LML_hf_list[count] = m.models[1].log_likelihood()
    count += 1

print(LML_hf_list)

LML_hf = LML_hf_list.reshape(np.shape(a))

# print(m.models[1].log_likelihood())

### Visualization
vis = False
if vis:
    ### Prediction (MAKE SURE ALL HYPERPARAMETERS ARE SET CORRECTLY)
    mu, sigma = m.predict(x)

    plt.plot(x, mu[0], color='b', label='MF cheap GPR (regular GPR)')
    plt.plot(x, mu[0] + 2 * sigma[0], color='k', lw=.5)
    plt.plot(x, mu[0] - 2 * sigma[0], color='k', lw=.5)
    plt.fill_between(x.flatten(), mu[0].flatten() - 2 * sigma[0].flatten(), mu[0].flatten() + 2 * sigma[0].flatten(), alpha=0.2, color='b')

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

    # m.plot()
    plt.show()