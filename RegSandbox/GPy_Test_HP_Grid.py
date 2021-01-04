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
m = GPy_MF.models.multiGPRegression(X, Y)

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

# noise_var_lf_list = np.linspace(0, 15, 10)
# rbf_var_lf_list = np.linspace(0, 10 ** rbf_len_scale_mag, 10)
# rbf_len_scale_lf_list = np.linspace(0, 10 ** rbf_var_mag, 10)

noise_var_lf_list = np.linspace(0.5, 1, 10)
rbf_var_lf_list = np.linspace(10, 40, 10)
rbf_len_scale_lf_list = np.linspace(0.1, 5, 10)

a_lf, b_lf, c_lf = np.meshgrid(noise_var_lf_list, rbf_var_lf_list, rbf_len_scale_lf_list)
hp_lf_list = np.array([a_lf.reshape(-1, 1), b_lf.reshape(-1, 1), c_lf.reshape(-1, 1)]).squeeze().T

# noise_var_hf_list = np.linspace(0, 15, 10)
# rbf_var_hf_list = np.linspace(0, 10 ** rbf_len_scale_mag, 10)
# rbf_len_scale_hf_list = np.linspace(0, 10 ** rbf_var_mag, 10)

noise_var_hf_list = np.linspace(0.5, 1, 10)
rbf_var_hf_list = np.linspace(10, 40, 10)
rbf_len_scale_hf_list = np.linspace(0.1, 5, 10)

a_hf, b_hf, c_hf = np.meshgrid(noise_var_hf_list, rbf_var_hf_list, rbf_len_scale_hf_list)
hp_hf_list = np.array([a_hf.reshape(-1, 1), b_hf.reshape(-1, 1), c_hf.reshape(-1, 1)]).squeeze().T

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

# print(LML_hf_list)

LML_hf = LML_hf_list.reshape(np.shape(a_hf))

# print(m.models[1].log_likelihood())

### Visualization
vis = True
if vis:
    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')

    a_plt, b_plt = np.meshgrid(noise_var_lf_list, rbf_var_lf_list)

    ax1.plot_surface(a_plt, b_plt, LML_hf[:, :, 0], cmap='viridis')
    ax1.set_xlabel('Noise var.')
    ax1.set_ylabel('RBF kernel var.')
    ax1.set_title('LML, RBF kernel length scale = ' + str(round(rbf_len_scale_lf_list[0], 2)))

    ax2.plot_surface(a_plt, b_plt, LML_hf[:, :, 3], cmap='viridis')
    ax2.set_xlabel('Noise var.')
    ax2.set_ylabel('RBF kernel var.')
    ax2.set_title('LML, RBF kernel length scale = ' + str(round(rbf_len_scale_lf_list[3], 2)))

    ax3.plot_surface(a_plt, b_plt, LML_hf[:, :, 6], cmap='viridis')
    ax3.set_xlabel('Noise var.')
    ax3.set_ylabel('RBF kernel var.')
    ax3.set_title('LML, RBF kernel length scale = ' + str(round(rbf_len_scale_lf_list[6], 2)))

    ax4.plot_surface(a_plt, b_plt, LML_hf[:, :, 9], cmap='viridis')
    ax4.set_xlabel('Noise var.')
    ax4.set_ylabel('RBF kernel var.')
    ax4.set_title('LML, RBF kernel length scale = ' + str(round(rbf_len_scale_lf_list[9], 2)))

    ### Prediction (MAKE SURE ALL HYPERPARAMETERS ARE SET CORRECTLY)
    # mu, sigma = m.predict(x)
    #
    # plt.plot(x, mu[0], color='b', label='MF cheap GPR (regular GPR)')
    # plt.plot(x, mu[0] + 2 * sigma[0], color='k', lw=.5)
    # plt.plot(x, mu[0] - 2 * sigma[0], color='k', lw=.5)
    # plt.fill_between(x.flatten(), mu[0].flatten() - 2 * sigma[0].flatten(), mu[0].flatten() + 2 * sigma[0].flatten(), alpha=0.2, color='b')
    #
    # plt.plot(x, mu[1], color='orange', label='MF expensive GPR')
    # plt.plot(x, mu[1] + 2 * sigma[1], color='k', lw=.5)
    # plt.plot(x, mu[1] - 2 * sigma[1], color='k', lw=.5)
    # plt.fill_between(x.flatten(), mu[1].flatten() - 2 * sigma[1].flatten(), mu[1].flatten() + 2 * sigma[1].flatten(), alpha=0.2, color='orange')
    #
    # plt.plot(x, fc(x), '--', color='b', label='Exact cheap function')
    # plt.plot(x, fe(x), '--', color='orange', label='Exact expensive function')
    #
    # plt.legend()
    #
    # plt.scatter(Xl, Yl, color='b')
    # plt.scatter(Xh, Yh, color='orange')
    #
    # plt.grid()
    #
    # # m.plot()
    plt.show()