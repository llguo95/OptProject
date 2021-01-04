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
# optimizer_bool = True
# optimizer_string = 'lbfgsb'
# num_of_restarts = 10
# DoE_set = 1
#
# noise_var_lf = 0.5
# noise_var_hf = 0

def noise_experiment(HPO_bool, HPO_string, num_of_restarts, DoE_set, noise_var_lf, noise_var_hf, vis_bool, save_bool):
    ###########
    # Inference grid
    x = np.linspace(0, 1, 100).reshape(-1, 1)

    # DoE input selection
    if DoE_set == 1:
        Xl = np.linspace(0, 1, 11).reshape(-1, 1)
        Xh = np.array([0, 0.4, 0.6, 0.8, 1]).reshape(-1, 1)

    if DoE_set == 2:
        Xl = np.linspace(0, 1, 6).reshape(-1, 1)
        Xh = np.array([0, 0.4, 1]).reshape(-1, 1)

    if DoE_set == 3:
        Xl = np.linspace(0, 1, 11).reshape(-1, 1)
        Xh = np.linspace(0, 1, 5).reshape(-1, 1)

    # DoE output calculation
    Yl = fc(Xl)
    Yh = fe(Xh)

    # Input and output
    X = [Xl, Xh]
    Y = [Yl, Yh]

    # MFGP regression object
    m = GPy_MF.models.multiGPRegression(X, Y)

    # Select HP optimizer
    m.models[0].preferred_optimizer = HPO_string
    m.models[1].preferred_optimizer = HPO_string

    ### HP constraints

    # m.models[0]['Gaussian_noise.variance'].constrain_bounded(0, 1)
    # m.models[0]['rbf.variance'].constrain_bounded(1, 20)
    # m.models[0]['rbf.lengthscale'].constrain_bounded(0.1, 5)

    # m.models[1]['Gaussian_noise.variance'].constrain_bounded(0, 1)
    # m.models[1]['rbf.variance'].constrain_bounded(1, 5)
    # m.models[1]['rbf.lengthscale'].constrain_bounded(1, 100)

    ### HP fixing

    m.models[0]['Gaussian_noise.variance'].fix(noise_var_lf)
    # m.models[0]['rbf.variance'].fix(1.5)
    # m.models[0]['rbf.lengthscale'].fix(0.1)

    m.models[1]['Gaussian_noise.variance'].fix(noise_var_hf)
    # m.models[1]['rbf.variance'].fix(0.1)
    # m.models[1]['rbf.lengthscale'].fix(0.1)

    if HPO_bool:
        # Optimize
        m.optimize_restarts(restarts=num_of_restarts, verbose=False)

    # print()
    # print(m.models[1].log_likelihood())
    #
    # print(m)

    ### Prediction (MAKE SURE ALL HYPERPARAMETERS ARE SET CORRECTLY)
    mu, sigma = m.predict(x)

    ### Visualization
    # if vis_bool:
    #     plt.figure()
    #     plt.plot(x, mu[0], color='b', label='MF cheap GPR (regular GPR)')
    #     plt.plot(x, mu[0] + 2 * sigma[0], color='k', lw=.5)
    #     plt.plot(x, mu[0] - 2 * sigma[0], color='k', lw=.5)
    #     plt.fill_between(x.flatten(), mu[0].flatten() - 2 * sigma[0].flatten(), mu[0].flatten() + 2 * sigma[0].flatten(), alpha=0.2, color='b')
    #
    #     plt.plot(x, mu[1], color='orange', label='MF expensive GPR')
    #     plt.plot(x, mu[1] + 2 * sigma[1], color='k', lw=.5)
    #     plt.plot(x, mu[1] - 2 * sigma[1], color='k', lw=.5)
    #     plt.fill_between(x.flatten(), mu[1].flatten() - 2 * sigma[1].flatten(), mu[1].flatten() + 2 * sigma[1].flatten(), alpha=0.2, color='orange')
    #
    #     plt.plot(x, fc(x), '--', color='b', label='Exact cheap function')
    #     plt.plot(x, fe(x), '--', color='orange', label='Exact expensive function')
    #
    #     plt.legend()
    #
    #     plt.scatter(Xl, Yl, color='b')
    #     plt.scatter(Xh, Yh, color='orange')
    #
    #     plt.grid()
    #
    #     if save_bool:
    #         plt.savefig('noise_experiment_Opt-%s_DoE%s_%s_%s.svg' % (HPO_string, DoE_set, noise_var_lf, noise_var_hf))
        # m.plot()
        # plt.show()

    return mu, sigma, x, Xl, Yl, Xh, Yh, HPO_string, DoE_set
    ###########

# Noise experiments
nl_list = [0.1, 0.5, 1]
nh_list = [0, 0.05, 0.1]
d_list = [3]

count_i = 0
count_j = 0

save_bool = False

fig, axs = plt.subplots(3, 3, figsize=(15, 10))

for nl in nl_list:
    for nh in nh_list:
        for d in d_list:
            mu, sigma, x, Xl, Yl, Xh, Yh, HPO_string, DoE_set = noise_experiment(HPO_bool=True, HPO_string='lbfgsb', num_of_restarts=10, DoE_set=d,
                 noise_var_lf=nl, noise_var_hf=nh, vis_bool=False, save_bool=False)
            p = axs[count_i, count_j]

            p.plot(x, mu[0], color='b', label='MF cheap GPR (regular GPR)')
            p.plot(x, mu[0] + 2 * sigma[0], color='k', lw=.5)
            p.plot(x, mu[0] - 2 * sigma[0], color='k', lw=.5)
            p.fill_between(x.flatten(), mu[0].flatten() - 2 * sigma[0].flatten(),
                             mu[0].flatten() + 2 * sigma[0].flatten(), alpha=0.2, color='b')

            p.plot(x, mu[1], color='orange', label='MF expensive GPR')
            p.plot(x, mu[1] + 2 * sigma[1], color='k', lw=.5)
            p.plot(x, mu[1] - 2 * sigma[1], color='k', lw=.5)
            p.fill_between(x.flatten(), mu[1].flatten() - 2 * sigma[1].flatten(),
                             mu[1].flatten() + 2 * sigma[1].flatten(), alpha=0.2, color='orange')

            p.plot(x, fc(x), '--', color='b', label='Exact cheap function')
            p.plot(x, fe(x), '--', color='orange', label='Exact expensive function')

            if count_i == 0 and count_j == 0:
                p.legend()

            p.scatter(Xl, Yl, color='b')
            p.scatter(Xh, Yh, color='orange')

            p.grid()
            p.set_title('LF noise ' + str(nl) + ', HF noise ' + str(nh))

            # if save_bool:
            #     plt.savefig('noise_experiment_Opt-%s_DoE%s.svg' % (HPO_string, DoE_set))
        count_i += 1
        count_i %= 3
    count_j += 1
plt.tight_layout()
plt.show()
