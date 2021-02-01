import numpy as np
import matplotlib.pyplot as plt
import GPy.models

np.random.seed(123)

n = 4
def f(x):
    # if x.any() == 0:
    #     res = np.ones(np.shape(x))
    # else:
    #     res = np.sin(x) + np.sin(n * np.pi * x) / (n * np.pi * x)
    # return res
    return - 10 * x * np.cos(x * 11)

def acqUCB(x, gpr, kappa=2):
    mu, sigma = gpr.predict(np.array(x).reshape(-1, 1))
    return mu + kappa * sigma

des_grid = np.linspace(0, 1, 100).reshape(-1, 1)

n_features = 1
# n_it = 5
for n_it in range(8, 9):
    X = np.array([[0.5]])
    Y = f(X)
    for i in range(n_it):
        gpr_step = GPy.models.GPRegression(X, Y)

        ### HP fixture
        # gpr_step.parameters[0]['rbf.variance'].fix(10)
        # gpr_step.parameters[0]['rbf.lengthscale'].fix(0.2)
        # gpr_step.parameters[1]['Gaussian_noise.variance'].fix(0)

        ### HPO
        gpr_step.preferred_optimizer = 'lbfgsb'
        gpr_step.optimize_restarts(num_restarts=5, verbose=False)

        # print(acqUCB(des_grid, gpr_step))
        x = des_grid[np.argmax(acqUCB(des_grid, gpr_step))]
        y = f(x)
        X = np.append(X, x).reshape(-1, n_features)
        Y = np.append(Y, y).reshape(-1, 1)

    # print(X);
    # print()
    # print(Y);
    # print()
    # print(np.argmin(Y));
    # print();
    # print(min(Y));
    # print()
    # print(gpr_step)

    mu, sigma = gpr_step.predict(des_grid)

    fig1, axs1 = plt.subplots(2, 1, figsize=(5, 8))

    axs1[0].plot(des_grid, -f(des_grid), '--', color='red', lw=.75, label='True function')
    axs1[0].plot(des_grid, -mu, color='red', label='GPR mean')
    axs1[0].plot(des_grid, -mu + 2 * sigma, color='k', lw=.5)
    axs1[0].plot(des_grid, -mu - 2 * sigma, color='k', lw=.5)
    axs1[0].fill_between(des_grid.flatten(), (-mu - 2 * sigma).flatten(), (-mu + 2 * sigma).flatten(),
                         alpha=.2, color='red', label='GPR 95% Confidence bound')
    axs1[0].scatter(X[:-1], -Y[:-1], c='r')
    axs1[0].set_ylabel('y')
    # axs1[0].set_ylim([-1.25, 1.25])

    for i in range(n_it):
        axs1[0].annotate(i + 1, (X[i], -Y[i]), xytext=(-3.5, 5), textcoords='offset pixels')

    acq_des = acqUCB(des_grid, gpr_step)
    axs1[1].plot(des_grid, (acq_des - min(acq_des)) / (max(acq_des) - min(acq_des)), color='red')
    axs1[1].set_xlabel('x')
    axs1[1].set_ylabel('Acquisition (normalized)')

    plt.tight_layout()
    for ax in axs1: ax.grid()

    # plt.savefig('BO_step_%d.png' % n_it)
plt.show()