import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *


def acqEI(x_par, gpr, X_train, xi=0):
    mu_par, sigma_par = gpr.predict(np.array(x_par), return_std=True)
    mu_par = mu_par.flatten()
    f_max_X_train = max(g(X_train))

    z = (mu_par - f_max_X_train - xi) / sigma_par

    res_0 = (mu_par - f_max_X_train - xi) * norm.cdf(z) + sigma_par * norm.pdf(z)
    zero_array = np.zeros(np.shape(res_0))

    res = np.multiply(res_0, [np.argmax(a) for a in zip(zero_array, sigma_par)])

    return res

# def acqUCB(x_par, X_train, kappa):
#     # mu_par, sigma_par = gpr.predict(np.array(x_par).reshape(-1, 1), return_std=True)
#     return mu_par.flatten() + kappa * sigma_par

def f(x):
    return -x * np.cos(x)

# def g(x):
#     x = x.T
#     y = rosen(x)
#     res = - np.array(y).reshape(-1, 1)
#     return res

def g(x):
    if x.ndim == 1:
        res = np.cos(np.pi / 2 * x[0]) * np.cos(np.pi / 4 * x[1])
    else:
        res = np.cos(np.pi / 2 * x[:, 0]) * np.cos(np.pi / 4 * x[:, 1])
    return res

### 2D
X = np.array([[0.5, 0.4]])
y = g(X)

des_grid_x = np.linspace(-2, 2, 100)
des_grid_y = np.linspace(-2, 2, 100)
des_grid_xx, des_grid_yy = np.meshgrid(des_grid_x, des_grid_y)
des_grid = np.array([des_grid_xx.reshape(-1, 1), des_grid_yy.reshape(-1, 1)]).squeeze().T

# print(np.shape(des_grid))

### Loop

x = X[0]

n_features = 2
k = 15 # number of iterations
for i in range(k): # optimization loop
    gpr_step = GaussianProcessRegressor().fit(X, y)
    mu_par, sigma_par = gpr_step.predict(np.array(x).reshape((1, n_features)), return_std=True)

    x = des_grid[np.argmax(acqEI(des_grid, gpr_step, X))]
    y_step = g(x)
    X = np.append(X, x).reshape(-1, n_features)
    y = np.append(y, y_step).reshape(-1, 1)

y_pred, sigma_pred = gpr_step.predict(des_grid, return_std=True)

### Visualization ###

fig2, axs2 = plt.subplots(1, 4, figsize=(16, 5))

# clev = np.linspace(min(-g(des_grid)), max(-g(des_grid)), 100).flatten()
# clev = 10 ** np.linspace(-1, 4, 10)
# clev2 = np.append(-100, 10 ** np.linspace(-1, 4, 100))

axs2[0].contourf(des_grid_xx, des_grid_yy, -g(des_grid).reshape(np.shape(des_grid_xx))) #, cmap=cm.coolwarm, locator=ticker.LogLocator())
axs2[0].contour(des_grid_xx, des_grid_yy, -g(des_grid).reshape(np.shape(des_grid_xx))) #, locator=ticker.LogLocator())

axs2[1].contourf(des_grid_xx, des_grid_yy, -y_pred.reshape(np.shape(des_grid_xx))) #, cmap=cm.coolwarm)
axs2[1].contour(des_grid_xx, des_grid_yy, -y_pred.reshape(np.shape(des_grid_xx))) #, locator=ticker.LogLocator())
axs2[1].scatter(X[:, 0], X[:, 1], color='r')

axs2[2].contourf(des_grid_xx, des_grid_yy, sigma_pred.reshape(np.shape(des_grid_xx)))
axs2[2].scatter(X[:, 0], X[:, 1], color='r')

axs2[3].contourf(des_grid_xx, des_grid_yy, acqEI(des_grid, gpr_step, X).reshape(np.shape(des_grid_xx)))
axs2[3].scatter(X[:, 0], X[:, 1], color='r')
# axs2[3].imshow(acqEI(des_grid, gpr_step, X).reshape(np.shape(des_grid_xx)), extent=[min(des_grid_y), max(des_grid_y), min(des_grid_x), max(des_grid_x)])

print(X)
print('Minimizer = ', X[np.argmin(-y)])
print(-y)
print('Minimum = ', min(-y))

# plt.savefig('step_%d.png' % (k - 1))
plt.show()