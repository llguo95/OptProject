import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *

from scipy.stats import norm

def f(x):
    return x * np.cos(x)

X = np.array([-2, -1]).reshape(-1, 1)
y = f(X)

D = np.linspace(-2, 6, 100).reshape(-1, 1)

kernel = RBF()
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X, y)
y_pred, sigma = gpr.predict(D, return_std=True)

def acqEI(x_par, X_train, xi=0):
    mu_par, sigma_par = gpr.predict(np.array(x_par).reshape(-1, 1), return_std=True)
    mu_par = mu_par.flatten()
    f_max_X_train = max(f(X_train))
    z = (mu_par - f_max_X_train - xi) / sigma_par
    # if sigma_par > 0:
    #     res = (mu_par - f_min_X_train - xi) * norm.cdf(z) + sigma_par * norm.pdf(z)
    # else:
    #     res = 0
    return (mu_par - f_max_X_train - xi) * norm.cdf(z) + sigma_par * norm.pdf(z)

def acqUCB(x_par, X_train, kappa):
    mu_par, sigma_par = gpr.predict(np.array(x_par).reshape(-1, 1), return_std=True)
    return mu_par.flatten() + kappa * sigma_par

# x = 0.1
# # X = X.flatten()
# # print(X)
#
for i in range(2):
    x = D[np.argmax(acqEI(D, X))]; y_step = f(x)
    X = np.append(X, x).reshape(-1, 1); y = np.append(y, y_step)
    gpr_step = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X, y)
    # print(x); print(y_step)

# print(acqEI(D, X))
# print(D[np.argmax(acqEI(D, X))])

# plt.plot(D, f(D))
# plt.plot(D, -y_pred)
print(X)
print(y)
plt.plot(D, acqEI(D, X))
# plt.scatter(X, y, color='r')
plt.grid()
plt.show()