import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
import GPy.models
from sklearn.preprocessing import StandardScaler

np.random.seed(123)

import os
folder_path = os.getcwd()

# def acqEI(x_par, gpr, X_train, xi=0):
#     mu_par, sigma_par = gpr.predict(np.array(x_par))
#
#     f_max_X_train = max(g(X_train))
#
#     z = (mu_par - f_max_X_train - xi) / sigma_par
#     res_0 = (mu_par - f_max_X_train - xi) * norm.cdf(z) + sigma_par * norm.pdf(z)
#
#     zero_array = np.zeros(np.shape(res_0))
#
#     res = np.multiply(res_0, np.array([np.argmax(a) for a in zip(zero_array, sigma_par)]).reshape(np.shape(res_0)))
#
#     return res
#
# def acqUCB(x_par, gpr, X_train, kappa=10):
#     mu_par, sigma_par = gpr.predict(np.array(x_par))
#
#     return mu_par + kappa * sigma_par

def g(x):
    return np.cos(np.pi / 2 * x[0]) * np.cos(np.pi / 4 * x[1])

### 2D
X = np.array([0.5, 0.4])
# Y = g(X).reshape(-1, 1)

# des_grid_x = np.linspace(-2, 2, 100)
# des_grid_y = np.linspace(-2, 2, 100)
# des_grid_xx, des_grid_yy = np.meshgrid(des_grid_x, des_grid_y)
# des_grid = np.array([des_grid_xx.reshape(-1, 1), des_grid_yy.reshape(-1, 1)]).squeeze().T

# scaler = StandardScaler()
# scaler.fit(des_grid)
#
# X_scaled = scaler.transform(X)
# des_grid_scaled = scaler.transform(des_grid)
#
# ### Loop
#
# x = X_scaled[0]

bds = scipy.optimize.Bounds(np.array([-2, -2]), np.array([2, 2]))

# min_NM = scipy.optimize.minimize(g, X, method='Nelder-Mead', options={'maxfev': 20, 'return_all': True})
# min_NM = scipy.optimize.minimize(g, X, method='L-BFGS-B', bounds=bds, options={'maxfun': 20})
min_NM = scipy.optimize.differential_evolution(g, bounds=bds, maxiter=1, popsize=1)
# print(np.array(min_NM['allvecs']))
print(min_NM)

# n_features = 2
# k = 5 # number of iterations
# for i in range(k): # optimization loop
#     gpr_step = GPy.models.GPRegression(X_scaled, Y)
#     mu, sigma = gpr_step.predict(np.array(x).reshape((1, n_features)))
#
#     x = des_grid_scaled[np.argmax(acqEI(des_grid_scaled, gpr_step, X))].reshape(-1, n_features)
#     y_step = g(x)
#     X_scaled = np.append(X_scaled, x).reshape(-1, n_features)
#     Y = np.append(Y, y_step).reshape(-1, 1)
#
#     ## Progress ##
#     np.savetxt('ProgTxt/in_iter_%d.csv' % (i + 1), x)
#     np.savetxt('ProgTxt/out_iter_%d.csv' % (i + 1), -y_step)
#     np.savetxt('ProgTxt/mu_iter_%d.csv' % (i + 1), gpr_step.predict(des_grid)[0].reshape(np.shape(des_grid_xx)), delimiter=",")
#     np.savetxt('ProgTxt/sigma_iter_%d.csv' % (i + 1), gpr_step.predict(des_grid)[1].reshape(np.shape(des_grid_xx)), delimiter=",")
#
# y_pred, sigma_pred = gpr_step.predict(des_grid)
#
# X = scaler.inverse_transform(X_scaled)

# np.savetxt('ProgTxt/in_history.csv', X)
# np.savetxt('ProgTxt/out_history.csv', Y)
# np.savetxt('ProgTxt/in_min.csv', X[np.argmin(-Y)])
# np.savetxt('ProgTxt/out_min.csv', min(-Y))