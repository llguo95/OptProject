import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import GPy.models
from sklearn.preprocessing import StandardScaler

import os
folder_path = os.getcwd()

def acqEI(x_par, gpr, X_train, xi=0):
    mu_par, sigma_par = gpr.predict(np.array(x_par))

    f_max_X_train = max(g(X_train))

    z = (mu_par - f_max_X_train - xi) / sigma_par
    res_0 = (mu_par - f_max_X_train - xi) * norm.cdf(z) + sigma_par * norm.pdf(z)

    zero_array = np.zeros(np.shape(res_0))

    res = np.multiply(res_0, np.array([np.argmax(a) for a in zip(zero_array, sigma_par)]).reshape(np.shape(res_0)))

    return res

def acqUCB(x_par, gpr, X_train, kappa=10):
    mu_par, sigma_par = gpr.predict(np.array(x_par))

    return mu_par + kappa * sigma_par

def g(x):
    return np.cos(np.pi / 2 * x[:, 0]) * np.cos(np.pi / 4 * x[:, 1])

### 2D
X = np.array([[0.5, 0.4]])
Y = g(X).reshape(-1, 1)

des_grid_x = np.linspace(-2, 2, 100)
des_grid_y = np.linspace(-2, 2, 100)
des_grid_xx, des_grid_yy = np.meshgrid(des_grid_x, des_grid_y)
des_grid = np.array([des_grid_xx.reshape(-1, 1), des_grid_yy.reshape(-1, 1)]).squeeze().T

scaler = StandardScaler()
scaler.fit(des_grid)

X_scaled = scaler.transform(X)
des_grid_scaled = scaler.transform(des_grid)

### Loop

x = X_scaled[0]

n_features = 2
k = 16 # number of iterations
for i in range(k): # optimization loop
    gpr_step = GPy.models.GPRegression(X_scaled, Y)
    mu, sigma = gpr_step.predict(np.array(x).reshape((1, n_features)))

    x = des_grid_scaled[np.argmax(acqEI(des_grid_scaled, gpr_step, X))].reshape(-1, n_features)
    y_step = g(x)
    X_scaled = np.append(X_scaled, x).reshape(-1, n_features)
    Y = np.append(Y, y_step).reshape(-1, 1)

    ## Progress ##
    # np.savetxt('ProgTxt/in_iter_%d.csv' % (i + 1), x)
    # np.savetxt('ProgTxt/out_iter_%d.csv' % (i + 1), -y_step)
    # np.savetxt('ProgTxt/mu_iter_%d.csv' % (i + 1), gpr_step.predict(des_grid)[0].reshape(np.shape(des_grid_xx)), delimiter=",")
    # np.savetxt('ProgTxt/sigma_iter_%d.csv' % (i + 1), gpr_step.predict(des_grid)[1].reshape(np.shape(des_grid_xx)), delimiter=",")

y_pred, sigma_pred = gpr_step.predict(des_grid)

X = scaler.inverse_transform(X_scaled)

## Visualization ###

fig2, axs2 = plt.subplots(1, 4, figsize=(16, 5))

axs2[0].contourf(des_grid_xx, des_grid_yy, -g(des_grid).reshape(np.shape(des_grid_xx))) #, cmap=cm.coolwarm, locator=ticker.LogLocator())
axs2[0].contour(des_grid_xx, des_grid_yy, -g(des_grid).reshape(np.shape(des_grid_xx))) #, locator=ticker.LogLocator())

axs2[1].contourf(des_grid_xx, des_grid_yy, -y_pred.reshape(np.shape(des_grid_xx))) #, cmap=cm.coolwarm)
axs2[1].contour(des_grid_xx, des_grid_yy, -y_pred.reshape(np.shape(des_grid_xx))) #, locator=ticker.LogLocator())
axs2[1].scatter(X[:, 0], X[:, 1], color='r')

axs2[2].contourf(des_grid_xx, des_grid_yy, sigma_pred.reshape(np.shape(des_grid_xx)))
axs2[2].scatter(X[:, 0], X[:, 1], color='r')

axs2[3].contourf(des_grid_xx, des_grid_yy, acqUCB(des_grid, gpr_step, X).reshape(np.shape(des_grid_xx)))
axs2[3].scatter(X[:, 0], X[:, 1], color='r')
#
print(X)

# np.savetxt('ProgTxt/in_history.csv', X)
# np.savetxt('ProgTxt/out_history.csv', Y)
# np.savetxt('ProgTxt/in_min.csv', X[np.argmin(-Y)])
# np.savetxt('ProgTxt/out_min.csv', min(-Y))

print('Predicted minimizer = ', X[np.argmin(-Y)])
print(-Y)
print('Predicted minimum = ', min(-Y))
#
plt.show()