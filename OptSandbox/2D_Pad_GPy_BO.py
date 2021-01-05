import numpy as np
from scipy.stats import norm
import GPy.models
from sklearn.preprocessing import StandardScaler

# from matplotlib import pyplot as plt

cmdl = "abaqus cae noGUI=2D-get.py"

def acqEI(x_par, gpr, Y_train, xi=0):
    mu_par, sigma_par = gpr.predict(np.array(x_par))

    f_max_X_train = max(Y_train)

    z = (mu_par - f_max_X_train - xi) / sigma_par
    res_0 = (mu_par - f_max_X_train - xi) * norm.cdf(z) + sigma_par * norm.pdf(z)

    zero_array = np.zeros(np.shape(res_0))

    res = np.multiply(res_0, np.array([np.argmax(a) for a in zip(zero_array, sigma_par)]).reshape(np.shape(res_0)))

    return res

a_Design_var1_txt_file = "a_Design_var1.txt"
a_Design_var2_txt_file = "a_Design_var2.txt"
def g(x):
    open(a_Design_var1_txt_file, "w").write(str(x[0][0]))
    open(a_Design_var2_txt_file, "w").write(str(x[0][1]))
    b_Objective_1_txt_file = "b_Objective_c_gap.txt"
    # os.system(cmdl)
    return np.array(float(open(b_Objective_1_txt_file, "r").read().strip()))

# def g(x):
#     return - np.cos(np.pi / 2 * x[:, 0]) * np.cos(np.pi / 4 * x[:, 1]) + 1

### 2D
X = np.array([[32., 150.]])
Y = np.array(g(X)).reshape(-1, 1)

des_grid_x = np.linspace(30.0, 34.18, 100)
des_grid_y = np.linspace(50.0, 200.0, 100)
des_grid_xx, des_grid_yy = np.meshgrid(des_grid_x, des_grid_y)
des_grid = np.array([des_grid_xx.reshape(-1, 1), des_grid_yy.reshape(-1, 1)]).squeeze().T

scaler = StandardScaler()
scaler.fit(des_grid)

X_scaled = scaler.transform(X)
des_grid_scaled = scaler.transform(des_grid)

### Loop

x = X_scaled[0]

n_features = 2
k = 10 # number of iterations
for i in range(k): # optimization loop
    gpr_step = GPy.models.GPRegression(X_scaled, Y)
    mu, sigma = gpr_step.predict(np.array(x).reshape((1, n_features)))

    x = des_grid_scaled[np.argmax(acqEI(des_grid_scaled, gpr_step, Y))].reshape(-1, n_features)
    y_step = g(scaler.inverse_transform(x))
    X_scaled = np.append(X_scaled, x).reshape(-1, n_features)
    Y = np.append(Y, y_step).reshape(-1, 1)

    ## Progress ##
    np.savetxt('ProgDir/in_iter_%d.csv' % (i + 1), scaler.inverse_transform(x), delimiter=",")
    np.savetxt('ProgDir/out_iter_%d.csv' % (i + 1), [y_step])
    np.savetxt('ProgDir/mu_iter_%d.csv' % (i + 1), gpr_step.predict(des_grid_scaled)[0].reshape(np.shape(des_grid_xx)), delimiter=",")
    np.savetxt('ProgDir/sigma_iter_%d.csv' % (i + 1), gpr_step.predict(des_grid_scaled)[1].reshape(np.shape(des_grid_xx)), delimiter=",")

X = scaler.inverse_transform(X_scaled)

## Save history + record ##
np.savetxt('ProgDir/in_history.csv', X)
np.savetxt('ProgDir/out_history.csv', Y)
np.savetxt('ProgDir/in_min_pred.csv', X[np.argmin(Y)])
np.savetxt('ProgDir/out_min_pred.csv', min(Y))

# fig2, axs2 = plt.subplots(1, 4, figsize=(16, 5))

# y_pred, sigma_pred = gpr_step.predict(des_grid)

# axs2[0].contourf(des_grid_xx, des_grid_yy, -g(des_grid).reshape(np.shape(des_grid_xx))) #, cmap=cm.coolwarm, locator=ticker.LogLocator())
# axs2[0].contour(des_grid_xx, des_grid_yy, -g(des_grid).reshape(np.shape(des_grid_xx))) #, locator=ticker.LogLocator())
#
# axs2[1].contourf(des_grid_xx, des_grid_yy, -y_pred.reshape(np.shape(des_grid_xx))) #, cmap=cm.coolwarm)
# axs2[1].contour(des_grid_xx, des_grid_yy, -y_pred.reshape(np.shape(des_grid_xx))) #, locator=ticker.LogLocator())
# axs2[1].scatter(X[:, 0], X[:, 1], color='r')
#
# axs2[2].contourf(des_grid_xx, des_grid_yy, sigma_pred.reshape(np.shape(des_grid_xx)))
# axs2[2].scatter(X[:, 0], X[:, 1], color='r')
#
# axs2[3].contourf(des_grid_xx, des_grid_yy, acqEI(des_grid, gpr_step, X).reshape(np.shape(des_grid_xx)))
# axs2[3].scatter(X[:, 0], X[:, 1], color='r')
#
# plt.show()