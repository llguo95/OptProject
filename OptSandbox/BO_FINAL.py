import numpy as np
from scipy.stats import norm
import GPy_MF.models
from sklearn.preprocessing import StandardScaler

def acqEI(x_par, gpr, Y_train, xi=0):
    mu_par, sigma_par = gpr.predict(np.array(x_par))

    f_max_X_train = max(Y_train)

    z = (mu_par - f_max_X_train - xi) / sigma_par
    res_0 = (mu_par - f_max_X_train - xi) * norm.cdf(z) + sigma_par * norm.pdf(z)

    zero_array = np.zeros(np.shape(res_0))

    res = np.multiply(res_0, np.array([np.argmax(a) for a in zip(zero_array, sigma_par)]).reshape(np.shape(res_0)))

    return res

def gap_size(x):
    open("a_Design_var1.txt", "w").write(str(x[0][0]))
    open("a_Design_var2.txt", "w").write(str(x[0][1]))
#     os.system("abaqus cae noGUI=2D-get.py")
    return np.array(float(open("b_Objective_c_gap.txt", "r").read().strip()))

X = np.array([[32., 150.]])
Y = np.array(gap_size(X)).reshape(-1, 1)

des_grid_x = np.linspace(30.0, 34.18, 100)
des_grid_y = np.linspace(50.0, 200.0, 100)
des_grid_xx, des_grid_yy = np.meshgrid(des_grid_x, des_grid_y)
des_grid = np.array([des_grid_xx.reshape(-1, 1), des_grid_yy.reshape(-1, 1)]).squeeze().T

scaler = StandardScaler()
scaler.fit(des_grid)

X_scaled = scaler.transform(X)
des_grid_scaled = scaler.transform(des_grid)

x = X_scaled[0]

n_features = 2
k = 20 # number of iterations

for i in range(k): # optimization loop
    gpr_step = GPy_MF.models.GPRegression(X_scaled, Y)
    mu, sigma = gpr_step.predict(np.array(x).reshape((1, n_features)))

    x = des_grid_scaled[np.argmax(acqEI(des_grid_scaled, gpr_step, Y))].reshape(-1, n_features)
    y_step = gap_size(scaler.inverse_transform(x))
    X_scaled = np.append(X_scaled, x).reshape(-1, n_features)
    Y = np.append(Y, y_step).reshape(-1, 1)

    np.savetxt('ProgDir/in_iter_%d.csv' % (i + 1), scaler.inverse_transform(x), delimiter=",")
    np.savetxt('ProgDir/out_iter_%d.csv' % (i + 1), [y_step])
    np.savetxt('ProgDir/mu_iter_%d.csv' % (i + 1), gpr_step.predict(des_grid_scaled)[0].reshape(np.shape(des_grid_xx)), delimiter=",")
    np.savetxt('ProgDir/sigma_iter_%d.csv' % (i + 1), gpr_step.predict(des_grid_scaled)[1].reshape(np.shape(des_grid_xx)), delimiter=",")

X = scaler.inverse_transform(X_scaled)

np.savetxt('ProgDir/in_history.csv', X)
np.savetxt('ProgDir/out_history.csv', Y)
np.savetxt('ProgDir/in_min_pred.csv', X[np.argmin(Y)])
np.savetxt('ProgDir/out_min_pred.csv', min(Y))