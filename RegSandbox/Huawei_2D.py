import numpy as np
import matplotlib.pyplot as plt
import GPy_MF.models
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import os

np.random.seed(123)

os.chdir('C:\\Users\\Leo\\PycharmProjects\\OptProject\\Data')
# os.chdir('/home/leoguo/PycharmProjects/OptProject/Data')
# os.chdir('C:\\Users\\leoli\\PycharmProjects\\OptProject\\Data')

################################################################################
# Expensive Function
def fe(x):
    return (6.0 * x - 2.) ** 2 * np.sin(12 * x - 4)


# Cheap Function
def fc(x):
    A = 0.5
    B = 10
    C = 5
    return A * fe(x) + B * (x - 0.5) - C

# Expensive Function
def ge(x):
    return np.array(x[:, 0] ** 2 + x[:, 1] ** 2).reshape(-1, 1)

# Cheap Function
def gc(x):
    A = 0.5
    B = 10
    C = 5
    return A * ge(x) + B * (x[:, 1].reshape(-1, 1) - 0.5) - C

################################################################################

CheapResp = np.loadtxt('c_Resp_Surface_case5.txt')
ExpensiveResp = np.loadtxt('c_Resp_Surface_case1.txt')

Xc1 = np.linspace(50, 200, 10).reshape(-1,1)
Xc2 = np.linspace(30, 34.18, 10).reshape(-1,1)
xxc, yyc = np.meshgrid(Xc1, Xc2)
Xc = np.transpose(np.array([xxc.ravel(), yyc.ravel()]))

scaler = StandardScaler()

scaler.fit(Xc)
Xc_scaled = scaler.transform(Xc)

des_grid_x = np.linspace(50, 200, 100)
des_grid_y = np.linspace(30, 34.18, 100)
des_grid_xx, des_grid_yy = np.meshgrid(des_grid_x, des_grid_y)
des_grid = np.array([des_grid_xx.reshape(-1, 1), des_grid_yy.reshape(-1, 1)]).squeeze().T

des_grid_scaled = scaler.transform(des_grid)

Yc = CheapResp.reshape(-1, 1)
Ye_full = ExpensiveResp.reshape(-1, 1)

Expensive_DoE_indices = np.random.choice(100, 10, replace=False)

Xe = Xc[Expensive_DoE_indices]
Xe_scaled = Xc_scaled[Expensive_DoE_indices]

Ye = Ye_full[Expensive_DoE_indices]

###

x = des_grid_scaled

Xl = Xc_scaled
Xh = Xe_scaled

X = [Xl, Xh]

Yl = Yc
Yh = Ye

Y = [Yl, Yh]

###

mExp = GPy_MF.models.GPRegression(Xl, Ye_full)

mExp.optimize(max_iters=4)
# mExp.param_array[2] = 0.05
# mExp.parameters[1]['Gaussian_noise.variance'].fix(0.5)
# print(mExp.parameters[1]['Gaussian_noise.variance'])
# print()

muExp, sigmaExp = mExp.predict(x)
# print(mExp)
# print(mExp['Gaussian_noise.variance'])

m = GPy_MF.models.multiGPRegression(X, Y)

m.optimize_restarts(restarts=4, verbose=False)

# m.models[0]['Gaussian_noise.variance'].fix(0.1)
# m.models[0]['rbf.variance'].fix(1)
# m.models[0]['rbf.lengthscale'].fix(1)
# m.models[1]['Gaussian_noise.variance'].fix(0.05)
# m.models[1]['rbf.variance'].fix(1)
# m.models[1]['rbf.lengthscale'].fix(1)

mu, sigma = m.predict(x)

# print(m)
# print(m.models[0]['Gaussian_noise.variance'])

# (MF)GPR hyperparameters
print(mExp)
print(m)

print()

# Coefficients of determination
print('R-squared score of full expensive GPR surface compared with...')
print('Cheap GPR surface:', r2_score(muExp, mu[0]))
print('Multi-fidelity GPR surface:', r2_score(muExp, mu[1]))

mu = [a.reshape(np.shape(des_grid_xx)) for a in mu]
sigma = [a.reshape(np.shape(des_grid_xx)) for a in sigma]

### Visualization

fig1, axs1 = plt.subplots(2, 3, figsize=(8, 5))

axs1[0, 0].contourf(xxc, yyc, CheapResp.reshape(np.shape(xxc)))
axs1[1, 0].contourf(xxc, yyc, ExpensiveResp.reshape(np.shape(xxc)))
axs1[0, 1].contourf(des_grid_xx, des_grid_yy, mu[0])
axs1[1, 1].contourf(des_grid_xx, des_grid_yy, mu[1])
axs1[0, 2].contourf(des_grid_xx, des_grid_yy, sigma[0])
axs1[1, 2].contourf(des_grid_xx, des_grid_yy, sigma[1])

plt.tight_layout()

fig2 = plt.figure(figsize=(10, 10))

ax1 = fig2.add_subplot(2, 2, 1, projection='3d')
ax1.view_init(20, 225)
ax1.plot_surface(xxc, yyc, CheapResp.reshape(np.shape(xxc)), cmap='viridis')
ax1.set_title('Cheap response surface')

ax2 = fig2.add_subplot(2, 2, 2, projection='3d')
ax2.view_init(20, 225)
ax2.plot_surface(des_grid_xx, des_grid_yy, mu[0], cmap='viridis')
ax2.set_title('Cheap GPR surface')

ax3 = fig2.add_subplot(2, 2, 3, projection='3d')
ax3.view_init(20, 225)
ax3.plot_surface(xxc, yyc, ExpensiveResp.reshape(np.shape(xxc)), cmap='viridis')
ax3.set_title('Expensive response surface')

ax4 = fig2.add_subplot(2, 2, 4, projection='3d')
ax4.view_init(20, 225)
ax4.plot_surface(des_grid_xx, des_grid_yy, mu[1], cmap='viridis')
ax4.set_title('MFGPR surface')

fig3 = plt.figure()
axEx = fig3.add_subplot(1, 1, 1, projection='3d')
axEx.view_init(20, 225)
axEx.plot_surface(des_grid_xx, des_grid_yy, muExp.reshape(np.shape(des_grid_xx)), cmap='viridis')
axEx.set_title('Full GPR surface')

plt.show()