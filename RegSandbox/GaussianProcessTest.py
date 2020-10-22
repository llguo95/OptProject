import numpy as np
from matplotlib import pyplot as plt
np.random.seed(123)

x_inference = np.linspace(-5, 5, 100)

x1_inference = np.linspace(-2, 2, 25)
x2_inference = np.linspace(-2, 2, 25)
xx = np.meshgrid(x1_inference, x2_inference)
xx_inference = np.array([x.reshape(-1, 1) for x in xx]).squeeze().T

def k(a, b, k_type):
    # inputs a, b: (# of points, dimension) array
    # output: (len(a), len(b)) array
    r = np.sum(np.abs(np.expand_dims(a, 1) - np.expand_dims(b, 0)), 2)
    res = np.exp(- r ** 2 / 2)
    if k_type == 'SE':
        res = np.exp(- r ** 2 / 2)
    if k_type == 'IQ':
        res = 1 / (r ** 2 + 1)
    return res

# def k(a, b, k_type):
#     res = np.exp(- (a - b) ** 2 / 2)
#     if k_type == 'SE':
#         res = np.exp(- (a - b) ** 2 / 2)
#     if k_type == 'IQ':
#         res = 1 / ((a - b) ** 2 + 1)
#     return res

k_type = 'SE'
varn = 0

### Prior ###

# xi_inference, xj_inference = np.meshgrid(x_inference, x_inference)
K_inference = k(x_inference.reshape(-1, 1), x_inference.reshape(-1, 1), k_type)
K_inference2 = k(xx_inference, xx_inference, k_type)

mean_zero = np.zeros(len(x_inference))
mean_zero2 = np.zeros(len(xx_inference))

y_prior = np.random.multivariate_normal(mean_zero, K_inference, 20).T
y_prior2 = np.random.multivariate_normal(mean_zero2, K_inference2, 20).T

###
# t1_inference = np.linspace(-5, 5, 100)
#
# tt_inference = t1_inference.reshape(-1, 1)
#
# # test1 = np.sum(np.abs(np.expand_dims(tt_inference, 1) - np.expand_dims(tt_inference, 0)) ** 2, 2)
#
# x1_inference = np.linspace(-2, 2, 20)
# x2_inference = np.linspace(-2, 2, 20)
# xx1, xx2 = np.meshgrid(x1_inference, x2_inference)
#
# xx_inference = np.array([xx1.reshape(-1, 1), xx2.reshape(-1, 1)]).squeeze().T
# # test2 = np.sum(np.abs(np.expand_dims(xx_inference, 1) - np.expand_dims(xx_inference, 0)) ** 2, 2)
#
# y1_inference = np.linspace(-1, 1, 20)
# y2_inference = np.linspace(-1, 1, 20)
# y3_inference = np.linspace(-1, 1, 20)
# yy1, yy2, yy3 = np.meshgrid(y1_inference, y2_inference, y3_inference)
#
# blabla = np.meshgrid(y1_inference, y2_inference, y3_inference)
# blablareshape = np.array([bla.reshape(-1, 1) for bla in blabla]).squeeze().T
#
# yy_inference = np.array([yy1.reshape(-1, 1), yy2.reshape(-1, 1), yy3.reshape(-1, 1)]).squeeze().T
# # test3 = np.sum(np.abs(np.expand_dims(yy_inference, 1) - np.expand_dims(yy_inference, 0)) ** 2, 2)

### Posterior ###

x_train = np.array([-3, 0, 1, 4])
y_train = np.array([-2, 2, -1, 1])

x_train2 = np.array([[-1.5, -1], [0, 1.5], [-1, 0], [1, 1], [1, -0.5], [-1.5, 0.5]])
y_train2 = np.array([1, 1, 0, 1, -1.5, -1])

K_train = k(x_train.reshape(-1, 1), x_train.reshape(-1, 1), k_type)
K_train2 = k(x_train2, x_train2, k_type)

K_cross = k(x_train.reshape(-1, 1), x_inference.reshape(-1, 1), k_type)
K_cross2 = k(x_train2, xx_inference, k_type)

mean_pred = np.matmul(np.matmul(K_cross.T, np.linalg.inv(K_train + varn * np.eye(len(x_train)))), y_train)
covariance_pred = K_inference - np.matmul(np.matmul(K_cross.T, np.linalg.inv(K_train + varn * np.eye(len(x_train)))), K_cross)
y_pred = np.random.multivariate_normal(mean_pred, covariance_pred, 20).T

mean_pred2 = np.matmul(np.matmul(K_cross2.T, np.linalg.inv(K_train2 + varn * np.eye(len(x_train2)))), y_train2)
covariance_pred2 = K_inference2 - np.matmul(np.matmul(K_cross2.T, np.linalg.inv(K_train2 + varn * np.eye(len(x_train2)))), K_cross2)
y_pred2 = np.random.multivariate_normal(mean_pred2, covariance_pred2, 20).T

### Visualization ###

plt.figure()
plt.plot(x_inference, mean_zero, lw=3, c='r')
plt.plot(x_inference, 2 * np.ones(len(x_inference)), lw=1, c='k')
plt.plot(x_inference, -2 * np.ones(len(x_inference)), lw=1, c='k')
plt.fill_between(x_inference, -2 * np.ones(len(x_inference)), 2 * np.ones(len(x_inference)), alpha=0.2, color='r')
plt.plot(x_inference, y_prior, ':', lw=1)
plt.xlim(-5, 5)
plt.ylim(-3, 3)
plt.grid()
plt.title("Prior sampling")

plt.figure()
plt.plot(x_inference, mean_pred, lw=3, c='r')
plt.plot(x_inference, mean_pred + 2 * np.diag(covariance_pred), lw=1, c='k')
plt.plot(x_inference, mean_pred - 2 * np.diag(covariance_pred), lw=1, c='k')
plt.fill_between(x_inference, mean_pred - 2 * np.diag(covariance_pred), mean_pred + 2 * np.diag(covariance_pred), alpha=0.2, color='r')
plt.plot(x_inference, y_pred, ':', lw=1)
plt.xlim(-5, 5)
plt.ylim(-4, 4)
plt.grid()
plt.scatter(x_train, y_train, c='b', zorder=3)
plt.title("Posterior sampling with variance " + str(varn))

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(20):
    ax.contourf(xx[0], xx[1], y_prior2.T[i].reshape((25, 25)), alpha=0.05)

fig = plt.figure(figsize=plt.figaspect(1/3))
ax = fig.add_subplot(131)
for i in range(20):
    ax.contourf(xx[0], xx[1], y_pred2.T[i].reshape((25, 25)), alpha=0.05)

ax = fig.add_subplot(132)
m2 = ax.contourf(xx[0], xx[1], mean_pred2.reshape((25, 25)))
fig.colorbar(m2)

ax = fig.add_subplot(133)
c2 = ax.contourf(xx[0], xx[1], np.diag(covariance_pred2).reshape((25, 25)))
fig.colorbar(c2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx[0], xx[1], mean_pred2.reshape((25, 25)), cmap='viridis', alpha=0.5)
ax.scatter(x_train2[:, 0], x_train2[:, 1], zs=y_train2, s=20, c='r', alpha=1)

plt.show()