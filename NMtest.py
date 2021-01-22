import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import *
from test_funs import *

from scipy.stats import multivariate_normal

from sklearn.preprocessing import StandardScaler
scaler_bds = [[-2., -2.], [-2., 2.], [2., -2.], [2., 2.]]
scaler = StandardScaler()
scaler.fit(scaler_bds)

# def rosenbrock_scipy(x):
#     X = np.array([x])
#     Y = rosenbrock(X)[0][0][0]
#     return Y

def f_scipy(x):
    x = scaler.inverse_transform(x)
    X = np.array([x])
    Y = rosenbrock(X)[0][0][0]
    return Y

def bell(x):
    x = scaler.inverse_transform(x)
    y = multivariate_normal.pdf(x, mean=[0, 0])
    return y

n_eval = 1
def callbackfun(x):
    global n_eval
    print('iteration', n_eval,
          '\ninput', scaler.inverse_transform(x),
          '\noutput', f_scipy(x),
          '\n')
    x_hist.append(scaler.inverse_transform(x))
    # np.savetxt('ProgDir/in_iter_%d.csv' % n_eval, scaler.inverse_transform(x), delimiter=",")
    # np.savetxt('ProgDir/out_iter_%d.csv' % n_eval, np.array([g(x)]))
    n_eval += 1

x0_orig = np.array([0, 0])
x_hist = [x0_orig]
x0 = scaler.transform(np.array([x0_orig]))
min_obj = minimize(f_scipy, x0, method='Nelder-Mead', callback=callbackfun, options={'maxfev': 50, 'return_all': True})
# print(min_obj)
# print(scaler.inverse_transform(min_obj['final_simplex'][0]))
# print(np.array(a))

x1 = np.linspace(-5, 5, 200); x2 = np.linspace(-5, 5, 200)

x_mesh = np.meshgrid(x1, x2)
x_arr = np.array([layer.reshape(-1, 1) for layer in x_mesh]).squeeze().T

test = multivariate_normal.pdf(x_arr, mean=[0, 0])

y_arr = ackley(x_arr)[0]
y_surf = y_arr.reshape(np.shape(x_mesh[0]))
# ite = 7
# x_hist = [0].append(x_hist)
# for ite in range(len(x_hist)):
#     fig = plt.figure(figsize=(3, 2))
#     cf = plt.contourf(x_mesh[0], x_mesh[1], y_surf, levels=np.linspace(min(y_arr), max(y_arr), 50).flatten())
#     plt.contour(cf, colors='k', linewidths=.1)
#     plt.scatter(x_hist[ite][0], x_hist[ite][1], color='red')
#     # plt.title('Minimizing Ackley, iteration ' + str(ite))
#     plt.xlabel('$x_1$')
#     plt.ylabel('$x_2$')
#     plt.xlim([-.5, 2.5])
#     plt.ylim([-.5, 2.5])
    # plt.savefig('imgs/NM_2D_img/NM_Ackley_fev_%d.png' % ite)
# plt.show()

print(np.array(x_hist))