import numpy as np
import matplotlib.pyplot as plt
import GPy.models

from sklearn.gaussian_process import GaussianProcessRegressor

np.random.seed(123)

def f(x):
    return np.sin(2 * np.pi * x) * np.exp(-x)

X = np.linspace(-1, 1, 10).reshape(-1, 1)
Y = f(X)

m = GPy.models.GPRegression(X, Y)

# print(m.parameters[0][0])

### HP fixtures
# m.parameters[0]['rbf.variance'].fix(1)
# m.parameters[0]['rbf.lengthscale'].fix(0.2)
m.parameters[1]['Gaussian_noise.variance'].fix(0.1)

m.optimize_restarts(num_restarts=10)

print(m)

x = np.linspace(-1, 1, 100).reshape(-1, 1)
mu, sigma = m.predict(x)

plt.plot(x, f(x), '--', color='red', lw=.75, label='True function')
plt.plot(x, mu, color='red', label='GPR mean')
plt.plot(x, mu + 2 * sigma, color='k', lw=.5)
plt.plot(x, mu - 2 * sigma, color='k', lw=.5)
plt.fill_between(x.flatten(), (mu - 2 * sigma).flatten(),  (mu + 2 * sigma).flatten(),
                 alpha=.2, color='red', label='GPR 95% Confidence bound')

plt.scatter(X, Y, color='red')
plt.grid()
plt.legend()
plt.show()