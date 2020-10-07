import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize

np.random.seed(123)

def f(x):
    return (np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) *
            np.random.randn() * 0.1)
res = gp_minimize(f, [(-2.0, 2.0)], n_calls=20)

print(res.models[0])

dom = np.linspace(-2, 2, 100).reshape(1, -1)
plt.plot(dom.T, f(dom))
plt.grid()
plt.show()