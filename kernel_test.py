import numpy as np
import matplotlib.pyplot as plt
import GPy_MF.kern

k1 = GPy_MF.kern.RBF(input_dim=1)
k2 = GPy_MF.kern.RBF(input_dim=1, lengthscale=.5)

fig, axs = plt.subplots()

a = k1.K_of_r(1)
print(a)

k1.plot(ax=axs, color='r')
k2.plot(ax=axs)

plt.grid()
plt.show()