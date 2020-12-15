import numpy as np
from matplotlib import pyplot as plt

num_res = 17

in_results = np.zeros((num_res, 2))
out_results = np.zeros((num_res, 1))
mu_results = np.loadtxt('../Results/mu_iter_17.csv', delimiter=',')

for i in range(17):
    in_results[i] = np.loadtxt('../Results/in_iter_%d.csv' % (i + 1), delimiter=',')
    out_results[i] = np.loadtxt('../Results/out_iter_%d.csv' % (i + 1), delimiter=',')

print(in_results)
print(out_results)

des_grid_x = np.linspace(30.0, 34.18, 100)
des_grid_y = np.linspace(50.0, 200.0, 100)
des_grid_xx, des_grid_yy = np.meshgrid(des_grid_x, des_grid_y)
des_grid = np.array([des_grid_xx.reshape(-1, 1), des_grid_yy.reshape(-1, 1)]).squeeze().T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(30, 225)
ax.plot_surface(des_grid_xx, des_grid_yy, mu_results, cmap='viridis')
ax.scatter(in_results[:, 0], in_results[:, 1], out_results)
plt.show()