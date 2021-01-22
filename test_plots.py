import numpy as np
import matplotlib.pyplot as plt
from test_funs import *

for f in [ackley, levy, rosenbrock, schwefel]:
    f_name = f([[0]])[1]
    if f_name is 'Ackley':     x1 = np.linspace(-5, 5, 200);     x2 = np.linspace(-5, 5, 200)
    if f_name is 'Levy':       x1 = np.linspace(-10, 10, 200);   x2 = np.linspace(-10, 10, 200)
    if f_name is 'Rosenbrock': x1 = np.linspace(-2, 2, 200);     x2 = np.linspace(-2, 2, 200)
    if f_name is 'Schwefel':   x1 = np.linspace(-500, 500, 200); x2 = np.linspace(-500, 500, 200)

    x_mesh = np.meshgrid(x1, x2)
    x_arr = np.array([layer.reshape(-1, 1) for layer in x_mesh]).squeeze().T

    y_arr = f(x_arr)[0]
    y_surf = y_arr.reshape(np.shape(x_mesh[0]))

    x_1D = x1.reshape(-1, 1)
    y_1D = f(x_1D)[0]

    if f_name is not 'Rosenbrock':
        fig = plt.figure(figsize=(5, 9))
        ax_0 = fig.add_subplot(211)
        ax_0.plot(x_1D, y_1D)
        ax_0.set_xlabel('$x_1$')
        ax_0.set_ylabel('$f(x_1)$')
        plt.grid()

        ax_1 = fig.add_subplot(212, projection='3d')
        ax_1.plot_surface(x_mesh[0], x_mesh[1], y_surf, cmap='viridis')
        ax_1.set_xlabel('$x_1$')
        ax_1.set_ylabel('$x_2$')
        ax_1.set_zlabel('$f(x_1, x_2)$')

        fig.suptitle(f_name + ' function', fontsize=20)
    else:
        fig = plt.figure()

        ax_1 = fig.add_subplot(111, projection='3d')
        ax_1.plot_surface(x_mesh[0], x_mesh[1], y_surf, cmap='viridis')
        ax_1.set_xlabel('$x_1$')
        ax_1.set_ylabel('$x_2$')
        ax_1.set_zlabel('$f(x_1, x_2)$')

        fig.suptitle(f_name + ' function', fontsize=20)
    # plt.savefig('imgs/test_fun_img/' + f_name + '_plot.pdf')

plt.show()