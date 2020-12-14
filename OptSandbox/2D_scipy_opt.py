import numpy as np
import scipy.optimize

import os
folder_path = os.getcwd()

cmdl = "cd {}/ && abaqus cae noGUI=2D-get.py".format(folder_path)

np.random.seed(123)

# from sklearn.preprocessing import StandardScaler
# scaler_bds = [[30., 50.], [34.18, 50.], [30., 200.], [34.18, 200.]]
# scaler = StandardScaler()
# scaler.fit(scaler_bds)
#
# a_Design_var1_txt_file = "{}/a_Design_var1.txt".format(folder_path)
# a_Design_var2_txt_file = "{}/a_Design_var2.txt".format(folder_path)
# def g(x):
#     x = scaler.inverse_transform(x)
#     open(a_Design_var1_txt_file, "w").write(str(x[0]))
#     open(a_Design_var2_txt_file, "w").write(str(x[1]))
#     b_Objective_1_txt_file = "{}/b_Objective_c_gap".format(folder_path) + '.txt'
#     # os.system(cmdl)
#     return np.array(float(open(b_Objective_1_txt_file, "r").read().strip()))

def g(x):
    return - np.cos(np.pi / 2 * x[0]) * np.cos(np.pi / 4 * x[1])

n_eval = 1
def callbackfun(x):
    global n_eval
    np.savetxt('ProgDir/in_iter_%d.csv' % n_eval, x, delimiter=",")
    # np.savetxt('ProgDir/out_iter_%d.csv' % n_eval, np.array([g(x)]))
    n_eval += 1

### 2D
# x0 = scaler.transform(np.array([[32., 150.]]))
x0 = np.array([0.5, 0.4])

# bds = scipy.optimize.Bounds(np.array([30., 50.]), np.array([34.18, 200.]))
bds = scipy.optimize.Bounds(np.array([-2, -2]), np.array([2, 2]))

# min_NM = scipy.optimize.minimize(g, x0, method='Nelder-Mead', callback=callbackfun, options={'maxfev': 20, 'return_all': True})
min_NM = scipy.optimize.minimize(g, x0, method='L-BFGS-B', callback=callbackfun, bounds=bds, options={'maxfun': 20})
# min_NM = scipy.optimize.differential_evolution(g, bounds=bds, callback=callbackfun, maxiter=1, popsize=1)

# print(np.array(min_NM['allvecs']))

print(min_NM)
open('ProgDir/min_obj.txt', "w").write(str(min_NM))

# np.savetxt('ProgDir/in_history.csv', min_NM['fun'], delimiter=',')