import numpy as np
import scipy.optimize

import os
folder_path = os.getcwd()

np.random.seed(123)

from sklearn.preprocessing import StandardScaler
scaler_bds = [[30., 50.], [34.18, 50.], [30., 200.], [34.18, 200.]]
scaler = StandardScaler()
scaler.fit(scaler_bds)

def g(x):
    x = scaler.inverse_transform(x)
    open("a_Design_var1.txt", "w").write(str(x[0]))
    open("a_Design_var2.txt", "w").write(str(x[1]))
    # os.system("abaqus cae noGUI=2D-get.py")
    return np.array(float(open("b_Objective_c_gap.txt", "r").read().strip()))

n_eval = 1
def callbackfun(x):
    global n_eval
    np.savetxt('ProgDir/in_iter_%d.csv' % n_eval, scaler.inverse_transform(x), delimiter=",")
    # np.savetxt('ProgDir/out_iter_%d.csv' % n_eval, np.array([g(x)]))
    n_eval += 1

### 2D
x0 = scaler.transform(np.array([[32., 150.]]))

min_NM = scipy.optimize.minimize(g, x0, method='Nelder-Mead', callback=callbackfun, options={'maxfev': 20, 'return_all': True})

open('ProgDir/min_obj.txt', "w").write(str(min_NM))
np.savetxt('ProgDir/in_history.csv', scaler.inverse_transform(min_NM['allvecs']), delimiter=',')