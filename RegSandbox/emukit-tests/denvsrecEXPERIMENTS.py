from denvsrecFUNCTION import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

n_fid_list = np.arange(2, 6)
x_dim_list = np.arange(1, 5)

times_arr_den = np.zeros((len(x_dim_list), len(n_fid_list)))
times_arr_rec = np.zeros((len(x_dim_list), len(n_fid_list)))
i = 0
for n_fid in n_fid_list:
    print('number of fidelities', n_fid)
    n = [5 * (n_fid - i) for i in range(n_fid)]
    j = 0
    for x_dim in x_dim_list:
        print('number of dimensions', x_dim)
        times_arr_den[j, i] = denvsrecmain(n_fid=n_fid, x_dim=x_dim, n=n)[0][0]
        times_arr_rec[j, i] = denvsrecmain(n_fid=n_fid, x_dim=x_dim, n=n)[0][1]
        j += 1
    i += 1

pd.DataFrame(times_arr_den).to_csv('data/den2.csv')
pd.DataFrame(times_arr_rec).to_csv('data/rec2.csv')

# times_arr_den = [[2.5925824642181396,6.651495456695557,20.34791326522827],
#                  [1.6793770790100098,6.8601648807525635,25.924055814743042],
#                  [2.6705992221832275,10.487486839294434,31.09686040878296],
#                  [1, 2, 3]]
# times_arr_rec = [[0.20104503631591797,0.2360095977783203,0.4451000690460205],
#                  [0.2810628414154053,0.439119815826416,0.6361429691314697],
#                  [0.46210312843322754,1.4453356266021729,1.875391960144043],
#                  [1, 2, 3]]

print(times_arr_den)
print(times_arr_rec)

fig, axs = plt.subplots(2, 2)

count = 0
for i in range(2):
    for j in range(2):
        axs[i, j].plot(n_fid_list, times_arr_den[count], 'r-o')
        axs[i, j].plot(n_fid_list, times_arr_rec[count], 'b-o')
        plt.grid()
        count += 1

plt.show()