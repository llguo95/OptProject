from denvsrecFUNCTION import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

n_fid_list = np.arange(2, 6)
x_dim_list = np.arange(1, 5)

# times_arr_den = np.zeros((len(x_dim_list), len(n_fid_list)))
# times_arr_rec = np.zeros((len(x_dim_list), len(n_fid_list)))
# i = 0
# for n_fid in n_fid_list:
#     print('number of fidelities', n_fid)
#     n = [5 * (n_fid - i) for i in range(n_fid)]
#     j = 0
#     for x_dim in x_dim_list:
#         print('number of dimensions', x_dim)
#         times_arr_den[j, i] = denvsrecmain(n_fid=n_fid, x_dim=x_dim, n=n)[0][0]
#         times_arr_rec[j, i] = denvsrecmain(n_fid=n_fid, x_dim=x_dim, n=n)[0][1]
#         j += 1
#     i += 1

# pd.DataFrame(times_arr_den).to_csv('data/den2.csv')
# pd.DataFrame(times_arr_rec).to_csv('data/rec2.csv')

times_arr_den = [[2.628610610961914,6.523576736450195,35.45032525062561,97.32035088539124],
                 [1.676396131515503,9.321655988693237,32.76026654243469,129.54090309143066],
                 [1.4643292427062988,7.5816309452056885,22.905601024627686,76.10079407691956],
                 [1.2679476737976074,8.109960317611694,52.32569670677185,80.55250835418701]]
times_arr_rec = [[0.2024538516998291,0.44161534309387207,0.9556682109832764,1.13162660598754],
                 [0.2810637950897217,0.45375752449035645,0.5083625316619873,0.90677523612976],
                 [0.25905823707580566,0.512293815612793,1.1229281425476074,1.404814958572387],
                 [0.24605512619018555,0.7020845413208008,0.8561923503875732,1.58235383033752]]

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