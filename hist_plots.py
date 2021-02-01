import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from test_funs import *
from GPy_BO_func import *

######

xls = pd.ExcelFile('Benchmark_data/Optimus_Benchmark_Data.xls')
df = [None] * 16
fun_inits = ['A', 'L', 'R', 'S']
fun_names = ['Ackley', 'Levy', 'Rosenbrock', 'Schwefel']
algo_inits = ['A', 'B', 'D', 'P']
algo_names = ['Annealing', 'BO', 'Diff. Ev.', 'PSO']
df_index = 0
for fi in fun_inits:
    for ai in algo_inits:
        df[df_index] = pd.read_excel(xls, fi + '_' + ai, header=None)
        df_index += 1

out_hists = [None] * 16
for i in range(16):
    out_hists[i] = df[i][2]

out_hists = np.array(np.array_split(out_hists, 4))
best_out_hists = np.minimum.accumulate(out_hists, axis=2)
norm_best_out_hists = best_out_hists / np.max(best_out_hists, axis=2).reshape((4, 4, 1))

fig, axs = plt.subplots(1, 4, figsize=(12, 3))
fig2, axs2 = plt.subplots(1, 4, figsize=(12, 3))

count = 0
for i in range(4):
    for j in range(4):
        axs[i].plot(np.arange(1, 51), norm_best_out_hists[i][j])
        axs[i].set_ylim(bottom=-.05, top=1.05)
        axs[i].set_title(fun_names[i])
        axs[i].set_xlabel('no. of function evaluations')
        # if i == 0:
        #     axs[i].legend(algo_names, loc='lower left', fontsize='x-small')
        #     if j == 0:
        #         axs[i].set_ylabel('normalized output')

        axs[i].grid(True)

        axs2[j].plot(np.arange(1, 51), norm_best_out_hists[i][j])
        axs2[j].set_ylim(bottom=-.05, top=1.05)
        axs2[j].set_title(algo_names[j])
        axs2[j].set_xlabel('no. of function evaluations')
        # if i == 0:
        #     if j == 0:
        #         axs2[j].legend(fun_names, loc='lower left', fontsize='x-small')
        #         axs2[i].set_ylabel('normalized output')

        axs2[j].grid(True)
    count += 4

axs[0].legend(algo_names, loc='lower left', fontsize='x-small')
axs[0].set_ylabel('normalized output')
axs2[0].legend(fun_names, loc='lower left', fontsize='x-small')
axs2[0].set_ylabel('normalized output')

####
# Different BO settings
# data_BO_fix = [None] * 3
# count = 0
# for s in ['R_B_HP-fix', 'L_B_HP-fix', 'S_B_HP-fix']:
#     best = np.minimum.accumulate(pd.read_excel(xls, s, header=None)[2])
#     data_BO_fix[count] = best / np.max(best)
#     count += 1
#
# for i in range(3):
#     axs[i + 1].plot(np.arange(1, 51), data_BO_fix[i], '--')

plt.tight_layout()
plt.savefig('Benchmark_algos.png')
plt.close(fig2)
plt.tight_layout()
plt.savefig('Benchmark_funs.png')

# plt.show()