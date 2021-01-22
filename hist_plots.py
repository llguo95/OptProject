import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

xls = pd.ExcelFile('Optimus_Benchmark_Data.xls')
df = [None] * 12
fun_inits = ['A', 'L', 'R', 'S']
fun_names = ['Ackley', 'Levy', 'Rosenbrock', 'Schwefel']
algo_inits = ['A', 'D', 'P']
algo_names = ['Annealing', 'Diff. Ev.', 'PSO']
df_index = 0
for fi in fun_inits:
    for ai in algo_inits:
        df[df_index] = pd.read_excel(xls, fi + '_' + ai, header=None)
        df_index += 1

out_hists = [None] * 12
for i in range(12):
    out_hists[i] = df[i][2]

out_hists = np.array(np.array_split(out_hists, 4))
best_out_hists = np.minimum.accumulate(out_hists, axis=2)
norm_best_out_hists = best_out_hists / np.max(best_out_hists, axis=2).reshape((4, 3, 1))

print(norm_best_out_hists)

fig, axs = plt.subplots(1, 4, figsize=(12, 3))
fig2, axs2 = plt.subplots(1, 3, figsize=(9, 3))

count = 0
for i in range(4):
    for j in range(3):
        axs[i].plot(np.arange(1, 51), norm_best_out_hists[i][j])
        axs[i].set_ylim(bottom=0)
        axs[i].set_title(fun_names[i])
        axs[i].legend(algo_names, loc='lower left', fontsize='x-small')
        axs[i].grid(True)

        axs2[j].plot(np.arange(1, 51), norm_best_out_hists[i][j])
        axs2[j].set_ylim(bottom=0)
        axs2[j].set_title(algo_names[j])
        axs2[j].legend(fun_names, loc='lower left', fontsize='x-small')
        axs2[j].grid(True)
    count += 3

plt.show()