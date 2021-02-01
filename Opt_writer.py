# from test_funs import *
import pandas as pd
from GPy_BO_func import *
from openpyxl import load_workbook

path = r'Benchmark_data/newdata.xlsx'
book = load_workbook(path)
writer = pd.ExcelWriter(path)
writer.book = book

fun = schwefel
x0 = [0, 0]
n_it = 50
HP_args = {'HP_fix': True,
           'rbf.variance': 1e6,
           'rbf.lengthscale': 1,
           'Gaussian_noise.variance': 0,
           'HPO': False,
           'HPO_opt': 'lbfgsb',
           'HPO_n': 4}

X, Y = BO(fun, x0, n_it, HP_args=HP_args, no_repeats=False)

BO_data = np.hstack((X, Y))
df = pd.DataFrame(BO_data)

df.to_excel(writer, 'S_B_HP-fix', index=False, header=False)
# writer.save()