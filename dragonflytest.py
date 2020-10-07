# from dragonfly import minimise_function, maximise_function
# func = lambda x: x ** 2
# domain = [[-10, 10]]
# max_capital = 100
# min_val, min_pt, history = minimise_function(func, domain, max_capital)
# print(min_val, min_pt)
# max_val, max_pt, history = maximise_function(lambda x: -func(x), domain, max_capital)
# print(max_val, max_pt)

from dragonfly.exd import domains
from dragonfly.exd.experiment_caller import CPFunctionCaller, EuclideanFunctionCaller
from dragonfly.opt import random_optimiser, cp_ga_optimiser, gp_bandit

max_capital = 100
objective = lambda x: x[0] ** 4 - x[0]**2 + 0.1 * x[0]
domain = domains.EuclideanDomain([[-10, 10]])
func_caller = EuclideanFunctionCaller(None, domain)
opt = gp_bandit.EuclideanGPBandit(func_caller, ask_tell_mode=True)
opt.initialise()

for i in range(max_capital):
    x = opt.ask()
    y = objective(x)
    print("x:", x, ", y:", y)
    opt.tell([(x, y)])
