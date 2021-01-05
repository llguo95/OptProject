import numpy as np
import matplotlib.pyplot as plt
import GPy.models

np.random.seed(123)

################################################################################
# Expensive Function
def fe(x):
    return (6.0*x-2.)**2*np.sin(12*x-4)
# Cheap Function
def fc(x):
    A = 0.5; B=10; C=5
    return A*fe(x) +B*(x-0.5)-C
################################################################################
x  = np.linspace(0,1,100).reshape(-1,1)
Xl  =   np.linspace(0,1,11).reshape(-1,1)
Xh  =   np.array([0,0.4,0.6,0.8,1]).reshape(-1,1)
X   =   [Xl,Xh]
Yl  =   fc(Xl)
Yh  =   fe(Xh)
Y   =   [Yl,Yh]
m   =   GPy.models.multiGPRegression(X, Y)
# m.models[0]['Gaussian_noise.variance'].fix(0)
# m.models[1]['Gaussian_noise.variance'].fix(0)
# m.optimize_restarts(restarts=100)
# m.models[1]['Gaussian_noise.variance'] = 0.
print(m)

mu, sigma = m.models[1].predict(x)

plt.plot(x, mu)

m.plot()
plt.show()