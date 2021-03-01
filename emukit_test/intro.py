import numpy as np
import GPy

from emukit.model_wrappers import GPyModelWrapper
from emukit.experimental_design.experimental_design_loop import ExperimentalDesignLoop
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.loop import UserFunctionWrapper

x_min = -30.0
x_max = 30.0

X = np.random.uniform(x_min, x_max, (10, 1))
Y = np.sin(X) + np.random.randn(10, 1) * 0.05
gpy_model = GPy.models.GPRegression(X, Y)