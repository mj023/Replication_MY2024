import optimagic as om
from model_function import simulate_moments
import pandas as pd
import numpy as np
from utils import retransform_params
from model_function import simulate_moments

""" fig = om.criterion_plot(['optim.db'])
fig.show() """
reader = om.SQLiteLogReader('nelder_mead_run_2.db')
history = reader.read_history()
min_ind = np.argmin(np.asarray(history.fun))
min_params = retransform_params(history.params[min_ind])
print(simulate_moments(min_params))
