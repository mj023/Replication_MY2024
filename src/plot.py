import optimagic as om
from model_function import simulate_moments
import pandas as pd

fig = om.params_plot("optim.db")
fig.show()

res = pd.read_pickle("nelder_mead.pkl")

moments = simulate_moments(res.params)
print(moments)
print(res.fun)
print(res.params)