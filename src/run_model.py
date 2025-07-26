import jax
from Mahler_Yum_2024 import MODEL_CONFIG
from jax import numpy as jnp
from lcm.dispatchers import _base_productmap
from lcm.entry_point import get_lcm_function
from model_function import model_solve_and_simulate
import nvtx
from plot import start_params

solve,_ = get_lcm_function(MODEL_CONFIG, targets='solve')
with nvtx.annotate('solve', color='green'):
    for i in range(1):
        res = model_solve_and_simulate(**start_params)
        print(res.loc[res['_period'] == 32,'alive'].to_string())