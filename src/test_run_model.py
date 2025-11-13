import jax
from jax import numpy as jnp
from model_function import model_solve_and_simulate
import nvtx
from Mahler_Yum_2024 import START_PARAMS
import pytest

def test_run():
    with nvtx.annotate('solve', color='green'):
        for i in range(1):
            res = model_solve_and_simulate(START_PARAMS)
            print(res.loc[res['_period'] == 32,'alive'].to_string())

test_run()