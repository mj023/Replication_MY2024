import jax
from jax import numpy as jnp
from model_function import model_solve_and_simulate, simulate_moments
import nvtx
from Mahler_Yum_2024 import START_PARAMS
import pytest
from dags import tree

def test_run():
    """     for i in range(2):
            with nvtx.annotate('full_run', color='green'):
                new_params = START_PARAMS
                if i == 1:
                    new_params['beta_mean'] = 0.2
                res = model_solve_and_simulate(new_params)
                print(res["dead"].groupby("period")["in_regime"].count()) """
    for i in range(2):
        simulate_moments(START_PARAMS)


test_run()