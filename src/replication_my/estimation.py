import time

import numpy as np
import optimagic as om
import pandas as pd

from replication_my.mahler_yum_2024 import START_PARAMS
from replication_my.moments import empirical_moments, moment_sd, simulate_moments

W_var = np.diag(1 / moment_sd.values**2)
W_root = np.sqrt(W_var)

algo_pounders = om.algos.tao_pounders(stopping_maxiter=400)
log_opts = om.SQLiteLogOptions(path="pd_var_2.db", if_database_exists="replace")

start_params = START_PARAMS.copy()

_wd_ages = ("27", "41", "51", "65")
_ec_ages = ("27", "49", "65", "87")

lower_bounds = {
    "work_disutility": {
        "bad": dict.fromkeys(_wd_ages, 0.0),
        "good": dict.fromkeys(_wd_ages, 0.0),
    },
    "education_disutility_adjustment": 0.0,
    "effort_cost": {
        "low": {
            "bad": dict.fromkeys(_ec_ages, 0.0),
            "good": dict.fromkeys(_ec_ages, 0.0),
        },
        "high": {
            "bad": dict.fromkeys(_ec_ages, 0.0),
            "good": dict.fromkeys(_ec_ages, 0.0),
        },
    },
    "income_process": {
        "y1": pd.Series({"low": 0.0, "high": 0.0}),
        "yt_s": pd.Series({"low": 0.0, "high": 0.0}),
        "yt_sq": pd.Series({"low": -0.15, "high": -0.15}),
        "wagep": pd.Series({"low": 0.0, "high": 0.0}),
        "sigx": 0.0,
    },
    "adjustment_cost": [0.0, 0.0],
    "discount_factor": pd.Series({"mean": 0.87, "std": 0.005}),
    "effort_elasticity": 0.0,
    "utility_constant": 7,
    "health_consumption_penalty": 0.65,
    "pension_replacement_rate": 0.2,
}

upper_bounds = {
    "work_disutility": {
        "bad": dict.fromkeys(_wd_ages, 4.0),
        "good": dict.fromkeys(_wd_ages, 4.0),
    },
    "education_disutility_adjustment": 1.5,
    "effort_cost": {
        "low": {
            "bad": dict.fromkeys(_ec_ages, 3.0),
            "good": dict.fromkeys(_ec_ages, 3.0),
        },
        "high": {
            "bad": dict.fromkeys(_ec_ages, 3.0),
            "good": dict.fromkeys(_ec_ages, 3.0),
        },
    },
    "income_process": {
        "y1": pd.Series({"low": 2.0, "high": 2.0}),
        "yt_s": pd.Series({"low": 0.3, "high": 0.3}),
        "yt_sq": pd.Series({"low": 0.2, "high": 0.2}),
        "wagep": pd.Series({"low": 0.3, "high": 0.3}),
        "sigx": 0.1,
    },
    "adjustment_cost": [0.01, 0.3],
    "discount_factor": pd.Series({"mean": 0.97, "std": 0.09}),
    "effort_elasticity": 2.0,
    "utility_constant": 13,
    "health_consumption_penalty": 1.0,
    "pension_replacement_rate": 0.6,
}


def criterion_func(params):
    sim_moments = simulate_moments(params=params)
    e = (sim_moments - empirical_moments).values
    return e.T @ W_var @ e


@om.mark.least_squares
def criterion_func_sqr(params):
    sim_moments = simulate_moments(params=params)
    e = (sim_moments - empirical_moments).values
    return e @ W_root


bounds = om.Bounds(lower=lower_bounds, upper=upper_bounds)

print(start_params)
start_time = time.time()

res = om.minimize(
    criterion_func_sqr,
    start_params,
    algo_pounders,
    bounds=bounds,
    scaling=om.ScalingOptions(method="bounds", clipping_value=0.0001),
    logging=log_opts,
)
res.to_pickle("pd_full_model_run_1.pkl")
optim_time = time.time() - start_time
simulate_moments(params=start_params)
start_time = time.time()
simulate_moments(params=start_params)
one_iter = time.time() - start_time
timings = {"full_opt": [optim_time], "one_iter": one_iter}
time_df = pd.DataFrame(timings)
time_df.to_csv("pd_optim_timings_2.csv")
