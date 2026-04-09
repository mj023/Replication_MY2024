import time

import numpy as np
import optimagic as om
import pandas as pd
from lcm.params import MappingLeaf

from Mahler_Yum_2024 import START_PARAMS
from model_function import simulate_moments

empirical_moments = np.asarray(
    [
        0.6508581,
        0.7660204,
        0.8232445,
        0.6193264,
        0.5055072,
        0.5830671,
        0.6008949,
        0.4091998,
        0.6777659,
        0.6769325,
        0.6802505,
        0.6992036,
        0.7301746,
        0.7237555,
        0.6426084,
        0.6227545,
        0.627258,
        0.6552106,
        0.6968261,
        0.6921402,
        0.7790819,
        0.7702285,
        0.7660254,
        0.7634262,
        0.779154,
        0.7724553,
        0.7517721,
        0.7435739,
        0.736526,
        0.7381558,
        0.750504,
        0.734436,
        0.0619297,
        0.516081,
        1.165899,
        1.651459,
        1.567324,
        1.006182,
        1.237489,
        0.2672905,
        0.3283083,
        0.4041793,
        8.49264942390098,
        0.1610319,
        0.7456731,
        1.163207,
        35.39329,
        49.37886,
        55.95501,
        42.21932,
        24.94774,
        33.16593,
        36.69067,
        25.31111,
        59.48338,
        89.53806,
        107.9282,
        98.27698,
        50.38816,
        66.25301,
        78.31755,
        63.1325,
        0.5952184,
        0.4770515,
    ]
)

moment_sd = np.asarray(
    [
        0.0022079,
        0.001673,
        0.0015903,
        0.0024375,
        0.0078668,
        0.0054486,
        0.0045718,
        0.0045788,
        0.0019615,
        0.0016137,
        0.0016517,
        0.0018318,
        0.0016836,
        0.0022494,
        0.0066662,
        0.0047753,
        0.0035851,
        0.0031197,
        0.0027306,
        0.0025937,
        0.0024741,
        0.0019636,
        0.0019423,
        0.0022411,
        0.0024561,
        0.0037815,
        0.0107689,
        0.0082543,
        0.0063126,
        0.0051546,
        0.0050761,
        0.0057938,
        0.0031501,
        0.0146831,
        0.023547,
        0.037393,
        0.042682,
        0.0473329,
        0.0029621,
        0.0037247,
        0.0030799,
        0.0039969,
        0.594830904063775,
        0.0004399,
        0.0035907,
        0.0221391,
        0.1955369,
        0.2318309,
        0.2660378,
        0.3528976,
        0.5630693,
        0.5187444,
        0.4988166,
        0.4986972,
        0.4875483,
        0.631705,
        0.7607303,
        1.108492,
        1.848723,
        1.65571,
        1.688008,
        1.78551,
        0.0023382,
        0.0015815,
    ]
)

W_var = np.diag(1 / moment_sd**2)
W_root = np.sqrt(W_var)

algo_pounders = om.algos.tao_pounders(stopping_maxiter=400)
log_opts = om.SQLiteLogOptions(path="pd_var_2.db", if_database_exists="replace")

start_params = START_PARAMS.copy()

lower_bounds = {
    "work_disutility": pd.DataFrame(
        {"bad": [0.0] * 4, "good": [0.0] * 4},
        index=[1, 8, 13, 20],
    ),
    "education_disutility_adj": 0.0,
    "effort_cost": MappingLeaf({
        "low": {"bad": [0.0] * 4, "good": [0.0] * 4},
        "high": {"bad": [0.0] * 4, "good": [0.0] * 4},
    }),
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
    "work_disutility": pd.DataFrame(
        {"bad": [4.0] * 4, "good": [4.0] * 4},
        index=[1, 8, 13, 20],
    ),
    "education_disutility_adj": 1.5,
    "effort_cost": MappingLeaf({
        "low": {"bad": [3.0] * 4, "good": [3.0] * 4},
        "high": {"bad": [3.0] * 4, "good": [3.0] * 4},
    }),
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
    sim_moments = simulate_moments(params)
    e = sim_moments - empirical_moments
    return e.T @ W_var @ e


@om.mark.least_squares
def criterion_func_sqr(params):
    sim_moments = simulate_moments(params)
    e = sim_moments - empirical_moments
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
simulate_moments(start_params)
start_time = time.time()
simulate_moments(start_params)
one_iter = time.time() - start_time
timings = {"full_opt": [optim_time], "one_iter": one_iter}
time_df = pd.DataFrame(timings)
time_df.to_csv("pd_optim_timings_2.csv")
