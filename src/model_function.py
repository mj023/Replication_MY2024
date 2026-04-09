from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import random
from lcm.pandas_utils import initial_conditions_from_dataframe
from lcm.utils.dispatchers import productmap
from scipy.interpolate import interp1d as scipy_interp1d

from Mahler_Yum_2024 import (
    MAHLER_YUM_MODEL,
    ages,
    effort_grid,
    n_periods,
    prod_shock_grid,
    productivity_type_multiplier,
    retirement_period,
    shock_persistence,
    wealth_to_level,
)
from utils import gini

_DATA_DIR = Path(__file__).parent

model = MAHLER_YUM_MODEL

_wealth_normalization = jnp.array([43978, 48201])


def create_work_disutility_grid(work_disutility, education_disutility_adj):
    """Interpolate work disutility knots to full period grid.

    Args:
        work_disutility: DataFrame with columns "bad", "good" and period index.
        education_disutility_adj: Scalar education adjustment factor.

    """
    grid = jnp.zeros((retirement_period + 1, 2, 2))
    for j, health in enumerate(["bad", "good"]):
        spline = scipy_interp1d(
            np.asarray(work_disutility.index),
            np.asarray(work_disutility[health]),
            kind="cubic",
        )
        interp_points = jnp.arange(1, retirement_period + 2)
        temp_grid = jnp.asarray(spline(interp_points))
        grid = grid.at[:, 0, j].set(temp_grid * jnp.exp(education_disutility_adj))
        grid = grid.at[:, 1, j].set(temp_grid)
    return grid


def create_effort_cost_grid(effort_cost):
    """Interpolate effort cost knots to full period grid.

    Args:
        effort_cost: DataFrame with MultiIndex columns (education, health)
            and period index.

    """
    grid = jnp.zeros((n_periods, 2, 2))
    for i, edu in enumerate(["low", "high"]):
        for j, health in enumerate(["bad", "good"]):
            knots = np.asarray(effort_cost[(edu, health)])
            spline = scipy_interp1d(np.asarray(effort_cost.index), knots, kind="cubic")
            interp_points = np.arange(1, 31)
            temp_grid = jnp.asarray(spline(interp_points))
            grid = grid.at[0:30, i, j].set(temp_grid)
            grid = grid.at[30:n_periods, i, j].set(knots[-1])
    return grid


def create_adjustment_cost_envelope(adjustment_cost):
    t = jnp.arange(38)
    return jnp.maximum(adjustment_cost[0] * jnp.exp(adjustment_cost[1] * t), 0)


def create_base_income_grid(income_process):
    """Build base income grid from income process parameters."""
    sigx = income_process["sigx"]
    sdztemp = ((sigx**2.0) / (1.0 - shock_persistence**2.0)) ** 0.5
    j = jnp.arange(20)
    health = jnp.arange(2)
    education = jnp.arange(2)

    y1 = income_process["y1"]
    yt_s = income_process["yt_s"]
    yt_sq = income_process["yt_sq"]
    wagep = income_process["wagep"]

    def calc_base(_period, health, education):
        yt = jnp.where(
            education == 1,
            (
                y1["high"]
                * jnp.exp(yt_s["high"] * _period + yt_sq["high"] * _period**2.0)
            )
            * (1.0 - wagep["high"] * (1 - health)),
            (y1["low"] * jnp.exp(yt_s["low"] * _period + yt_sq["low"] * _period**2.0))
            * (1.0 - wagep["low"] * (1 - health)),
        )
        return yt / (
            jnp.exp(((jnp.log(productivity_type_multiplier[1]) ** 2.0) ** 2.0) / 2.0)
            * jnp.exp(((sdztemp**2.0) ** 2.0) / 2.0)
        )

    variables = ("_period", "health", "education")
    mapped = productmap(
        func=calc_base, variables=variables, batch_sizes=dict.fromkeys(variables, 0)
    )
    return mapped(_period=j, health=health, education=education)


# Utility arrays for initial type draws
_discount = jnp.zeros((16), dtype=jnp.int8)
_prod = jnp.zeros((16), dtype=jnp.int8)
_ht = jnp.zeros((16), dtype=jnp.int8)
_ed = jnp.zeros((16), dtype=jnp.int8)
for _i in range(1, 3):
    for _j in range(1, 3):
        for _k in range(1, 3):
            _index = (_i - 1) * 2 * 2 + (_j - 1) * 2 + _k - 1
            _discount = _discount.at[_index].set(_i - 1)
            _prod = _prod.at[_index].set(_j - 1)
            _ht = _ht.at[_index].set(1 - (_k - 1))
            _discount = _discount.at[_index + 8].set(_i - 1)
            _prod = _prod.at[_index + 8].set(_j - 1)
            _ht = _ht.at[_index + 8].set(1 - (_k - 1))
            _ed = _ed.at[_index + 8].set(1)
_init_distr = jnp.array(np.loadtxt(_DATA_DIR / "init_distr_2b2t2h.txt"))
_initial_dists = jnp.diff(_init_distr[:, 0], prepend=0)

_HEALTH_LABELS = {0: "bad", 1: "good"}
_EDUCATION_LABELS = {0: "low", 1: "high"}
_PRODUCTIVITY_LABELS = {0: "low", 1: "high"}
_HEALTH_TYPE_LABELS = {0: "low", 1: "high"}
_EFFORT_LABELS = {i: f"class{i}" for i in range(40)}


def create_inputs(seed, n_simulation_subjects, params):
    """Build model params and initial conditions from structured parameters."""
    base_income = create_base_income_grid(params["income_process"])
    cost_envelope = create_adjustment_cost_envelope(params["adjustment_cost"])
    xvalues = prod_shock_grid.get_gridpoints()
    xtrans = prod_shock_grid.get_transition_probs()
    ec_grid = create_effort_cost_grid(params["effort_cost"])
    wd_grid = create_work_disutility_grid(
        params["work_disutility"], params["education_disutility_adj"]
    )

    model_params = {
        "work_disutility": {"work_disutility_grid": wd_grid},
        "effort_cost": {
            "effort_elasticity": params["effort_elasticity"],
            "effort_cost_grid": ec_grid,
        },
        "consumption_utility": {
            "utility_constant": params["utility_constant"],
            "health_consumption_penalty": params["health_consumption_penalty"],
        },
        "income": {"base_income_grid": base_income},
        "pension": {
            "base_income_grid": base_income,
            "pension_replacement_rate": params["pension_replacement_rate"],
        },
        "adjustment_cost_penalty": {"adjustment_cost_envelope": cost_envelope},
        "scaled_productivity_shock": {
            "productivity_shock_scale": jnp.sqrt(params["income_process"]["sigx"])
        },
    }

    n = n_simulation_subjects
    key = random.key(seed)
    types = random.choice(key, jnp.arange(16), (n,), p=_initial_dists)
    new_keys = random.split(key=key, num=3)
    health_draw = random.uniform(new_keys[0], (n,))
    health_thresholds = _init_distr[:, 1][types]
    initial_health = jnp.where(health_draw > health_thresholds, 0, 1)
    initial_health_type = 1 - _ht[types]
    initial_education = _ed[types]
    initial_productivity = _prod[types]
    initial_discount = _discount[types]
    initial_effort = jnp.searchsorted(effort_grid, _init_distr[:, 2][types])
    prod_dist = jax.lax.fori_loop(
        0,
        1000000,
        lambda _i, a: a @ xtrans.T,
        jnp.full(5, 1 / 5),
    )
    initial_adjustment_cost = np.asarray(random.uniform(new_keys[1], (n,)))
    initial_productivity_shock = np.asarray(
        xvalues[random.choice(new_keys[2], jnp.arange(5), (n,), p=prod_dist)]
    )

    initial_conditions_df = pd.DataFrame(
        {
            "regime": "alive",
            "age": ages.values[0],
            "wealth": np.zeros(n),
            "health": pd.Categorical(
                [_HEALTH_LABELS[int(v)] for v in initial_health],
                categories=["bad", "good"],
            ),
            "lagged_effort": pd.Categorical(
                [_EFFORT_LABELS[int(v)] for v in initial_effort],
                categories=[f"class{i}" for i in range(40)],
            ),
            "education": pd.Categorical(
                [_EDUCATION_LABELS[int(v)] for v in initial_education],
                categories=["low", "high"],
            ),
            "productivity": pd.Categorical(
                [_PRODUCTIVITY_LABELS[int(v)] for v in initial_productivity],
                categories=["low", "high"],
            ),
            "health_type": pd.Categorical(
                [_HEALTH_TYPE_LABELS[int(v)] for v in initial_health_type],
                categories=["low", "high"],
            ),
            "productivity_shock": initial_productivity_shock,
            "adjustment_cost": initial_adjustment_cost,
        }
    )

    return model_params, initial_conditions_df, np.asarray(initial_discount)


def model_solve_and_simulate(params):
    seed = 32
    n_subjects = 10000
    params_without_beta = {k: v for k, v in params.items() if k != "discount_factor"}
    common_params, initial_conditions_df, discount_factor_type = create_inputs(
        seed, n_simulation_subjects=n_subjects, params=params_without_beta
    )

    beta = params["discount_factor"]
    beta_mean = beta["mean"]
    beta_std = beta["std"]

    dfs = []
    for beta_val, type_id in [
        (beta_mean - beta_std, 0),
        (beta_mean + beta_std, 1),
    ]:
        mask = discount_factor_type == type_id
        type_df = initial_conditions_df.loc[mask].reset_index(drop=True)

        type_initial = initial_conditions_from_dataframe(df=type_df, model=model)

        result = model.simulate(
            params={"alive": {"discount_factor": beta_val, **common_params}},
            initial_conditions=type_initial,
            period_to_regime_to_V_arr=None,
            seed=42,
            log_level="off",
        )
        df = result.to_dataframe(
            additional_targets=[
                "utility",
                "effort_cost",
                "pension",
                "income",
                "consumption",
            ],
            use_labels=False,
        )
        df["discount_type"] = type_id
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def simulate_moments(params):
    res = model_solve_and_simulate(params)
    res = res[res["regime"] == "alive"].copy()
    moments = np.zeros(64)
    res["effort"] = np.asarray(effort_grid[res["effort"].to_numpy().astype(int)])
    res["lagged_effort"] = np.asarray(
        effort_grid[res["lagged_effort"].to_numpy().astype(int)]
    )
    res["wealth"] = np.asarray(wealth_to_level(res["wealth"].to_numpy()))
    res["saving"] = np.asarray(wealth_to_level(res["saving"].to_numpy()))
    for health in range(2):
        for interval in range(4):
            mask = (
                (res["period"] >= (interval * 5))
                & (res["period"] < ((interval + 1) * 5))
                & (res["health"] == health)
            )
            working_pct = (res.loc[mask, ["labor_supply"]].sum() / 2) / (
                res.loc[mask, "health"].count()
            )
            moments[(interval + 4 * (1 - health))] = working_pct.iloc[0]
    for health in range(2):
        for education in range(2):
            for interval in range(6):
                mask = (
                    (res["period"] >= (interval * 5))
                    & (res["period"] < ((interval + 1) * 5))
                    & (res["health"] == health)
                    & (res["education"] == education)
                )
                avg_effort = res.loc[mask, "effort"].sum() / (
                    res.loc[mask, "effort"].count()
                )
                moments[(interval + 6 * (1 - health) + education * 6 * 2) + 8] = (
                    avg_effort
                )
                if interval < 4:
                    avg_income = res.loc[mask, "income"].sum() / (
                        res.loc[mask, "income"].count()
                    )
                    moments[(interval + 4 * (1 - health) + education * 4 * 2) + 46] = (
                        avg_income * _wealth_normalization[1] / 1000
                    )
    for interval in range(6):
        mask = (res["period"] >= (interval * 5)) & (
            res["period"] < ((interval + 1) * 5)
        )
        median_wealth = res.loc[mask, ["wealth"]].median()
        moments[interval + 32] = median_wealth.iloc[0]
    avgemp_low = (res.loc[(res["education"] == 0), ["labor_supply"]].sum() / 2) / (
        res.loc[(res["education"] == 0), "labor_supply"].count()
    )
    avgemp_high = (res.loc[(res["education"] == 1), ["labor_supply"]].sum() / 2) / (
        res.loc[(res["education"] == 1), "labor_supply"].count()
    )
    moments[38] = avgemp_high.iloc[0] / avgemp_low.iloc[0]
    for interval in range(3):
        mask = (res["period"] >= (interval * 10)) & (
            res["period"] < ((interval + 1) * 10)
        )
        non_adjusters = (
            res.loc[mask & (res["effort"] == res["lagged_effort"])].count()
        ) / (res.loc[mask].count())
        moments[interval + 39] = non_adjusters.iloc[0]
    avg_kappa = (
        (res.loc[(res["health"] == 1)].count())
        + (res.loc[(res["health"] == 0)].count()) * params["health_consumption_penalty"]
    ) / (len(res))
    avg_cons = res["consumption"].mean()
    avg_utility = res["utility"].mean()
    vsly = avg_utility / avg_kappa.iloc[0] * (avg_cons**-2)
    moments[42] = vsly
    moments[43] = res["effort"].std()
    moments[44] = gini(jnp.asarray(res["wealth"].to_numpy()))
    cons_ratio = (
        res.loc[(res["health"] == 1), "consumption"].sum()
        / res.loc[(res["health"] == 1), "consumption"].count()
    ) / (
        res.loc[(res["health"] == 0), "consumption"].sum()
        / res.loc[(res["health"] == 0), "consumption"].count()
    )
    moments[45] = cons_ratio
    log_earnings = np.log(
        res.loc[
            (res["period"] <= retirement_period) & (res["labor_supply"] > 0), "income"
        ]
        * productivity_type_multiplier[1]
    )
    moments[62] = log_earnings.var()
    pension_avg = (
        res.loc[(res["period"] == retirement_period + 1), "pension"].sum()
        / res.loc[(res["period"] == retirement_period + 1), "pension"].count()
    )
    avg_income = (
        res.loc[(res["period"] < retirement_period + 1), "income"].sum()
        / res.loc[(res["period"] < retirement_period + 1), "income"].count()
    )
    moments[63] = pension_avg / avg_income
    print(moments)
    return moments


def simulate_wealth(params):
    res = model_solve_and_simulate(params)
    res = res[res["regime"] == "alive"].copy()
    moments = np.zeros(10)
    res["effort"] = np.asarray(effort_grid[res["effort"].to_numpy().astype(int)])
    res["lagged_effort"] = np.asarray(
        effort_grid[res["lagged_effort"].to_numpy().astype(int)]
    )
    res["wealth"] = np.asarray(wealth_to_level(res["wealth"].to_numpy()))
    res["saving"] = np.asarray(wealth_to_level(res["saving"].to_numpy()))
    for interval in range(1, 6):
        median_wealth_h = res.loc[
            (res["period"] >= (interval * 5))
            & (res["period"] < ((interval + 1) * 5))
            & (res["health"] == 1),
            ["wealth"],
        ].median()
        median_wealth_uh = res.loc[
            (res["period"] >= (interval * 5))
            & (res["period"] < ((interval + 1) * 5))
            & (res["health"] == 0),
            ["wealth"],
        ].median()
        moments[interval - 1] = median_wealth_h.iloc[0]
        moments[interval - 1 + 5] = median_wealth_uh.iloc[0]
    return moments
