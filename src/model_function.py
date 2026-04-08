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
    calc_savingsgrid,
    prod_shock_grid,
    rho,
)
from utils import gini

_DATA_DIR = Path(__file__).parent

model = MAHLER_YUM_MODEL

avrgearn = 57706.57
theta_val = jnp.array([jnp.exp(-0.2898), jnp.exp(0.2898)])
n = 38
retirement_age = 19
winit = jnp.array([43978, 48201])
avrgearn = avrgearn / winit[1]


def create_phigrid(nu, nu_ad):
    """Interpolate work disutility knots to full period grid.

    Args:
        nu: DataFrame with columns "unhealthy", "healthy" and period index.
        nu_ad: Scalar education adjustment factor.

    """
    phigrid = jnp.zeros((retirement_age + 1, 2, 2))
    for j, health in enumerate(["unhealthy", "healthy"]):
        spline = scipy_interp1d(
            np.asarray(nu.index), np.asarray(nu[health]), kind="cubic"
        )
        interp_points = jnp.arange(1, retirement_age + 2)
        temp_grid = jnp.asarray(spline(interp_points))
        # education=0 (low): apply nu_ad adjustment
        phigrid = phigrid.at[:, 0, j].set(temp_grid * jnp.exp(nu_ad))
        # education=1 (high): no adjustment
        phigrid = phigrid.at[:, 1, j].set(temp_grid)
    return phigrid


def create_xigrid(xi):
    """Interpolate effort disutility knots to full period grid.

    Args:
        xi: DataFrame with MultiIndex columns (education, health) and period index.

    """
    xigrid = jnp.zeros((n, 2, 2))
    edu_labels = ["low", "high"]
    health_labels = ["unhealthy", "healthy"]
    for i, edu in enumerate(edu_labels):
        for j, health in enumerate(health_labels):
            knots = np.asarray(xi[(edu, health)])
            spline = scipy_interp1d(np.asarray(xi.index), knots, kind="cubic")
            interp_points = np.arange(1, 31)
            temp_grid = jnp.asarray(spline(interp_points))
            xigrid = xigrid.at[0:30, i, j].set(temp_grid)
            xigrid = xigrid.at[30:n, i, j].set(knots[-1])
    return xigrid


def create_chimaxgrid(chi):
    t = jnp.arange(38)
    return jnp.maximum(chi[0] * jnp.exp(chi[1] * t), 0)


def create_income_grid(income_process):
    """Build base income grid from income process parameters.

    Args:
        income_process: Dict with "y1", "yt_s", "yt_sq", "wagep" (each a
            pd.Series indexed by education), and "sigx" (scalar).

    """
    sigx = income_process["sigx"]
    sdztemp = ((sigx**2.0) / (1.0 - rho**2.0)) ** 0.5
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
            jnp.exp(((jnp.log(theta_val[1]) ** 2.0) ** 2.0) / 2.0)
            * jnp.exp(((sdztemp**2.0) ** 2.0) / 2.0)
        )

    variables = ("_period", "health", "education")
    mapped = productmap(
        func=calc_base, variables=variables, batch_sizes=dict.fromkeys(variables, 0)
    )
    return mapped(_period=j, health=health, education=education)


eff_grid = jnp.linspace(0, 1, 40)

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
    """Build model params and initial conditions from structured parameters.

    Args:
        seed: Random seed for initial condition draws.
        n_simulation_subjects: Number of agents.
        params: Structured parameter dict (without beta).

    """
    income_grid = create_income_grid(params["income_process"])
    chimax_grid = create_chimaxgrid(params["chi"])
    xvalues = prod_shock_grid.get_gridpoints()
    xtrans = prod_shock_grid.get_transition_probs()
    xi_grid = create_xigrid(params["xi"])
    phi_grid = create_phigrid(params["nu"], params["nu_ad"])

    model_params = {
        "disutil": {"phigrid": phi_grid},
        "fcost": {"psi": params["psi"], "xigrid": xi_grid},
        "cons_util": {"bb": params["bb"], "kappa": params["conp"]},
        "income": {"income_grid": income_grid},
        "pension": {"income_grid": income_grid, "penre": params["penre"]},
        "scaled_adjustment_cost": {"chimaxgrid": chimax_grid},
        "scaled_productivity_shock": {
            "sigx": jnp.sqrt(params["income_process"]["sigx"])
        },
    }

    # Draw initial conditions
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
    initial_effort = jnp.searchsorted(eff_grid, _init_distr[:, 2][types])
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
            "effort_t_1": pd.Categorical(
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
    params_without_beta = {k: v for k, v in params.items() if k != "beta"}
    common_params, initial_conditions_df, discount_factor_type = create_inputs(
        seed, n_simulation_subjects=n_subjects, params=params_without_beta
    )

    beta_mean = params["beta"]["mean"]
    beta_std = params["beta"]["std"]

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
            additional_targets=["utility", "fcost", "pension", "income", "cnow"],
            use_labels=False,
        )
        df["discount_type"] = type_id
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def simulate_moments(params):
    res = model_solve_and_simulate(params)
    res = res[res["regime"] == "alive"].copy()
    moments = np.zeros(64)
    res["effort"] = np.asarray(eff_grid[res["effort"].to_numpy().astype(int)])
    res["effort_t_1"] = np.asarray(eff_grid[res["effort_t_1"].to_numpy().astype(int)])
    res["wealth"] = np.asarray(calc_savingsgrid(res["wealth"].to_numpy()))
    res["saving"] = np.asarray(calc_savingsgrid(res["saving"].to_numpy()))
    for health in range(2):
        for interval in range(4):
            mask = (
                (res["period"] >= (interval * 5))
                & (res["period"] < ((interval + 1) * 5))
                & (res["health"] == health)
            )
            working_pct_10years = (res.loc[mask, ["working"]].sum() / 2) / (
                res.loc[mask, "health"].count()
            )
            moments[(interval + 4 * (1 - health))] = working_pct_10years.iloc[0]
    for health in range(2):
        for education in range(2):
            for interval in range(6):
                mask = (
                    (res["period"] >= (interval * 5))
                    & (res["period"] < ((interval + 1) * 5))
                    & (res["health"] == health)
                    & (res["education"] == education)
                )
                avg_effort_10years = res.loc[mask, "effort"].sum() / (
                    res.loc[mask, "effort"].count()
                )
                moments[(interval + 6 * (1 - health) + education * 6 * 2) + 8] = (
                    avg_effort_10years
                )
                if interval < 4:
                    avg_income_10years = res.loc[mask, "income"].sum() / (
                        res.loc[mask, "income"].count()
                    )
                    moments[(interval + 4 * (1 - health) + education * 4 * 2) + 46] = (
                        avg_income_10years * winit[1] / 1000
                    )
    for interval in range(6):
        mask = (res["period"] >= (interval * 5)) & (
            res["period"] < ((interval + 1) * 5)
        )
        median_wealth_10y = res.loc[mask, ["wealth"]].median()
        moments[interval + 32] = median_wealth_10y.iloc[0]
    avgemp_HS = (res.loc[(res["education"] == 0), ["working"]].sum() / 2) / (
        res.loc[(res["education"] == 0), "working"].count()
    )
    avgemp_CL = (res.loc[(res["education"] == 1), ["working"]].sum() / 2) / (
        res.loc[(res["education"] == 1), "working"].count()
    )
    moments[38] = avgemp_CL.iloc[0] / avgemp_HS.iloc[0]
    for interval in range(3):
        mask = (res["period"] >= (interval * 10)) & (
            res["period"] < ((interval + 1) * 10)
        )
        non_adjusters = (
            res.loc[mask & (res["effort"] == res["effort_t_1"])].count()
        ) / (res.loc[mask].count())
        moments[interval + 39] = non_adjusters.iloc[0]
    avg_kappa = (
        (res.loc[(res["health"] == 1)].count())
        + (res.loc[(res["health"] == 0)].count()) * params["conp"]
    ) / (len(res))
    avg_cons = res["cnow"].mean()
    avg_utility = res["utility"].mean()
    vsly = avg_utility / avg_kappa.iloc[0] * (avg_cons**-2)
    moments[42] = vsly
    moments[43] = res["effort"].std()
    moments[44] = gini(jnp.asarray(res["wealth"].to_numpy()))
    cons_ratio = (
        res.loc[(res["health"] == 1), "cnow"].sum()
        / res.loc[(res["health"] == 1), "cnow"].count()
    ) / (
        res.loc[(res["health"] == 0), "cnow"].sum()
        / res.loc[(res["health"] == 0), "cnow"].count()
    )
    moments[45] = cons_ratio
    log_earnings = np.log(
        res.loc[(res["period"] <= retirement_age) & (res["working"] > 0), "income"]
        * theta_val[1]
    )
    moments[62] = log_earnings.var()
    pension_avg = (
        res.loc[(res["period"] == retirement_age + 1), "pension"].sum()
        / res.loc[(res["period"] == retirement_age + 1), "pension"].count()
    )
    avg_income = (
        res.loc[(res["period"] < retirement_age + 1), "income"].sum()
        / res.loc[(res["period"] < retirement_age + 1), "income"].count()
    )
    moments[63] = pension_avg / avg_income
    print(moments)
    return moments


def simulate_wealth(params):
    res = model_solve_and_simulate(params)
    res = res[res["regime"] == "alive"].copy()
    moments = np.zeros(10)
    res["effort"] = np.asarray(eff_grid[res["effort"].to_numpy().astype(int)])
    res["effort_t_1"] = np.asarray(eff_grid[res["effort_t_1"].to_numpy().astype(int)])
    res["wealth"] = np.asarray(calc_savingsgrid(res["wealth"].to_numpy()))
    res["saving"] = np.asarray(calc_savingsgrid(res["saving"].to_numpy()))
    for interval in range(1, 6):
        median_wealth_10y_h = res.loc[
            (res["period"] >= (interval * 5))
            & (res["period"] < ((interval + 1) * 5))
            & (res["health"] == 1),
            ["wealth"],
        ].median()
        median_wealth_10y_uh = res.loc[
            (res["period"] >= (interval * 5))
            & (res["period"] < ((interval + 1) * 5))
            & (res["health"] == 0),
            ["wealth"],
        ].median()
        moments[interval - 1] = median_wealth_10y_h.iloc[0]
        moments[interval - 1 + 5] = median_wealth_10y_uh.iloc[0]
    return moments
