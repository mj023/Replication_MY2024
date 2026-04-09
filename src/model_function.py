import jax.numpy as jnp
import numpy as np
import pandas as pd
from lcm.pandas_utils import initial_conditions_from_dataframe

from Mahler_Yum_2024 import MAHLER_YUM_MODEL, create_inputs
from utils import gini

model = MAHLER_YUM_MODEL

_productivity_type_multiplier_high = float(
    model.fixed_params["alive"]["productivity_type_multiplier"][1]  # ty: ignore[not-subscriptable, invalid-argument-type]
)
_WEALTH_NORMALIZATION = 48201
_RETIREMENT_PERIOD = 19
_INTERVAL_LABELS_4 = ["25-34", "35-44", "45-54", "55-64"]
_INTERVAL_LABELS_6 = [*_INTERVAL_LABELS_4, "65-74", "75-84"]
_INTERVAL_LABELS_10 = ["25-44", "45-64", "65-84"]
_HEALTH_LABELS = ["bad", "good"]
_EDUCATION_LABELS = ["low", "high"]

_ADDITIONAL_TARGETS = [
    "utility",
    "effort_cost",
    "pension",
    "income",
    "consumption",
    "effort_value",
    "lagged_effort_value",
    "wealth_level",
    "saving_level",
]


def _build_moment_index():  # noqa: C901
    """Build the labeled index for the 64 target moments.

    The ordering matches the original position arithmetic exactly.

    """
    idx = [None] * 64

    for health_code, health in enumerate(_HEALTH_LABELS):
        for i, interval in enumerate(_INTERVAL_LABELS_4):
            idx[i + 4 * (1 - health_code)] = ("working_pct", health, interval)

    for health_code, health in enumerate(_HEALTH_LABELS):
        for edu_code, edu in enumerate(_EDUCATION_LABELS):
            for i, interval in enumerate(_INTERVAL_LABELS_6):
                idx[i + 6 * (1 - health_code) + edu_code * 12 + 8] = (
                    "avg_effort",
                    edu,
                    health,
                    interval,
                )

    for i, interval in enumerate(_INTERVAL_LABELS_6):
        idx[32 + i] = ("median_wealth", interval)

    idx[38] = ("employment_ratio",)

    for i, interval in enumerate(_INTERVAL_LABELS_10):
        idx[39 + i] = ("non_adjusters", interval)

    idx[42] = ("vsly",)
    idx[43] = ("effort_std",)
    idx[44] = ("wealth_gini",)
    idx[45] = ("consumption_ratio",)

    for health_code, health in enumerate(_HEALTH_LABELS):
        for edu_code, edu in enumerate(_EDUCATION_LABELS):
            for i, interval in enumerate(_INTERVAL_LABELS_4):
                idx[i + 4 * (1 - health_code) + edu_code * 8 + 46] = (
                    "avg_income",
                    edu,
                    health,
                    interval,
                )

    idx[62] = ("log_earnings_var",)
    idx[63] = ("pension_income_ratio",)

    return pd.Index([str(t) for t in idx])


MOMENT_INDEX = _build_moment_index()


def model_solve_and_simulate(params):
    """Solve and simulate for both discount factor types."""
    (
        common_params,
        initial_conditions_df,
        discount_types,
        discount_factor_small,
        discount_factor_large,
    ) = create_inputs(seed=32, n_simulation_subjects=10000, params=params)

    dfs = []
    for discount_factor, type_id in [
        (discount_factor_small, 0),
        (discount_factor_large, 1),
    ]:
        mask = discount_types == type_id
        type_df = initial_conditions_df.loc[mask].reset_index(drop=True)
        type_initial = initial_conditions_from_dataframe(df=type_df, model=model)

        result = model.simulate(
            params={"alive": {"discount_factor": discount_factor, **common_params}},
            initial_conditions=type_initial,
            period_to_regime_to_V_arr=None,
            seed=42,
            log_level="off",
        )
        df = result.to_dataframe(additional_targets=_ADDITIONAL_TARGETS)
        df["discount_type"] = type_id
        dfs.append(df)

    res = pd.concat(dfs, ignore_index=True)
    return res.loc[res["regime"] == "alive"].copy()


def _assign_intervals(res):
    """Add interval columns for groupby aggregation."""
    res["interval_5"] = pd.cut(
        res["period"],
        bins=[-1, 5, 10, 15, 20, 25, 30],
        labels=_INTERVAL_LABELS_6,
    )
    res["interval_4"] = pd.cut(
        res["period"],
        bins=[-1, 5, 10, 15, 20],
        labels=_INTERVAL_LABELS_4,
    )
    res["interval_10"] = pd.cut(
        res["period"],
        bins=[-1, 10, 20, 30],
        labels=_INTERVAL_LABELS_10,
    )
    return res


def _fill_grouped_moments(moments, grouped, name, keys):
    """Fill moments Series from a groupby result."""
    for key in keys:
        label = (name, *key) if isinstance(key, tuple) else (name, key)
        moments[str(label)] = grouped.loc[key]


def simulate_moments(params):
    """Compute 64 target moments from simulated data."""
    res = model_solve_and_simulate(params)
    res = _assign_intervals(res)
    moments = pd.Series(0.0, index=MOMENT_INDEX)

    # Working pct by (health, 4 intervals)
    working = res.groupby(["health", "interval_4"])["labor_supply"].agg(
        lambda x: (x != "retired").sum() / len(x)
    )
    _fill_grouped_moments(
        moments,
        working,
        "working_pct",
        [(h, i) for h in _HEALTH_LABELS for i in _INTERVAL_LABELS_4],
    )

    # Avg effort by (education, health, 6 intervals)
    avg_effort = res.groupby(["education", "health", "interval_5"])[
        "effort_value"
    ].mean()
    _fill_grouped_moments(
        moments,
        avg_effort,
        "avg_effort",
        [
            (e, h, i)
            for e in _EDUCATION_LABELS
            for h in _HEALTH_LABELS
            for i in _INTERVAL_LABELS_6
        ],
    )

    # Avg income by (education, health, 4 intervals), scaled
    avg_income = res.groupby(["education", "health", "interval_4"])["income"].mean()
    avg_income = avg_income * _WEALTH_NORMALIZATION / 1000
    _fill_grouped_moments(
        moments,
        avg_income,
        "avg_income",
        [
            (e, h, i)
            for e in _EDUCATION_LABELS
            for h in _HEALTH_LABELS
            for i in _INTERVAL_LABELS_4
        ],
    )

    # Median wealth by (6 intervals)
    median_wealth = res.groupby("interval_5")["wealth_level"].median()
    _fill_grouped_moments(moments, median_wealth, "median_wealth", _INTERVAL_LABELS_6)

    # Employment ratio (high edu / low edu)
    emp_by_edu = res.groupby("education")["labor_supply"].agg(
        lambda x: (x != "retired").sum() / len(x)
    )
    moments[str(("employment_ratio",))] = emp_by_edu.loc["high"] / emp_by_edu.loc["low"]

    # Non-adjusters by 10-year intervals
    res["is_non_adjuster"] = res["effort_value"] == res["lagged_effort_value"]
    non_adj = res.groupby("interval_10")["is_non_adjuster"].mean()
    _fill_grouped_moments(moments, non_adj, "non_adjusters", _INTERVAL_LABELS_10)

    # Scalar moments
    health_good_count = (res["health"] == "good").sum()
    health_bad_count = (res["health"] == "bad").sum()
    avg_kappa = (
        health_good_count + health_bad_count * params["health_consumption_penalty"]
    ) / len(res)
    moments[str(("vsly",))] = (
        res["utility"].mean() / avg_kappa * (res["consumption"].mean() ** -2)
    )
    moments[str(("effort_std",))] = res["effort_value"].std()
    moments[str(("wealth_gini",))] = gini(jnp.asarray(res["wealth_level"].to_numpy()))

    cons_by_health = res.groupby("health")["consumption"].mean()
    moments[str(("consumption_ratio",))] = (
        cons_by_health.loc["good"] / cons_by_health.loc["bad"]
    )

    working_mask = (res["period"] <= _RETIREMENT_PERIOD) & (
        res["labor_supply"] != "retired"
    )
    log_earnings = np.log(
        res.loc[working_mask, "income"] * _productivity_type_multiplier_high
    )
    moments[str(("log_earnings_var",))] = log_earnings.var()

    pension_avg = res.loc[res["period"] == _RETIREMENT_PERIOD + 1, "pension"].mean()
    avg_income_pre_ret = res.loc[
        res["period"] < _RETIREMENT_PERIOD + 1, "income"
    ].mean()
    moments[str(("pension_income_ratio",))] = pension_avg / avg_income_pre_ret

    print(moments.values)
    return moments


def simulate_wealth(params):
    """Compute wealth moments by health status."""
    res = model_solve_and_simulate(params)
    res = _assign_intervals(res)
    intervals = _INTERVAL_LABELS_6[1:]  # skip first interval (ages 25-34)
    moments = pd.Series(
        0.0,
        index=[
            f"median_wealth_{health}_{interval}"
            for health in _HEALTH_LABELS
            for interval in intervals
        ],
    )
    median_wealth = res.groupby(["health", "interval_5"])["wealth_level"].median()
    for health in _HEALTH_LABELS:
        for interval in intervals:
            moments[f"median_wealth_{health}_{interval}"] = median_wealth.loc[
                health, interval
            ]
    return moments
