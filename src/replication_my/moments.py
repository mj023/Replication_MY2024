"""Moment specification and computation for MSM estimation."""

import dataclasses
import logging
from collections.abc import Sequence
from typing import cast

import jax.numpy as jnp
import numpy as np
import pandas as pd

from replication_my.mahler_yum_2024 import (
    Education,
    Health,
    _wealth_normalization,
    model_solve_and_simulate,
    retirement_period,
)
from replication_my.utils import gini

_log = logging.getLogger("lcm")

_productivity_type_multiplier_high = float(
    np.exp(0.2898)  # high-type multiplier, matches jnp.exp(0.2898)
)

# Maps labor supply labels to intensive margin (fraction of full-time)
_LABOR_INTENSITY = {"retired": 0.0, "part_time": 0.5, "full_time": 1.0}

_HEALTH_FIELDS = [f.name for f in dataclasses.fields(Health)]
_EDUCATION_FIELDS = [f.name for f in dataclasses.fields(Education)]
_INTERVAL_LABELS_4 = ["25-34", "35-44", "45-54", "55-64"]
_INTERVAL_LABELS_6 = [*_INTERVAL_LABELS_4, "65-74", "75-84"]
_INTERVAL_LABELS_10 = ["25-44", "45-64", "65-84"]


def _build_moment_index() -> pd.Index:  # noqa: C901
    """Build the labeled index for the 64 target moments.

    The ordering matches the original position arithmetic exactly.

    """
    idx = [None] * 64

    for health_code, health in enumerate(_HEALTH_FIELDS):
        for i, interval in enumerate(_INTERVAL_LABELS_4):
            idx[i + 4 * (1 - health_code)] = ("working_pct", health, interval)

    for health_code, health in enumerate(_HEALTH_FIELDS):
        for edu_code, edu in enumerate(_EDUCATION_FIELDS):
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

    for health_code, health in enumerate(_HEALTH_FIELDS):
        for edu_code, edu in enumerate(_EDUCATION_FIELDS):
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

empirical_moments = pd.Series(
    np.array(
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
    ),
    index=MOMENT_INDEX,
)

moment_sd = pd.Series(
    np.array(
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
    ),
    index=MOMENT_INDEX,
)


def _assign_intervals(*, res: pd.DataFrame) -> pd.DataFrame:
    """Add interval columns for groupby aggregation."""
    res["interval_5"] = pd.cut(
        res["period"],
        bins=[-1, 5, 10, 15, 20, 25, 30],
        labels=_INTERVAL_LABELS_6,
        right=False,
    )
    res["interval_4"] = pd.cut(
        res["period"],
        bins=[-1, 5, 10, 15, 20],
        labels=_INTERVAL_LABELS_4,
        right=False,
    )
    res["interval_10"] = pd.cut(
        res["period"],
        bins=[-1, 10, 20, 30],
        labels=_INTERVAL_LABELS_10,
        right=False,
    )
    return res


def _fill_grouped_moments(
    *,
    moments: pd.Series,
    grouped: pd.Series,
    name: str,
    keys: Sequence[tuple[str, ...] | str],
) -> None:
    """Fill moments Series from a groupby result."""
    for key in keys:
        label = (name, *key) if isinstance(key, tuple) else (name, key)
        moments[str(label)] = grouped.loc[key]


def simulate_moments(*, params: dict) -> pd.Series:
    """Compute 64 target moments from simulated data."""
    res = model_solve_and_simulate(params=params)
    res = _assign_intervals(res=res)
    moments = pd.Series(0.0, index=MOMENT_INDEX)
    res["labor_supply"] = res["labor_supply"].map(_LABOR_INTENSITY).astype("float")
    # Working pct by (health, 4 intervals) — intensive margin
    working = res.groupby(["health", "interval_4"])["labor_supply"].agg(np.mean)
    _fill_grouped_moments(
        moments=moments,
        grouped=working,
        name="working_pct",
        keys=[(h, i) for h in _HEALTH_FIELDS for i in _INTERVAL_LABELS_4],
    )

    # Avg effort by (education, health, 6 intervals)
    avg_effort = res.groupby(["education", "health", "interval_5"])[
        "effort_value"
    ].mean()
    _fill_grouped_moments(
        moments=moments,
        grouped=avg_effort,
        name="avg_effort",
        keys=[
            (e, h, i)
            for e in _EDUCATION_FIELDS
            for h in _HEALTH_FIELDS
            for i in _INTERVAL_LABELS_6
        ],
    )

    # Avg income by (education, health, 4 intervals), scaled
    avg_income = res.groupby(["education", "health", "interval_4"])["income"].mean()
    avg_income = avg_income * _wealth_normalization[1] / 1000
    _fill_grouped_moments(
        moments=moments,
        grouped=avg_income,
        name="avg_income",
        keys=[
            (e, h, i)
            for e in _EDUCATION_FIELDS
            for h in _HEALTH_FIELDS
            for i in _INTERVAL_LABELS_4
        ],
    )

    # Median wealth by (6 intervals). `SeriesGroupBy.median()` on a single
    # selected column returns a Series; the pandas stubs over-approximate it.
    median_wealth = cast("pd.Series", res.groupby("interval_5")["wealth"].median())
    _fill_grouped_moments(
        moments=moments,
        grouped=median_wealth,
        name="median_wealth",
        keys=_INTERVAL_LABELS_6,
    )

    # Employment ratio (high edu / low edu) — intensive margin
    emp_by_edu = res.groupby("education")["labor_supply"].agg(np.mean)
    moments[str(("employment_ratio",))] = emp_by_edu.loc["high"] / emp_by_edu.loc["low"]

    # Non-adjusters by 20-year intervals
    res["is_non_adjuster"] = res["effort_value"] == res["lagged_effort_value"]
    non_adj = res.groupby("interval_10")["is_non_adjuster"].mean()
    _fill_grouped_moments(
        moments=moments,
        grouped=non_adj,
        name="non_adjusters",
        keys=_INTERVAL_LABELS_10,
    )

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
    moments[str(("wealth_gini",))] = gini(jnp.asarray(res["wealth"].to_numpy()))

    cons_by_health = res.groupby("health")["consumption"].mean()
    moments[str(("consumption_ratio",))] = (
        cons_by_health.loc["good"] / cons_by_health.loc["bad"]
    )

    working_mask = (res["period"] < retirement_period) & (res["labor_supply"] > 0)
    log_earnings = np.log(
        res.loc[working_mask, "income"] * _productivity_type_multiplier_high
    )
    moments[str(("log_earnings_var",))] = log_earnings.var()

    pension_avg = res.loc[res["period"] == retirement_period, "pension"].mean()
    avg_income_pre_ret = res.loc[res["period"] < retirement_period, "income"].mean()
    moments[str(("pension_income_ratio",))] = pension_avg / avg_income_pre_ret

    _log.info(moments.values)
    return moments
