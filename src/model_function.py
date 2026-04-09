import numpy as np
import pandas as pd
from lcm.pandas_utils import initial_conditions_from_dataframe

from Mahler_Yum_2024 import MAHLER_YUM_MODEL, create_inputs, wealth_to_level
from utils import gini

model = MAHLER_YUM_MODEL

_effort_grid = np.asarray(model.fixed_params["alive"]["effort_grid"])  # ty: ignore[not-subscriptable]
_productivity_type_multiplier = np.asarray(
    model.fixed_params["alive"]["productivity_type_multiplier"]  # ty: ignore[not-subscriptable]
)
_wealth_normalization = np.array([43978, 48201])
retirement_period = 19


def model_solve_and_simulate(params):
    """Solve and simulate for both discount factor types, return converted DataFrame."""
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

    res = pd.concat(dfs, ignore_index=True)
    res = res[res["regime"] == "alive"].copy()

    # Convert indices to human-readable values
    res["effort"] = _effort_grid[res["effort"].to_numpy().astype(int)]
    res["lagged_effort"] = _effort_grid[res["lagged_effort"].to_numpy().astype(int)]
    res["wealth"] = np.asarray(wealth_to_level(res["wealth"].to_numpy()))
    res["saving"] = np.asarray(wealth_to_level(res["saving"].to_numpy()))

    return res


def simulate_moments(params):
    """Compute 64 target moments from simulated data."""
    res = model_solve_and_simulate(params)
    moments = np.zeros(64)
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
    moments[44] = gini(np.asarray(res["wealth"]))
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
        * _productivity_type_multiplier[1]
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
    """Compute wealth moments by health status."""
    res = model_solve_and_simulate(params)
    moments = np.zeros(10)
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
