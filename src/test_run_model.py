from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lcm.pandas_utils import initial_conditions_from_dataframe

from Mahler_Yum_2024 import (
    _EFFORT_FIELD_NAMES,
    MAHLER_YUM_MODEL,
    START_PARAMS,
    ages,
    create_inputs,
    model_solve_and_simulate,
    prod_shock_grid,
    retirement_period,
)

_REGRESSION_DIR = Path(__file__).parent.parent / "regression_data"


def _make_initial_conditions(*, df):
    return initial_conditions_from_dataframe(
        df=df,
        regimes=MAHLER_YUM_MODEL.regimes,
        regime_names_to_ids=MAHLER_YUM_MODEL.regime_names_to_ids,
    )


def test_model_solves_and_simulates():
    """Smoke test: model runs end-to-end with small n."""
    common_params, ic_df, _, discount_factor_small, _ = create_inputs(
        seed=0, n_simulation_subjects=4, params=START_PARAMS
    )
    initial_conditions = _make_initial_conditions(df=ic_df)
    result = MAHLER_YUM_MODEL.simulate(
        params={"alive": {"discount_factor": discount_factor_small, **common_params}},
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        seed=12345,
        log_level="off",
    )
    df = result.to_dataframe()
    assert len(df) > 0
    assert "period" in df.columns
    assert "wealth" in df.columns
    assert "labor_supply" in df.columns


# --------------- Regression fixtures ---------------


@pytest.fixture(scope="module")
def simulation_result():
    """Full simulation with START_PARAMS (seed=32, n=10000)."""
    res = model_solve_and_simulate(params=START_PARAMS)
    return res[res["regime"] == "alive"].copy()


# --------------- Period-0 policy regression ---------------


_HAS_OLD_IC = (_REGRESSION_DIR / "old_initial_health.npy").exists()


@pytest.mark.skipif(not _HAS_OLD_IC, reason="Regression data not yet generated")
def test_period_0_policy_matches_old_pylcm():
    """Regression: period-0 labor supply matches pylcm 167a3a6 output.

    Uses initial conditions from the old code to ensure identical period-0
    labor supply choices. Period 0 is deterministic (no stochastic transitions
    yet), so any difference indicates a genuine policy change.
    """
    old_health = np.load(_REGRESSION_DIR / "old_initial_health.npy")
    old_effort = np.load(_REGRESSION_DIR / "old_initial_effort.npy")
    old_discount = np.load(_REGRESSION_DIR / "old_initial_discount.npy")
    old_prodshock = np.load(_REGRESSION_DIR / "old_initial_prodshock.npy")
    old_adjcost = np.load(_REGRESSION_DIR / "old_initial_adjcost.npy")

    common_params, new_ic_df, _, discount_factor_small, discount_factor_large = (
        create_inputs(seed=32, n_simulation_subjects=10000, params=START_PARAMS)
    )

    xvalues = prod_shock_grid.get_gridpoints()
    uniform_gridpoints = np.linspace(0, 1, 5)

    old_ic_df = pd.DataFrame(
        {
            "regime": "alive",
            "age": ages.values[0],
            "wealth": np.zeros(10000),
            "health": np.where(old_health == 0, "bad", "good"),
            "lagged_effort": _EFFORT_FIELD_NAMES[old_effort.astype(int)],
            "education": new_ic_df["education"],
            "productivity": new_ic_df["productivity"],
            "health_type": new_ic_df["health_type"],
            "productivity_shock": np.asarray(xvalues[old_prodshock]),
            "adjustment_cost": uniform_gridpoints[old_adjcost],
        }
    )

    discount_factor_type = old_discount

    all_labor_supply = []
    for discount_factor, type_id in [
        (discount_factor_small, 0),
        (discount_factor_large, 1),
    ]:
        mask = discount_factor_type == type_id
        type_df = old_ic_df.loc[mask].reset_index(drop=True)
        type_initial = _make_initial_conditions(df=type_df)

        result = MAHLER_YUM_MODEL.simulate(
            params={"alive": {"discount_factor": discount_factor, **common_params}},
            initial_conditions=type_initial,
            period_to_regime_to_V_arr=None,
            seed=42,
            log_level="off",
        )
        df = result.to_dataframe(use_labels=False)
        p0 = df[(df["regime"] == "alive") & (df["period"] == 0)]
        all_labor_supply.append(p0["labor_supply"].values)

    labor_supply = np.concatenate(all_labor_supply)

    # Period-0 labor supply distribution must approximately match old pylcm 167a3a6.
    # Small deviations (1-2 agents) are expected from IrregSpacedGrid float precision.
    assert abs((labor_supply == 0).sum() - 109) <= 2
    assert abs((labor_supply == 1).sum() - 5406) <= 2
    assert abs((labor_supply == 2).sum() - 4485) <= 2
    np.testing.assert_allclose(labor_supply.mean(), 1.4376, atol=1e-3)


# --------------- Labor supply distribution ---------------


_LABOR_MAP = {"retired": 0.0, "part_time": 1.0, "full_time": 2.0}


@pytest.mark.parametrize(
    ("period", "expected_retired", "expected_part_time", "expected_full_time"),
    [
        (0, 177, 6283, 3540),
        (1, 435, 7182, 2376),
        (2, 382, 6993, 2614),
        (3, 323, 6495, 3160),
        (4, 270, 6324, 3374),
    ],
)
def test_labor_supply_distribution(
    simulation_result,
    period,
    expected_retired,
    expected_part_time,
    expected_full_time,
):
    """Labor supply counts per period must match reference within tolerance."""
    p = simulation_result[simulation_result["period"] == period]
    vc = p["labor_supply"].value_counts()
    assert abs(vc.get("retired", 0) - expected_retired) <= 5
    assert abs(vc.get("part_time", 0) - expected_part_time) <= 5
    assert abs(vc.get("full_time", 0) - expected_full_time) <= 5


# --------------- Wealth accumulation profile ---------------


@pytest.mark.parametrize(
    ("period", "expected_mean_wealth"),
    [
        (0, 0.0),
        (5, 0.3906),
        (10, 1.2593),
        (15, 2.5898),
        (20, 3.0429),
        (25, 2.0246),
        (30, 0.9496),
    ],
)
def test_mean_wealth_profile(simulation_result, period, expected_mean_wealth):
    """Mean wealth at key periods must match reference."""
    p = simulation_result[simulation_result["period"] == period]
    np.testing.assert_allclose(p["wealth"].mean(), expected_mean_wealth, atol=0.01)


# --------------- Health transitions ---------------


@pytest.mark.parametrize(
    ("period", "expected_good_frac"),
    [
        (0, 0.928),
        (10, 0.9094),
        (20, 0.8327),
        (30, 0.6723),
    ],
)
def test_health_good_fraction(simulation_result, period, expected_good_frac):
    """Fraction in good health must decline with age as expected."""
    p = simulation_result[simulation_result["period"] == period]
    np.testing.assert_allclose(
        (p["health"] == "good").mean(), expected_good_frac, atol=0.005
    )


# --------------- Survival / mortality ---------------


@pytest.mark.parametrize(
    ("period", "expected_alive"),
    [
        (10, 9877),
        (20, 9151),
        (30, 4959),
        (37, 450),
    ],
)
def test_survival_counts(simulation_result, period, expected_alive):
    """Number of surviving agents must match reference."""
    n = len(simulation_result[simulation_result["period"] == period])
    assert abs(n - expected_alive) <= 5


# --------------- Effort behavior ---------------


def test_effort_statistics(simulation_result):
    """Mean and std of effort_value across all periods must match reference."""
    np.testing.assert_allclose(
        simulation_result["effort_value"].mean(), 0.8514, atol=0.005
    )
    np.testing.assert_allclose(
        simulation_result["effort_value"].std(), 0.1860, atol=0.005
    )


# --------------- Consumption by health ---------------


def test_consumption_by_health(simulation_result):
    """Consumption must be higher for good health than bad health."""
    cons = simulation_result.groupby("health")["consumption"].mean()
    np.testing.assert_allclose(cons.loc["good"], 0.8623, atol=0.005)
    np.testing.assert_allclose(cons.loc["bad"], 0.7629, atol=0.005)
    assert cons.loc["good"] > cons.loc["bad"]


# --------------- Income by education ---------------


def test_income_by_education(simulation_result):
    """Mean income during working life must be higher for high education."""
    working = simulation_result[simulation_result["period"] < retirement_period]
    inc = working.groupby("education")["income"].mean()
    np.testing.assert_allclose(inc.loc["low"], 1.0395, atol=0.01)
    np.testing.assert_allclose(inc.loc["high"], 1.9014, atol=0.01)
    assert inc.loc["high"] > inc.loc["low"]


# --------------- Retirement behavior ---------------


def test_all_retired_after_retirement_period(simulation_result):
    """All agents must choose retired labor supply at and after retirement."""
    post_ret = simulation_result[simulation_result["period"] >= retirement_period]
    assert (post_ret["labor_supply"] == "retired").all()


def test_no_income_after_retirement(simulation_result):
    """Labor income must be zero after retirement."""
    post_ret = simulation_result[simulation_result["period"] >= retirement_period]
    np.testing.assert_allclose(post_ret["income"].values, 0.0, atol=1e-10)


# --------------- Structural consistency ---------------


def test_total_alive_rows(simulation_result):
    """Total number of alive-regime rows must match reference."""
    assert abs(len(simulation_result) - 293992) <= 50


def test_wealth_non_negative(simulation_result):
    """Wealth must be non-negative (borrowing constraint)."""
    assert (simulation_result["wealth"] >= -1e-6).all()


def test_consumption_positive(simulation_result):
    """Consumption must be positive."""
    assert (simulation_result["consumption"] > 0).all()


if __name__ == "__main__":
    print("Running smoke test...")
    test_model_solves_and_simulates()
    print("Smoke test passed!")
