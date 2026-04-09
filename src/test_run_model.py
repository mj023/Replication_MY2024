from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lcm.pandas_utils import initial_conditions_from_dataframe

from Mahler_Yum_2024 import (
    MAHLER_YUM_MODEL,
    START_PARAMS,
    ages,
    prod_shock_grid,
)
from model_function import create_inputs

_REGRESSION_DIR = Path(__file__).parent.parent / "regression_data"


def test_model_solves_and_simulates():
    """Smoke test: model runs end-to-end with small n."""
    params_without_beta = {
        k: v for k, v in START_PARAMS.items() if k != "discount_factor"
    }
    common_params, initial_conditions_df, _discount_type = create_inputs(
        seed=0, n_simulation_subjects=4, params=params_without_beta
    )
    beta = START_PARAMS["discount_factor"]
    assert isinstance(beta, pd.Series)
    initial_conditions = initial_conditions_from_dataframe(
        df=initial_conditions_df, model=MAHLER_YUM_MODEL
    )
    result = MAHLER_YUM_MODEL.simulate(
        params={"alive": {"discount_factor": beta["mean"], **common_params}},
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        seed=12345,
        log_level="off",
    )
    df = result.to_dataframe(use_labels=False)
    assert len(df) > 0
    assert "period" in df.columns
    assert "wealth" in df.columns
    assert "labor_supply" in df.columns


@pytest.mark.skipif(
    not (_REGRESSION_DIR / "old_initial_health.npy").exists(),
    reason="Regression data not yet generated",
)
def test_period_0_policy_matches_old_pylcm():
    """Regression test: period-0 policy matches pylcm 167a3a6 output.

    Uses initial conditions from the old code (pylcm commit 167a3a6) to ensure
    that the ported model produces identical period-0 actions. Period 0 is
    deterministic (no stochastic transitions yet), so any difference indicates
    a genuine policy difference rather than random seed noise.
    """
    old_health = np.load(_REGRESSION_DIR / "old_initial_health.npy")
    old_effort = np.load(_REGRESSION_DIR / "old_initial_effort.npy")
    old_discount = np.load(_REGRESSION_DIR / "old_initial_discount.npy")
    old_prodshock = np.load(_REGRESSION_DIR / "old_initial_prodshock.npy")
    old_adjcost = np.load(_REGRESSION_DIR / "old_initial_adjcost.npy")

    params_without_beta = {
        k: v for k, v in START_PARAMS.items() if k != "discount_factor"
    }
    common_params, new_ic_df, _ = create_inputs(
        seed=32, n_simulation_subjects=10000, params=params_without_beta
    )

    health_labels = {0: "bad", 1: "good"}
    effort_labels = {i: f"class{i}" for i in range(40)}

    xvalues = prod_shock_grid.get_gridpoints()
    uniform_gridpoints = np.linspace(0, 1, 5)

    old_ic_df = pd.DataFrame(
        {
            "regime": "alive",
            "age": ages.values[0],
            "wealth": np.zeros(10000),
            "health": pd.Categorical(
                [health_labels[int(v)] for v in old_health],
                categories=["bad", "good"],
            ),
            "lagged_effort": pd.Categorical(
                [effort_labels[int(v)] for v in old_effort],
                categories=[f"class{i}" for i in range(40)],
            ),
            "education": new_ic_df["education"],
            "productivity": new_ic_df["productivity"],
            "health_type": new_ic_df["health_type"],
            "productivity_shock": np.asarray(xvalues[old_prodshock]),
            "adjustment_cost": uniform_gridpoints[old_adjcost],
        }
    )

    beta = START_PARAMS["discount_factor"]
    assert isinstance(beta, pd.Series)
    discount_factor_type = old_discount

    all_labor_supply = []
    for beta_val, type_id in [
        (beta["mean"] - beta["std"], 0),
        (beta["mean"] + beta["std"], 1),
    ]:
        mask = discount_factor_type == type_id
        type_df = old_ic_df.loc[mask].reset_index(drop=True)
        type_initial = initial_conditions_from_dataframe(
            df=type_df, model=MAHLER_YUM_MODEL
        )

        result = MAHLER_YUM_MODEL.simulate(
            params={"alive": {"discount_factor": beta_val, **common_params}},
            initial_conditions=type_initial,
            period_to_regime_to_V_arr=None,
            seed=42,
            log_level="off",
        )
        df = result.to_dataframe(use_labels=False)
        p0 = df[(df["regime"] == "alive") & (df["period"] == 0)]
        all_labor_supply.append(p0["labor_supply"].values)

    labor_supply = np.concatenate(all_labor_supply)

    # Period-0 labor supply distribution must match old pylcm 167a3a6 exactly.
    assert (labor_supply == 0).sum() == 109
    assert (labor_supply == 1).sum() == 5406
    assert (labor_supply == 2).sum() == 4485
    np.testing.assert_allclose(labor_supply.mean(), 1.4376, atol=1e-4)


if __name__ == "__main__":
    print("Running smoke test...")
    test_model_solves_and_simulates()
    print("Smoke test passed!")
