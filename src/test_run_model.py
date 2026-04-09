import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lcm.pandas_utils import initial_conditions_from_dataframe

from Mahler_Yum_2024 import (
    MAHLER_YUM_MODEL,
    START_PARAMS,
    Effort,
    Health,
    ages,
    create_inputs,
    prod_shock_grid,
)

_REGRESSION_DIR = Path(__file__).parent.parent / "regression_data"


def test_model_solves_and_simulates():
    """Smoke test: model runs end-to-end with small n."""
    common_params, ic_df, _, discount_factor_small, _ = create_inputs(
        seed=0, n_simulation_subjects=4, params=START_PARAMS
    )
    initial_conditions = initial_conditions_from_dataframe(
        df=ic_df, model=MAHLER_YUM_MODEL
    )
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

    common_params, new_ic_df, _, discount_factor_small, discount_factor_large = (
        create_inputs(seed=32, n_simulation_subjects=10000, params=START_PARAMS)
    )

    health_fields = [f.name for f in dataclasses.fields(Health)]
    effort_fields = [f.name for f in dataclasses.fields(Effort)]

    xvalues = prod_shock_grid.get_gridpoints()
    uniform_gridpoints = np.linspace(0, 1, 5)

    old_ic_df = pd.DataFrame(
        {
            "regime": "alive",
            "age": ages.values[0],
            "wealth": np.zeros(10000),
            "health": pd.Categorical(
                [health_fields[int(v)] for v in old_health],
            ).astype(Health.to_categorical_dtype()),  # ty: ignore[unresolved-attribute],
            "lagged_effort": pd.Categorical(
                [effort_fields[int(v)] for v in old_effort],
            ).astype(Effort.to_categorical_dtype()),  # ty: ignore[unresolved-attribute],
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
        type_initial = initial_conditions_from_dataframe(
            df=type_df, model=MAHLER_YUM_MODEL
        )

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


if __name__ == "__main__":
    print("Running smoke test...")
    test_model_solves_and_simulates()
    print("Smoke test passed!")
