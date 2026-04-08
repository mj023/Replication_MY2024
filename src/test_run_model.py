from pathlib import Path

import numpy as np
import pytest
from jax import numpy as jnp

from Mahler_Yum_2024 import MAHLER_YUM_MODEL, START_PARAMS, ages, prod_shock_grid
from model_function import create_inputs

_REGRESSION_DIR = Path(__file__).parent.parent / "regression_data"


def test_model_solves_and_simulates():
    """Smoke test: model runs end-to-end with small n."""
    start_params_without_beta = {
        k: v for k, v in START_PARAMS.items() if k not in ("beta_mean", "beta_std")
    }
    common_params, initial_states, _discount_factor_type = create_inputs(
        seed=0,
        n_simulation_subjects=4,
        **start_params_without_beta,
    )
    params = {"alive": {"discount_factor": START_PARAMS["beta_mean"], **common_params}}
    initial_conditions = {
        **initial_states,
        "regime": jnp.full(
            4, MAHLER_YUM_MODEL.regime_names_to_ids["alive"], dtype=jnp.int32
        ),
    }
    result = MAHLER_YUM_MODEL.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        seed=12345,
        log_level="off",
    )
    df = result.to_dataframe(use_labels=False)
    assert len(df) > 0
    assert "period" in df.columns
    assert "wealth" in df.columns
    assert "working" in df.columns


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
    old_prodshock = np.load(_REGRESSION_DIR / "old_initial_prodshock.npy")
    old_adjcost = np.load(_REGRESSION_DIR / "old_initial_adjcost.npy")
    old_discount = np.load(_REGRESSION_DIR / "old_initial_discount.npy")

    start_params_without_beta = {
        k: v for k, v in START_PARAMS.items() if k not in ("beta_mean", "beta_std")
    }
    common_params, new_initial_states, _ = create_inputs(
        seed=32,
        n_simulation_subjects=10000,
        **start_params_without_beta,
    )

    xvalues = prod_shock_grid.get_gridpoints()
    uniform_gridpoints = jnp.linspace(0, 1, 5)

    initial_states = {
        "age": jnp.full(10000, ages.values[0]),
        "wealth": jnp.full(10000, 0, dtype=jnp.int8),
        "health": jnp.array(old_health),
        "health_type": new_initial_states["health_type"],
        "effort_t_1": jnp.array(old_effort),
        "productivity_shock": xvalues[old_prodshock],
        "adjustment_cost": uniform_gridpoints[old_adjcost],
        "education": new_initial_states["education"],
        "productivity": new_initial_states["productivity"],
    }
    discount_factor_type = jnp.array(old_discount)

    beta_mean = START_PARAMS["beta_mean"]
    beta_std = START_PARAMS["beta_std"]

    all_working = []
    for beta_val, type_id in [(beta_mean - beta_std, 0), (beta_mean + beta_std, 1)]:
        mask = discount_factor_type == type_id
        n_type = int(mask.sum())
        type_initial = {k: v[mask] for k, v in initial_states.items()}
        type_initial["regime"] = jnp.full(
            n_type, MAHLER_YUM_MODEL.regime_names_to_ids["alive"], dtype=jnp.int32
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
        all_working.append(p0["working"].values)

    working = np.concatenate(all_working)

    # Period-0 working distribution must match old pylcm 167a3a6 exactly.
    # Actions at period 0 are deterministic (no stochastic transitions yet).
    assert (working == 0).sum() == 109
    assert (working == 1).sum() == 5406
    assert (working == 2).sum() == 4485
    np.testing.assert_allclose(working.mean(), 1.4376, atol=1e-4)


if __name__ == "__main__":
    print("Running smoke test...")
    test_model_solves_and_simulates()
    print("Smoke test passed!")
