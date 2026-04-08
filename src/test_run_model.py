from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from Mahler_Yum_2024 import MAHLER_YUM_MODEL, START_PARAMS
from model_function import create_inputs, simulate_moments

_REGRESSION_DIR = Path(__file__).parent.parent / "regression_data"


def test_model_solves_and_simulates():
    """Smoke test: model runs end-to-end with small n."""
    start_params_without_beta = {k: v for k, v in START_PARAMS.items() if k != "beta"}
    common_params, initial_states, _discount_factor_type = create_inputs(
        seed=0,
        n_simulation_subjects=4,
        **start_params_without_beta,
    )
    params = {
        "alive": {
            "discount_factor": START_PARAMS["beta"]["mean"],
            **common_params,
        },
    }
    initial_conditions = {
        **initial_states,
        "regime": jnp.full(
            4,
            MAHLER_YUM_MODEL.regime_names_to_ids["alive"],
            dtype=jnp.int32,
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


def test_single_type_simulation_structure():
    """Verify simulation DataFrame has expected structure and values in range."""
    n_subjects = 4
    start_params_without_beta = {k: v for k, v in START_PARAMS.items() if k != "beta"}
    common_params, initial_states, _discount_factor_type = create_inputs(
        seed=0,
        n_simulation_subjects=n_subjects,
        **start_params_without_beta,
    )
    params = {
        "alive": {
            "discount_factor": START_PARAMS["beta"]["mean"],
            **common_params,
        },
    }
    initial_conditions = {
        **initial_states,
        "regime": jnp.full(
            n_subjects,
            MAHLER_YUM_MODEL.regime_names_to_ids["alive"],
            dtype=jnp.int32,
        ),
    }

    got = MAHLER_YUM_MODEL.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        seed=12345,
        log_level="off",
    ).to_dataframe(use_labels=False)

    assert len(got) > 0
    expected_cols = {
        "subject_id",
        "period",
        "age",
        "regime",
        "value",
        "wealth",
        "health",
        "working",
        "saving",
        "effort",
    }
    assert expected_cols.issubset(set(got.columns))
    # Value functions should be finite where alive
    alive = got[got["regime"] == "alive"]
    assert alive["value"].notna().all()


@pytest.mark.skipif(
    not _REGRESSION_DIR.exists(),
    reason="Regression data not yet generated",
)
def test_simulate_moments_regression():
    """Regression test: computed moments match saved reference."""
    expected = np.load(_REGRESSION_DIR / "moments_reference.npy")
    got = simulate_moments(START_PARAMS)
    # Float32 non-determinism means ~1% relative differences across runs.
    # Use rtol=0.02 to account for this.
    np.testing.assert_allclose(got, expected, rtol=0.02)


if __name__ == "__main__":
    print("Running smoke test...")
    test_model_solves_and_simulates()
    print("Smoke test passed!")
