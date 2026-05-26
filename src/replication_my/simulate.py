"""Standard solve-and-simulate run for the Mahler & Yum replication."""

import pandas as pd
from lcm_examples.mahler_yum_2024 import MAHLER_YUM_MODEL, create_inputs

_ADDITIONAL_TARGETS = [
    "utility",
    "effort_cost",
    "pension",
    "income",
    "consumption",
    "effort_value",
    "lagged_effort_value",
]


def model_solve_and_simulate(*, params: dict) -> pd.DataFrame:
    """Solve and simulate the model (one pass, both discount types)."""
    common_params, initial_conditions_df = create_inputs(
        seed=32, n_simulation_subjects=10000, params=params
    )
    result = MAHLER_YUM_MODEL.simulate(
        params={"alive": common_params},
        initial_conditions=initial_conditions_df,
        period_to_regime_to_V_arr=None,
        seed=42,
        log_level="off",
    )
    res = result.to_dataframe(additional_targets=_ADDITIONAL_TARGETS)
    return res.loc[res["regime_name"] == "alive"].copy()
