@.ai-instructions/profiles/tier-a.md @.ai-instructions/modules/jax.md
@.ai-instructions/modules/optimagic.md

# Replication of Mahler & Yum (2024)

## Overview

Replication of "Lifestyle Behaviors and Wealth-Health Gaps in Germany" (Econometrica,
2024\) using pylcm. Consumption-savings model with health, exercise, and heterogeneous
discount factors.

## Build & Test

- `pixi run tests` — run tests
- `pixi run ty` — type checking
- `prek run --all-files` — run all pre-commit hooks

## Architecture

- `src/Mahler_Yum_2024.py` — model definition (regimes, functions, grids)
- `src/model_function.py` — solve, simulate, compute moments
- `src/estimation.py` — MSM estimation via optimagic
- `src/test_run_model.py` — regression tests
- `regression_data/` — reference data for regression tests
