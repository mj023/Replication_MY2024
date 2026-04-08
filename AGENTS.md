@.ai-instructions/profiles/tier-a.md @.ai-instructions/modules/jax.md

# Replication of Mahler & Yum (2024)

## Overview

Replication of "Lifestyle Behaviors and Wealth-Health Gaps in Germany" (Econometrica,
2024\) using pylcm. Consumption-savings model with health, exercise, and heterogeneous
discount factors.

## Build & Test

- `pixi run tests` - Run tests
- `pixi run ty` - Type checking
- `prek run --all-files` - Run all pre-commit hooks

## Architecture

- `src/Mahler_Yum_2024.py` - Model definition (regimes, functions, grids)
- `src/model_function.py` - Solve, simulate, compute moments
- `src/estimation.py` - MSM estimation via optimagic
- `src/test_run_model.py` - Regression tests
- `regression_data/` - Reference data for regression tests
