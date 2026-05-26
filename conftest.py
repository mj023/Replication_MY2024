"""Install beartype's runtime type-checking claw on `replication_my` for tests.

The claw must be registered before `replication_my` (or any submodule) is first
imported, so it lives in this repo-root `conftest.py` — pytest loads it before
collecting `tests/`. It cannot live inside the package: importing a conftest
under `src/replication_my/` would run `replication_my/__init__.py` first, too
late for the claw to instrument it.
"""

import jax
from beartype import BeartypeConf
from beartype.claw import beartype_packages

# The Mahler & Yum model is float32-unstable: its large state space accumulates
# ~1e-3 errors under XLA's float32 fused kernels. Run at float64, matching pylcm.
jax.config.update("jax_enable_x64", val=True)

# `is_pep484_tower=True`: let `int` satisfy `float`-annotated slots, matching the
# implicit numeric conversions the replication relies on. Default `O1` strategy:
# the claw decorates every function in the package, so constant-time checking
# keeps the test suite fast; deep-container correctness is covered by the
# regression tests.
beartype_packages(
    ("replication_my",),
    conf=BeartypeConf(is_pep484_tower=True),
)
