"""Example specification for a consumption-savings model with health and exercise.

Replication of Mahler & Yum (2024, Econometrica): "Lifestyle Behaviors and
Wealth-Health Gaps in Germany".

DISCREPANCIES vs. pylcm example (lcm_examples.mahler_yum_2024):
- discount_factor was a state variable with manual beta^period discounting;
  now handled via framework's discount_factor param (two-solve approach).
- bb in START_PARAMS is 11 (starting point), not 13.1 (estimated value).
- Terminal period survival forced to 0 (matching pylcm example).
- Stationary distribution uses 1,000,000 iterations (replication value, vs 200).
- interpax replaced by scipy.interpolate (setup-only, not inside JIT).
- Productivity shock: discrete indices + lookup -> Rouwenhorst shock grid +
  scaled_productivity_shock function (equivalent math).
- Adjustment cost: discrete 0-4 / 4 -> Uniform(0,1) shock grid (equivalent).
"""

from dataclasses import make_dataclass
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import lcm
from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    Period,
)

_DATA_DIR = Path(__file__).parent

avrgearn = 57706.57
theta_val = jnp.array([jnp.exp(-0.2898), jnp.exp(0.2898)])
ages = AgeGrid(start=25, stop=101, step="2Y")
n = ages.n_periods
retirement_age = 19
taul = 0.128
lamda = 1.0 - 0.321
rho = 0.975
r = 1.04**2.0
tt0 = 0.115
winit = jnp.array([43978, 48201])
avrgearn = avrgearn / winit[1]
mincon0 = 0.10
mincon = mincon0 * avrgearn


def calc_savingsgrid(x):
    x = ((jnp.log(10.0**2) - jnp.log(10.0**0)) / 49) * x
    x = jnp.exp(x)
    xgrid = x - 10.0 ** (0.0)
    xgrid = xgrid / (10.0**2 - 10.0**0.0)
    return xgrid * (30 - 0) + 0


@categorical(ordered=True)
class WorkingStatus:
    retired: int
    part: int
    full: int


@categorical(ordered=True)
class EducationStatus:
    low: int
    high: int


Effort = make_dataclass(
    "HealthEffort", [("class" + str(i), int, int(i)) for i in range(40)]
)


@categorical(ordered=True)
class Health:
    bad: int
    good: int


@categorical(ordered=True)
class ProductivityType:
    low: int
    high: int


@categorical(ordered=True)
class HealthType:
    low: int
    high: int


@categorical(ordered=False)
class RegimeId:
    alive: int
    dead: int


eff_grid = jnp.linspace(0, 1, 40)

prod_shock_grid = lcm.shocks.ar1.Rouwenhorst(n_points=5, rho=rho, mu=0, sigma=1)


def utility(
    scaled_adjustment_cost: FloatND,
    fcost: FloatND,
    disutil: FloatND,
    cons_util: FloatND,
) -> FloatND:
    return cons_util - disutil - fcost - scaled_adjustment_cost


def disutil(
    working: DiscreteAction,
    health: DiscreteState,
    education: DiscreteState,
    period: Period,
    phigrid: FloatND,
) -> FloatND:
    return phigrid[period, education, health] * ((working / 2) ** (2)) / 2


def scaled_adjustment_cost(
    period: Period,
    adjustment_cost: ContinuousState,
    effort: DiscreteAction,
    effort_t_1: DiscreteState,
    chimaxgrid: FloatND,
) -> FloatND:
    return jnp.where(
        jnp.logical_not(effort == effort_t_1),
        adjustment_cost * chimaxgrid[period],
        0,
    )


def cnow(
    net_income: FloatND, wealth: ContinuousState, saving: ContinuousAction,
) -> FloatND:
    wealth = calc_savingsgrid(wealth)
    saving = calc_savingsgrid(saving)
    return jnp.maximum(net_income + (wealth) * r - (saving), mincon)


def cons_util(
    health: DiscreteState, cnow: FloatND, kappa: float, sigma: float, bb: float,
) -> FloatND:
    mucon = jnp.where(health, 1, kappa)
    return mucon * (((cnow) ** (1.0 - sigma)) / (1.0 - sigma)) + mucon * bb


def fcost(
    period: Period,
    education: DiscreteState,
    health: DiscreteState,
    effort: DiscreteAction,
    psi: float,
    xigrid: FloatND,
) -> FloatND:
    return (
        xigrid[period, education, health]
        * (eff_grid[effort] ** (1 + (1 / psi)))
        / (1 + (1 / psi))
    )


def net_income(
    benefits: FloatND, taxed_income: FloatND, pension: FloatND,
) -> FloatND:
    return taxed_income + pension + benefits


def scaled_productivity_shock(
    productivity_shock: ContinuousState, sigx: float,
) -> FloatND:
    return productivity_shock * sigx


def income(
    working: DiscreteAction,
    period: Period,
    health: DiscreteState,
    education: DiscreteState,
    productivity: DiscreteState,
    scaled_productivity_shock: FloatND,
    income_grid: FloatND,
) -> FloatND:
    return (
        income_grid[period, health, education]
        * (working / 2)
        * theta_val[productivity]
        * jnp.exp(scaled_productivity_shock)
    )


def taxed_income(income: FloatND) -> FloatND:
    return lamda * (income ** (1.0 - taul)) * (avrgearn**taul)


def benefits(
    period: Period, health: DiscreteState, working: DiscreteAction,
) -> FloatND:
    eligible = jnp.logical_and(health == 0, working == 0)
    return jnp.where(
        jnp.logical_and(eligible, period <= retirement_age), tt0 * avrgearn, 0
    )


def pension(
    period: Period,
    education: DiscreteState,
    productivity: DiscreteState,
    income_grid: FloatND,
    penre: float,
) -> FloatND:
    return jnp.where(
        period > retirement_age,
        income_grid[19, 1, education] * theta_val[productivity] * penre,
        0,
    )


def next_wealth(saving: ContinuousAction) -> ContinuousState:
    return saving


def next_health(
    period: Period,
    health: DiscreteState,
    effort: DiscreteAction,
    effort_t_1: DiscreteState,
    education: DiscreteState,
    health_type: DiscreteState,
    probs_array: FloatND,
) -> FloatND:
    return probs_array[period, health, effort, effort_t_1, education, health_type]


def next_effort_t_1(effort: DiscreteAction) -> DiscreteState:
    return effort


surv_HS = jnp.array(np.loadtxt(_DATA_DIR / "surv_HS.txt"))
surv_CL = jnp.array(np.loadtxt(_DATA_DIR / "surv_CL.txt"))
spgrid = jnp.zeros((38, 2, 2))
spgrid = spgrid.at[:, 0, 0].set(surv_HS[:, 1])
spgrid = spgrid.at[:, 1, 0].set(surv_CL[:, 1])
spgrid = spgrid.at[:, 0, 1].set(surv_HS[:, 0])
spgrid = spgrid.at[:, 1, 1].set(surv_CL[:, 0])
# Certain death at the terminal period (age 101 is inactive for alive)
spgrid = spgrid.at[-1].set(0.0)


def next_regime(
    period: Period,
    education: DiscreteState,
    health: DiscreteState,
    probs_array: FloatND,
) -> FloatND:
    """Return probability array [P(alive), P(dead)] indexed by RegimeId."""
    survival_prob = probs_array[period, education, health]
    return jnp.array([survival_prob, 1 - survival_prob])


def retirement_constraint(period: Period, working: DiscreteAction) -> BoolND:
    return jnp.logical_not(
        jnp.logical_and(period > retirement_age, working > 0)
    )


def savings_constraint(
    net_income: FloatND, wealth: ContinuousState, saving: ContinuousAction,
) -> BoolND:
    wealth = calc_savingsgrid(wealth)
    saving = calc_savingsgrid(saving)
    return net_income + (wealth) * r >= (saving)


def alive_is_active(age: int, final_age_alive: float) -> bool:
    return age <= final_age_alive


def dead_is_active(age: int, initial_age: float) -> bool:
    return age > initial_age


ALIVE_REGIME = Regime(
    transition=MarkovTransition(next_regime),
    active=partial(alive_is_active, final_age_alive=ages.values[-2]),
    states={
        "wealth": LinSpacedGrid(start=0, stop=49, n_points=50),
        "health": DiscreteGrid(Health),
        "productivity_shock": prod_shock_grid,
        "effort_t_1": DiscreteGrid(Effort),
        "adjustment_cost": lcm.shocks.iid.Uniform(n_points=5, start=0, stop=1),
        "education": DiscreteGrid(EducationStatus),
        "productivity": DiscreteGrid(ProductivityType),
        "health_type": DiscreteGrid(HealthType),
    },
    state_transitions={
        "wealth": next_wealth,
        "health": MarkovTransition(next_health),
        "effort_t_1": next_effort_t_1,
        "education": None,
        "productivity": None,
        "health_type": None,
    },
    actions={
        "working": DiscreteGrid(WorkingStatus),
        "saving": LinSpacedGrid(start=0, stop=49, n_points=50),
        "effort": DiscreteGrid(Effort),
    },
    functions={
        "utility": utility,
        "disutil": disutil,
        "fcost": fcost,
        "cons_util": cons_util,
        "cnow": cnow,
        "income": income,
        "benefits": benefits,
        "scaled_adjustment_cost": scaled_adjustment_cost,
        "net_income": net_income,
        "taxed_income": taxed_income,
        "pension": pension,
        "scaled_productivity_shock": scaled_productivity_shock,
    },
    constraints={
        "retirement_constraint": retirement_constraint,
        "savings_constraint": savings_constraint,
    },
)

DEAD_REGIME = Regime(
    transition=None,
    active=partial(dead_is_active, initial_age=ages.values[0]),
    functions={"utility": lambda: 0.0},
)

MAHLER_YUM_MODEL = Model(
    regimes={"alive": ALIVE_REGIME, "dead": DEAD_REGIME},
    ages=ages,
    regime_id_class=RegimeId,
)


START_PARAMS = {
    "nu": {
        "h": [2.63390750888379, 1.66602983591164, 1.27839561280412, 1.71439043350863],
        "u": [2.41177758126754, 1.8133670880598, 1.39103558901915, 2.41466980231321],
        "ad": 0.807247922589072,
    },
    "xi": {
        "hs": {
            "h": [
                0.146075197675677,
                0.55992411008533,
                1.04795036000287,
                1.60294886005945,
            ],
            "u": [
                0.628031290227532,
                1.36593242946612,
                1.64963812690034,
                0.734873142494319,
            ],
        },
        "cl": {
            "h": [
                0.091312997289004,
                0.302477689083851,
                0.739843441095022,
                1.36582077051777,
            ],
            "u": [
                0.46921037985024,
                0.996665589702672,
                1.65388250352532,
                1.08866246911941,
            ],
        },
    },
    "income_process": {
        "hs": {
            "y1": 0.899399488241831,
            "yt_s": 0.0615804210614531,
            "yt_sq": -0.00250769285750586,
            "wagep": 0.17769766414897,
        },
        "cl": {
            "y1": 1.1654726432446,
            "yt_s": 0.0874283672769353,
            "yt_sq": -0.00293713499239749,
            "wagep": 0.144836058314823,
        },
        "sigx": 0.0289408524185787,
    },
    "chi": [0.000120437772838191, 0.14468204213946],
    "beta": {"mean": 0.942749393405227, "std": 0.0283688760224992},
    "psi": 1.11497911620865,
    # NOTE: bb=11 here is a starting point for estimation, not the final estimate
    # (13.1079320277342). Kept as in the original replication.
    "bb": 11,
    "conp": 0.871503495423925,
    "penre": 0.358766004066242,
    "sigma": 2,
}
