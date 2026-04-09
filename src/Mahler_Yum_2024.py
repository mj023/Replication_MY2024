"""Replication of Mahler & Yum (2024): Lifestyle Behaviors and Wealth-Health Gaps."""

from dataclasses import make_dataclass
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import lcm
import numpy as np
import pandas as pd
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
from lcm.params import MappingLeaf
from lcm.utils.dispatchers import productmap

_DATA_DIR = Path(__file__).parent

avg_earnings_raw = 57706.57
productivity_type_multiplier = jnp.array([jnp.exp(-0.2898), jnp.exp(0.2898)])
ages = AgeGrid(start=25, stop=101, step="2Y")
n_periods = ages.n_periods
retirement_period = 19
labor_tax_rate = 0.128
tax_scale = 1.0 - 0.321
shock_persistence = 0.975
gross_interest_rate = 1.04**2.0
benefit_rate = 0.115
_wealth_normalization = jnp.array([43978, 48201])
avg_earnings = avg_earnings_raw / _wealth_normalization[1]
min_consumption_share = 0.10
min_consumption = min_consumption_share * avg_earnings


def wealth_to_level(x):
    """Convert grid index to wealth level via log-spaced transformation."""
    x = ((jnp.log(10.0**2) - jnp.log(10.0**0)) / 49) * x
    x = jnp.exp(x)
    xgrid = x - 10.0 ** (0.0)
    xgrid = xgrid / (10.0**2 - 10.0**0.0)
    return xgrid * (30 - 0) + 0


@categorical(ordered=True)
class LaborSupply:
    retired: int
    part_time: int
    full_time: int


@categorical(ordered=True)
class Education:
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


effort_grid = jnp.linspace(0, 1, 40)

prod_shock_grid = lcm.shocks.ar1.Rouwenhorst(
    n_points=5, rho=shock_persistence, mu=0, sigma=1
)

risk_aversion = 2.0

health_intercept: float = -0.906
health_age_effects = jnp.asarray(
    [0.0, -0.289, -0.644, -0.881, -1.138, -1.586, -1.586, -1.586]
)
health_effort_coefficients = jnp.asarray([0.693, 0.734])
good_health_coefficient: float = 2.311
health_type_coefficient: float = 0.632
college_coefficient: float = 0.238


def _health_transition(period, health, eff, eff_1, edu, ht):
    y = (
        health_intercept
        + health_age_effects[period]
        + edu * college_coefficient
        + health * good_health_coefficient
        + ht * health_type_coefficient
        + effort_grid[eff] * health_effort_coefficients[0]
        + effort_grid[eff_1] * health_effort_coefficients[1]
    )
    return jnp.exp(y) / (1.0 + jnp.exp(y))


_health_trans_variables = ("period", "health", "eff", "eff_1", "edu", "ht")
_mapped_health_transition = productmap(
    func=_health_transition,
    variables=_health_trans_variables,
    batch_sizes=dict.fromkeys(_health_trans_variables, 0),
)

health_transition_probs = jnp.zeros((38, 2, 40, 40, 2, 2, 2))
_age_groups = jnp.floor_divide(jnp.arange(38), 5)
health_transition_probs = health_transition_probs.at[:, :, :, :, :, :, 1].set(
    _mapped_health_transition(
        period=_age_groups,
        health=jnp.arange(2),
        eff=jnp.arange(40),
        eff_1=jnp.arange(40),
        edu=jnp.arange(2),
        ht=jnp.arange(2),
    )
)
health_transition_probs = health_transition_probs.at[:, :, :, :, :, :, 0].set(
    1.0 - health_transition_probs[:, :, :, :, :, :, 1]
)


def _load_survival_probs():
    """Load survival probabilities as array (period x education x health)."""
    surv_low_edu = np.loadtxt(_DATA_DIR / "surv_HS.txt")
    surv_high_edu = np.loadtxt(_DATA_DIR / "surv_CL.txt")
    probs = jnp.zeros((38, 2, 2))
    probs = probs.at[:, 0, 0].set(surv_low_edu[:, 1])  # low edu, bad health
    probs = probs.at[:, 1, 0].set(surv_high_edu[:, 1])  # high edu, bad health
    probs = probs.at[:, 0, 1].set(surv_low_edu[:, 0])  # low edu, good health
    probs = probs.at[:, 1, 1].set(surv_high_edu[:, 0])  # high edu, good health
    return probs.at[-1].set(0.0)  # certain death at terminal period


survival_probs = _load_survival_probs()


def utility(
    adjustment_cost_penalty: FloatND,
    effort_cost: FloatND,
    work_disutility: FloatND,
    consumption_utility: FloatND,
) -> FloatND:
    return consumption_utility - work_disutility - effort_cost - adjustment_cost_penalty


def work_disutility(
    labor_supply: DiscreteAction,
    health: DiscreteState,
    education: DiscreteState,
    period: Period,
    work_disutility_grid: FloatND,
) -> FloatND:
    return (
        work_disutility_grid[period, education, health]
        * ((labor_supply / 2) ** (2))
        / 2
    )


def adjustment_cost_penalty(
    period: Period,
    adjustment_cost: ContinuousState,
    effort: DiscreteAction,
    lagged_effort: DiscreteState,
    adjustment_cost_envelope: FloatND,
) -> FloatND:
    return jnp.where(
        jnp.logical_not(effort == lagged_effort),
        adjustment_cost * adjustment_cost_envelope[period],
        0,
    )


def consumption(
    net_income: FloatND, wealth: ContinuousState, saving: ContinuousAction
) -> FloatND:
    wealth = wealth_to_level(wealth)
    saving = wealth_to_level(saving)
    return jnp.maximum(
        net_income + wealth * gross_interest_rate - saving, min_consumption
    )


def consumption_utility(
    health: DiscreteState,
    consumption: FloatND,
    health_consumption_penalty: float,
    sigma: float,
    utility_constant: float,
) -> FloatND:
    mucon = jnp.where(health, 1, health_consumption_penalty)
    return (
        mucon * (consumption ** (1.0 - sigma) / (1.0 - sigma))
        + mucon * utility_constant
    )


def effort_cost(
    period: Period,
    education: DiscreteState,
    health: DiscreteState,
    effort: DiscreteAction,
    effort_elasticity: float,
    effort_cost_grid: FloatND,
) -> FloatND:
    return (
        effort_cost_grid[period, education, health]
        * (effort_grid[effort] ** (1 + (1 / effort_elasticity)))
        / (1 + (1 / effort_elasticity))
    )


def net_income(benefits: FloatND, taxed_income: FloatND, pension: FloatND) -> FloatND:
    return taxed_income + pension + benefits


def scaled_productivity_shock(
    productivity_shock: ContinuousState, productivity_shock_scale: float
) -> FloatND:
    return productivity_shock * productivity_shock_scale


def income(
    labor_supply: DiscreteAction,
    period: Period,
    health: DiscreteState,
    education: DiscreteState,
    productivity: DiscreteState,
    scaled_productivity_shock: FloatND,
    base_income_grid: FloatND,
) -> FloatND:
    return (
        base_income_grid[period, health, education]
        * (labor_supply / 2)
        * productivity_type_multiplier[productivity]
        * jnp.exp(scaled_productivity_shock)
    )


def taxed_income(income: FloatND) -> FloatND:
    return (
        tax_scale * (income ** (1.0 - labor_tax_rate)) * (avg_earnings**labor_tax_rate)
    )


def benefits(
    period: Period, health: DiscreteState, labor_supply: DiscreteAction
) -> FloatND:
    eligible = jnp.logical_and(health == 0, labor_supply == 0)
    return jnp.where(
        jnp.logical_and(eligible, period <= retirement_period),
        benefit_rate * avg_earnings,
        0,
    )


def pension(
    period: Period,
    education: DiscreteState,
    productivity: DiscreteState,
    base_income_grid: FloatND,
    pension_replacement_rate: float,
) -> FloatND:
    return jnp.where(
        period > retirement_period,
        base_income_grid[19, 1, education]
        * productivity_type_multiplier[productivity]
        * pension_replacement_rate,
        0,
    )


def next_wealth(saving: ContinuousAction) -> ContinuousState:
    return saving


def next_health(
    period: Period,
    health: DiscreteState,
    effort: DiscreteAction,
    lagged_effort: DiscreteState,
    education: DiscreteState,
    health_type: DiscreteState,
    transition_probs: FloatND,
) -> FloatND:
    return transition_probs[
        period, health, effort, lagged_effort, education, health_type
    ]


def next_lagged_effort(effort: DiscreteAction) -> DiscreteState:
    return effort


def next_regime(
    period: Period,
    education: DiscreteState,
    health: DiscreteState,
    transition_probs: FloatND,
) -> FloatND:
    """Return probability array [P(alive), P(dead)] indexed by RegimeId."""
    survival_prob = transition_probs[period, education, health]
    return jnp.array([survival_prob, 1 - survival_prob])


def retirement_constraint(period: Period, labor_supply: DiscreteAction) -> BoolND:
    return jnp.logical_not(
        jnp.logical_and(period > retirement_period, labor_supply > 0)
    )


def savings_constraint(
    net_income: FloatND, wealth: ContinuousState, saving: ContinuousAction
) -> BoolND:
    wealth = wealth_to_level(wealth)
    saving = wealth_to_level(saving)
    return net_income + wealth * gross_interest_rate >= saving


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
        "lagged_effort": DiscreteGrid(Effort),
        "adjustment_cost": lcm.shocks.iid.Uniform(n_points=5, start=0, stop=1),
        "education": DiscreteGrid(Education),
        "productivity": DiscreteGrid(ProductivityType),
        "health_type": DiscreteGrid(HealthType),
    },
    state_transitions={
        "wealth": next_wealth,
        "health": MarkovTransition(next_health),
        "lagged_effort": next_lagged_effort,
        "education": None,
        "productivity": None,
        "health_type": None,
    },
    actions={
        "labor_supply": DiscreteGrid(LaborSupply),
        "saving": LinSpacedGrid(start=0, stop=49, n_points=50),
        "effort": DiscreteGrid(Effort),
    },
    functions={
        "utility": utility,
        "work_disutility": work_disutility,
        "effort_cost": effort_cost,
        "consumption_utility": consumption_utility,
        "consumption": consumption,
        "income": income,
        "benefits": benefits,
        "adjustment_cost_penalty": adjustment_cost_penalty,
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
    fixed_params={
        "alive": {
            "consumption_utility": {"sigma": risk_aversion},
            "next_health": {"transition_probs": health_transition_probs},
            "next_regime": {"transition_probs": survival_probs},
        },
    },
)


START_PARAMS = {
    # Work disutility knot values at periods [1, 8, 13, 20]
    "work_disutility": pd.DataFrame(
        {
            "bad": [
                2.41177758126754,
                1.8133670880598,
                1.39103558901915,
                2.41466980231321,
            ],
            "good": [
                2.63390750888379,
                1.66602983591164,
                1.27839561280412,
                1.71439043350863,
            ],
        },
        index=[1, 8, 13, 20],
    ),
    "education_disutility_adj": 0.807247922589072,
    # Effort cost knot values at periods [1, 12, 20, 31]
    "effort_cost": MappingLeaf({
        "low": {
            "bad": [0.628031290227532, 1.36593242946612, 1.64963812690034, 0.734873142494319],
            "good": [0.146075197675677, 0.55992411008533, 1.04795036000287, 1.60294886005945],
        },
        "high": {
            "bad": [0.46921037985024, 0.996665589702672, 1.65388250352532, 1.08866246911941],
            "good": [0.091312997289004, 0.302477689083851, 0.739843441095022, 1.36582077051777],
        },
    }),
    "income_process": {
        "y1": pd.Series({"low": 0.899399488241831, "high": 1.1654726432446}),
        "yt_s": pd.Series({"low": 0.0615804210614531, "high": 0.0874283672769353}),
        "yt_sq": pd.Series({"low": -0.00250769285750586, "high": -0.00293713499239749}),
        "wagep": pd.Series({"low": 0.17769766414897, "high": 0.144836058314823}),
        "sigx": 0.0289408524185787,
    },
    "adjustment_cost": [0.000120437772838191, 0.14468204213946],
    "discount_factor": pd.Series(
        {"mean": 0.942749393405227, "std": 0.0283688760224992}
    ),
    "effort_elasticity": 1.11497911620865,
    "utility_constant": 11,
    "health_consumption_penalty": 0.871503495423925,
    "pension_replacement_rate": 0.358766004066242,
}
