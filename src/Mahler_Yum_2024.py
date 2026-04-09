"""Replication of Mahler & Yum (2024): Lifestyle Behaviors and Wealth-Health Gaps."""

from dataclasses import make_dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import lcm
import numpy as np
import pandas as pd
from jax import random
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
from scipy.interpolate import interp1d as scipy_interp1d

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

health_intercept = -0.906
health_age_effects = pd.Series(
    {
        25: 0.0,
        35: -0.289,
        45: -0.644,
        55: -0.881,
        65: -1.138,
        75: -1.586,
        85: -1.586,
        95: -1.586,
    },
).reindex(np.asarray(ages.values), method="ffill")
health_age_effects.index.name = "age"
health_effort_coefficient = 0.693
lagged_health_effort_coefficient = 0.734
good_health_coefficient = 2.311
health_type_coefficient = 0.632
college_coefficient = 0.238


def _load_survival_probs():
    """Load survival probabilities as labeled Series (age x education x health)."""
    surv_hs = np.loadtxt(_DATA_DIR / "surv_HS.txt")
    surv_cl = np.loadtxt(_DATA_DIR / "surv_CL.txt")
    age_values = np.asarray(ages.values)
    n_data_rows = len(surv_hs)
    records = []
    for period_idx, age in enumerate(age_values):
        for edu_label, surv_data in [("low", surv_hs), ("high", surv_cl)]:
            for health_idx, health_label in enumerate(["good", "bad"]):
                if period_idx >= n_data_rows - 1:
                    prob = 0.0  # certain death at terminal transition
                else:
                    prob = surv_data[period_idx, health_idx]
                records.append((age, edu_label, health_label, prob))
    df = pd.DataFrame(
        records, columns=["age", "education", "health", "survival_probability"]
    )
    return df.set_index(["age", "education", "health"])["survival_probability"]


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
    effort_grid: FloatND,
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


def base_income(
    period: Period,
    health: DiscreteState,
    education: DiscreteState,
    y1: FloatND,
    yt_s: FloatND,
    yt_sq: FloatND,
    wagep: FloatND,
    income_normalization: float,
) -> FloatND:
    """Compute base income for a given (period, health, education) combination."""
    yt = (
        y1[education]
        * jnp.exp(yt_s[education] * period + yt_sq[education] * period**2.0)
        * (1.0 - wagep[education] * (1.0 - health))
    )
    return yt / income_normalization


def income(
    labor_supply: DiscreteAction,
    productivity: DiscreteState,
    scaled_productivity_shock: FloatND,
    base_income: FloatND,
    productivity_type_multiplier: FloatND,
) -> FloatND:
    return (
        base_income
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
    pension_base: FloatND,
    pension_replacement_rate: float,
    productivity_type_multiplier: FloatND,
) -> FloatND:
    return jnp.where(
        period > retirement_period,
        pension_base[education]
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
    health_intercept: float,
    health_age_effects: FloatND,
    good_health_coefficient: float,
    health_type_coefficient: float,
    college_coefficient: float,
    health_effort_coefficient: float,
    lagged_health_effort_coefficient: float,
    effort_grid: FloatND,
) -> FloatND:
    """Compute health transition probabilities via logit model."""
    y = (
        health_intercept
        + health_age_effects[period]
        + education * college_coefficient
        + health * good_health_coefficient
        + health_type * health_type_coefficient
        + effort_grid[effort] * health_effort_coefficient
        + effort_grid[lagged_effort] * lagged_health_effort_coefficient
    )
    prob_good = jnp.exp(y) / (1.0 + jnp.exp(y))
    return jnp.array([1.0 - prob_good, prob_good])


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
        "base_income": base_income,
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
            "effort_grid": effort_grid,
            "productivity_type_multiplier": productivity_type_multiplier,
            "consumption_utility": {"sigma": risk_aversion},
            "next_health": {
                "health_intercept": health_intercept,
                "health_age_effects": health_age_effects,
                "good_health_coefficient": good_health_coefficient,
                "health_type_coefficient": health_type_coefficient,
                "college_coefficient": college_coefficient,
                "health_effort_coefficient": health_effort_coefficient,
                "lagged_health_effort_coefficient": lagged_health_effort_coefficient,
            },
            "next_regime": {"transition_probs": survival_probs},
        },
    },
)


START_PARAMS = {
    # Work disutility knot values at ages 27, 41, 51, 65
    "work_disutility": {
        "bad": {
            "27": 2.41177758126754,
            "41": 1.8133670880598,
            "51": 1.39103558901915,
            "65": 2.41466980231321,
        },
        "good": {
            "27": 2.63390750888379,
            "41": 1.66602983591164,
            "51": 1.27839561280412,
            "65": 1.71439043350863,
        },
    },
    "education_disutility_adjustment": 0.807247922589072,
    # Effort cost knot values at ages 27, 49, 65, 87
    "effort_cost": {
        "low": {
            "bad": {
                "27": 0.628031290227532,
                "49": 1.36593242946612,
                "65": 1.64963812690034,
                "87": 0.734873142494319,
            },
            "good": {
                "27": 0.146075197675677,
                "49": 0.55992411008533,
                "65": 1.04795036000287,
                "87": 1.60294886005945,
            },
        },
        "high": {
            "bad": {
                "27": 0.46921037985024,
                "49": 0.996665589702672,
                "65": 1.65388250352532,
                "87": 1.08866246911941,
            },
            "good": {
                "27": 0.091312997289004,
                "49": 0.302477689083851,
                "65": 0.739843441095022,
                "87": 1.36582077051777,
            },
        },
    },
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


# ---------------------------------------------------------------------------
# Grid creation and initial conditions
# ---------------------------------------------------------------------------


def _age_keys_to_periods(age_keyed_dict):
    """Convert {"27": val, "41": val, ...} to period-indexed arrays.

    The grid creation functions use period indexing for the interpolation knots.

    """
    start_age = int(ages.values[0])
    step = int(ages.values[1] - ages.values[0])
    knot_ages = np.array([int(k) for k in age_keyed_dict])
    knot_periods = (knot_ages - start_age) // step
    values = np.array(list(age_keyed_dict.values()))
    return knot_periods, values


def create_work_disutility_grid(work_disutility, education_disutility_adjustment):
    """Interpolate work disutility knots to full period grid.

    Args:
        work_disutility: Dict {"bad": {"27": v, ...}, "good": {"27": v, ...}}.
        education_disutility_adjustment: Scalar education adjustment factor.

    """
    grid = jnp.zeros((retirement_period + 1, 2, 2))
    for j, health in enumerate(["bad", "good"]):
        knot_periods, knot_values = _age_keys_to_periods(work_disutility[health])
        spline = scipy_interp1d(knot_periods, knot_values, kind="cubic")
        interp_points = jnp.arange(1, retirement_period + 2)
        temp_grid = jnp.asarray(spline(interp_points))
        grid = grid.at[:, 0, j].set(
            temp_grid * jnp.exp(education_disutility_adjustment)
        )
        grid = grid.at[:, 1, j].set(temp_grid)
    return grid


def create_effort_cost_grid(effort_cost):
    """Interpolate effort cost knots to full period grid.

    Args:
        effort_cost: Nested dict
            {"low": {"bad": {"27": v, ...}, ...}, "high": {...}}.

    """
    grid = jnp.zeros((n_periods, 2, 2))
    for i, edu in enumerate(["low", "high"]):
        for j, health in enumerate(["bad", "good"]):
            knot_periods, knot_values = _age_keys_to_periods(effort_cost[edu][health])
            spline = scipy_interp1d(knot_periods, knot_values, kind="cubic")
            interp_points = np.arange(1, 31)
            temp_grid = jnp.asarray(spline(interp_points))
            grid = grid.at[0:30, i, j].set(temp_grid)
            grid = grid.at[30:n_periods, i, j].set(knot_values[-1])
    return grid


def create_adjustment_cost_envelope(adjustment_cost):
    """Build exponential adjustment cost envelope over periods."""
    t = jnp.arange(n_periods)
    return jnp.maximum(adjustment_cost[0] * jnp.exp(adjustment_cost[1] * t), 0)


# Initial type distribution arrays
_discount = jnp.zeros((16), dtype=jnp.int8)
_prod = jnp.zeros((16), dtype=jnp.int8)
_ht = jnp.zeros((16), dtype=jnp.int8)
_ed = jnp.zeros((16), dtype=jnp.int8)
for _i in range(1, 3):
    for _j in range(1, 3):
        for _k in range(1, 3):
            _index = (_i - 1) * 2 * 2 + (_j - 1) * 2 + _k - 1
            _discount = _discount.at[_index].set(_i - 1)
            _prod = _prod.at[_index].set(_j - 1)
            _ht = _ht.at[_index].set(1 - (_k - 1))
            _discount = _discount.at[_index + 8].set(_i - 1)
            _prod = _prod.at[_index + 8].set(_j - 1)
            _ht = _ht.at[_index + 8].set(1 - (_k - 1))
            _ed = _ed.at[_index + 8].set(1)
_init_distr = jnp.array(np.loadtxt(_DATA_DIR / "init_distr_2b2t2h.txt"))
_initial_dists = jnp.diff(_init_distr[:, 0], prepend=0)

_HEALTH_LABELS = {0: "bad", 1: "good"}
_EDUCATION_LABELS = {0: "low", 1: "high"}
_PRODUCTIVITY_LABELS = {0: "low", 1: "high"}
_HEALTH_TYPE_LABELS = {0: "low", 1: "high"}
_EFFORT_LABELS = {i: f"class{i}" for i in range(40)}


def _compute_income_normalization(sigx):
    """Compute the income normalization denominator from shock variance."""
    sdztemp = ((sigx**2.0) / (1.0 - shock_persistence**2.0)) ** 0.5
    return jnp.exp(
        ((jnp.log(productivity_type_multiplier[1]) ** 2.0) ** 2.0) / 2.0
    ) * jnp.exp(((sdztemp**2.0) ** 2.0) / 2.0)


def _compute_pension_base(income_process, income_normalization):
    """Compute base income at retirement (period 19, good health) by education."""
    y1 = income_process["y1"]
    yt_s = income_process["yt_s"]
    yt_sq = income_process["yt_sq"]
    wagep = income_process["wagep"]
    period = 19.0
    health = 1.0  # good health
    pension_base = jnp.zeros(2)
    for edu_idx, edu_key in enumerate(["low", "high"]):
        yt = (
            y1[edu_key]
            * jnp.exp(yt_s[edu_key] * period + yt_sq[edu_key] * period**2.0)
            * (1.0 - wagep[edu_key] * (1.0 - health))
        )
        pension_base = pension_base.at[edu_idx].set(yt / income_normalization)
    return pension_base


def create_inputs(seed, n_simulation_subjects, params):
    """Build model params and initial conditions from structured parameters."""
    cost_envelope = create_adjustment_cost_envelope(params["adjustment_cost"])
    xvalues = prod_shock_grid.get_gridpoints()
    xtrans = prod_shock_grid.get_transition_probs()
    ec_grid = create_effort_cost_grid(params["effort_cost"])
    wd_grid = create_work_disutility_grid(
        params["work_disutility"], params["education_disutility_adjustment"]
    )

    income_process = params["income_process"]
    sigx = income_process["sigx"]
    income_norm = _compute_income_normalization(sigx)
    pension_base = _compute_pension_base(income_process, income_norm)

    model_params = {
        "work_disutility": {"work_disutility_grid": wd_grid},
        "effort_cost": {
            "effort_elasticity": params["effort_elasticity"],
            "effort_cost_grid": ec_grid,
        },
        "consumption_utility": {
            "utility_constant": params["utility_constant"],
            "health_consumption_penalty": params["health_consumption_penalty"],
        },
        "base_income": {
            "y1": jnp.array(
                [income_process["y1"]["low"], income_process["y1"]["high"]]
            ),
            "yt_s": jnp.array(
                [income_process["yt_s"]["low"], income_process["yt_s"]["high"]]
            ),
            "yt_sq": jnp.array(
                [income_process["yt_sq"]["low"], income_process["yt_sq"]["high"]]
            ),
            "wagep": jnp.array(
                [income_process["wagep"]["low"], income_process["wagep"]["high"]]
            ),
            "income_normalization": income_norm,
        },
        "pension": {
            "pension_base": pension_base,
            "pension_replacement_rate": params["pension_replacement_rate"],
        },
        "adjustment_cost_penalty": {"adjustment_cost_envelope": cost_envelope},
        "scaled_productivity_shock": {"productivity_shock_scale": jnp.sqrt(sigx)},
    }

    n = n_simulation_subjects
    key = random.key(seed)
    types = random.choice(key, jnp.arange(16), (n,), p=_initial_dists)
    new_keys = random.split(key=key, num=3)
    health_draw = random.uniform(new_keys[0], (n,))
    health_thresholds = _init_distr[:, 1][types]
    initial_health = jnp.where(health_draw > health_thresholds, 0, 1)
    initial_health_type = 1 - _ht[types]
    initial_education = _ed[types]
    initial_productivity = _prod[types]
    initial_discount = _discount[types]
    initial_effort = jnp.searchsorted(effort_grid, _init_distr[:, 2][types])
    prod_dist = jax.lax.fori_loop(
        0,
        1000000,
        lambda _i, a: a @ xtrans.T,
        jnp.full(5, 1 / 5),
    )
    initial_adjustment_cost = np.asarray(random.uniform(new_keys[1], (n,)))
    initial_productivity_shock = np.asarray(
        xvalues[random.choice(new_keys[2], jnp.arange(5), (n,), p=prod_dist)]
    )

    initial_conditions_df = pd.DataFrame(
        {
            "regime": "alive",
            "age": ages.values[0],
            "wealth": np.zeros(n),
            "health": pd.Categorical(
                [_HEALTH_LABELS[int(v)] for v in initial_health],
                categories=["bad", "good"],
            ),
            "lagged_effort": pd.Categorical(
                [_EFFORT_LABELS[int(v)] for v in initial_effort],
                categories=[f"class{i}" for i in range(40)],
            ),
            "education": pd.Categorical(
                [_EDUCATION_LABELS[int(v)] for v in initial_education],
                categories=["low", "high"],
            ),
            "productivity": pd.Categorical(
                [_PRODUCTIVITY_LABELS[int(v)] for v in initial_productivity],
                categories=["low", "high"],
            ),
            "health_type": pd.Categorical(
                [_HEALTH_TYPE_LABELS[int(v)] for v in initial_health_type],
                categories=["low", "high"],
            ),
            "productivity_shock": initial_productivity_shock,
            "adjustment_cost": initial_adjustment_cost,
        }
    )

    return model_params, initial_conditions_df, np.asarray(initial_discount)
