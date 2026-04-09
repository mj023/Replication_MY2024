"""Replication of Mahler & Yum (2024): Lifestyle Behaviors and Wealth-Health Gaps."""

import dataclasses
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
    IrregSpacedGrid,
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


_WEALTH_GRID_POINTS = tuple(
    float(30.0 * (jnp.exp(jnp.log(100.0) / 49 * i) - 1.0) / 99.0) for i in range(50)
)


@categorical(ordered=True)
class LaborSupply:
    retired: int
    part_time: int
    full_time: int


@categorical(ordered=True)
class Education:
    low: int
    high: int


Effort = categorical(ordered=True)(
    make_dataclass("Effort", [(f"level_{i}", int) for i in range(40)])
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
                    prob = 0.0
                else:
                    prob = surv_data[period_idx, health_idx]
                records.append((age, edu_label, health_label, prob))
    df = pd.DataFrame(
        records, columns=["age", "education", "health", "survival_probability"]
    )
    return df.set_index(["age", "education", "health"])["survival_probability"]


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


def effort_value(effort: DiscreteAction, effort_grid: FloatND) -> FloatND:
    """Map effort class index to continuous [0, 1] value."""
    return effort_grid[effort]


def lagged_effort_value(lagged_effort: DiscreteState, effort_grid: FloatND) -> FloatND:
    """Map lagged effort class index to continuous [0, 1] value."""
    return effort_grid[lagged_effort]


def consumption(
    net_income: FloatND, wealth: ContinuousState, saving: ContinuousAction
) -> FloatND:
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
    effort_value: FloatND,
    effort_elasticity: float,
    effort_cost_grid: FloatND,
) -> FloatND:
    return (
        effort_cost_grid[period, education, health]
        * (effort_value ** (1 + (1 / effort_elasticity)))
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
    effort_value: FloatND,
    lagged_effort_value: FloatND,
    education: DiscreteState,
    health_type: DiscreteState,
    health_intercept: float,
    health_age_effects: FloatND,
    good_health_coefficient: float,
    health_type_coefficient: float,
    college_coefficient: float,
    health_effort_coefficient: float,
    lagged_health_effort_coefficient: float,
) -> FloatND:
    """Compute health transition probabilities via logit model."""
    y = (
        health_intercept
        + health_age_effects[period]
        + education * college_coefficient
        + health * good_health_coefficient
        + health_type * health_type_coefficient
        + effort_value * health_effort_coefficient
        + lagged_effort_value * lagged_health_effort_coefficient
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
    """Return regime transition probabilities indexed by RegimeId."""
    survival_prob = transition_probs[period, education, health]
    probs = jnp.zeros(2).at[RegimeId.alive].set(survival_prob)
    return probs.at[RegimeId.dead].set(1.0 - survival_prob)


def retirement_constraint(period: Period, labor_supply: DiscreteAction) -> BoolND:
    return jnp.logical_not(
        jnp.logical_and(period > retirement_period, labor_supply > 0)
    )


def savings_constraint(
    net_income: FloatND, wealth: ContinuousState, saving: ContinuousAction
) -> BoolND:
    return net_income + wealth * gross_interest_rate >= saving


def alive_is_active(age: int, final_age_alive: float) -> bool:
    return age <= final_age_alive


def dead_is_active(age: int, initial_age: float) -> bool:
    return age > initial_age


ALIVE_REGIME = Regime(
    transition=MarkovTransition(next_regime),
    active=partial(alive_is_active, final_age_alive=ages.values[-2]),
    states={
        "wealth": IrregSpacedGrid(points=_WEALTH_GRID_POINTS),
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
        "saving": IrregSpacedGrid(points=_WEALTH_GRID_POINTS),
        "effort": DiscreteGrid(Effort),
    },
    functions={
        "utility": utility,
        "effort_value": effort_value,
        "lagged_effort_value": lagged_effort_value,
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
            "next_regime": {"transition_probs": _load_survival_probs()},
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
    """Convert {"27": val, "41": val, ...} to period-indexed arrays."""
    start_age = int(ages.values[0])
    step = int(ages.values[1] - ages.values[0])
    knot_ages = np.array([int(k) for k in age_keyed_dict])
    knot_periods = (knot_ages - start_age) // step
    values = np.array(list(age_keyed_dict.values()))
    return knot_periods, values


def _interpolate_knots(age_keyed_dict, period_range, flat_after=None):
    """Cubic spline interpolation of age-keyed knots over a period range.

    Args:
        age_keyed_dict: Dict mapping age strings to values.
        period_range: Array of periods to interpolate over.
        flat_after: If set, extend the last knot value for periods beyond this.

    """
    knot_periods, knot_values = _age_keys_to_periods(age_keyed_dict)
    spline = scipy_interp1d(knot_periods, knot_values, kind="cubic")
    values = np.asarray(spline(period_range))
    if flat_after is not None:
        values[period_range >= flat_after] = knot_values[-1]
    return values


def create_work_disutility_grid(work_disutility, education_disutility_adjustment):
    """Interpolate work disutility knots to a labeled Series."""
    age_values = np.asarray(ages.values)
    period_range = np.arange(1, retirement_period + 2)
    records = []
    for health in ["bad", "good"]:
        values = _interpolate_knots(work_disutility[health], period_range)
        for period_idx, age in enumerate(age_values):
            for edu in ["low", "high"]:
                if period_idx <= retirement_period:
                    factor = (
                        np.exp(education_disutility_adjustment) if edu == "low" else 1.0
                    )
                    val = values[period_idx] * factor
                else:
                    val = 0.0
                records.append((age, edu, health, val))
    df = pd.DataFrame(records, columns=["age", "education", "health", "value"])
    return df.set_index(["age", "education", "health"])["value"]


def create_effort_cost_grid(effort_cost):
    """Interpolate effort cost knots to a labeled Series (age x education x health)."""
    age_values = np.asarray(ages.values)
    period_range = np.arange(1, 31)
    records = []
    for edu in ["low", "high"]:
        for health in ["bad", "good"]:
            values = _interpolate_knots(effort_cost[edu][health], period_range)
            last_knot = _age_keys_to_periods(effort_cost[edu][health])[1][-1]
            for period_idx, age in enumerate(age_values):
                if period_idx < len(period_range):
                    val = values[period_idx]
                else:
                    val = last_knot
                records.append((age, edu, health, val))
    df = pd.DataFrame(records, columns=["age", "education", "health", "value"])
    return df.set_index(["age", "education", "health"])["value"]


def create_adjustment_cost_envelope(adjustment_cost):
    """Build exponential adjustment cost envelope as a labeled Series (age)."""
    age_values = np.asarray(ages.values)
    t = np.arange(n_periods)
    values = np.maximum(adjustment_cost[0] * np.exp(adjustment_cost[1] * t), 0)
    return pd.Series(values, index=pd.Index(age_values, name="age"))


_DISCOUNT_LABELS = ["small", "large"]


def _field_name(categorical_class, code):
    """Get the field name for an integer code from a @categorical class."""
    return dataclasses.fields(categorical_class)[code].name


def _build_type_distribution():
    """Build initial type distribution as a DataFrame.

    Each row is one of 16 types (discount x productivity x health_type x education).

    """
    raw = np.loadtxt(_DATA_DIR / "init_distr_2b2t2h.txt")
    probabilities = jnp.diff(jnp.array(raw[:, 0]), prepend=0)

    records = []
    for idx in range(16):
        edu_code = idx // 8
        remainder = idx % 8
        discount_code = remainder // 4
        prod_code = (remainder % 4) // 2
        health_type_code = remainder % 2
        records.append(
            {
                "probability": float(probabilities[idx]),
                "health_threshold": float(raw[idx, 1]),
                "initial_effort": float(raw[idx, 2]),
                "discount_type": _DISCOUNT_LABELS[discount_code],
                "education": _field_name(Education, edu_code),
                "productivity": _field_name(ProductivityType, prod_code),
                "health_type": _field_name(HealthType, health_type_code),
            }
        )
    return pd.DataFrame(records)


_TYPE_DISTRIBUTION = _build_type_distribution()


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
    """Build model params, initial conditions, and discount factors.

    Returns:
        Tuple of (model_params, initial_conditions_df, discount_types,
        discount_factor_small, discount_factor_large).

    """
    discount_factor = params["discount_factor"]
    discount_factor_small = discount_factor["mean"] - discount_factor["std"]
    discount_factor_large = discount_factor["mean"] + discount_factor["std"]

    income_process = params["income_process"]
    income_norm = _compute_income_normalization(income_process["sigx"])

    model_params = {
        "work_disutility": {
            "work_disutility_grid": create_work_disutility_grid(
                params["work_disutility"], params["education_disutility_adjustment"]
            ),
        },
        "effort_cost": {
            "effort_elasticity": params["effort_elasticity"],
            "effort_cost_grid": create_effort_cost_grid(params["effort_cost"]),
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
            "pension_base": _compute_pension_base(income_process, income_norm),
            "pension_replacement_rate": params["pension_replacement_rate"],
        },
        "adjustment_cost_penalty": {
            "adjustment_cost_envelope": create_adjustment_cost_envelope(
                params["adjustment_cost"]
            ),
        },
        "scaled_productivity_shock": {
            "productivity_shock_scale": jnp.sqrt(income_process["sigx"]),
        },
    }

    td = _TYPE_DISTRIBUTION
    key = random.key(seed)
    type_indices = np.asarray(
        random.choice(
            key,
            jnp.arange(len(td)),
            (n_simulation_subjects,),
            p=jnp.array(td["probability"].values),
        )
    )

    keys = random.split(key=key, num=3)
    health_draw = random.uniform(keys[0], (n_simulation_subjects,))
    health_thresholds = td["health_threshold"].values[type_indices]
    initial_health_codes = np.where(health_draw > health_thresholds, 0, 1)

    initial_effort_values = td["initial_effort"].values[type_indices]
    initial_effort_codes = np.searchsorted(
        np.asarray(effort_grid), initial_effort_values
    )

    shock_gridpoints = prod_shock_grid.get_gridpoints()
    stationary_shock_dist = jax.lax.fori_loop(
        0,
        1000000,
        lambda _i, a: a @ prod_shock_grid.get_transition_probs().T,
        jnp.full(5, 1 / 5),
    )

    initial_conditions_df = pd.DataFrame(
        {
            "regime": "alive",
            "age": ages.values[0],
            "wealth": np.zeros(n_simulation_subjects),
            "health": pd.Categorical(
                [_field_name(Health, c) for c in initial_health_codes],
            ).astype(Health.to_categorical_dtype()),  # ty: ignore[unresolved-attribute],
            "lagged_effort": pd.Categorical(
                [_field_name(Effort, c) for c in initial_effort_codes],
            ).astype(Effort.to_categorical_dtype()),  # ty: ignore[unresolved-attribute],
            "education": pd.Categorical(
                td["education"].values[type_indices],
            ).astype(Education.to_categorical_dtype()),  # ty: ignore[unresolved-attribute],
            "productivity": pd.Categorical(
                td["productivity"].values[type_indices],
            ).astype(ProductivityType.to_categorical_dtype()),  # ty: ignore[unresolved-attribute],
            "health_type": pd.Categorical(
                td["health_type"].values[type_indices],
            ).astype(HealthType.to_categorical_dtype()),  # ty: ignore[unresolved-attribute],
            "productivity_shock": np.asarray(
                shock_gridpoints[
                    random.choice(
                        keys[2],
                        jnp.arange(5),
                        (n_simulation_subjects,),
                        p=stationary_shock_dist,
                    )
                ]
            ),
            "adjustment_cost": np.asarray(
                random.uniform(keys[1], (n_simulation_subjects,))
            ),
        }
    )
    discount_types = np.array(
        [_DISCOUNT_LABELS.index(d) for d in td["discount_type"].values[type_indices]]
    )

    return (
        model_params,
        initial_conditions_df,
        discount_types,
        discount_factor_small,
        discount_factor_large,
    )
