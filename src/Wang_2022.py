"""Example specification for a consumption-savings model with health and exercise."""

from dataclasses import dataclass

import jax.numpy as jnp

from lcm import DiscreteGrid, LinspaceGrid, Model

# ======================================================================================
# Model functions
# ======================================================================================


# --------------------------------------------------------------------------------------
# Categorical variables
# --------------------------------------------------------------------------------------
@dataclass
class Working:
    unemp: int = 0
    half: int = 1
    full: int = 2

@dataclass
class Married:
    unmarried: int = 0
    married: int = 1

@dataclass
class Fecund:
    not_fecund: int = 0
    fecund: int = 1

@dataclass
class Fertility:
    no_child: int = 0
    child: int = 1

@dataclass
class JobOffer:
    no_offer: int = 0
    offer: int = 1

@dataclass
class IncomeShock:
    zero: int = 0
    one: int = 1
    two: int = 2
    three: int = 3
    four: int = 4


# --------------------------------------------------------------------------------------
# Utility function
# --------------------------------------------------------------------------------------
def utility(consumption, married, ):
    return 


# --------------------------------------------------------------------------------------
# Auxiliary variables
# --------------------------------------------------------------------------------------
def labor_income(wage, working):
    return wage * working


def wage(age):
    return 1 + 0.1 * age


def age(_period):
    return _period + 22


# --------------------------------------------------------------------------------------
# State transitions
# --------------------------------------------------------------------------------------
def next_wealth(wealth, consumption, labor_income, interest_rate):
    return (1 + interest_rate) * (wealth + labor_income - consumption)


def next_health(health, exercise, working):
    return health * (1 + exercise - working / 2)


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def consumption_constraint(consumption, wealth, labor_income):
    return consumption <= wealth + labor_income


# ======================================================================================
# Model specification and parameters
# ======================================================================================
RETIREMENT_AGE = 65


MODEL_CONFIG = Model(
    n_periods=RETIREMENT_AGE - 22,
    functions={
        "utility": utility,
        "next_wealth": next_wealth,
        "next_health": next_health,
        "consumption_constraint": consumption_constraint,
        "labor_income": labor_income,
        "wage": wage,
        "age": age,
    },
    actions={
        "working": DiscreteGrid(Working),
        "fertility": DiscreteGrid(Fertility)
    },
    states={
        "fecundity_shock": DiscreteGrid(Fecund),
        "offer_shock": DiscreteGrid(JobOffer),
        "marrital_shock": DiscreteGrid(FecShock),
        "fertility_pref_shock": DiscreteGrid(FecShock),
        "labor_pref": DiscreteGrid(FecShock),
        "income_shock": DiscreteGrid(IncomeShock),
        "income_shock_h": DiscreteGrid(IncomeShock),
        "married" :DiscreteGrid(Married),
    },
)

PARAMS = {
    "beta": 0.95,
    "utility": {"disutility_of_work": 0.05},
    "next_wealth": {"interest_rate": 0.05},
}
