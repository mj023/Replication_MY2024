"""Example specification for a consumption-savings model with health and exercise."""

from dataclasses import dataclass

import jax.numpy as jnp

from lcm import DiscreteGrid, Model

# ======================================================================================
# Model functions
# ======================================================================================


# --------------------------------------------------------------------------------------
# Categorical variables
# --------------------------------------------------------------------------------------
@dataclass
class Gender:
    male: int = 0
    female: int = 1
@dataclass
class Working:
    unemp: float = 0
    half: float = 1
    full: float = 2
@dataclass
class Health:
    good: int = 0
    fair: int = 1
    poor: int = 2
@dataclass
class Education:
    hsd: int = 0
    hsg: int = 1
    sc: int = 2
    cg: int = 3
    pc: int = 4
@dataclass
class Experience:
    hsd: int = 0
    hsg: int = 1
    sc: int = 2
    cg: int = 3
    pc: int = 4
@dataclass
class Children:
    none: int = 0
    one: int = 1
    two: int = 2
    three: int = 3
@dataclass
class TasteLeisure:
    zero: int = 0
    one: int = 1
    two: int = 2

@dataclass
class Skill:
    zero: int = 0
    one: int = 1
    two: int = 2

@dataclass
class Pregnancy:
    np: int = 0
    p: int = 1

@dataclass
class ParentEdu:
    none: int = 0
    college: int = 1


# --------------------------------------------------------------------------------------
# Utility function
# --------------------------------------------------------------------------------------

def utility_married(working_w, working_h, consumption, alpha, beta, gamma, theta, pi, pregnancy, mu, a_f, a_m,a_g, A_m, rho, net_income, num_children):
    return ((1/alpha)*(0.707*consumption)**alpha)+ value_of_leisure(working_w, beta, gamma, mu)+ theta + pregnancy*pi + A_m * Q(working_w,working_h,net_income, num_children, a_f,a_m, rho) 
def utility_single(working_w, working_h, consumption, alpha, beta, gamma, theta, pi, pregnancy, mu, a_f, a_m,a_g, A_m, rho, net_income, num_children):
    ((1/alpha)*(consumption)**alpha)+ value_of_leisure(working_w, beta, gamma, mu)+ theta + pregnancy*pi + A_m * Q(working_w,working_h,net_income, num_children, a_f,a_m, rho) 
def value_of_leisure(working_w, beta, gamma, mu):
    return (beta/gamma)*((1-working_w)*2000)**gamma + mu * (1-working_w)*2000
def Q(working_w, working_h, net_income, num_children, a_f, a_m,a_g, rho):
    leisure_w = (1-working_w)*2000
    leisure_h = (1-working_h)*2000
    return ((a_f*leisure_w**rho)+(a_m*leisure_h**rho)+(a_g*(net_income*(jnp.sqrt(2/(2+num_children))))**rho) + (1-a_f-a_m-a_g)*num_children**rho)**(1/rho)


# --------------------------------------------------------------------------------------
# Auxiliary variables
# --------------------------------------------------------------------------------------
def age(_period):
    return _period + 17
def net_income(working_w, working_h, wage_w, wage_h, benefits_w, benefits_h, num_children):
    gross_labor_income = (working_w*2000*wage_w)+(working_h*2000*wage_h)
    return (gross_labor_income+jnp.where(working_w == 0, benefits_w, 0)+jnp.where(working_h == 0, benefits_h, 0)+jnp.where(working_w == 0, benefits_w, 0)) - net(gross_labor_income, num_children)
def net(income, num_children):
    return income
def age(_period):
    return _period + 18


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
def budget_constraint(consumption, net_income, num_children):
    return consumption == net_income*(jnp.sqrt(2/(2+num_children)))
def working_constraint(working_h, working_w, offer_h, offer_w):
    return jnp.logical_and(working_w <= offer_w, working_h <= offer_h)
def male_constraint(pregnancy, gender):
    jnp.logical_and(pregnancy == 0, gender == 0)


# ======================================================================================
# Model specification and parameters
# ======================================================================================
RETIREMENT_AGE = 65


MODEL_CONFIG = Model(
    n_periods=RETIREMENT_AGE - 18,
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
        "working_w": DiscreteGrid(Working),
        "working_h": DiscreteGrid(Working),
        "pregnant": DiscreteGrid(Pregnancy),
    },
    states={
        "gender": DiscreteGrid(Gender),
        "education": DiscreteGrid(Education),
        "experience": DiscreteGrid(Experience),
        "taste_leisure": DiscreteGrid(TasteLeisure),
        "skill": DiscreteGrid(Skill),
        "parent_edu": DiscreteGrid(ParentEdu),
        "lagged_working": DiscreteGrid(Working),
        "lagged_pregnancy": DiscreteGrid(Pregnancy),
        "health": DiscreteGrid(Health)
    },
)

PARAMS = {
    "beta": 0.95,
    "utility": {"disutility_of_work": 0.05},
    "next_wealth": {"interest_rate": 0.05},
}
