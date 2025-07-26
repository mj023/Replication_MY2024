"""Example specification for a consumption-savings model with health and exercise."""

from dataclasses import dataclass, make_dataclass
import jax.numpy as jnp
import numpy as np

from lcm import DiscreteGrid, Model, LinspaceGrid
import lcm
import nvtx

# --------------------------------------------------------------------------------------
# Fixed Parameters
# --------------------------------------------------------------------------------------
avrgearn = 57706.57
theta_val = jnp.array([jnp.exp(-0.2898),jnp.exp(0.2898)])
n = 38
retirement_age = 19
taul = 0.128     
lamda = 1.0 - 0.321
rho = 0.975
r = 1.04**2.0
tt0 = 0.115
winit = jnp.array([43978,48201])
avrgearn = avrgearn/winit[1]
mincon0 = 0.10
mincon = mincon0 * avrgearn

def calc_savingsgrid(x):
    x = ((jnp.log(10.0**2)-jnp.log(10.0**0))/49)*x
    x = jnp.exp(x)
    xgrid = x - 10.0**(0.0)
    xgrid = xgrid/(10.0**2 - 10.0**0.0)
    xgrid = xgrid*(30-0) + 0 
    return xgrid
# --------------------------------------------------------------------------------------
# Categorical variables
# --------------------------------------------------------------------------------------
@dataclass
class WorkingStatus:
    retired: int = 0
    part: int = 1
    full: int = 2
@dataclass
class EducationStatus:
    low: int = 0
    high: int = 1

AdjustmentCost = make_dataclass('AdjustmentCost', [("class" + str(i), int, int(i)) for i in range(5)])
Effort = make_dataclass('HealthEffort', [("class" + str(i), int, int(i)) for i in range(40)])
@dataclass
class DiscountFactor:
    low: int = 0
    high: int = 1

@dataclass
class Health:
    bad: int = 0
    good: int = 1
@dataclass
class Alive:
    dead: int = 0
    alive: int = 1

@dataclass
class ProductivityType:
    low: int = 0
    high: int = 1
@dataclass
class HealthType:
    low: int = 0
    high: int = 1
@dataclass
class ProductivityShock:
    val0: int = 0
    val1: int = 1
    val2: int = 2
    val3: int = 3
    val4: int = 4
    # --------------------------------------------------------------------------------------
    # Grid Creation
    # --------------------------------------------------------------------------------------
eff_grid = jnp.linspace(0,1,40)
# ======================================================================================
# Model functions
# ======================================================================================
# --------------------------------------------------------------------------------------
# Utility function
# --------------------------------------------------------------------------------------
def utility(_period,health_type,education,adj_cost, fcost,disutil,cons_util,discount_factor, beta_mean, beta_std):
    beta = beta_mean + jnp.where(discount_factor, beta_std, -beta_std)
    f = cons_util - disutil - fcost- adj_cost
    return f * (beta**_period)
def disutil(working, health,education, _period, phigrid):
    return phigrid[_period,education,health] * ((working/2)**(2))/2
def adj_cost(_period,adjustment_cost, effort, effort_t_1, chimaxgrid):
    cost = jnp.where(jnp.logical_not(effort == effort_t_1), adjustment_cost*(chimaxgrid[_period]/4), 0)
    return cost
def cnow(net_income, wealth, saving):
    wealth = calc_savingsgrid(wealth)
    saving = calc_savingsgrid(saving)
    cnow = jnp.maximum(net_income + (wealth)*r - (saving), mincon)
    return cnow
def cons_util( health,cnow, kappa, sigma, bb):
    mucon = jnp.where(health, 1, kappa)
    return mucon*(((cnow)**(1.0-sigma))/(1.0-sigma)) + mucon*bb
def fcost(_period,education,health,effort, psi, xigrid):
    return xigrid[_period,education,health] * (eff_grid[effort]**(1+(1/psi)))/(1+(1/psi))

# --------------------------------------------------------------------------------------
# Income Calculation
# --------------------------------------------------------------------------------------
def net_income(benefits, taxed_income, pension):

    return taxed_income + pension + benefits
def income(working, _period, health, education, productivity , productivity_shock, xvalues, income_grid):
    return income_grid[ _period, health, education]*(working/2)*theta_val[productivity]*jnp.exp(xvalues[productivity_shock])
def taxed_income(income):
    return lamda*(income**(1.0-taul))*(avrgearn**taul)
def benefits(_period,health, working):
    eligible = jnp.logical_and(health == 0, working == 0)
    return jnp.where(jnp.logical_and(eligible,_period <= retirement_age), tt0*avrgearn,0)
def pension(_period,education,productivity, income_grid, penre):
    return jnp.where(_period > retirement_age,income_grid[19,1,education]*theta_val[productivity]*penre, 0)



# --------------------------------------------------------------------------------------
# State transitions
# --------------------------------------------------------------------------------------
def next_wealth(saving):
    return saving
def next_discount_factor(discount_factor):
    return discount_factor
@lcm.mark.stochastic
def next_alive(alive, _period, education, health):
    pass
@lcm.mark.stochastic
def next_health(_period, health, effort,effort_t_1, education, health_type):
    pass
def next_productivity(productivity):
    return productivity
def next_health_type(health_type):
    return health_type
def next_effort_t_1(effort):
    return effort
def next_education(education):
    return education
@lcm.mark.stochastic
def next_adjustment_cost(adjustment_cost):
    pass
@lcm.mark.stochastic
def next_productivity_shock(productivity_shock):
    pass
# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def retirement_constraint(_period, working):
    return jnp.logical_not(jnp.logical_and(_period > retirement_age, working > 0))
def savings_constraint(net_income, wealth, saving):
    wealth = calc_savingsgrid(wealth)
    saving = calc_savingsgrid(saving)
    return  net_income + (wealth)*r >= (saving)

# ======================================================================================
# Model specification and parameters
# ======================================================================================


MODEL_CONFIG = Model(
    n_periods=38,
    functions={
        "utility": utility,
        "disutil" : disutil,
        "fcost": fcost,
        "cons_util": cons_util,
        "cnow": cnow,
        "next_wealth": next_wealth,
        "next_health": next_health,
        "next_productivity_shock" : next_productivity_shock,
        "next_discount_factor": next_discount_factor,
        "next_adjustment_cost": next_adjustment_cost,
        "next_effort_t_1": next_effort_t_1,
        "next_health_type": next_health_type,
        "next_education": next_education,
        "next_productivity": next_productivity,
        "income": income,
        "benefits": benefits,
        "adj_cost": adj_cost,
        "net_income": net_income,
        "taxed_income" : taxed_income,
        "pension": pension,
        "retirement_constraint": retirement_constraint,
        "savings_constraint": savings_constraint,
    },
    actions={
        "working": DiscreteGrid(WorkingStatus),
        "saving": LinspaceGrid(start=0,stop=49,n_points=50),
        "effort": DiscreteGrid(Effort),
    },
    states={
        "wealth":  LinspaceGrid(start=0,stop=49,n_points=50),
        "health": DiscreteGrid(Health),
        "productivity_shock": DiscreteGrid(ProductivityShock),
        "effort_t_1": DiscreteGrid(Effort),
        "adjustment_cost": DiscreteGrid(AdjustmentCost),
        "education": DiscreteGrid(EducationStatus),
        "discount_factor": DiscreteGrid(DiscountFactor),
        "productivity": DiscreteGrid(ProductivityType),
        "health_type": DiscreteGrid(HealthType),

    },
)