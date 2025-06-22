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
# Health Techonology
# --------------------------------------------------------------------------------------
const_healthtr = -0.906
age_const = jnp.asarray([0.0,-0.289,-0.644,-0.881,-1.138,-1.586,-1.586,-1.586])
eff_param = jnp.asarray([0.693,0.734])            
eff_sq = 0
healthy_dummy = 2.311
htype_dummy = 0.632
college_dummy = 0.238
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

AdjustmentCost = make_dataclass('AdjustmentCost', [("class" + str(i), int, int(i)) for i in range(10)])
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
with nvtx.annotate("grids", color = "green"):
    # --------------------------------------------------------------------------------------
    # Grid Creation
    # --------------------------------------------------------------------------------------
    surv_HS = jnp.array(np.loadtxt("surv_HS.txt"))
    surv_CL = jnp.array(np.loadtxt("surv_CL.txt"))
    spgrid = jnp.zeros((39,2,2))
    spgrid = spgrid.at[0,:,:].set(1)
    spgrid = spgrid.at[1:,0,0].set(surv_HS[:,1])
    spgrid = spgrid.at[1:,1,0].set(surv_CL[:,1])
    spgrid = spgrid.at[1:,0,1].set(surv_HS[:,0])
    spgrid = spgrid.at[1:,1,1].set(surv_CL[:,0])
    eff_grid = jnp.linspace(0,1,40)
# ======================================================================================
# Model functions
# ======================================================================================

# --------------------------------------------------------------------------------------
# Utility function
# --------------------------------------------------------------------------------------
def utility(_period, lagged_health,health_type,education,adj_cost, fcost,disutil,cons_util,discount_factor, beta_mean, beta_std):
    beta = beta_mean + jnp.where(discount_factor, beta_std, -beta_std)
    f = cons_util - disutil - fcost- adj_cost   
    return f * spgrid[_period,education,lagged_health]*(beta**(_period))
def disutil(working, health,education, _period, phigrid):
    return phigrid[_period,education,health] * ((working/2)**(2))/2
def adj_cost(_period,adjustment_cost, effort, effort_t_1, chimaxgrid):
    cost = jnp.where(jnp.logical_not(effort == effort_t_1), adjustment_cost*(chimaxgrid[_period]/9), 0)
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
    return ((eff_grid[effort]**(1.0+(1.0/psi)))/(1.0+(1.0/psi))) * xigrid[_period,education,health]

# --------------------------------------------------------------------------------------
# Income Calculation
# --------------------------------------------------------------------------------------
def net_income(working, taxed_income, _period, health, pension):
    return taxed_income + jnp.where(_period > retirement_age, pension,jnp.where(jnp.logical_and(health == 0, working == 0), tt0*avrgearn,0))
def income(working, _period, health, education, productivity , productivity_shock, xvalues, income_grid):
    return income_grid[ _period, health, education]*(working/2)*theta_val[productivity]*jnp.exp(xvalues[productivity_shock])
def taxed_income(income):
    return lamda*(income**(1.0-taul))*(avrgearn**taul)
def pension(education,productivity, income_grid, penre):
    return income_grid[19,1,education]*theta_val[productivity]*penre



# --------------------------------------------------------------------------------------
# State transitions
# --------------------------------------------------------------------------------------
def next_wealth(saving):
    return saving
def next_discount_factor(discount_factor):
    return discount_factor
def next_lagged_health(health):
    return health
@lcm.mark.stochastic
def next_health(health, _period, effort, effort_t_1, education, health_type):
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
        "next_lagged_health": next_lagged_health,
        "next_health_type": next_health_type,
        "next_education": next_education,
        "next_productivity": next_productivity,
        "income": income,
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
        "wealth": LinspaceGrid(start=0,stop=49,n_points=50),
        "health": DiscreteGrid(Health),
        "lagged_health": DiscreteGrid(Health),
        "productivity_shock": DiscreteGrid(ProductivityShock),
        "effort_t_1": DiscreteGrid(Effort),
        "adjustment_cost": DiscreteGrid(AdjustmentCost),
        "education": DiscreteGrid(EducationStatus),
        "discount_factor": DiscreteGrid(DiscountFactor),
        "productivity": DiscreteGrid(ProductivityType),
        "health_type": DiscreteGrid(HealthType),

    },
)