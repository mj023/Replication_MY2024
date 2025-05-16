"""Example specification for a consumption-savings model with health and exercise."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from lcm import DiscreteGrid, LinspaceGrid, Model, LogspaceGrid
import lcm

# --------------------------------------------------------------------------------------
# Fixed Parameters
# --------------------------------------------------------------------------------------
avrgearn = 57706.57
theta_val = (jnp.exp(-0.2898),jnp.exp(0.2898))
n = 38
retirement_age = 21
taul = 0.128     
lamda = 1.0 - 0.321
rho = 0.975
mincon0 = 0.10
r = 1.04**2.0
tt0 = 0.115
winit =(43978,48201)
avrgearn = avrgearn/winit[1]
mincon0 = 0.10
mincon = mincon0 * avrgearn

# --------------------------------------------------------------------------------------
# Health Techonology
# --------------------------------------------------------------------------------------
const_healthtr = -0.906
age_const = (0.0,-0.289,-0.644,-0.881,-1.138,-1.586,-1.586,-1.586)
eff_param = (0.693,0.734)            
eff_sq = 0
healthy_dummy = 2.311
htype_dummy = 0.632
college_dummy = 0.238
# --------------------------------------------------------------------------------------
# Categorical variables
# --------------------------------------------------------------------------------------
@dataclass
class WorkingStatus:
    retired: float = 0.0
    part: float = 1.0
    full: float = 2.0

@dataclass
class Alive:
    dead: float = 0.0
    alive: float = 1.0
@dataclass
class EducationStatus:
    low: float = 0.0
    high: float = 1.0

@dataclass
class DiscountFactor:
    low: float = 0.0
    high: float = 1.0

@dataclass
class Health:
    bad: float = 0.0
    good: float = 1.0

@dataclass
class ProductivityType:
    low: float = 0.0
    high: float = 1.0

@dataclass
class ProductivityShock:
    low: float = 0.0
    high: float = 1.0

# --------------------------------------------------------------------------------------
# Grid Creation
# --------------------------------------------------------------------------------------
phi_interp_values = jnp.array([1,8,13,20])
def create_phigrid(nu,nu_e):
    phigrid = jnp.zeros((retirement_age-1, 2,2))
    for i in range(2):
        for j in range(2):
            temp_grid = jnp.arange(1,retirement_age)
            temp_grid = jnp.interp(temp_grid,phi_interp_values, nu[j])
            temp_grid = jnp.where(i == 0, temp_grid*nu_e, temp_grid)
            phigrid = phigrid.at[...,i,j].set(temp_grid)
    return phigrid

xi_interp_values = jnp.array([1,12,20,31])
def create_xigrid(xi):
    xigrid = jnp.zeros((n, 2,2))
    for i in range(2):
        for j in range(2):
            temp_grid = jnp.arange(1,31)
            temp_grid = jnp.interp(temp_grid,xi_interp_values, xi[i][j])
            xigrid = xigrid.at[0:31,i,j].set(temp_grid)
            xigrid = xigrid.at[31:retirement_age,i,j].set(xi[i][j][3])
    return xigrid
def create_chimaxgrid(chi_1,chi_2, chi_3):
    t = jnp.arange(1,39)
    chimax = jnp.max(chi_1*jnp.exp(chi_2*(t)-1.0) + chi_3*((t)-1.0)**2.0, axis = 1, initial=0)
    return chimax
chimaxgrid = jnp
surv_HS = jnp.array(np.loadtxt("surv_HS.txt"))
surv_CL = jnp.array(np.loadtxt("surv_CLS.txt"))

spgrid = jnp.zeros((38,2,2))
for i in range(2):
    spgrid = spgrid.at[:,0,i].set(surv_HS[i,:])
    spgrid = spgrid.at[:,1,i].set(surv_CL[i,:])

# period, health, effort, effort_t-1, education, health_type
eff_grid = np.linspace(0,1,40).tolist()
tr2yp_grid = jnp.zeros((38,2,40,40,2,2))
j = jnp.floor_divide(jnp.arange(38), 5)
for health in range(2):
    for eff in eff_grid:
        for eff_1 in eff_grid:
            for edu in range(2):
                for ht in range(2):
                    y = const_healthtr + age_const[j] + edu*college_dummy + health*healthy_dummy + ht*htype_dummy + eff*eff_param[1] + eff_1*eff_param[2] + eff**2 * eff_sq
                    tr2yp = jnp.exp(y) / (1.0 + jnp.exp(y))
                    tr2yp_grid = tr2yp_grid.at[:,health,eff,eff_1,edu,ht].set(y)



        
# ======================================================================================
# Model functions
# ======================================================================================

# --------------------------------------------------------------------------------------
# Utility function
# --------------------------------------------------------------------------------------
def utility(lagged_health, wealth, saving,working,health,education,adjustment_cost,  health_effort, effort_t_1, fcost,net_income, xigrid, sigma, bb, mincon, kappa, disutil, _period, chimaxgrid):
    adj_cost = jnp.where(jnp.logical_not(health_effort == effort_t_1), adjustment_cost*(chimaxgrid[_period]/200), 0)
    cnow = jnp.max( net_income + wealth*r - saving, mincon)
    mucon = jnp.where(health, 1, kappa)
    f = mucon*((cnow)**(1.0-sigma))/(1.0-sigma) + mucon*bb - disutil - xigrid[_period,education,health]*fcost- adj_cost   
    return f * spgrid[_period,lagged_health,education]
def disutil(working, health,education, _period, phigrid):
    return phigrid[_period,education,health] * ((working/2)**(2))/2
def fcost(health_effort, psi):
    return (health_effort**(1.0+(1.0/psi)))/(1.0+(1.0/psi))

# --------------------------------------------------------------------------------------
# Income Calculation
# --------------------------------------------------------------------------------------
def net_income(working, taxed_income, _period, health, pension):
    return taxed_income + jnp.where(_period >= retirement_age-1, pension,jnp.where(jnp.logical_and(health == 0, working == 0), tt0*avrgearn,0))
def income(working, _period, education, health, productivity, y1_HS,y1_CL,ytHS_s,ytHS_sq,wagep_HS,wagep_CL,ytCL_s,ytCL_sq, sigx):
    sdztemp = ((sigx**2.0)/(1.0-rho**2.0))**0.5
    yt = jnp.where(education==1, (y1_CL*jnp.exp( ytCL_s*(_period-1.0) + ytCL_sq*(_period-1.0)**2.0 ))*(1.0-wagep_CL*(1-health)),(y1_HS*jnp.exp( ytHS_s*(_period-1.0) + ytHS_sq*(_period-1.0)**2.0 ))*(1.0-wagep_HS*(1-health)))
    return (working/2)*(yt/(jnp.exp( ((jnp.log(theta_val[1])**2.0)**2.0)/2.0 )*jnp.exp( ((sdztemp**2.0)**2.0)/2.0)))*theta_val[productivity]       
def taxed_income(income, productivity_shock):
    return income*productivity_shock - lamda*(income**(1.0-taul))*(avrgearn**taul)
def pension(education,productivity, y1_HS,y1_CL,ytHS_s,ytHS_sq,wagep_HS,wagep_CL,ytCL_s,ytCL_sq, sigx, penre):
    return income(2,20,education,0,productivity,y1_HS,y1_CL,ytHS_s,ytHS_sq,wagep_HS,wagep_CL,ytCL_s,ytCL_sq, sigx)*penre



# --------------------------------------------------------------------------------------
# State transitions
# --------------------------------------------------------------------------------------
def next_wealth(saving):
    return saving


def next_lagged_health(health):
    return health

@lcm.mark.stochastic
def next_health(_period, health, effort, effort_t_1, education, health_type):
    pass

def next_productivity(productivity):
    return productivity

def next_effort_t_1(health_effort):
    return health_effort

def next_education(education):
    return education

@lcm.mark.stochastic
def next_adjustment_cost():
    pass
@lcm.mark.stochastic
def next_productivity_shock(productivity_shock):
    pass
# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def consumption_constraint(consumption, wealth, labor_income):
    return consumption <= wealth + labor_income


# ======================================================================================
# Model specification and parameters
# ======================================================================================


MODEL_CONFIG = Model(
    n_periods=38,
    functions={
        "utility": utility,
        "next_wealth": next_wealth,
        "next_health": next_health,
        "consumption_constraint": consumption_constraint,
        "income": income,
    },
    actions={
        "working": DiscreteGrid(WorkingStatus),
        "saving": LogspaceGrid(
            start=0,
            stop=30.0,
            n_points=50,
        ),
        "health_effort": LinspaceGrid(
            start=0,
            stop=1,
            n_points=40,
        ),
    },
    states={
        "wealth": LogspaceGrid(
            start=0,
            stop=30.0,
            n_points=50,
        ),
        "health": DiscreteGrid(Health),
        "lagged_health": DiscreteGrid(Health),
        "productivity": DiscreteGrid(ProductivityType),
        "productivity_shock": DiscreteGrid(ProductivityShock),
        "education": DiscreteGrid(EducationStatus),
        "discount_factor": DiscreteGrid(DiscountFactor),
        "effort_t_1": LinspaceGrid(
            start=0,
            stop=1,
            n_points=40,
        ),
        "adjustment_cost": DiscreteGrid(
            start=0,
            stop=1,
            n_points=200,
        ),

    },
)

PARAMS = {
    "beta": 0.95,
    "utility": {"disutility_of_work": 0.05},
    "next_wealth": {"interest_rate": 0.05},
}
