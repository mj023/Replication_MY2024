"""Example specification for a consumption-savings model with health and exercise."""

from dataclasses import dataclass, make_dataclass
from utils import rouwenhorst
import jax.numpy as jnp
import numpy as np

from lcm import DiscreteGrid, Model, LinspaceGrid
import lcm
import nvtx

from lcm.dispatchers import _base_productmap
# --------------------------------------------------------------------------------------
# Fixed Parameters
# --------------------------------------------------------------------------------------
avrgearn = 57706.57
theta_val = jnp.array([jnp.exp(-0.2898),jnp.exp(0.2898)])
n = 38
retirement_age = 21
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
    x = ((jnp.log(10.0**2)-jnp.log(10.0**0))/50)*x
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
Savings = make_dataclass('Savings', [("class" + str(i), int, int(i)) for i in range(50)])
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

    spgrid = jnp.zeros((38,2,2))
    for i in range(2):
        spgrid = spgrid.at[:,0,i].set(surv_HS[:,i])
        spgrid = spgrid.at[:,1,i].set(surv_CL[:,i])
    eff_grid = jnp.linspace(0,1,40)
# ======================================================================================
# Model functions
# ======================================================================================

# --------------------------------------------------------------------------------------
# Utility function
# --------------------------------------------------------------------------------------
def utility(_period, lagged_health, wealth, saving,working,health,education,adjustment_cost,  effort, effort_t_1, health_type, fcost,disutil,net_income,cons_util, xigrid, sigma, bb, kappa, chimaxgrid, discount_factor, beta_mean, beta_std):
    adj_cost = jnp.where(jnp.logical_not(effort == effort_t_1), adjustment_cost*(chimaxgrid[_period]/10), 0)
    beta = beta_mean + jnp.where(discount_factor, -beta_std, beta_std)
    f = cons_util - disutil - xigrid[_period,education,health]*fcost- adj_cost   
    return f * spgrid[_period,education,lagged_health]*(beta**_period)
def disutil(working, health,education, _period, phigrid):
    return phigrid[_period,education,health] * ((working/2)**(2))/2
def cons_util(net_income, wealth, saving, health, kappa, sigma, bb):
    wealth = calc_savingsgrid(wealth)
    saving = calc_savingsgrid(saving)
    cnow = jnp.maximum(net_income + (wealth)*r - (saving), mincon)
    mucon = jnp.where(health, 1, kappa)
    return mucon*((cnow)**(1.0-sigma))/(1.0-sigma) + mucon*bb
def fcost(effort, psi):
    return (eff_grid[effort]**(1.0+(1.0/psi)))/(1.0+(1.0/psi))

# --------------------------------------------------------------------------------------
# Income Calculation
# --------------------------------------------------------------------------------------
def net_income(working, taxed_income, _period, health, pension):
    return taxed_income + jnp.where(_period > retirement_age, pension,jnp.where(jnp.logical_and(health == 0, working == 0), tt0*avrgearn,0))
def income( _period, health, education, income_grid):
    return income_grid[ _period, health, education]
def taxed_income(income, productivity_shock, xvalues):
    income = income*jnp.exp(xvalues[productivity_shock])
    return income - lamda*(income**(1.0-taul))*(avrgearn**taul)
def pension(education,productivity, income_grid, penre):
    return income_grid[2,20,1,education,productivity]*penre



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
    return jnp.logical_not(jnp.logical_and(_period > 21, working > 0))
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
# Parameters for disutility of work    
# healthy
nuh_1 = 2.63390750888379
nuh_2 = 1.66602983591164
nuh_3 = 1.27839561280412
nuh_4 = 1.71439043350863

# unhealthy
nuu_1 = 2.41177758126754
nuu_2 = 1.8133670880598
nuu_3 = 1.39103558901915
nuu_4 = 2.41466980231321
    
nuad = 0.807247922589072         
nuh = jnp.array([nuh_1, nuh_2, nuh_3, nuh_4])
nuu = jnp.array([nuu_1, nuu_2, nuu_3, nuu_4])
nu = [nuu, nuh]
# direct utility cost of effort
# HS-Healthy
xiHSh_1 = 0.146075197675677
xiHSh_2 = 0.55992411008533
xiHSh_3 = 1.04795036000287
xiHSh_4 = 1.60294886005945


# HS-Unhealthy    
xiHSu_1 = 0.628031290227532
xiHSu_2 = 1.36593242946612
xiHSu_3 = 1.64963812690034
xiHSu_4 = 0.734873142494319
                                                                                            

# CL-Healthy
xiCLh_1 = 0.091312997289004
xiCLh_2 = 0.302477689083851
xiCLh_3 = 0.739843441095022
xiCLh_4 = 1.36582077051777


# CL-Unhealthy    
xiCLu_1 = 0.46921037985024
xiCLu_2 = 0.996665589702672
xiCLu_3 = 1.65388250352532
xiCLu_4 = 1.08866246911941

xi_HSh = jnp.array([xiHSh_1,
xiHSh_2,
xiHSh_3,
xiHSh_4])
xi_HSu = jnp.array([xiHSu_1,
xiHSu_2,
xiHSu_3,
xiHSu_4])
xi_CLu = jnp.array([xiCLu_1,
xiCLu_2,
xiCLu_3,
xiCLu_4])
xi_CLh = jnp.array([xiCLh_1,
xiCLh_2,
xiCLh_3,
xiCLh_4])

xi = [[xi_HSu, xi_HSh], [xi_CLu, xi_CLh]]



# effort habit adjustment cost max
chi_1 = 0.000120437772838191          
chi_2 = 0.14468204213946              

sigx = 0.0289408524185787               

penre = 0.358766004066242           

betamean0 = 0.942749393405227        
betadev0 = 0.0283688760224992        

bb = 13.1079320277342          

conp = 0.871503495423925       
psi = 1.11497911620865         

# Wage profile for HS + healthy
ytHS_s = 0.0615804210614531           
ytHS_sq = -0.00250769285750586  

# Wage profile for CL + healthy
ytCL_s = 0.0874283672769353  
ytCL_sq = -0.00293713499239749 

# wage penalty: depends on education and age
wagep_HS = 0.17769766414897      
wagep_CL = 0.144836058314823         
    

# Initial yt(1) for HS relative to CL
y1_HS = 0.899399488241831
y1_CL = 1.1654726432446

sigma = 2.0
           
haddft = 0.0
        
sdxi= 0.0
chi_3 = 0.0 
