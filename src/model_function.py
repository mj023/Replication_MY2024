import jax
from jax import numpy as jnp
from lcm.dispatchers import _base_productmap
from lcm.entry_point import get_lcm_function
from lcm import LogspaceGrid
from utils import rouwenhorst,gini
from Mahler_Yum_2024 import MODEL_CONFIG, calc_savingsgrid
from jax import random
from interpax import interp1d
import numpy as np
import pandas as pd

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
sigma = 2
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
# Grid Creation Functions
# --------------------------------------------------------------------------------------

phi_interp_values = jnp.array([1,8,13,20])
def create_phigrid(nu,nu_e):
    phigrid = jnp.zeros((retirement_age+1, 2,2))
    for i in range(2):
        for j in range(2):
            temp_grid = jnp.arange(1,retirement_age+2)
            temp_grid = interp1d(temp_grid,phi_interp_values, nu[j], method='cubic2')
            temp_grid = jnp.where(i == 0, temp_grid*jnp.exp(nu_e), temp_grid)
            phigrid = phigrid.at[:,i,j].set(temp_grid)
    return phigrid

xi_interp_values = jnp.array([1,12,20,31])
def create_xigrid(xi):
    xigrid = jnp.zeros((n, 2,2))
    for i in range(2):
        for j in range(2):
            temp_grid = jnp.arange(1,31)
            temp_grid = interp1d(temp_grid,xi_interp_values, xi[i][j], method='cubic2')
            xigrid = xigrid.at[0:30,i,j].set(temp_grid)
            xigrid = xigrid.at[30:n,i,j].set(xi[i][j][3])
    return xigrid
def create_chimaxgrid(chi_1,chi_2):
    t = jnp.arange(38)
    chimax = jnp.maximum(chi_1*jnp.exp(chi_2*t),0)
    return chimax
def create_income_grid(y1_HS,y1_CL,ytHS_s,ytHS_sq,wagep_HS,wagep_CL,ytCL_s,ytCL_sq, sigx):
    sdztemp = ((sigx**2.0)/(1.0-rho**2.0))**0.5
    j = jnp.arange(20)
    health = jnp.arange(2)
    education = jnp.arange(2)
    def calc_base( _period, health, education):
        yt = jnp.where(education==1, (y1_CL*jnp.exp( ytCL_s*(_period) + ytCL_sq*(_period)**2.0 ))*(1.0-wagep_CL*(1-health)),(y1_HS*jnp.exp( ytHS_s*(_period) + ytHS_sq*(_period)**2.0 ))*(1.0-wagep_HS*(1-health)))
        return (yt/(jnp.exp(((jnp.log(theta_val[1])**2.0)**2.0)/2.0 )*jnp.exp( ((sdztemp**2.0)**2.0)/2.0)))
    mapped = _base_productmap(calc_base, ("_period", "health", "education"))
    return mapped(j,health,education)

# --------------------------------------------------------------------------------------
# Create static Grids
# --------------------------------------------------------------------------------------
eff_grid = jnp.linspace(0,1,40)
tr2yp_grid = jnp.zeros((38,2,40,40,2,2,2))
j = jnp.floor_divide(jnp.arange(38), 5)
def health_trans(period,health,eff,eff_1,edu,ht):
    y = const_healthtr + age_const[period] + edu*college_dummy + health*healthy_dummy + ht*htype_dummy + eff_grid[eff]*eff_param[0] + eff_grid[eff_1]*eff_param[1]
    return jnp.exp(y) / (1.0 + jnp.exp(y))
mapped_health_trans = _base_productmap(health_trans, ("period","health","eff","eff_1","edu","ht"))

tr2yp_grid = tr2yp_grid.at[:,:,:,:,:,:,1].set(mapped_health_trans(j,jnp.arange(2), jnp.arange(40),jnp.arange(40),jnp.arange(2),jnp.arange(2)))
tr2yp_grid = tr2yp_grid.at[:,:,:,:,:,:,0].set(1.0 - tr2yp_grid[:,:,:,:,:,:,1])

# Utility arrays for initial draws
discount = jnp.zeros((16),dtype=jnp.int8)
prod = jnp.zeros((16),dtype=jnp.int8)
ht = jnp.zeros((16),dtype=jnp.int8)
ed = jnp.zeros((16),dtype=jnp.int8)
for i in range(1,3):
    for j in range(1,3):
        for k in range(1,3):
            index = (i-1)*2*2 + (j-1)*2 + k - 1
            discount = discount.at[index].set(i-1)
            prod = prod.at[index].set(j-1)               
            ht = ht.at[index].set(1-(k-1))
            discount = discount.at[index+8].set(i-1)
            prod = prod.at[index+8].set(j-1)               
            ht = ht.at[index+8].set(1-(k-1))
            ed = ed.at[index+8].set(1)
init_distr_2b2t2h = jnp.array(np.loadtxt("init_distr_2b2t2h.txt"))
initial_dists = jnp.diff(init_distr_2b2t2h[:,0],prepend=0)

solve_and_simulate , _ = get_lcm_function(model=MODEL_CONFIG,jit = True, targets="solve_and_simulate")
solve , _ = get_lcm_function(model=MODEL_CONFIG,jit = True, targets="solve")

surv_HS = jnp.array(np.loadtxt("surv_HS.txt"))
surv_CL = jnp.array(np.loadtxt("surv_CL.txt"))
spgrid = jnp.zeros((2,38,2,2,2))
spgrid = spgrid.at[1,:,0,0,1].set(surv_HS[:,1])
spgrid = spgrid.at[1,:,1,0,1].set(surv_CL[:,1])
spgrid = spgrid.at[1,:,0,1,1].set(surv_HS[:,0])
spgrid = spgrid.at[1,:,1,1,1].set(surv_CL[:,0])
spgrid = spgrid.at[0,:,:,:,1].set(0)
spgrid = spgrid.at[:,:,:,:,0].set(1-spgrid[:,:,:,:,1])

def create_inputs(nuh_1, nuh_2, nuh_3, nuh_4,nuu_1, nuu_2, nuu_3, nuu_4,xiHSh_1,xiHSh_2,xiHSh_3,xiHSh_4,xiHSu_1,xiHSu_2,xiHSu_3,xiHSu_4,xiCLu_1,xiCLu_2,xiCLu_3,xiCLu_4,xiCLh_1,xiCLh_2,xiCLh_3,xiCLh_4,y1_HS,y1_CL,ytHS_s,ytHS_sq,wagep_HS,wagep_CL,ytCL_s,ytCL_sq, sigx, chi_1,chi_2, psi, nuad, bb, conp, penre, beta_mean, beta_std):
    nuh = jnp.array([nuh_1, nuh_2, nuh_3, nuh_4])
    nuu = jnp.array([nuu_1, nuu_2, nuu_3, nuu_4])
    nu = [nuu, nuh]
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
    income_grid = create_income_grid(y1_HS,y1_CL,ytHS_s,ytHS_sq,wagep_HS,wagep_CL,ytCL_s,ytCL_sq, sigx)
    chimax_grid = create_chimaxgrid(chi_1,chi_2)
    xvalues, xtrans = rouwenhorst(rho, jnp.sqrt(sigx), 5)
    prod_dist = jax.lax.fori_loop(0,1000000, lambda i,a: a @ xtrans.T, jnp.full(5,1/5))
    xi_grid = create_xigrid(xi)
    phi_grid = create_phigrid(nu, nuad)
    print(phi_grid[:,0,0])
    params = {
    "beta": 1,
    "disutil": {"phigrid": phi_grid},
    "fcost" : {"psi": psi, "xigrid":xi_grid},
    "cons_util": {"sigma": sigma, "bb": bb, "kappa" : conp},
    "utility": { "beta_mean": beta_mean, "beta_std": beta_std},
    "income": {"income_grid": income_grid, "xvalues" : xvalues},
    "pension": {"income_grid": income_grid, "penre":penre},
    "adj_cost": {"chimaxgrid": chimax_grid},
    "shocks" : {
        "productivity_shock": xtrans.T,
        "health": tr2yp_grid,
        "adjustment_cost": jnp.full((5, 5), 1/5),
        "alive": spgrid
    }}
    n = 1000
    seed = 32
    eff_grid = jnp.linspace(0,1,40)
    key = random.key(seed)
    initial_wealth = jnp.full((n), 0, dtype=jnp.int8)
    types = random.choice(key, jnp.arange(16), (n,), p=initial_dists)
    new_keys = random.split(key=key, num=3)
    health_draw = random.uniform(new_keys[0],(n,))
    health_thresholds = init_distr_2b2t2h[:,1][types]
    initial_health = jnp.where(health_draw > health_thresholds, 0, 1)
    initial_health_type = ht[types]
    initial_alive = jnp.ones(n, dtype=jnp.int8)
    initial_education = ed[types]
    initial_productivity = prod[types]
    initial_discount = discount[types]
    initial_effort = jnp.searchsorted(eff_grid,init_distr_2b2t2h[:,2][types])
    initial_adjustment_cost = random.choice(new_keys[1], jnp.arange(10), (n,))
    initial_productivity_shock = random.choice(new_keys[2], jnp.arange(5), (n,), p = prod_dist)

    initial_states = {"wealth": initial_wealth, "health": initial_health, "health_type": initial_health_type, "effort_t_1": initial_effort, 
                      "productivity_shock": initial_productivity_shock, "adjustment_cost": initial_adjustment_cost,
                      "education": initial_education, "alive":initial_alive, "productivity": initial_productivity, "discount_factor": initial_discount
                      }
    return params, initial_states

jitted_create_inputs = jax.jit(create_inputs)

def model_solve_and_simulate(nuh_1, nuh_2, nuh_3, nuh_4,nuu_1, nuu_2, nuu_3, nuu_4,xiHSh_1,xiHSh_2,xiHSh_3,xiHSh_4,xiHSu_1,xiHSu_2,xiHSu_3,xiHSu_4,xiCLu_1,xiCLu_2,xiCLu_3,xiCLu_4,xiCLh_1,xiCLh_2,xiCLh_3,xiCLh_4,y1_HS,y1_CL,ytHS_s,ytHS_sq,wagep_HS,wagep_CL,ytCL_s,ytCL_sq, sigx, chi_1,chi_2, psi, nuad, bb, conp, penre, beta_mean, beta_std):
    
    params, initial_states = jitted_create_inputs(nuh_1, nuh_2, nuh_3, nuh_4,nuu_1, nuu_2, nuu_3, nuu_4,xiHSh_1,xiHSh_2,xiHSh_3,xiHSh_4,xiHSu_1,xiHSu_2,xiHSu_3,xiHSu_4,xiCLu_1,xiCLu_2,xiCLu_3,xiCLu_4,xiCLh_1,xiCLh_2,xiCLh_3,xiCLh_4,y1_HS,y1_CL,ytHS_s,ytHS_sq,wagep_HS,wagep_CL,ytCL_s,ytCL_sq, sigx, chi_1,chi_2, psi, nuad, bb, conp, penre, beta_mean, beta_std)
    return solve_and_simulate(params=params,initial_states=initial_states,additional_targets=["utility","fcost","pension","income","cnow"])

def simulate_moments(params):
    res = model_solve_and_simulate(**params)
    moments = np.zeros(63)
    res['effort'] = np.asarray(eff_grid[res['effort'].to_numpy()])
    res['effort_t_1'] = np.asarray(eff_grid[res['effort_t_1'].to_numpy()])
    res['wealth'] = np.asarray(calc_savingsgrid(res['wealth'].to_numpy()))
    res['saving'] = np.asarray(calc_savingsgrid(res['saving'].to_numpy()))
    for health in range(2):
        for interval in range(4):
            working_pct_10years = (res.loc[(res['_period'] >= (interval*5)) & (res['_period'] < ((interval+1)*5)) & (res['alive'] == 1) & (res['health'] == health), ["working"]].sum()/2)/(res.loc[(res['_period'] >= (interval*5)) & (res['_period'] < ((interval+1)*5)) & (res['alive'] == 1)  &(res['health'] == health), "health"].count())
            moments[(interval+4*health)] = working_pct_10years.iloc[0]
    for health in range(2):
        for education in range(2):
            for interval in range(6):
                avg_effort_10years = (res.loc[(res['_period'] >= (interval*5)) & (res['_period'] < ((interval+1)*5)) & (res['alive'] == 1) & (res['health'] == health)& (res['education'] == education), ["effort"]].sum())/(res.loc[(res['_period'] >= (interval*5)) & (res['_period'] < ((interval+1)*5)) &(res['health'] == health) & (res['alive'] == 1) & (res['education'] == education), "effort"].count())
                moments[(interval+6*health+education*6*2)+8] = avg_effort_10years.iloc[0]
                if interval < 4:
                    avg_income_10years = (res.loc[(res['_period'] >= (interval*5)) & (res['_period'] < ((interval+1)*5)) & (res['alive'] == 1) & (res['health'] == health)& (res['education'] == education), ["income"]].sum())/(res.loc[(res['_period'] >= (interval*5)) & (res['_period'] < ((interval+1)*5)) &(res['health'] == health) & (res['alive'] == 1) & (res['education'] == education), "income"].count())
                    moments[(interval+4*health+education*4*2)+45] = avg_income_10years.iloc[0]*winit[1]/1000

    for interval in range(6):
                median_wealth_10y = res.loc[(res['_period'] >= (interval*5)) & (res['_period'] < ((interval+1)*5)) & (res['alive'] == 1) & (res['health'] == health)& (res['education'] == education), ["wealth"]].median()
                moments[interval+32] = median_wealth_10y.iloc[0]
    avgemp_HS = (res.loc[(res['alive'] == 1) & (res['education'] == 0), ["working"]].sum()/2)/(res.loc[ (res['alive'] == 1)  &(res['education'] == 0), "working"].count())
    avgemp_CL = (res.loc[(res['alive'] == 1) & (res['education'] == 1), ["working"]].sum()/2)/(res.loc[ (res['alive'] == 1)  &(res['education'] == 1), "working"].count())
    moments[38] = avgemp_CL.iloc[0]/avgemp_HS.iloc[0]
    for interval in range(3):
        non_adjusters = (res.loc[(res['_period'] >= (interval*10)) & (res['_period'] < ((interval+1)*10)) & (res['alive'] == 1) & (res['effort'] == res['effort_t_1'])].count())/ (res.loc[(res['_period'] >= (interval*10)) & (res['_period'] < ((interval+1)*10)) & (res['alive'] == 1)].count())
        moments[interval+39] = non_adjusters.iloc[0]
    std_effort = res.loc[res['alive'] == 1, 'effort'].std()
    moments[42] = std_effort
    cons_ratio = (res.loc[(res['alive'] == 1) & (res['health'] == 1), 'cnow'].sum()/res.loc[(res['alive'] == 1) & (res['health'] == 1), 'cnow'].count())/ (res.loc[(res['alive'] == 1) & (res['health'] == 0), 'cnow'].sum()/res.loc[(res['alive'] == 1) & (res['health'] == 0), 'cnow'].count())
    moments[43] = cons_ratio
    log_earnings = np.log(res.loc[(res['alive'] == 1) & (res['_period']<= retirement_age) & (res['working'] > 0), 'income'] * theta_val[1])
    moments[60] = log_earnings.var()
    pension_sum = (res.loc[(res['alive'] == 1) & (res['_period'] == retirement_age + 1), 'pension'].sum()/ res.loc[(res['alive'] == 1) & (res['_period'] == retirement_age+1), 'pension'].count())
    avg_income = (res.loc[(res['alive'] == 1) & (res['_period'] < retirement_age + 1), 'income'].sum()/ res.loc[(res['alive'] == 1) & (res['_period'] < retirement_age+1), 'income'].count())
    moments[61] = (pension_sum/avg_income)
    moments[62] = gini(jnp.asarray(res.loc[res['alive']== 1, 'wealth'].to_numpy()))
    return moments

