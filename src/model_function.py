from jax import numpy as jnp
from lcm.dispatchers import _base_productmap
from lcm.entry_point import get_lcm_function
from lcm import LogspaceGrid
from utils import rouwenhorst
from Mahler_Yum_2024 import MODEL_CONFIG
from jax import random
import jax
from interpax import interp1d
import numpy as np

# --------------------------------------------------------------------------------------
# Fixed Parameters
# --------------------------------------------------------------------------------------
avrgearn = 57706.57
theta_val = jnp.array([jnp.exp(-0.2898),jnp.exp(0.2898)])
n = 38
retirement_age = 20
taul = 0.128     
lamda = 1.0 - 0.321
rho = 0.975
r = 1.04**2.0
tt0 = 0.115
winit = jnp.array([43978,48201])
avrgearn = avrgearn/winit[1]
mincon0 = 0.10
mincon = mincon0 * avrgearn
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


phi_interp_values = jnp.array([1,8,13,20])
def create_phigrid(nu,nu_e):
    phigrid = jnp.zeros((retirement_age, 2,2))
    for i in range(2):
        for j in range(2):
            temp_grid = jnp.arange(1,retirement_age+1)
            temp_grid = interp1d(temp_grid,phi_interp_values, nu[j], method='cubic2')
            temp_grid = jnp.where(i == 0, temp_grid*jnp.exp(nu_e), temp_grid)
            phigrid = phigrid.at[...,i,j].set(temp_grid)
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
def create_chimaxgrid(chi_1,chi_2, chi_3):
    t = jnp.arange(1,39)
    chimax = jnp.maximum(chi_1*jnp.exp(chi_2*(t)-1.0) + chi_3*((t)-1.0)**2.0,0)
    return chimax
def create_income_grid(y1_HS,y1_CL,ytHS_s,ytHS_sq,wagep_HS,wagep_CL,ytCL_s,ytCL_sq, sigx):
    sdztemp = ((sigx**2.0)/(1.0-rho**2.0))**0.5
    j = jnp.arange(38)
    health = jnp.arange(2)
    education = jnp.arange(2)
    def calc_base( _period, health, education):
        yt = jnp.where(education==1, (y1_CL*jnp.exp( ytCL_s*(_period) + ytCL_sq*(_period)**2.0 ))*(1.0-wagep_CL*(1-health)),(y1_HS*jnp.exp( ytHS_s*(_period) + ytHS_sq*(_period)**2.0 ))*(1.0-wagep_HS*(1-health)))
        return (yt/(jnp.exp( ((jnp.log(theta_val[1])**2.0)**2.0)/2.0 )*jnp.exp( ((sdztemp**2.0)**2.0)/2.0)))
    mapped = _base_productmap(calc_base, ("_period", "health", "education"))
    return mapped(j,health,education)

eff_grid = jnp.linspace(0,1,40)
tr2yp_grid = jnp.zeros((2,38,40,40,2,2,2))
j = jnp.floor_divide(jnp.arange(38), 5)

def health_trans(health,period,eff,eff_1,edu,ht):
    y = const_healthtr + age_const[period] + edu*college_dummy + health*healthy_dummy + ht*htype_dummy + eff_grid[eff]*eff_param[0] + eff_grid[eff_1]*eff_param[1]
    return jnp.exp(y) / (1.0 + jnp.exp(y))
mapped_health_trans = _base_productmap(health_trans, ("health","period","eff","eff_1","edu","ht"))

tr2yp_grid = tr2yp_grid.at[:,:,:,:,:,:,1].set(mapped_health_trans(jnp.arange(2),j, jnp.arange(40),jnp.arange(40),jnp.arange(2),jnp.arange(2)))
tr2yp_grid = tr2yp_grid.at[:,:,:,:,:,:,0].set(1.0 - tr2yp_grid[:,:,:,:,:,:,1])


# Utility arrays for initial draws
discount = jnp.zeros((16),dtype=jnp.int32)
prod = jnp.zeros((16),dtype=jnp.int32)
ht = jnp.zeros((16),dtype=jnp.int32)
ed = jnp.zeros((16),dtype=jnp.int32)
for i in range(1,3):
    for j in range(1,3):
        for k in range(1,3):
            index = (i-1)*2*2 + (j-1)*2 + k - 1
            discount = discount.at[index].set(i-1)
            prod = prod.at[index].set(j-1)               
            ht = ht.at[index].set(k-1)
            discount = discount.at[index+8].set(i-1)
            prod = prod.at[index+8].set(j-1)               
            ht = ht.at[index+8].set(k-1)
            ed = ed.at[index+8].set(1)
init_distr_2b2t2h = jnp.array(np.loadtxt("init_distr_2b2t2h.txt"))
initial_dists = jnp.diff(init_distr_2b2t2h[:,0],prepend=0)

solve_and_simulate , _ = get_lcm_function(model=MODEL_CONFIG,jit = True, targets="solve_and_simulate")

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

beta_mean = 0.942749393405227     
beta_std = 0.0283688760224992

# effort habit adjustment cost max
chi_1 = 0.000120437772838191          
chi_2 = 0.14468204213946              

sigx = 0.0289408524185787               

penre = 0.358766004066242           
     

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

def draw_alive(simulation_result):

    return
def model_solve_and_simulate(nuh_1, nuh_2, nuh_3, nuh_4,nuu_1, nuu_2, nuu_3, nuu_4,xiHSh_1,xiHSh_2,xiHSh_3,xiHSh_4,xiHSu_1,xiHSu_2,xiHSu_3,xiHSu_4,xiCLu_1,xiCLu_2,xiCLu_3,xiCLu_4,xiCLh_1,xiCLh_2,xiCLh_3,xiCLh_4,y1_HS,y1_CL,ytHS_s,ytHS_sq,wagep_HS,wagep_CL,ytCL_s,ytCL_sq, sigx, chi_1,chi_2,chi_3, psi, nuad, sigma, bb, conp, penre, beta_mean, beta_std):
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
    """ for w in range(3):
        for e in range(2):
            for h in range(2):
                for p in range(2):
                    print(f"w:{w},e:{e},h:{h},p:{p}")
                    print(income_grid[w,:,h,e,p]) """
    chimax_grid = create_chimaxgrid(chi_1,chi_2,chi_3)
    xvalues, xtrans = rouwenhorst(rho, jnp.sqrt(sigx), 5)
    print(income_grid)
    prod_dist = jax.lax.fori_loop(0,100000, lambda i,a: a @ xtrans.T, jnp.full(5,1/5))
    xi_grid = create_xigrid(xi)
    phi_grid = create_phigrid(nu, nuad)
    params = {
    "beta": 1,
    "disutil": {"phigrid": phi_grid},
    "fcost" : {"psi": psi},
    "cons_util": {"sigma": sigma, "bb": bb, "kappa" : conp},
    "utility": {"xigrid": xi_grid, "sigma": sigma, "bb": bb, "kappa" : conp, "chimaxgrid": chimax_grid, "beta_mean": beta_mean, "beta_std": beta_std},
    "income": {"income_grid": income_grid},
    "pension": {"income_grid": income_grid, "penre":penre},
    "taxed_income": {"xvalues" : xvalues},
    "shocks" : {
        "productivity_shock": xtrans.T,
        "health": tr2yp_grid,
        "adjustment_cost": jnp.full((10, 10), 1/10)
    }}
    n = 1000
    seed = 56
    eff_grid = jnp.linspace(0,1,40)
    key = random.key(seed)
    initial_wealth = jnp.full((n), 0)
    types = random.choice(key, jnp.arange(16), (n,), p=initial_dists)
    new_keys = random.split(key=key, num=3)
    health_draw = random.uniform(new_keys[0],(n,))
    health_thresholds = init_distr_2b2t2h[:,1][types]
    initial_health = jnp.where(health_draw > health_thresholds, 0, 1)
    initial_health_type = ht[types]
    initial_education = ed[types]
    initial_productivity = prod[types]
    initial_discount = discount[types]
    initial_effort = jnp.searchsorted(eff_grid,init_distr_2b2t2h[:,2][types])
    initial_lagged_health = initial_health
    initial_adjustment_cost = random.choice(new_keys[1], jnp.arange(10), (n,))
    initial_productivity_shock = random.choice(new_keys[2], jnp.arange(5), (n,), p = prod_dist)

    initial_states = {"wealth": initial_wealth, "health": initial_health, "health_type": initial_health_type, "effort_t_1": initial_effort, 
                      "productivity_shock": initial_productivity_shock, "adjustment_cost": initial_adjustment_cost, "lagged_health": initial_lagged_health,
                      "education": initial_education, "productivity": initial_productivity, "discount_factor": initial_discount
                      }
    
    #return solve_and_simulate(params,initial_states,additional_targets=["utility","cons_util","disutil","income"])

res = model_solve_and_simulate(nuh_1, nuh_2, nuh_3, nuh_4,nuu_1, nuu_2, nuu_3, nuu_4,xiHSh_1,xiHSh_2,xiHSh_3,xiHSh_4,xiHSu_1,xiHSu_2,xiHSu_3,xiHSu_4,xiCLu_1,xiCLu_2,xiCLu_3,xiCLu_4,xiCLh_1,xiCLh_2,xiCLh_3,xiCLh_4,y1_HS,y1_CL,ytHS_s,ytHS_sq,wagep_HS,wagep_CL,ytCL_s,ytCL_sq, sigx, chi_1,chi_2,chi_3, psi, nuad, sigma, bb, conp, penre, beta_mean, beta_std)
print(res.loc[(slice(None),[90]),:].to_string())
print((res.loc[(res['_period'] < 5) & (res['health'] == 0), ["working"]].sum()/2)/(res.loc[(res['_period'] < 5) &(res['health'] == 0), "health"].count()))
print((res.loc[(res['_period'] < 5) & (res['health'] == 1), ["working"]].sum()/2)/(res.loc[(res['_period'] < 5) &(res['health'] == 1), "health"].count()))
