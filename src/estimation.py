import jax
from jax import numpy as jnp
import numpy as np
import time
import pandas as pd
import optimagic as om

from model_function import simulate_moments

########################
# Mahler & Yum Params  #
########################

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

start_params = {'nuh_1':nuh_1, 'nuh_2':nuh_2, 'nuh_3':nuh_3, 'nuh_4':nuh_4,'nuu_1':nuu_1, 'nuu_2':nuu_2, 'nuu_3': nuu_3, 'nuu_4':nuu_4,
                'xiHSh_1':xiHSh_1,'xiHSh_2':xiHSh_2,'xiHSh_3':xiHSh_3,'xiHSh_4':xiHSh_4,'xiHSu_1':xiHSu_1,'xiHSu_2':xiHSu_2,
                'xiHSu_3':xiHSu_3,'xiHSu_4':xiHSu_4,'xiCLu_1':xiCLu_1,'xiCLu_2':xiCLu_2,'xiCLu_3':xiCLu_3,'xiCLu_4':xiCLu_4,
                'xiCLh_1':xiCLh_1,'xiCLh_2':xiCLh_2,'xiCLh_3':xiCLh_3,'xiCLh_4':xiCLh_4,'y1_HS':y1_HS,'y1_CL': y1_CL,'ytHS_s':ytHS_s,
                'ytHS_sq':ytHS_sq,'wagep_HS':wagep_HS,'wagep_CL':wagep_CL,'ytCL_s':ytCL_s,'ytCL_sq':ytCL_sq, 'sigx':sigx,
                'chi_1': chi_1,'chi_2':chi_2, 'psi':psi, 'nuad':nuad, 'bb':10, 'conp':conp, 'penre':penre,
                'beta_mean':beta_mean, 'beta_std':beta_std}
empirical_moments = np.asarray([0.6508581,0.7660204,0.8232445,0.6193264, 
        0.5055072,0.5830671,0.6008949,0.4091998,                                              
    
        0.6777659,0.6769325,0.6802505,0.6992036,0.7301746,0.7237555,   
        0.6426084,0.6227545,0.627258,0.6552106,0.6968261,0.6921402,   
        0.7790819,0.7702285,0.7660254,0.7634262,0.779154,0.7724553,   
        0.7517721,0.7435739,0.736526,0.7381558,0.750504,0.734436,     
        0.0619297,0.516081,1.165899,1.651459, 1.567324, 1.006182,     
        1.237489,             
        0.2672905,0.3283083,0.4041793, 
        8.49264942390098,                             
        0.1610319,                         
        0.7456731,                         
        1.163207,                        

        35.39329, 49.37886, 55.95501, 42.21932,	 
        24.94774, 33.16593, 36.69067, 25.31111,   
        59.48338, 89.53806, 107.9282, 98.27698,   
        50.38816, 66.25301, 78.31755, 63.1325,    
        0.5952184,                           
        0.4770515 ])

moment_sd = np.asarray([0.0022079,0.001673,0.0015903,0.0024375,
        0.0078668,0.0054486,0.0045718,0.0045788,      
        0.0019615,0.0016137,0.0016517,0.0018318,0.0016836,0.0022494,
        0.0066662,0.0047753,0.0035851,0.0031197,0.0027306,0.0025937,
        0.0024741,0.0019636,0.0019423,0.0022411,0.0024561,0.0037815, 
        0.0107689,0.0082543,0.0063126,0.0051546,0.0050761,0.0057938, 
        0.0031501,0.0146831,0.023547,0.037393, 0.042682, 0.0473329,                   
        0.0029621,      
        0.0037247,0.0030799,0.0039969,            
        0.594830904063775, 
        0.0004399,                            
        0.0035907,                        
        0.0221391,                       
                                                               
        0.1955369, 0.2318309, 0.2660378, 0.3528976,	
        0.5630693, 0.5187444, 0.4988166, 0.4986972, 
        0.4875483, 0.631705, 0.7607303, 1.108492, 
        1.848723, 1.65571, 1.688008, 1.78551,
        0.0023382,                          
        0.0015815 ])                     

W_var = np.diag(1/moment_sd**2)
W_root = np.sqrt(W_var)
W_ones = np.diag(np.ones(64))
algo_nm = om.algos.scipy_neldermead(
    stopping_maxfun=2000
)
algo_pounders = om.algos.tao_pounders(
    stopping_maxiter=400
)
log_opts = om.SQLiteLogOptions(
    path= "pd_var_2.db",
    if_database_exists='replace'
)
key = jax.random.key(33)

vals = jax.random.uniform(key, (42), maxval=0.4)
start_params_keys = list(start_params.keys())
for i in range(42):
    start_params[start_params_keys[i]] = start_params[start_params_keys[i]] + start_params[start_params_keys[i]]*(vals[i]-0.2)
def criterion_func(params):
    sim_moments = simulate_moments(params)
    e = sim_moments - empirical_moments
    g_theta = e.T @ W_var @ e
    return g_theta

@om.mark.least_squares
def criterion_func_sqr(params):
    sim_moments = simulate_moments(params)
    e = sim_moments - empirical_moments
    residuals = e @ W_root
    return residuals
start_params['conp'] = 0.9
start_params['beta_mean'] = 0.93

lower_bounds = {'nuh_1':0, 'nuh_2':0, 'nuh_3':0, 'nuh_4':0,'nuu_1':0, 'nuu_2':0, 'nuu_3': 0, 'nuu_4':0,
                'xiHSh_1':0,'xiHSh_2':0,'xiHSh_3':0,'xiHSh_4':0.0,'xiHSu_1':0.0,'xiHSu_2':0.0,
                'xiHSu_3':0.0,'xiHSu_4':0.0,'xiCLu_1':0.0,'xiCLu_2':0.0,'xiCLu_3':0.0,'xiCLu_4':0.0,
                'xiCLh_1':0.0,'xiCLh_2':0.0,'xiCLh_3':0.0,'xiCLh_4':0.0,'y1_HS':0,'y1_CL': 0,'ytHS_s':0,
                'ytHS_sq':-0.15,'wagep_HS':0,'wagep_CL':0,'ytCL_s':0,'ytCL_sq':-0.15, 'sigx':0,
                'chi_1': 0.0,'chi_2':0.0, 'psi':0.0, 'nuad':0, 'bb':6, 'conp':0.65, 'penre':0.2,
                'beta_mean':0.8, 'beta_std':0.005}
upper_bounds = {'nuh_1':4, 'nuh_2':4, 'nuh_3':4, 'nuh_4':4,'nuu_1':4, 'nuu_2':4, 'nuu_3': 4, 'nuu_4':4,
                'xiHSh_1':3,'xiHSh_2':3,'xiHSh_3':3,'xiHSh_4':3,'xiHSu_1':3,'xiHSu_2':3,
                'xiHSu_3':3,'xiHSu_4':3,'xiCLu_1':3,'xiCLu_2':3,'xiCLu_3':3,'xiCLu_4':3,
                'xiCLh_1':3,'xiCLh_2':3,'xiCLh_3':3,'xiCLh_4':3,'y1_HS':2,'y1_CL': 2,'ytHS_s':0.3,
                'ytHS_sq':0.2,'wagep_HS':0.3,'wagep_CL':0.3,'ytCL_s':0.3,'ytCL_sq':0.2, 'sigx':0.1,
                'chi_1': 0.01,'chi_2':0.3, 'psi':2, 'nuad':1.5, 'bb':15, 'conp':1, 'penre':0.6,
                'beta_mean':1.05, 'beta_std':0.09}
bounds = om.Bounds(lower=lower_bounds, upper=upper_bounds)

start_time = time.time()

res = om.minimize(criterion_func_sqr,start_params, algo_pounders, bounds=bounds, scaling=om.ScalingOptions(method='bounds', clipping_value=0.0001), logging=log_opts)
res.to_pickle('pd_full_model_run_1.pkl')
optim_time = time.time() - start_time
simulate_moments(start_params)
start_time = time.time()
simulate_moments(start_params)
one_iter = time.time() - start_time
timings = {"full_opt": [optim_time], "one_iter" : one_iter}
time_df = pd.DataFrame(timings)
time_df.to_csv("pd_optim_timings_2.csv")


