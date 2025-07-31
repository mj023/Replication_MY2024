import jax
from jax import numpy as jnp
import numpy as np
import pickle
import pandas as pd
import optimagic as om
from estimagic.estimate_msm import get_msm_optimization_functions
from estimagic.msm_weighting import get_weighting_matrix
import estimagic as em
from model_function import simulate_moments

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
reader = om.SQLiteLogReader('../optim_results/pd_var_1.db')
history = reader.read_history()
min_ind = np.argmin(np.asarray(history.fun))
min_params = history.params[min_ind]

""" G_hat = om.first_derivative(simulate_moments, min_params, method='forward')
dbfile = open('G_hat', 'ab')
pickle.dump(G_hat, dbfile)
dbfile.close() """

dbfile = open('G_hat', 'rb')    
G_hat = pickle.load(dbfile)
print(G_hat)

G_hat_inf = np.linalg.inv(G_hat.derivative)