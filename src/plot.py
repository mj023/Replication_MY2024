import optimagic as om
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from jax import numpy as jnp
import plotly.graph_objects as go
from model_function import simulate_moments

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

########################
# Mahler & Yum Params  #
########################
winit = jnp.array([43978,48201])

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
start_params = {'nuh_1':nuh_1, 'nuh_2':nuh_2, 'nuh_3':nuh_3, 'nuh_4':nuh_4,'nuu_1':nuu_1, 'nuu_2':nuu_2, 'nuu_3': nuu_3, 'nuu_4':nuu_4,  'nuad':nuad, 
                'xiHSh_1':xiHSh_1,'xiHSh_2':xiHSh_2,'xiHSh_3':xiHSh_3,'xiHSh_4':xiHSh_4,'xiHSu_1':xiHSu_1,'xiHSu_2':xiHSu_2,
                'xiHSu_3':xiHSu_3,'xiHSu_4':xiHSu_4,'xiCLu_1':xiCLu_1,'xiCLu_2':xiCLu_2,'xiCLu_3':xiCLu_3,'xiCLu_4':xiCLu_4,
                'xiCLh_1':xiCLh_1,'xiCLh_2':xiCLh_2,'xiCLh_3':xiCLh_3,'xiCLh_4':xiCLh_4,'y1_HS':y1_HS,'ytHS_s':ytHS_s,'ytHS_sq':ytHS_sq,'wagep_HS':wagep_HS,'y1_CL': y1_CL,
                'ytCL_s':ytCL_s,'ytCL_sq':ytCL_sq,'wagep_CL':wagep_CL, 'sigx':sigx,
                'chi_1': chi_1,'chi_2':chi_2, 'psi':psi,'bb':11, 'conp':conp, 'penre':penre,
                'beta_mean':beta_mean, 'beta_std':beta_std}
fig = om.criterion_plot(['../optim_results/pd_rand_1.db'], monotone=True)
#fig.write_image("../plots/comp_algos.pdf")
fig.show('firefox')
reader = om.SQLiteLogReader('../optim_results/pd_rand_1.db')
history = reader.read_history()
min_ind = np.argmin(np.asarray(history.fun))
min_params = history.params[min_ind]
#print(np.round(((simulate_moments(min_params)- empirical_moments)/empirical_moments)*100,decimals=3 ))

""" optimal_moments = simulate_moments(min_params)
np.savetxt("optimal_moments.csv", optimal_moments, 
              delimiter = ",") """
optimal_moments = np.loadtxt("optimal_moments.csv")

np.asarray(optimal_moments)
emp_healthy = optimal_moments[0:4]
emp_unhealthy = optimal_moments[4:8]

fig = go.Figure()
trace = go.Scatter(
        x=['25-34', '35-44','45-54','55-64'],
        y=emp_healthy,
        name='Healthy',
        mode="lines+markers",
        marker={'size':6},
        legendgroup='Model',
        line_color='#1F77B4'
)
fig.add_trace(trace)
trace2 = go.Scatter(
        x=['25-34', '35-44','45-54','55-64'],
        y=emp_unhealthy,
        name='Unhealthy',
        mode="lines+markers",
        marker={'size':6},
        legendgroup='Model',
        legendgrouptitle_text='Model',
        line_color='#FF7F0E'
)
fig.add_trace(trace2)
trace3 = go.Scatter(
        x=['25-34', '35-44','45-54','55-64'],
        y=empirical_moments[0:4],
        name='Healthy',
        mode="lines+markers",
        marker={'size':6},
        marker_symbol= 'diamond',
        legendgroup='Data',
        line_dash='dot',
        line_color='#1F77B4'
        
)
fig.add_trace(trace3)
trace4 = go.Scatter(
        x=['25-34', '35-44','45-54','55-64'],
        y=empirical_moments[4:8],
        name='Unhealthy',
        mode="lines+markers",
        marker={'size':6},
        marker_symbol= 'diamond',
        legendgroup='Data',
        legendgrouptitle_text='Data',
        line_dash='dot',
        line_color='#FF7F0E'
)
fig.add_trace(trace4)
fig.update_layout(
        template='simple_white',
        xaxis_title_text="Age Group",
        yaxis_title_text="Employment share",
        yaxis_range=[0,1],
)
fig.write_image("../plots/emp.pdf",height=400,
        width=500,)

fig = make_subplots(cols=2, subplot_titles=['Non-college', 'College'])
trace = go.Scatter(
        x=['25-34', '35-44','45-54','55-64', '65-74','75-84'],
        y=optimal_moments[8:14],
        name='Healthy',
        mode="lines+markers",
        marker={'size':6},
        legendgroup='Model',
        line_color='#1F77B4'
)
fig.add_trace(trace,row=1, col=1)
trace2 = go.Scatter(
        x=['25-34', '35-44','45-54','55-64', '65-74','75-84'],
        y=optimal_moments[14:20],
        name='Unhealthy',
        mode="lines+markers",
        marker={'size':6},
        legendgroup='Model',
        legendgrouptitle_text='Model',
        line_color='#FF7F0E'
)
fig.add_trace(trace2, row=1,col=1)
trace3 = go.Scatter(
        x=['25-34', '35-44','45-54','55-64', '65-74','75-84'],
        y=empirical_moments[8:14],
        name='Healthy',
        mode="lines+markers",
        marker={'size':6},
        marker_symbol= 'diamond',
        legendgroup='Data',
        showlegend=False,
        line_dash='dot',
        line_color='#1F77B4'
        
)
fig.add_trace(trace3,row=1, col=1)
trace4 = go.Scatter(
        x=['25-34', '35-44','45-54','55-64', '65-74','75-84'],
        y=empirical_moments[14:20],
        name='Unhealthy',
        mode="lines+markers",
        marker={'size':6},
        marker_symbol= 'diamond',
        legendgroup='Data',
        showlegend=False,
        legendgrouptitle_text='Data',
        line_dash='dot',
        line_color='#FF7F0E'
)
fig.add_trace(trace4,row=1, col=1)
trace5 = go.Scatter(
        x=['25-34', '35-44','45-54','55-64', '65-74','75-84'],
        y=optimal_moments[20:26],
        name='Healthy',
        mode="lines+markers",
        marker={'size':6},
        showlegend=False,
        legendgroup='Model',
        line_color='#1F77B4'
)
fig.add_trace(trace5,row=1, col=2)
trace6 = go.Scatter(
        x=['25-34', '35-44','45-54','55-64', '65-74','75-84'],
        y=optimal_moments[26:32],
        name='Unhealthy',
        mode="lines+markers",
        marker={'size':6},
        showlegend=False,
        legendgroup='Model',
        legendgrouptitle_text='Model',
        line_color='#FF7F0E'
)
fig.add_trace(trace6, row=1,col=2)
trace7 = go.Scatter(
        x=['25-34', '35-44','45-54','55-64', '65-74','75-84'],
        y=empirical_moments[20:26],
        name='Healthy',
        mode="lines+markers",
        marker={'size':6},
        marker_symbol= 'diamond',
        legendgroup='Data',
        line_dash='dot',
        line_color='#1F77B4'
        
)
fig.add_trace(trace7,row=1, col=2)
trace8 = go.Scatter(
        x=['25-34', '35-44','45-54','55-64', '65-74','75-84'],
        y=empirical_moments[26:32],
        name='Unhealthy',
        mode="lines+markers",
        marker={'size':6},
        marker_symbol= 'diamond',
        legendgroup='Data',
        legendgrouptitle_text='Data',
        line_dash='dot',
        line_color='#FF7F0E'
)
fig.add_trace(trace8,row=1, col=2)
fig.update_layout(
        template='simple_white',
        xaxis_title_text="Age Group",
        yaxis_title_text="Effort",
        xaxis2_title_text="Age Group",
        yaxis2_title_text="Effort",
        yaxis_range=[0.5,1],
        yaxis2_range=[0.5,1],
)

fig.write_image("../plots/eff.pdf", height=400,
        width=1000)

fig = make_subplots(cols=2, subplot_titles=['Non-college', 'College'])
trace = go.Scatter(
        x=['25-34', '35-44','45-54','55-64'],
        y=optimal_moments[46:50],
        name='Healthy',
        mode="lines+markers",
        marker={'size':6},
        legendgroup='Model',
        line_color='#1F77B4'
)
fig.add_trace(trace,row=1, col=1)
trace2 = go.Scatter(
        x=['25-34', '35-44','45-54','55-64'],
        y=optimal_moments[50:54],
        name='Unhealthy',
        mode="lines+markers",
        marker={'size':6},
        legendgroup='Model',
        legendgrouptitle_text='Model',
        line_color='#FF7F0E'
)
fig.add_trace(trace2, row=1,col=1)
trace3 = go.Scatter(
        x=['25-34', '35-44','45-54','55-64'],
        y=empirical_moments[46:50],
        name='Healthy',
        mode="lines+markers",
        marker={'size':6},
        marker_symbol= 'diamond',
        legendgroup='Data',
        line_dash='dot',
        line_color='#1F77B4'
        
)
fig.add_trace(trace3,row=1, col=1)
trace4 = go.Scatter(
        x=['25-34', '35-44','45-54','55-64'],
        y=empirical_moments[50:54],
        name='Unhealthy',
        mode="lines+markers",
        marker={'size':6},
        marker_symbol= 'diamond',
        legendgroup='Data',
        legendgrouptitle_text='Data',
        line_dash='dot',
        line_color='#FF7F0E'
)
fig.add_trace(trace4,row=1, col=1)
trace5 = go.Scatter(
        x=['25-34', '35-44','45-54','55-64', ],
        y=optimal_moments[54:58],
        name='Healthy',
        mode="lines+markers",
        showlegend=False,
        marker={'size':6},
        legendgroup='Model',
        line_color='#1F77B4'
)
fig.add_trace(trace5,row=1, col=2)
trace6 = go.Scatter(
        x=['25-34', '35-44','45-54','55-64', ],
        y=optimal_moments[58:62],
        name='Unhealthy',
        mode="lines+markers",
        showlegend=False,
        marker={'size':6},
        legendgroup='Model',
        legendgrouptitle_text='Model',
        line_color='#FF7F0E'
)
fig.add_trace(trace6, row=1,col=2)
trace7 = go.Scatter(
        x=['25-34', '35-44','45-54','55-64', ],
        y=empirical_moments[54:58],
        name='Healthy',
        mode="lines+markers",
        showlegend=False,
        marker={'size':6},
        marker_symbol= 'diamond',
        legendgroup='Data',
        line_dash='dot',
        line_color='#1F77B4'
        
)
fig.add_trace(trace7,row=1, col=2)
trace8 = go.Scatter(
        x=['25-34', '35-44','45-54','55-64'],
        y=empirical_moments[58:62],
        name='Unhealthy',
        showlegend=False,
        mode="lines+markers",
        marker={'size':6},
        marker_symbol= 'diamond',
        legendgroup='Data',
        legendgrouptitle_text='Data',
        line_dash='dot',
        line_color='#FF7F0E'
)
fig.add_trace(trace8,row=1, col=2)
fig.update_layout(
        template='simple_white',
        xaxis_title_text="Age Group",
        yaxis_title_text="Labor Income (Ths.)",
        xaxis2_title_text="Age Group",
        yaxis2_title_text="Labor Income (Ths.)",
        yaxis_range=[0,120],
        yaxis2_range=[0,120],
)

fig.write_image("../plots/inc.pdf", height=400,
        width=1000,)

fig = go.Figure()
trace = go.Scatter(
        x=['25-34', '35-44','45-54','55-64', '65-74','75-84'],
        y=(optimal_moments[32:38]*winit[1])/1000,
        name='Model',
        mode="lines+markers",
        marker={'size':6},
        legendgroup='Model',
        line_color='#1F77B4'
)
fig.add_trace(trace)
trace2 = go.Scatter(
        x=['25-34', '35-44','45-54','55-64', '65-74','75-84'],
        y=(empirical_moments[32:38]*winit[1])/1000,
        name='Data',
        marker_symbol= 'diamond',
        line_dash='dot',
        mode="lines+markers",
        marker={'size':6},
        line_color='#1F77B4'
)
fig.add_trace(trace2)
fig.update_layout(
        template='simple_white',
        xaxis_title_text="Age Group",
        yaxis_title_text="Wealth (Ths.)",
        yaxis_range=[0,120],
)

fig.write_image("../plots/wealth.pdf",    height=400,
        width=500,)

sensitivity_data = (np.abs(np.loadtxt('../results/sens.txt')))

labels_disw = [r'$\nu_{1}^{h=1}$',r'$\nu_{8}^{h=1}$',r'$\nu_{13}^{h=1}$',r'$\nu_{20}^{h=1}$',r'$\nu_{1}^{h=0}$',r'$\nu_{8}^{h=0}$',r'$\nu_{13}^{h=0}$',r'$\nu_{20}^{h=0}$',r'$\nu_{e}$',]
labels_diseff = [r'$\xi_{1}^{h=1,e=0}$',r'$\xi_{12}^{h=1,e=0}$',r'$\xi_{20}^{h=1,e=0}$',r'$\xi_{31}^{h=1,e=0}$',r'$\xi_{1}^{h=0,e=0}$',r'$\xi_{12}^{h=0,e=0}$',r'$\xi_{20}^{h=0,e=0}$',r'$\xi_{31}^{h=0,e=0}$',r'$\xi_{1}^{h=1,e=1}$',r'$\xi_{12}^{h=1,e=1}$',r'$\xi_{20}^{h=1,e=1}$',r'$\xi_{31}^{h=1,e=1}$',r'$\xi_{1}^{h=0,e=1}$',r'$\xi_{12}^{h=0,e=1}$',r'$\xi_{20}^{h=0,e=1}$',r'$\xi_{31}^{h=0,e=1}$',r'$\psi$',]
labels_inc = [r'$\zeta_{0}^{e=0}$',r'$\zeta_{1}^{e=0}$',r'$\zeta_{2}^{e=0}$',r'$w_{p}^{e=0}$',r'$\zeta_{0}^{e=1}$',r'$\zeta_{1}^{e=1}$',r'$\zeta_{2}^{e=1}$',r'$w_{p}^{e=1}$',]
sns.set_theme( rc={'text.usetex' : True})
fig, axes = plt.subplots(2, 2, figsize=(15, 5))
sns.heatmap(ax=axes[0,0],data=sensitivity_data[:9,:],  cmap='viridis', linewidths=.5
                 )
axes[0,0].set_yticklabels( labels=labels_disw,rotation=0)
axes[0,0].add_patch(Rectangle((0, 0), 8, 8, fill=False, edgecolor='crimson', lw=3, clip_on=False))
sns.heatmap(ax=axes[0,1],data=sensitivity_data[9:26,:], cmap='viridis', linewidths=.5,yticklabels=labels_diseff
                 )
axes[0,1].add_patch(Rectangle((9, 0), 24, 16, fill=False, edgecolor='crimson', lw=3, clip_on=False))

sns.heatmap(ax=axes[1,0],data=sensitivity_data[26:34,:], cmap='viridis', linewidths=.5,yticklabels=labels_inc
                 )
axes[1,0].add_patch(Rectangle((34, 0), 8, 8, fill=False, edgecolor='crimson', lw=3, clip_on=False))

sns.heatmap(ax=axes[1,1],data=sensitivity_data[34:,:], cmap='viridis', linewidths=.5,
                 )
plt.show()