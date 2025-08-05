import os
import optimagic as om
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from jax import numpy as jnp
import plotly.graph_objects as go
from model_function import simulate_wealth
# Set working directory
path = '/home/mj023/Downloads/Soep/SOEP_V38'
os.chdir(f"{path}")

# Load raw data

pl = pd.read_stata("pl_clean.dta")
gather_frame = []

""" for pl in iter:
    # Keep only selected variables
    print(pl.columns)
    pl = pl[[
        'pid', 'syear','ple0008'
    ]]
    gather_frame.append(pl)

pl = pd.concat(gather_frame) """
# Merge with other datasets
def merge_data(base, filename, suffix):
    df = pd.read_stata(filename)
    merged = pd.merge(base, df, on=['pid', 'syear'], how='left', suffixes=('', f'_{suffix}'))
    return merged

pl = merge_data(pl, 'pgen.dta', 'pgen')
pl = merge_data(pl, 'health.dta', 'health')
pl = merge_data(pl, 'pwealth.dta', 'wealth')
pl = merge_data(pl, 'pequiv.dta', 'pequiv')

# Keep only 2004-2018
pl = pl[pl['syear'].between(2004, 2018)]
pl = pl[pl['d11101']>=0]
disc_f = pl['d11101'].cat.categories
pl['d11101'] = pl['d11101'].cat.codes
pl['d11101'] =  pd.to_numeric(np.asarray(disc_f)[np.asarray(pl['d11101'])])
pl['age'] = pl['d11101']


# Age categories
pl['long_age_cat'] = pd.cut(pl['age'], bins=[25, 45, 65, 85, np.inf], labels=[1, 2, 3, 4])

# 10-year age categories
pl['age_cat10'] = pd.cut(pl['age'], bins=[25, 35, 45, 55, 65, 75, 85, 99], right=False)
print(pl['d11108'].cat.categories)
pl['d11108'] = pl['d11108'].cat.codes
pl['college'] = np.where(pl['d11108'] == 3, 1, np.where(pl['d11108'].isin([1, 2]), 0, np.nan))
print(pl['ple0008'])
print(pl['ple0008'].cat.codes)
pl['ple0008'] = pl['ple0008'].cat.codes - 2

pl['health_cat'] = np.where(pl['ple0008'].isin([1,2,3]), 1, 0)
print(pl['health_cat'])
pl = pl[(pl['college'].notnull()) & (pl['health_cat'].notnull()) & (pl['age'] >= 25)]

disc_f = pl['y11101'].cat.categories
pl['y11101'] = pl['y11101'].cat.codes
pl['y11101'] =  pd.to_numeric(np.asarray(disc_f)[np.asarray(pl['y11101'])])


pl['discount'] = 100 / pl['y11101']

# Discounted imputed wealth variables
for i in ['a', 'b', 'c', 'd', 'e']:
    pl[f'w0111{i}15'] = pl[f'w0111{i}']*pl['discount']

# Average over imputations
pl['net_wealth'] = pl[[f'w0111{i}15' for i in ['a', 'b', 'c', 'd', 'e']]].mean(axis=1)
pl['net_wealth_full'] = pl['net_wealth']

# Drop bottom 0.5%
cutoff_wealth = pl['net_wealth'].quantile(0.005)
pl['net_wealth'] = pl['net_wealth'].where(pl['net_wealth'] >= cutoff_wealth)
pl = pl[pl['net_wealth'].notna()]
pl['net_wealth0'] = pl['net_wealth']
pl.loc[pl['net_wealth0'] < 0, 'net_wealth0'] = 0



# 2007,2012,2017
pl = pl[pl['syear'].isin([2007,2012,2017])]
pl['short_age_cat'] = pd.cut(pl['age'], bins=[35, 45,55, 65,75, 85], labels=[1, 2, 3, 4,5])

wealth_1_h = pl.loc[(pl['short_age_cat'] == 1) &(pl['health_cat'] == 1), 'net_wealth0'].median()

wealth_2_h = pl.loc[(pl['short_age_cat'] == 2) &(pl['health_cat'] == 1), 'net_wealth0'].median()
wealth_3_h = pl.loc[(pl['short_age_cat'] == 3) &(pl['health_cat'] == 1), 'net_wealth0'].median()
wealth_4_h = pl.loc[(pl['short_age_cat'] == 4) &(pl['health_cat'] == 1), 'net_wealth0'].median()
wealth_5_h = pl.loc[(pl['short_age_cat'] == 5) &(pl['health_cat'] == 1), 'net_wealth0'].median()
wealth_1_u = pl.loc[(pl['short_age_cat'] == 1) &(pl['health_cat'] == 0), 'net_wealth0'].median()
wealth_2_u = pl.loc[(pl['short_age_cat'] == 2) &(pl['health_cat'] == 0), 'net_wealth0'].median()
wealth_3_u = pl.loc[(pl['short_age_cat'] == 3) &(pl['health_cat'] == 0), 'net_wealth0'].median()
wealth_4_u = pl.loc[(pl['short_age_cat'] == 4) &(pl['health_cat'] == 0), 'net_wealth0'].median()
wealth_5_u = pl.loc[(pl['short_age_cat'] == 5) &(pl['health_cat'] == 0), 'net_wealth0'].median()

wealth_h = [wealth_1_h,wealth_2_h,wealth_3_h,wealth_4_h,wealth_5_h]
wealth_uh = [wealth_1_u,wealth_2_u,wealth_3_u,wealth_4_u,wealth_5_u]
path = '/home/mj023/Git/Econ_RL/src'
os.chdir(f"{path}")
fig = go.Figure()
trace = go.Scatter(
        x=['35-44','45-54','55-64', '65-74','75-84'],
        y=(np.asarray(wealth_h))/1000,
        name='Model',
        mode="lines+markers",
        marker={'size':6},
        legendgroup='Model',
        line_color='#1F77B4'
)
fig.add_trace(trace)
trace2 = go.Scatter(
        x=['35-44','45-54','55-64', '65-74','75-84'],
        y=(np.asarray(wealth_uh))/1000,
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
        margin=dict(l=20, r=20, t=0, b=60),
        height= 200,
        width= 400
)

fig.write_image("../plots/wealthgaps.pdf",    height=300,
        width=400,)

reader = om.SQLiteLogReader('../optim_results/pd_var_2.db')
history = reader.read_history()
min_ind = np.argmin(np.asarray(history.fun))
min_params = history.params[min_ind]
print(simulate_wealth(min_params))
