import textwrap

import numpy as np
import optimagic as om
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from plotly.subplots import make_subplots

WEALTH_NORMALIZATION = 48201


def wrap_labels(ax, width, *, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(
            textwrap.fill(text, width=width, break_long_words=break_long_words)
        )
    ax.set_xticklabels(labels, rotation=-45, ha="left")


empirical_moments = np.asarray(
    [
        0.6508581,
        0.7660204,
        0.8232445,
        0.6193264,
        0.5055072,
        0.5830671,
        0.6008949,
        0.4091998,
        0.6777659,
        0.6769325,
        0.6802505,
        0.6992036,
        0.7301746,
        0.7237555,
        0.6426084,
        0.6227545,
        0.627258,
        0.6552106,
        0.6968261,
        0.6921402,
        0.7790819,
        0.7702285,
        0.7660254,
        0.7634262,
        0.779154,
        0.7724553,
        0.7517721,
        0.7435739,
        0.736526,
        0.7381558,
        0.750504,
        0.734436,
        0.0619297,
        0.516081,
        1.165899,
        1.651459,
        1.567324,
        1.006182,
        1.237489,
        0.2672905,
        0.3283083,
        0.4041793,
        8.49264942390098,
        0.1610319,
        0.7456731,
        1.163207,
        35.39329,
        49.37886,
        55.95501,
        42.21932,
        24.94774,
        33.16593,
        36.69067,
        25.31111,
        59.48338,
        89.53806,
        107.9282,
        98.27698,
        50.38816,
        66.25301,
        78.31755,
        63.1325,
        0.5952184,
        0.4770515,
    ]
)
fig = om.criterion_plot(
    {
        "POUNDerS": "../optim_results/pd_var_2_test.db",
        "Nelder-Mead": "../optim_results/nm_var_1_real.db",
    },
    palette=["#FF7F0E", "#1F77B4"],
    monotone=True,
    max_evaluations=1500,
)
fig.update_layout(
    margin={"l": 20, "r": 20, "t": 0, "b": 60},
)
fig.write_image("../plots/comp_algos.pdf")
fig.show("firefox")
reader = om.SQLiteLogReader("../optim_results/pd_rand_1.db")
history = reader.read_history()
min_ind = np.argmin(np.asarray(history.fun))
min_params = history.params[min_ind]
optimal_moments = np.loadtxt("../results/optim_moments_boot.txt")

np.asarray(optimal_moments)
emp_healthy = optimal_moments[0:4]
emp_unhealthy = optimal_moments[4:8]

fig = go.Figure()
trace = go.Scatter(
    x=["25-34", "35-44", "45-54", "55-64"],
    y=emp_healthy,
    name="Healthy",
    mode="lines+markers",
    marker={"size": 6},
    legendgroup="Model",
    line_color="#1F77B4",
)
fig.add_trace(trace)
trace2 = go.Scatter(
    x=["25-34", "35-44", "45-54", "55-64"],
    y=emp_unhealthy,
    name="Unhealthy",
    mode="lines+markers",
    marker={"size": 6},
    legendgroup="Model",
    legendgrouptitle_text="Model",
    line_color="#FF7F0E",
)
fig.add_trace(trace2)
trace3 = go.Scatter(
    x=["25-34", "35-44", "45-54", "55-64"],
    y=empirical_moments[0:4],
    name="Healthy",
    mode="lines+markers",
    marker={"size": 6},
    marker_symbol="diamond",
    legendgroup="Data",
    line_dash="dot",
    line_color="#1F77B4",
)
fig.add_trace(trace3)
trace4 = go.Scatter(
    x=["25-34", "35-44", "45-54", "55-64"],
    y=empirical_moments[4:8],
    name="Unhealthy",
    mode="lines+markers",
    marker={"size": 6},
    marker_symbol="diamond",
    legendgroup="Data",
    legendgrouptitle_text="Data",
    line_dash="dot",
    line_color="#FF7F0E",
)
fig.add_trace(trace4)

fig.update_layout(
    template="simple_white",
    xaxis_title_text="Age Group",
    yaxis_title_text="Employment share",
    yaxis_range=[0, 1],
    margin={"l": 20, "r": 20, "t": 0, "b": 60},
    height=200,
    width=400,
)
fig.write_image(
    "../plots/emp.pdf",
    height=300,
    width=400,
)

fig = make_subplots(cols=2, subplot_titles=["Non-college", "College"])
trace = go.Scatter(
    x=["25-34", "35-44", "45-54", "55-64", "65-74", "75-84"],
    y=optimal_moments[8:14],
    name="Healthy",
    mode="lines+markers",
    marker={"size": 6},
    legendgroup="Model",
    line_color="#1F77B4",
)
fig.add_trace(trace, row=1, col=1)
trace2 = go.Scatter(
    x=["25-34", "35-44", "45-54", "55-64", "65-74", "75-84"],
    y=optimal_moments[14:20],
    name="Unhealthy",
    mode="lines+markers",
    marker={"size": 6},
    legendgroup="Model",
    legendgrouptitle_text="Model",
    line_color="#FF7F0E",
)
fig.add_trace(trace2, row=1, col=1)
trace3 = go.Scatter(
    x=["25-34", "35-44", "45-54", "55-64", "65-74", "75-84"],
    y=empirical_moments[8:14],
    name="Healthy",
    mode="lines+markers",
    marker={"size": 6},
    marker_symbol="diamond",
    legendgroup="Data",
    showlegend=False,
    line_dash="dot",
    line_color="#1F77B4",
)
fig.add_trace(trace3, row=1, col=1)
trace4 = go.Scatter(
    x=["25-34", "35-44", "45-54", "55-64", "65-74", "75-84"],
    y=empirical_moments[14:20],
    name="Unhealthy",
    mode="lines+markers",
    marker={"size": 6},
    marker_symbol="diamond",
    legendgroup="Data",
    showlegend=False,
    legendgrouptitle_text="Data",
    line_dash="dot",
    line_color="#FF7F0E",
)
fig.add_trace(trace4, row=1, col=1)
trace5 = go.Scatter(
    x=["25-34", "35-44", "45-54", "55-64", "65-74", "75-84"],
    y=optimal_moments[20:26],
    name="Healthy",
    mode="lines+markers",
    marker={"size": 6},
    showlegend=False,
    legendgroup="Model",
    line_color="#1F77B4",
)
fig.add_trace(trace5, row=1, col=2)
trace6 = go.Scatter(
    x=["25-34", "35-44", "45-54", "55-64", "65-74", "75-84"],
    y=optimal_moments[26:32],
    name="Unhealthy",
    mode="lines+markers",
    marker={"size": 6},
    showlegend=False,
    legendgroup="Model",
    legendgrouptitle_text="Model",
    line_color="#FF7F0E",
)
fig.add_trace(trace6, row=1, col=2)
trace7 = go.Scatter(
    x=["25-34", "35-44", "45-54", "55-64", "65-74", "75-84"],
    y=empirical_moments[20:26],
    name="Healthy",
    mode="lines+markers",
    marker={"size": 6},
    marker_symbol="diamond",
    legendgroup="Data",
    line_dash="dot",
    line_color="#1F77B4",
)
fig.add_trace(trace7, row=1, col=2)
trace8 = go.Scatter(
    x=["25-34", "35-44", "45-54", "55-64", "65-74", "75-84"],
    y=empirical_moments[26:32],
    name="Unhealthy",
    mode="lines+markers",
    marker={"size": 6},
    marker_symbol="diamond",
    legendgroup="Data",
    legendgrouptitle_text="Data",
    line_dash="dot",
    line_color="#FF7F0E",
)
fig.add_trace(trace8, row=1, col=2)
fig.update_layout(
    template="simple_white",
    xaxis_title_text="Age Group",
    yaxis_title_text="Effort",
    xaxis2_title_text="Age Group",
    yaxis2_title_text="Effort",
    yaxis_range=[0.5, 0.9],
    yaxis2_range=[0.5, 0.9],
    margin={"l": 20, "r": 20, "t": 20, "b": 80},
)

fig.write_image("../plots/eff.pdf", height=300, width=1000)

fig = make_subplots(cols=2, subplot_titles=["Non-college", "College"])
trace = go.Scatter(
    x=["25-34", "35-44", "45-54", "55-64"],
    y=optimal_moments[46:50],
    name="Healthy",
    mode="lines+markers",
    marker={"size": 6},
    legendgroup="Model",
    line_color="#1F77B4",
)
fig.add_trace(trace, row=1, col=1)
trace2 = go.Scatter(
    x=["25-34", "35-44", "45-54", "55-64"],
    y=optimal_moments[50:54],
    name="Unhealthy",
    mode="lines+markers",
    marker={"size": 6},
    legendgroup="Model",
    legendgrouptitle_text="Model",
    line_color="#FF7F0E",
)
fig.add_trace(trace2, row=1, col=1)
trace3 = go.Scatter(
    x=["25-34", "35-44", "45-54", "55-64"],
    y=empirical_moments[46:50],
    name="Healthy",
    mode="lines+markers",
    marker={"size": 6},
    marker_symbol="diamond",
    legendgroup="Data",
    line_dash="dot",
    line_color="#1F77B4",
)
fig.add_trace(trace3, row=1, col=1)
trace4 = go.Scatter(
    x=["25-34", "35-44", "45-54", "55-64"],
    y=empirical_moments[50:54],
    name="Unhealthy",
    mode="lines+markers",
    marker={"size": 6},
    marker_symbol="diamond",
    legendgroup="Data",
    legendgrouptitle_text="Data",
    line_dash="dot",
    line_color="#FF7F0E",
)
fig.add_trace(trace4, row=1, col=1)
trace5 = go.Scatter(
    x=[
        "25-34",
        "35-44",
        "45-54",
        "55-64",
    ],
    y=optimal_moments[54:58],
    name="Healthy",
    mode="lines+markers",
    showlegend=False,
    marker={"size": 6},
    legendgroup="Model",
    line_color="#1F77B4",
)
fig.add_trace(trace5, row=1, col=2)
trace6 = go.Scatter(
    x=[
        "25-34",
        "35-44",
        "45-54",
        "55-64",
    ],
    y=optimal_moments[58:62],
    name="Unhealthy",
    mode="lines+markers",
    showlegend=False,
    marker={"size": 6},
    legendgroup="Model",
    legendgrouptitle_text="Model",
    line_color="#FF7F0E",
)
fig.add_trace(trace6, row=1, col=2)
trace7 = go.Scatter(
    x=[
        "25-34",
        "35-44",
        "45-54",
        "55-64",
    ],
    y=empirical_moments[54:58],
    name="Healthy",
    mode="lines+markers",
    showlegend=False,
    marker={"size": 6},
    marker_symbol="diamond",
    legendgroup="Data",
    line_dash="dot",
    line_color="#1F77B4",
)
fig.add_trace(trace7, row=1, col=2)
trace8 = go.Scatter(
    x=["25-34", "35-44", "45-54", "55-64"],
    y=empirical_moments[58:62],
    name="Unhealthy",
    showlegend=False,
    mode="lines+markers",
    marker={"size": 6},
    marker_symbol="diamond",
    legendgroup="Data",
    legendgrouptitle_text="Data",
    line_dash="dot",
    line_color="#FF7F0E",
)
fig.add_trace(trace8, row=1, col=2)
fig.update_layout(
    template="simple_white",
    xaxis_title_text="Age Group",
    yaxis_title_text="Labor Income (Ths.)",
    xaxis2_title_text="Age Group",
    yaxis2_title_text="Labor Income (Ths.)",
    yaxis_range=[0, 120],
    yaxis2_range=[0, 120],
    margin={"l": 20, "r": 20, "t": 20, "b": 80},
)

fig.write_image(
    "../plots/inc.pdf",
    height=400,
    width=1000,
)

fig = go.Figure()
trace = go.Scatter(
    x=["25-34", "35-44", "45-54", "55-64", "65-74", "75-84"],
    y=(optimal_moments[32:38] * WEALTH_NORMALIZATION) / 1000,
    name="Model",
    mode="lines+markers",
    marker={"size": 6},
    legendgroup="Model",
    line_color="#1F77B4",
)
fig.add_trace(trace)
trace2 = go.Scatter(
    x=["25-34", "35-44", "45-54", "55-64", "65-74", "75-84"],
    y=(empirical_moments[32:38] * WEALTH_NORMALIZATION) / 1000,
    name="Data",
    marker_symbol="diamond",
    line_dash="dot",
    mode="lines+markers",
    marker={"size": 6},
    line_color="#1F77B4",
)
fig.add_trace(trace2)
fig.update_layout(
    template="simple_white",
    xaxis_title_text="Age Group",
    yaxis_title_text="Wealth (Ths.)",
    yaxis_range=[0, 120],
    margin={"l": 20, "r": 20, "t": 0, "b": 60},
    height=200,
    width=400,
)

fig.write_image(
    "../plots/wealth.pdf",
    height=300,
    width=400,
)

sensitivity_data = np.abs(np.loadtxt("../results/sens.txt"))
emp = np.expand_dims(np.max(sensitivity_data[:, 0:8], axis=1), axis=1)
eff = np.expand_dims(np.max(sensitivity_data[:, 8:32], axis=1), axis=1)
inc = np.expand_dims(np.max(sensitivity_data[:, 46:62], axis=1), axis=1)
wealth = np.expand_dims(np.average(sensitivity_data[:, 32:38], axis=1), axis=1)
non_adj = np.expand_dims(np.average(sensitivity_data[:, 39:42], axis=1), axis=1)
other = sensitivity_data[:, [38, 42, 43, 44, 45, 62, 63]]
sensitivity_data = np.hstack([emp, eff, inc, wealth, non_adj, other])
sensitivity_data_sum = np.sum(sensitivity_data, axis=1)
sensitivity_data_perc = sensitivity_data / np.expand_dims(sensitivity_data_sum, axis=1)
indexes = np.asarray(list(np.arange(0, 47)) + list(np.arange(61, 64)))
labels_disw = [
    r"$\nu_{1}^{h=1}$",
    r"$\nu_{8}^{h=1}$",
    r"$\nu_{13}^{h=1}$",
    r"$\nu_{20}^{h=1}$",
    r"$\nu_{1}^{h=0}$",
    r"$\nu_{8}^{h=0}$",
    r"$\nu_{13}^{h=0}$",
    r"$\nu_{20}^{h=0}$",
    r"$\nu_{e}$",
]
labels_diseff = [
    r"$\xi_{1}^{h=1,e=0}$",
    r"$\xi_{12}^{h=1,e=0}$",
    r"$\xi_{20}^{h=1,e=0}$",
    r"$\xi_{31}^{h=1,e=0}$",
    r"$\xi_{1}^{h=0,e=0}$",
    r"$\xi_{12}^{h=0,e=0}$",
    r"$\xi_{20}^{h=0,e=0}$",
    r"$\xi_{31}^{h=0,e=0}$",
    r"$\xi_{1}^{h=1,e=1}$",
    r"$\xi_{12}^{h=1,e=1}$",
    r"$\xi_{20}^{h=1,e=1}$",
    r"$\xi_{31}^{h=1,e=1}$",
    r"$\xi_{1}^{h=0,e=1}$",
    r"$\xi_{12}^{h=0,e=1}$",
    r"$\xi_{20}^{h=0,e=1}$",
    r"$\xi_{31}^{h=0,e=1}$",
    r"$\psi$",
]
labels_inc = [
    r"$\zeta_{0}^{e=0}$",
    r"$\zeta_{1}^{e=0}$",
    r"$\zeta_{2}^{e=0}$",
    r"$w_{p}^{e=0}$",
    r"$\zeta_{0}^{e=1}$",
    r"$\zeta_{1}^{e=1}$",
    r"$\zeta_{2}^{e=1}$",
    r"$w_{p}^{e=1}$",
]
labels_other = [
    r"$\chi_{0}$",
    r"$\chi_{1}$",
    r"$b$",
    r"$\kappa$",
    r"$\omega$",
    r"$\sigma_{z}$",
    r"$\mu_{\beta}$",
    r"$\sigma_{\beta}$",
]
sns.set_theme(rc={"text.usetex": True, "figure.figsize": (15, 12)})
ax = sns.heatmap(
    data=sensitivity_data_perc,
    linewidths=0.3,
    linecolor="black",
    fmt=".1f",
    annot=sensitivity_data,
    cmap=sns.light_palette("#1F77B4", as_cmap=True),
    cbar_kws={"label": r"$\%$ of Sum of Sensitivities"},
)

ax.set_yticklabels(labels_disw + labels_diseff + labels_inc + labels_other, rotation=0)
ax.add_patch(
    Rectangle((0, 0), 1, 8, fill=False, edgecolor="crimson", lw=3, clip_on=False)
)
ax.add_patch(
    Rectangle((5, 8), 1, 1, fill=False, edgecolor="crimson", lw=3, clip_on=False)
)
ax.add_patch(
    Rectangle((1, 9), 1, 16, fill=False, edgecolor="crimson", lw=3, clip_on=False)
)
ax.add_patch(
    Rectangle((7, 25), 1, 1, fill=False, edgecolor="crimson", lw=3, clip_on=False)
)
ax.add_patch(
    Rectangle((2, 26), 1, 8, fill=False, edgecolor="crimson", lw=3, clip_on=False)
)
ax.add_patch(
    Rectangle((4, 34), 1, 2, fill=False, edgecolor="crimson", lw=3, clip_on=False)
)

ax.add_patch(
    Rectangle((6, 36), 1, 1, fill=False, edgecolor="crimson", lw=3, clip_on=False)
)
ax.add_patch(
    Rectangle((9, 37), 1, 1, fill=False, edgecolor="crimson", lw=3, clip_on=False)
)
ax.add_patch(
    Rectangle((11, 38), 1, 1, fill=False, edgecolor="crimson", lw=3, clip_on=False)
)
ax.add_patch(
    Rectangle((10, 39), 1, 1, fill=False, edgecolor="crimson", lw=3, clip_on=False)
)
ax.add_patch(
    Rectangle((3, 40), 1, 1, fill=False, edgecolor="crimson", lw=3, clip_on=False)
)
ax.add_patch(
    Rectangle((8, 41), 1, 1, fill=False, edgecolor="crimson", lw=3, clip_on=False)
)

ax.set_xticks(np.arange(12) + 0.5)
ax.set_xticklabels(
    [
        "Avg. Employment (MAX)",
        "Avg. Effort (MAX)",
        "Avg. Income (MAX)",
        "Med. Wealth (AVG)",
        "Non-Adjusters (AVG)",
        "Employment Grad.",
        r"$VSLY/\hat c$",
        "Std. Eff.",
        "Wealth Gini",
        "Cons. Ratio",
        "Var. Log. Income",
        "Pension Repl.",
    ],
    rotation=-45,
    ha="left",
)
wrap_labels(ax, 15)
plt.savefig("../plots/sens.pdf", bbox_inches="tight")
plt.show()
