from pathlib import Path

import jax
import numpy as np
import pandas as pd
from jax import numpy as jnp
from jax import random
from lcm.utils.dispatchers import productmap
from scipy.interpolate import interp1d as scipy_interp1d

from Mahler_Yum_2024 import (
    MAHLER_YUM_MODEL,
    ages,
    calc_savingsgrid,
    prod_shock_grid,
    spgrid,
)
from utils import gini

_DATA_DIR = Path(__file__).parent

model = MAHLER_YUM_MODEL

# --------------------------------------------------------------------------------------
# Fixed Parameters
# --------------------------------------------------------------------------------------
avrgearn = 57706.57
theta_val = jnp.array([jnp.exp(-0.2898), jnp.exp(0.2898)])
n = 38
retirement_age = 19
taul = 0.128
lamda = 1.0 - 0.321
rho = 0.975
r = 1.04**2.0
tt0 = 0.115
winit = jnp.array([43978, 48201])
avrgearn = avrgearn / winit[1]
mincon0 = 0.10
mincon = mincon0 * avrgearn
sigma = 2
# --------------------------------------------------------------------------------------
# Health Techonology
# --------------------------------------------------------------------------------------
const_healthtr = -0.906
age_const = jnp.asarray([0.0, -0.289, -0.644, -0.881, -1.138, -1.586, -1.586, -1.586])
eff_param = jnp.asarray([0.693, 0.734])
eff_sq = 0
healthy_dummy = 2.311
htype_dummy = 0.632
college_dummy = 0.238

# --------------------------------------------------------------------------------------
# Grid Creation Functions
# --------------------------------------------------------------------------------------

phi_interp_values = jnp.array([1, 8, 13, 20])


def create_phigrid(nu, nu_e):
    phigrid = jnp.zeros((retirement_age + 1, 2, 2))
    for i in range(2):
        for j in range(2):
            interp_points = jnp.arange(1, retirement_age + 2)
            spline = scipy_interp1d(
                np.asarray(phi_interp_values), np.asarray(nu[j]), kind="cubic"
            )
            temp_grid = jnp.asarray(spline(interp_points))
            temp_grid = jnp.where(i == 0, temp_grid * jnp.exp(nu_e), temp_grid)
            phigrid = phigrid.at[:, i, j].set(temp_grid)
    return phigrid


xi_interp_values = jnp.array([1, 12, 20, 31])


def create_xigrid(xi):
    xigrid = jnp.zeros((n, 2, 2))
    for i in range(2):
        for j in range(2):
            interp_points = np.arange(1, 31)
            spline = scipy_interp1d(
                np.asarray(xi_interp_values), np.asarray(xi[i][j]), kind="cubic"
            )
            temp_grid = jnp.asarray(spline(interp_points))
            xigrid = xigrid.at[0:30, i, j].set(temp_grid)
            xigrid = xigrid.at[30:n, i, j].set(xi[i][j][3])
    return xigrid


def create_chimaxgrid(chi_1, chi_2):
    t = jnp.arange(38)
    return jnp.maximum(chi_1 * jnp.exp(chi_2 * t), 0)


def create_income_grid(
    y1_HS, y1_CL, ytHS_s, ytHS_sq, wagep_HS, wagep_CL, ytCL_s, ytCL_sq, sigx
):
    sdztemp = ((sigx**2.0) / (1.0 - rho**2.0)) ** 0.5
    j = jnp.arange(20)
    health = jnp.arange(2)
    education = jnp.arange(2)

    def calc_base(_period, health, education):
        yt = jnp.where(
            education == 1,
            (y1_CL * jnp.exp(ytCL_s * (_period) + ytCL_sq * (_period) ** 2.0))
            * (1.0 - wagep_CL * (1 - health)),
            (y1_HS * jnp.exp(ytHS_s * (_period) + ytHS_sq * (_period) ** 2.0))
            * (1.0 - wagep_HS * (1 - health)),
        )
        return yt / (
            jnp.exp(((jnp.log(theta_val[1]) ** 2.0) ** 2.0) / 2.0)
            * jnp.exp(((sdztemp**2.0) ** 2.0) / 2.0)
        )

    variables = ("_period", "health", "education")
    mapped = productmap(
        func=calc_base, variables=variables, batch_sizes=dict.fromkeys(variables, 0)
    )
    return mapped(_period=j, health=health, education=education)


# --------------------------------------------------------------------------------------
# Create static Grids
# --------------------------------------------------------------------------------------
eff_grid = jnp.linspace(0, 1, 40)
tr2yp_grid = jnp.zeros((38, 2, 40, 40, 2, 2, 2))
j = jnp.floor_divide(jnp.arange(38), 5)


def health_trans(period, health, eff, eff_1, edu, ht):
    y = (
        const_healthtr
        + age_const[period]
        + edu * college_dummy
        + health * healthy_dummy
        + ht * htype_dummy
        + eff_grid[eff] * eff_param[0]
        + eff_grid[eff_1] * eff_param[1]
    )
    return jnp.exp(y) / (1.0 + jnp.exp(y))


_health_trans_variables = ("period", "health", "eff", "eff_1", "edu", "ht")
mapped_health_trans = productmap(
    func=health_trans,
    variables=_health_trans_variables,
    batch_sizes=dict.fromkeys(_health_trans_variables, 0),
)

tr2yp_grid = tr2yp_grid.at[:, :, :, :, :, :, 1].set(
    mapped_health_trans(
        period=j,
        health=jnp.arange(2),
        eff=jnp.arange(40),
        eff_1=jnp.arange(40),
        edu=jnp.arange(2),
        ht=jnp.arange(2),
    )
)
tr2yp_grid = tr2yp_grid.at[:, :, :, :, :, :, 0].set(
    1.0 - tr2yp_grid[:, :, :, :, :, :, 1]
)

# Utility arrays for initial draws
discount = jnp.zeros((16), dtype=jnp.int8)
prod = jnp.zeros((16), dtype=jnp.int8)
ht = jnp.zeros((16), dtype=jnp.int8)
ed = jnp.zeros((16), dtype=jnp.int8)
for i in range(1, 3):
    for j in range(1, 3):
        for k in range(1, 3):
            index = (i - 1) * 2 * 2 + (j - 1) * 2 + k - 1
            discount = discount.at[index].set(i - 1)
            prod = prod.at[index].set(j - 1)
            ht = ht.at[index].set(1 - (k - 1))
            discount = discount.at[index + 8].set(i - 1)
            prod = prod.at[index + 8].set(j - 1)
            ht = ht.at[index + 8].set(1 - (k - 1))
            ed = ed.at[index + 8].set(1)
init_distr_2b2t2h = jnp.array(np.loadtxt(_DATA_DIR / "init_distr_2b2t2h.txt"))
initial_dists = jnp.diff(init_distr_2b2t2h[:, 0], prepend=0)


def create_inputs(
    seed,
    n_simulation_subjects,
    nuh_1,
    nuh_2,
    nuh_3,
    nuh_4,
    nuu_1,
    nuu_2,
    nuu_3,
    nuu_4,
    xiHSh_1,
    xiHSh_2,
    xiHSh_3,
    xiHSh_4,
    xiHSu_1,
    xiHSu_2,
    xiHSu_3,
    xiHSu_4,
    xiCLu_1,
    xiCLu_2,
    xiCLu_3,
    xiCLu_4,
    xiCLh_1,
    xiCLh_2,
    xiCLh_3,
    xiCLh_4,
    y1_HS,
    y1_CL,
    ytHS_s,
    ytHS_sq,
    wagep_HS,
    wagep_CL,
    ytCL_s,
    ytCL_sq,
    sigx,
    chi_1,
    chi_2,
    psi,
    nuad,
    bb,
    conp,
    penre,
):
    nuh = jnp.array([nuh_1, nuh_2, nuh_3, nuh_4])
    nuu = jnp.array([nuu_1, nuu_2, nuu_3, nuu_4])
    nu = [nuu, nuh]
    xi_HSh = jnp.array([xiHSh_1, xiHSh_2, xiHSh_3, xiHSh_4])
    xi_HSu = jnp.array([xiHSu_1, xiHSu_2, xiHSu_3, xiHSu_4])
    xi_CLu = jnp.array([xiCLu_1, xiCLu_2, xiCLu_3, xiCLu_4])
    xi_CLh = jnp.array([xiCLh_1, xiCLh_2, xiCLh_3, xiCLh_4])
    xi = [[xi_HSu, xi_HSh], [xi_CLu, xi_CLh]]
    print(xi)
    income_grid = create_income_grid(
        y1_HS, y1_CL, ytHS_s, ytHS_sq, wagep_HS, wagep_CL, ytCL_s, ytCL_sq, sigx
    )
    chimax_grid = create_chimaxgrid(chi_1, chi_2)
    xvalues = prod_shock_grid.get_gridpoints()
    xtrans = prod_shock_grid.get_transition_probs()
    prod_dist = jax.lax.fori_loop(
        0,
        1000000,
        lambda _i, a: a @ xtrans.T,
        jnp.full(5, 1 / 5),
    )
    xi_grid = create_xigrid(xi)
    phi_grid = create_phigrid(nu, nuad)
    params = {
        "disutil": {"phigrid": phi_grid},
        "fcost": {"psi": psi, "xigrid": xi_grid},
        "cons_util": {"sigma": sigma, "bb": bb, "kappa": conp},
        "income": {"income_grid": income_grid},
        "pension": {"income_grid": income_grid, "penre": penre},
        "scaled_adjustment_cost": {"chimaxgrid": chimax_grid},
        "scaled_productivity_shock": {"sigx": jnp.sqrt(sigx)},
        "next_health": {"probs_array": tr2yp_grid},
        "next_regime": {"probs_array": spgrid},
    }
    n = n_simulation_subjects
    eff_grid = jnp.linspace(0, 1, 40)
    key = random.key(seed)
    initial_wealth = jnp.full((n), 0, dtype=jnp.int8)
    types = random.choice(key, jnp.arange(16), (n,), p=initial_dists)
    new_keys = random.split(key=key, num=3)
    health_draw = random.uniform(new_keys[0], (n,))
    health_thresholds = init_distr_2b2t2h[:, 1][types]
    initial_health = jnp.where(health_draw > health_thresholds, 0, 1)
    initial_health_type = 1 - ht[types]
    initial_education = ed[types]
    initial_productivity = prod[types]
    initial_discount = discount[types]
    initial_effort = jnp.searchsorted(eff_grid, init_distr_2b2t2h[:, 2][types])
    initial_adjustment_cost = random.uniform(new_keys[1], (n,))
    initial_productivity_shock = xvalues[
        random.choice(new_keys[2], jnp.arange(5), (n,), p=prod_dist)
    ]
    initial_states = {
        "age": jnp.full(n, ages.values[0]),
        "wealth": initial_wealth,
        "health": initial_health,
        "health_type": initial_health_type,
        "effort_t_1": initial_effort,
        "productivity_shock": initial_productivity_shock,
        "adjustment_cost": initial_adjustment_cost,
        "education": initial_education,
        "productivity": initial_productivity,
    }

    return params, initial_states, initial_discount


def model_solve_and_simulate(params):
    seed = 32
    n_subjects = 10000
    start_params_without_beta = {
        k: v for k, v in params.items() if k not in ("beta_mean", "beta_std")
    }
    common_params, initial_states, discount_factor_type = create_inputs(
        seed, n_simulation_subjects=n_subjects, **start_params_without_beta
    )

    beta_mean = params["beta_mean"]
    beta_std = params["beta_std"]

    # Two-solve approach: simulate separately for each discount type, combine.
    # Replaces old approach where discount_factor was a state variable with
    # manual beta^period discounting. Mathematically equivalent.
    dfs = []
    for beta_val, type_id in [(beta_mean - beta_std, 0), (beta_mean + beta_std, 1)]:
        mask = discount_factor_type == type_id
        n_type = int(mask.sum())
        type_initial = {k: v[mask] for k, v in initial_states.items()}
        type_initial["regime"] = jnp.full(
            n_type, model.regime_names_to_ids["alive"], dtype=jnp.int32
        )

        result = model.simulate(
            params={"alive": {"discount_factor": beta_val, **common_params}},
            initial_conditions=type_initial,
            period_to_regime_to_V_arr=None,
            seed=42,
            log_level="off",
        )
        df = result.to_dataframe(
            additional_targets=["utility", "fcost", "pension", "income", "cnow"],
            use_labels=False,
        )
        df["discount_type"] = type_id
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def simulate_moments(params):

    res = model_solve_and_simulate(params)
    # Filter to alive regime only (regime column is always string names)
    res = res[res["regime"] == "alive"].copy()
    moments = np.zeros(64)
    res["effort"] = np.asarray(eff_grid[res["effort"].to_numpy().astype(int)])
    res["effort_t_1"] = np.asarray(eff_grid[res["effort_t_1"].to_numpy().astype(int)])
    res["wealth"] = np.asarray(calc_savingsgrid(res["wealth"].to_numpy()))
    res["saving"] = np.asarray(calc_savingsgrid(res["saving"].to_numpy()))
    for health in range(2):
        for interval in range(4):
            working_pct_10years = (
                res.loc[
                    (res["period"] >= (interval * 5))
                    & (res["period"] < ((interval + 1) * 5))
                    & (res["health"] == health),
                    ["working"],
                ].sum()
                / 2
            ) / (
                res.loc[
                    (res["period"] >= (interval * 5))
                    & (res["period"] < ((interval + 1) * 5))
                    & (res["health"] == health),
                    "health",
                ].count()
            )
            moments[(interval + 4 * (1 - health))] = working_pct_10years.iloc[0]
    for health in range(2):
        for education in range(2):
            for interval in range(6):
                avg_effort_10years = (
                    res.loc[
                        (res["period"] >= (interval * 5))
                        & (res["period"] < ((interval + 1) * 5))
                        & (res["health"] == health)
                        & (res["education"] == education),
                        ["effort"],
                    ].sum()
                ) / (
                    res.loc[
                        (res["period"] >= (interval * 5))
                        & (res["period"] < ((interval + 1) * 5))
                        & (res["health"] == health)
                        & (res["education"] == education),
                        "effort",
                    ].count()
                )
                moments[(interval + 6 * (1 - health) + education * 6 * 2) + 8] = (
                    avg_effort_10years.iloc[0]
                )
                if interval < 4:
                    avg_income_10years = (
                        res.loc[
                            (res["period"] >= (interval * 5))
                            & (res["period"] < ((interval + 1) * 5))
                            & (res["health"] == health)
                            & (res["education"] == education),
                            ["income"],
                        ].sum()
                    ) / (
                        res.loc[
                            (res["period"] >= (interval * 5))
                            & (res["period"] < ((interval + 1) * 5))
                            & (res["health"] == health)
                            & (res["education"] == education),
                            "income",
                        ].count()
                    )
                    moments[(interval + 4 * (1 - health) + education * 4 * 2) + 46] = (
                        avg_income_10years.iloc[0] * winit[1] / 1000
                    )
    for interval in range(6):
        median_wealth_10y = res.loc[
            (res["period"] >= (interval * 5)) & (res["period"] < ((interval + 1) * 5)),
            ["wealth"],
        ].median()
        moments[interval + 32] = median_wealth_10y.iloc[0]
    avgemp_HS = (res.loc[(res["education"] == 0), ["working"]].sum() / 2) / (
        res.loc[(res["education"] == 0), "working"].count()
    )
    avgemp_CL = (res.loc[(res["education"] == 1), ["working"]].sum() / 2) / (
        res.loc[(res["education"] == 1), "working"].count()
    )
    moments[38] = avgemp_CL.iloc[0] / avgemp_HS.iloc[0]
    for interval in range(3):
        non_adjusters = (
            res.loc[
                (res["period"] >= (interval * 10))
                & (res["period"] < ((interval + 1) * 10))
                & (res["effort"] == res["effort_t_1"])
            ].count()
        ) / (
            res.loc[
                (res["period"] >= (interval * 10))
                & (res["period"] < ((interval + 1) * 10))
            ].count()
        )
        moments[interval + 39] = non_adjusters.iloc[0]
    avg_kappa = (
        (res.loc[(res["health"] == 1)].count())
        + (res.loc[(res["health"] == 0)].count()) * params["conp"]
    ) / (len(res))
    avg_cons = res["cnow"].mean()
    # In the new two-solve approach, utility is flow utility (no beta^period factor)
    avg_utility = res["utility"].mean()
    vsly = avg_utility / avg_kappa.iloc[0] * (avg_cons**-2)
    moments[42] = vsly
    std_effort = res["effort"].std()
    moments[43] = std_effort
    moments[44] = gini(jnp.asarray(res["wealth"].to_numpy()))
    cons_ratio = (
        res.loc[(res["health"] == 1), "cnow"].sum()
        / res.loc[(res["health"] == 1), "cnow"].count()
    ) / (
        res.loc[(res["health"] == 0), "cnow"].sum()
        / res.loc[(res["health"] == 0), "cnow"].count()
    )
    moments[45] = cons_ratio
    log_earnings = np.log(
        res.loc[(res["period"] <= retirement_age) & (res["working"] > 0), "income"]
        * theta_val[1]
    )
    moments[62] = log_earnings.var()
    pension_avg = (
        res.loc[(res["period"] == retirement_age + 1), "pension"].sum()
        / res.loc[(res["period"] == retirement_age + 1), "pension"].count()
    )
    avg_income = (
        res.loc[(res["period"] < retirement_age + 1), "income"].sum()
        / res.loc[(res["period"] < retirement_age + 1), "income"].count()
    )
    moments[63] = pension_avg / avg_income
    print(moments)
    return moments
