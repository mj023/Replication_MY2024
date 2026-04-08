"""Example specification for a consumption-savings model with health and exercise."""

from dataclasses import make_dataclass
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import lcm
import numpy as np
from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    Period,
)
from lcm.utils.dispatchers import productmap

_DATA_DIR = Path(__file__).parent

# --------------------------------------------------------------------------------------
# Fixed Parameters
# --------------------------------------------------------------------------------------
avrgearn = 57706.57
theta_val = jnp.array([jnp.exp(-0.2898), jnp.exp(0.2898)])
ages = AgeGrid(start=25, stop=101, step="2Y")
n = ages.n_periods
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


def calc_savingsgrid(x):
    x = ((jnp.log(10.0**2) - jnp.log(10.0**0)) / 49) * x
    x = jnp.exp(x)
    xgrid = x - 10.0 ** (0.0)
    xgrid = xgrid / (10.0**2 - 10.0**0.0)
    return xgrid * (30 - 0) + 0


# --------------------------------------------------------------------------------------
# Categorical variables
# --------------------------------------------------------------------------------------
@categorical(ordered=True)
class WorkingStatus:
    retired: int
    part: int
    full: int


@categorical(ordered=True)
class EducationStatus:
    low: int
    high: int


Effort = make_dataclass(
    "HealthEffort", [("class" + str(i), int, int(i)) for i in range(40)]
)


@categorical(ordered=True)
class Health:
    bad: int
    good: int


@categorical(ordered=True)
class ProductivityType:
    low: int
    high: int


@categorical(ordered=True)
class HealthType:
    low: int
    high: int


@categorical(ordered=False)
class RegimeId:
    alive: int
    dead: int


eff_grid = jnp.linspace(0, 1, 40)

prod_shock_grid = lcm.shocks.ar1.Rouwenhorst(n_points=5, rho=rho, mu=0, sigma=1)

SIGMA = 2.0

const_healthtr: float = -0.906
age_const = jnp.asarray([0.0, -0.289, -0.644, -0.881, -1.138, -1.586, -1.586, -1.586])
eff_param = jnp.asarray([0.693, 0.734])
healthy_dummy: float = 2.311
htype_dummy: float = 0.632
college_dummy: float = 0.238


def _health_trans(period, health, eff, eff_1, edu, ht):
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
_mapped_health_trans = productmap(
    func=_health_trans,
    variables=_health_trans_variables,
    batch_sizes=dict.fromkeys(_health_trans_variables, 0),
)

tr2yp_grid = jnp.zeros((38, 2, 40, 40, 2, 2, 2))
_j = jnp.floor_divide(jnp.arange(38), 5)
tr2yp_grid = tr2yp_grid.at[:, :, :, :, :, :, 1].set(
    _mapped_health_trans(
        period=_j,
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


# --------------------------------------------------------------------------------------
# Utility function
# --------------------------------------------------------------------------------------
def utility(
    scaled_adjustment_cost: FloatND,
    fcost: FloatND,
    disutil: FloatND,
    cons_util: FloatND,
) -> FloatND:
    return cons_util - disutil - fcost - scaled_adjustment_cost


def disutil(
    working: DiscreteAction,
    health: DiscreteState,
    education: DiscreteState,
    period: Period,
    phigrid: FloatND,
) -> FloatND:
    return phigrid[period, education, health] * ((working / 2) ** (2)) / 2


def scaled_adjustment_cost(
    period: Period,
    adjustment_cost: ContinuousState,
    effort: DiscreteAction,
    effort_t_1: DiscreteState,
    chimaxgrid: FloatND,
) -> FloatND:
    return jnp.where(
        jnp.logical_not(effort == effort_t_1),
        adjustment_cost * chimaxgrid[period],
        0,
    )


def cnow(
    net_income: FloatND, wealth: ContinuousState, saving: ContinuousAction
) -> FloatND:
    wealth = calc_savingsgrid(wealth)
    saving = calc_savingsgrid(saving)
    return jnp.maximum(net_income + (wealth) * r - (saving), mincon)


def cons_util(
    health: DiscreteState, cnow: FloatND, kappa: float, sigma: float, bb: float
) -> FloatND:
    mucon = jnp.where(health, 1, kappa)
    return mucon * (((cnow) ** (1.0 - sigma)) / (1.0 - sigma)) + mucon * bb


def fcost(
    period: Period,
    education: DiscreteState,
    health: DiscreteState,
    effort: DiscreteAction,
    psi: float,
    xigrid: FloatND,
) -> FloatND:
    return (
        xigrid[period, education, health]
        * (eff_grid[effort] ** (1 + (1 / psi)))
        / (1 + (1 / psi))
    )


# --------------------------------------------------------------------------------------
# Income Calculation
# --------------------------------------------------------------------------------------
def net_income(benefits: FloatND, taxed_income: FloatND, pension: FloatND) -> FloatND:

    return taxed_income + pension + benefits


def scaled_productivity_shock(
    productivity_shock: ContinuousState, sigx: float
) -> FloatND:
    return productivity_shock * sigx


def income(
    working: DiscreteAction,
    period: Period,
    health: DiscreteState,
    education: DiscreteState,
    productivity: DiscreteState,
    scaled_productivity_shock: FloatND,
    income_grid: FloatND,
) -> FloatND:
    return (
        income_grid[period, health, education]
        * (working / 2)
        * theta_val[productivity]
        * jnp.exp(scaled_productivity_shock)
    )


def taxed_income(income: FloatND) -> FloatND:
    return lamda * (income ** (1.0 - taul)) * (avrgearn**taul)


def benefits(period: Period, health: DiscreteState, working: DiscreteAction) -> FloatND:
    eligible = jnp.logical_and(health == 0, working == 0)
    return jnp.where(
        jnp.logical_and(eligible, period <= retirement_age), tt0 * avrgearn, 0
    )


def pension(
    period: Period,
    education: DiscreteState,
    productivity: DiscreteState,
    income_grid: FloatND,
    penre: float,
) -> FloatND:
    return jnp.where(
        period > retirement_age,
        income_grid[19, 1, education] * theta_val[productivity] * penre,
        0,
    )


# --------------------------------------------------------------------------------------
# State transitions
# --------------------------------------------------------------------------------------
def next_wealth(saving: ContinuousAction) -> ContinuousState:
    return saving


def next_health(
    period: Period,
    health: DiscreteState,
    effort: DiscreteAction,
    effort_t_1: DiscreteState,
    education: DiscreteState,
    health_type: DiscreteState,
    probs_array: FloatND,
) -> FloatND:
    return probs_array[period, health, effort, effort_t_1, education, health_type]


def next_effort_t_1(effort: DiscreteAction) -> DiscreteState:
    return effort


# --------------------------------------------------------------------------------------
# Regime Transitions
# --------------------------------------------------------------------------------------

surv_HS = jnp.array(np.loadtxt(_DATA_DIR / "surv_HS.txt"))
surv_CL = jnp.array(np.loadtxt(_DATA_DIR / "surv_CL.txt"))
spgrid = jnp.zeros((38, 2, 2))
spgrid = spgrid.at[:, 0, 0].set(surv_HS[:, 1])
spgrid = spgrid.at[:, 1, 0].set(surv_CL[:, 1])
spgrid = spgrid.at[:, 0, 1].set(surv_HS[:, 0])
spgrid = spgrid.at[:, 1, 1].set(surv_CL[:, 0])
# Certain death at the terminal period (age 101 is inactive for alive)
spgrid = spgrid.at[-1].set(0.0)


def next_regime(
    period: Period,
    education: DiscreteState,
    health: DiscreteState,
    probs_array: FloatND,
) -> FloatND:
    """Return probability array [P(alive), P(dead)] indexed by RegimeId."""
    survival_prob = probs_array[period, education, health]
    return jnp.array([survival_prob, 1 - survival_prob])


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def retirement_constraint(period: Period, working: DiscreteAction) -> BoolND:
    return jnp.logical_not(jnp.logical_and(period > retirement_age, working > 0))


def savings_constraint(
    net_income: FloatND, wealth: ContinuousState, saving: ContinuousAction
) -> BoolND:
    wealth = calc_savingsgrid(wealth)
    saving = calc_savingsgrid(saving)
    return net_income + (wealth) * r >= (saving)


def alive_is_active(age: int, final_age_alive: float) -> bool:
    return age <= final_age_alive


def dead_is_active(age: int, initial_age: float) -> bool:
    return age > initial_age


# ======================================================================================
# Model specification and parameters
# ======================================================================================


ALIVE_REGIME = Regime(
    transition=MarkovTransition(next_regime),
    active=partial(alive_is_active, final_age_alive=ages.values[-2]),
    states={
        "wealth": LinSpacedGrid(start=0, stop=49, n_points=50),
        "health": DiscreteGrid(Health),
        "productivity_shock": prod_shock_grid,
        "effort_t_1": DiscreteGrid(Effort),
        "adjustment_cost": lcm.shocks.iid.Uniform(n_points=5, start=0, stop=1),
        "education": DiscreteGrid(EducationStatus),
        "productivity": DiscreteGrid(ProductivityType),
        "health_type": DiscreteGrid(HealthType),
    },
    state_transitions={
        "wealth": next_wealth,
        "health": MarkovTransition(next_health),
        "effort_t_1": next_effort_t_1,
        "education": None,
        "productivity": None,
        "health_type": None,
    },
    actions={
        "working": DiscreteGrid(WorkingStatus),
        "saving": LinSpacedGrid(start=0, stop=49, n_points=50),
        "effort": DiscreteGrid(Effort),
    },
    functions={
        "utility": utility,
        "disutil": disutil,
        "fcost": fcost,
        "cons_util": cons_util,
        "cnow": cnow,
        "income": income,
        "benefits": benefits,
        "scaled_adjustment_cost": scaled_adjustment_cost,
        "net_income": net_income,
        "taxed_income": taxed_income,
        "pension": pension,
        "scaled_productivity_shock": scaled_productivity_shock,
    },
    constraints={
        "retirement_constraint": retirement_constraint,
        "savings_constraint": savings_constraint,
    },
)

DEAD_REGIME = Regime(
    transition=None,
    active=partial(dead_is_active, initial_age=ages.values[0]),
    functions={"utility": lambda: 0.0},
)

MAHLER_YUM_MODEL = Model(
    regimes={"alive": ALIVE_REGIME, "dead": DEAD_REGIME},
    ages=ages,
    regime_id_class=RegimeId,
    fixed_params={
        "alive": {
            "cons_util": {"sigma": SIGMA},
            "next_health": {"probs_array": tr2yp_grid},
            "next_regime": {"probs_array": spgrid},
        },
    },
)


########################
# Mahler & Yum Params  #
########################
winit = jnp.array([43978, 48201])

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

xi_HSh = jnp.array([xiHSh_1, xiHSh_2, xiHSh_3, xiHSh_4])
xi_HSu = jnp.array([xiHSu_1, xiHSu_2, xiHSu_3, xiHSu_4])
xi_CLu = jnp.array([xiCLu_1, xiCLu_2, xiCLu_3, xiCLu_4])
xi_CLh = jnp.array([xiCLh_1, xiCLh_2, xiCLh_3, xiCLh_4])

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

START_PARAMS = {
    "nuh_1": nuh_1,
    "nuh_2": nuh_2,
    "nuh_3": nuh_3,
    "nuh_4": nuh_4,
    "nuu_1": nuu_1,
    "nuu_2": nuu_2,
    "nuu_3": nuu_3,
    "nuu_4": nuu_4,
    "nuad": nuad,
    "xiHSh_1": xiHSh_1,
    "xiHSh_2": xiHSh_2,
    "xiHSh_3": xiHSh_3,
    "xiHSh_4": xiHSh_4,
    "xiHSu_1": xiHSu_1,
    "xiHSu_2": xiHSu_2,
    "xiHSu_3": xiHSu_3,
    "xiHSu_4": xiHSu_4,
    "xiCLu_1": xiCLu_1,
    "xiCLu_2": xiCLu_2,
    "xiCLu_3": xiCLu_3,
    "xiCLu_4": xiCLu_4,
    "xiCLh_1": xiCLh_1,
    "xiCLh_2": xiCLh_2,
    "xiCLh_3": xiCLh_3,
    "xiCLh_4": xiCLh_4,
    "y1_HS": y1_HS,
    "ytHS_s": ytHS_s,
    "ytHS_sq": ytHS_sq,
    "wagep_HS": wagep_HS,
    "y1_CL": y1_CL,
    "ytCL_s": ytCL_s,
    "ytCL_sq": ytCL_sq,
    "wagep_CL": wagep_CL,
    "sigx": sigx,
    "chi_1": chi_1,
    "chi_2": chi_2,
    "psi": psi,
    "bb": 11,
    "conp": conp,
    "penre": penre,
    "beta_mean": beta_mean,
    "beta_std": beta_std,
}
