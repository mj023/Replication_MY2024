
import jax.numpy as jnp

import numpy as np
from scipy.optimize import linprog

def rouwenhorst(rho,sigma_eps,n):

    mu_eps = 0

    q = (rho+1)/2
    nu = jnp.sqrt((n-1)/(1-rho**2)) * sigma_eps
    P = jnp.zeros((n,n))

    P = P.at[0,0].set(q)
    P = P.at[0,1].set(1-q)
    P = P.at[1,0].set(1-q)
    P = P.at[1,1].set(q)

    for i in range(2,n):
        P11 = jnp.zeros((i+1,i+1))
        P12 = jnp.zeros((i+1,i+1))
        P21 = jnp.zeros((i+1,i+1))
        P22 = jnp.zeros((i+1,i+1))

        P11= P11.at[0:i,0:i].set(P[0:i,0:i])
        P12 = P12.at[0:i,1:i+1].set(P[0:i,0:i])
        P21 = P21.at[1:i+1,0:i].set(P[0:i,0:i])
        P22 = P22.at[1:i+1,1:i+1].set(P[0:i,0:i])

        P=P.at[0:i+1,0:i+1].set(q*P11 + (1-q)*P12 + (1-q)*P21 + q*P22)
        P = P.at[1:i,:].set(P[1:i,:]/2)
    return jnp.linspace(mu_eps/(1.0-rho)-nu,mu_eps/(1.0-rho)+nu,n), P.T

def gini(x):
    sorted_x = jnp.sort(x)
    n = len(x)
    cumx = jnp.cumsum(sorted_x, dtype=float)

    return (n + 1 - 2 * jnp.sum(cumx) / cumx[-1]) / n

def transform_params(params):
    return {name:jnp.log(value + 0.2) for name,value in params.items()}
def retransform_params(params):
    return {name:jnp.exp(value)-0.2 for name,value in params.items()}


def qreg(y, X, tau):
    """
    Quantile regression using linear programming.

    Parameters:
        y (numpy array): Outcome vector (n,)
        X (numpy array): Predictor matrix (n, m)
        tau (float): Quantile level (between 0 and 1)

    Returns:
        bhat (numpy array): Estimated regression coefficients (m,)
    """

    n, m = X.shape
    print(m)
    # Objective function: tau * u + (1 - tau) * v + 0 * beta
    f = np.concatenate([tau * np.ones(n), (1 - tau) * np.ones(n), np.zeros(m)])

    # Equality constraints: u - v + X*beta = y => A_eq @ z = y
    A_eq = np.hstack([np.eye(n), -np.eye(n), X])
    b_eq = y

    # Bounds: u >= 0, v >= 0, beta unrestricted
    bounds = [(0, None)] * n + [(0, None)] * n + [(None, None)] * m

    # Solve linear program
    res = linprog(c=f, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not res.success:
        raise ValueError("Linear programming failed: " + res.message)

    # Extract beta coefficients from solution vector
    bhat = res.x[-m:]

    return bhat


