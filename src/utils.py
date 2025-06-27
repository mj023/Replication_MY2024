import jax
import jax.numpy as jnp
import lcm

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
    # The above formula, with all weights equal to 1 simplifies to:
    return (n + 1 - 2 * jnp.sum(cumx) / cumx[-1]) / n