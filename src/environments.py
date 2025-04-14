"""JAX compatible version of Pendulum-v0 OpenAI gym environment.


Source: github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
"""

from typing import Any, Dict, Optional, Tuple, Union
from lcm.input_processing import process_model
import dataclasses as dc
import chex
from flax import struct
import jax
from jax import lax
import numpy as np
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces

ne            = 15
ssigma_eps    = 0.02058
llambda_eps   = 0.99
m             = 1.5
ssigma_y = jnp.sqrt(jnp.pow(ssigma_eps, 2) / (1 - jnp.pow(llambda_eps,2)))
estep = 2*ssigma_y*m / (ne-1)
mm = estep
egrid  = np.zeros(ne)
P  = np.zeros((ne, ne))
for i in range(0,ne):
    egrid[i] = (-m*jnp.sqrt(jnp.pow(ssigma_eps, 2) / (1 - jnp.pow(llambda_eps,2))) + i*estep)
for j in range(0,ne):
    for k in range(0,ne):
        if (k == 0):
            P[j, k] = jax.scipy.stats.norm.cdf((egrid[k] - llambda_eps*egrid[j] + (mm/2))/ssigma_eps)
        elif (k == ne-1):
            P[j, k] = 1 - jax.scipy.stats.norm.cdf((egrid[k] - llambda_eps*egrid[j] - (mm/2))/ssigma_eps)
        else:
            P[j, k] = jax.scipy.stats.norm.cdf((egrid[k] - llambda_eps*egrid[j] + (mm/2))/ssigma_eps) - jax.scipy.stats.norm.cdf((egrid[k] - llambda_eps*egrid[j] - (mm/2))/ssigma_eps)
P = jnp.array(P)
egrid = jnp.array(egrid)
egrid = jnp.exp(egrid)


@struct.dataclass
class EnvState(environment.EnvState):
    shock: jnp.ndarray
    savings: jnp.ndarray
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    P: jnp.ndarray = dc.field(default_factory=lambda: P)
    egrid: jnp.ndarray = dc.field(default_factory=lambda: egrid)
    max_steps_in_episode: int = 10
    ssigma: float = 2



class Paralell_Computing_Model(environment.Environment[EnvState, EnvParams]):
    """JAX Compatible version of Pendulum-v0 OpenAI gym environment."""

    def __init__(self):
        super().__init__()
        self.obs_shape = (3,)

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Pendulum-v0."""
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Integrate pendulum ODE and return transition."""
        consumption = jnp.clip(action, 0.1, 15.0)
        productivity_shock = jax.random.choice(key, jnp.arange(15), p= params.P[:,state.shock])
        income = income_func(params.egrid[productivity_shock])
        reward = u(state.savings, income, consumption, params.ssigma)
        reward = reward.squeeze()

        newsavings = state.savings + income - consumption
        newshock = productivity_shock

        # Update state dict and evaluate termination conditions
        state = EnvState(
            savings=newsavings,
            shock=newshock,
            time=state.time + 1,
        )
        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling theta, theta_dot."""
        high = jnp.array([4.0])
        low = jnp.array([0.1])
        savings = jax.random.uniform(key, minval=low, maxval=high)
        state = EnvState(
            savings=savings, shock=jax.random.randint(key, minval=0, maxval=14, shape=()), time=0
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Return angle in polar coordinates and change."""
        return jnp.array(
            [
                jnp.squeeze(state.savings),
                jnp.squeeze(state.shock),
            ]
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
        return jnp.array(done)

    @property
    def name(self) -> str:
        """Environment name."""
        return "Paralell_computing"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=0.1,
            high=4.0,
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array([4.0, 14.0], dtype=jnp.float32)
        return spaces.Box(0, high, shape=(2,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "savings": spaces.Box(
                    0,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "shock": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )


def u(savings, income, consumption, ssigma) -> jnp.ndarray:
    """Normalize the angle - radians."""
    
    return jnp.where(jnp.array((consumption<= income + savings*(1+0.04))), jnp.array((jnp.pow(consumption, 1-ssigma)-1)/(1-ssigma)), -100.0 )
def income_func(productivity_shock):
    print(egrid)
    return 5* productivity_shock


rng = jax.random.PRNGKey(54)
env = Paralell_Computing_Model()
params = env.default_params
_, state = env.reset_env(rng,params)
env.step_env(rng, state, 0.1, params)