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
from long_running import MODEL_CONFIG, PARAMS

def make_lcm_env(config, params):

    mod = process_model(config)
    vi = mod.variable_info
    




