from Mahler_Yum_2024 import MODEL_CONFIG, PARAMS
from lcm.entry_point import get_lcm_function
import nvtx
from jaxlib import xla_client
import jax


def todotgraph(x):
   return xla_client._xla.hlo_module_to_dot_graph(xla_client._xla.hlo_module_from_text(x))

with nvtx.annotate("create", color = "blue"):
    solve , _ = get_lcm_function(model=MODEL_CONFIG,jit = True, targets="solve")

res = solve.lower(PARAMS).compile().as_text()
with open("t.dot", "w") as f:
    f.write(todotgraph(res))

print('Starting Solve.')

with nvtx.annotate("run", color = "red"):
    print(solve(PARAMS))