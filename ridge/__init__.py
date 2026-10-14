"""ridge -- run a coding model on 8 GB of RAM, no CUDA, from arithmetic.

The package turns the derivations in `docs/MATH.md` into code:

  model     geometry and the exact RAM budget
  roofline  bytes/flops per token, which roof binds, tokens per second
  spec_decode  the acceptance identity and the optimal draft length
  jepa      a latent predictor and the readout that detects its collapse
  state     belief state (Kalman, reservoir) and its flat memory
  router    tier + draft length from measured agreement and context pressure

Nothing here downloads weights. It sizes, predicts and verifies.
"""
from .model import ModelSpec, bytes_per_weight, reference_specs, spec_from_hf_config
from .roofline import Machine, decode_time, tokens_per_second, plan_tiers, max_params_in_budget
from .spec_decode import (acceptance_rate, expected_tokens, speedup, optimal_gamma,
                          optimal_gamma_analytic, simulate, speedup_table, total_variation)
from .jepa import hankel, hankel_rank_gap, dmd, participation_ratio, train_latent, make_rank_r_series
from .state import kalman, steady_state_gain, Reservoir, state_memory_bytes, kv_memory_bytes
from .router import Foreman

__all__ = [
    "ModelSpec", "bytes_per_weight", "reference_specs", "spec_from_hf_config",
    "Machine", "decode_time", "tokens_per_second", "plan_tiers", "max_params_in_budget",
    "acceptance_rate", "expected_tokens", "speedup", "optimal_gamma", "optimal_gamma_analytic",
    "simulate", "speedup_table", "total_variation",
    "hankel", "hankel_rank_gap", "dmd", "participation_ratio", "train_latent", "make_rank_r_series",
    "kalman", "steady_state_gain", "Reservoir", "state_memory_bytes", "kv_memory_bytes",
    "Foreman",
]
__version__ = "0.1.0"
