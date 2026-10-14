"""The Foreman: pick the tier and the draft length from measured quantities.

The referenced IDE routes with keyword rules. That is a fixed, unmeasurable
policy. Here the routing decision is driven by the two numbers that actually
decide throughput — the acceptance rate alpha (how close the cheap tier is to the
target) and the draft's relative cost c — and by context pressure from the real
KV arithmetic. Given alpha and c, the optimal draft length is not a guess; it is
`spec_decode.optimal_gamma`.
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import ModelSpec
from .roofline import Machine, decode_time, fits
from .spec_decode import optimal_gamma, speedup


@dataclass
class Route:
    tier: str
    gamma: int
    predicted_speedup: float
    reason: str


class Foreman:
    """Choose between a resident target, a speculative pair, and a plain pass."""

    def __init__(self, target: ModelSpec, draft: ModelSpec, machine: Machine,
                 target_dtype: str = "q4", draft_dtype: str = "q2"):
        self.target = target
        self.draft = draft
        self.machine = machine
        self.target_dtype = target_dtype
        self.draft_dtype = draft_dtype
        self._t_target = decode_time(target.active_bytes(target_dtype),
                                     target.flops_per_token(), machine, dtype=target_dtype)
        self._t_draft = decode_time(draft.active_bytes(draft_dtype),
                                    draft.flops_per_token(), machine, dtype=draft_dtype)
        self.c = self._t_draft / self._t_target if self._t_target else 1.0

    def context_pressure(self, seq_len: int) -> float:
        """Fraction of the RAM budget the target's attention state would use at this length."""
        kv = self.target.kv_bytes(seq_len)
        return kv / self.machine.ram_bytes

    def route(self, alpha: float, seq_len: int, max_context_pressure: float = 0.5) -> Route:
        """alpha is the measured/predicted agreement between draft and target."""
        pressure = self.context_pressure(seq_len)
        if pressure > max_context_pressure:
            return Route("target-only", 0, 1.0,
                         f"attention state would use {pressure:.0%} of RAM at len {seq_len}; "
                         "speculation cannot pay for a cache it cannot hold")
        if self.c >= 1.0:
            return Route("target-only", 0, 1.0,
                         f"draft costs {self.c:.2f}x the target; a draft that expensive is a loss")
        g = optimal_gamma(alpha, self.c)
        s = speedup(alpha, g, self.c)
        if s <= 1.0:
            return Route("target-only", 0, 1.0,
                         f"agreement {alpha:.2f} too low for any draft length to beat one pass")
        return Route("speculative", g, round(s, 3),
                     f"alpha={alpha:.2f}, c={self.c:.3f} -> gamma*={g}, S={s:.2f}")

    def report(self) -> dict:
        return {
            "target": self.target.name,
            "draft": self.draft.name,
            "target_tokens_per_s": round(1.0 / self._t_target, 3),
            "draft_tokens_per_s": round(1.0 / self._t_draft, 3),
            "draft_relative_cost": round(self.c, 4),
        }
