"""Roofline cost model for a decode step, plus the three-tier planner.

All formulas are derived in `docs/MATH.md` Sections 1 and 6. This module does no
guessing: every rate it uses is passed in from a measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .model import ModelSpec, bytes_per_weight


@dataclass
class Machine:
    """Measured rates for one machine. Defaults are this box; overwrite from results/machine.json."""

    read_gbs: float = 35.4
    copy_gbs: float = 21.2
    fma_gflops: float = 108.7          # peak fp32 FMA rate, the compute roof of Section 1
    compute_gflops: float = 57.4       # achieved int8 VNNI mat-vec rate (ternary is 17.9)
    disk_direct_gbs: float = 0.177
    disk_buffered_gbs: float = 15.4
    ram_bytes: float = 8e9
    n_cores: int = 2

    @property
    def ridge_flop_per_byte(self) -> float:
        return self.fma_gflops / self.read_gbs


def bytes_per_token_stream(weight_bytes: float, kv_bytes_per_token: float) -> float:
    """Bytes that must cross the memory bus for one token at batch 1."""
    return weight_bytes + kv_bytes_per_token


def intensity(flops: float, byte_stream: float) -> float:
    return flops / byte_stream if byte_stream > 0 else float("inf")


def decode_time(weight_bytes: float, flops: float, machine: Machine,
                kv_bytes_per_token: float = 0.0, dtype: str = "fp16") -> float:
    """Seconds for one decode step, the max of the memory roof and the compute roof.

    The compute roof depends on the arithmetic: fp16/bf16 weights run on the FMA
    path (roof = `fma_gflops`), quantised weights run on the bit-sliced integer
    path (roof = `compute_gflops`, measured by the kernel, not assumed).
    """
    stream = bytes_per_token_stream(weight_bytes, kv_bytes_per_token)
    t_mem = stream / (machine.read_gbs * 1e9)
    peak = machine.fma_gflops if dtype in ("fp16", "bf16", "fp32") else machine.compute_gflops
    t_cmp = flops / (peak * 1e9)
    return max(t_mem, t_cmp)


def tokens_per_second(spec: ModelSpec, dtype: str, machine: Machine,
                      seq_len: int = 4096, kv_dtype: str = "fp16") -> dict:
    """Decode throughput for one model held at `dtype`, with its real per-token KV traffic."""
    w = spec.active_bytes(dtype)
    fl = spec.flops_per_token()
    kv_total = spec.kv_bytes(seq_len, kv_dtype)
    kv_per_tok = 0.0 if spec.attention != "full" else (
        spec.n_layer * 2 * spec.effective_kv_head * spec.d_head * bytes_per_weight(kv_dtype))
    stream = bytes_per_token_stream(w, kv_per_tok)
    t_mem = stream / (machine.read_gbs * 1e9)
    t = decode_time(w, fl, machine, kv_per_tok, dtype=dtype)
    return {
        "model": spec.name,
        "dtype": dtype,
        "attention": spec.attention,
        "weight_gb": round(w / 1e9, 3),
        "kv_total_gb": round(kv_total / 1e9, 3),
        "flops_per_token_gflop": round(fl / 1e9, 3),
        "intensity_flop_per_byte": round(intensity(fl, stream), 3),
        "bound": "memory" if t == t_mem else "compute",
        "s_per_token": round(t, 6),
        "tokens_per_s": round(1.0 / t, 3),
        "streamed_from_ssd_tokens_per_s": round(machine.disk_direct_gbs * 1e9 / w, 6) if w else None,
    }


def fits(weight_bytes: float, kv_bytes: float, machine: Machine,
         runtime_bytes: float = 0.6e9) -> bool:
    return weight_bytes + kv_bytes + runtime_bytes <= machine.ram_bytes


def plan_tiers(target: ModelSpec, draft: ModelSpec, machine: Machine,
               target_dtype: str, draft_dtype: str,
               seq_len: int = 4096, kv_dtype: str = "fp16") -> dict:
    """Check that a resident target + resident draft fit, and estimate the pair's speed.

    Speed uses the speculation identity from docs/MATH.md Section 3. The caller
    supplies the acceptance rate, because it depends on the models, not the box.
    """
    tw = target.active_bytes(target_dtype)
    dw = draft.active_bytes(draft_dtype)
    # KV is carried by the target only; the draft is constant-state in the intended design.
    kv = target.kv_bytes(seq_len, kv_dtype)
    total = tw + dw + kv + 0.6e9
    t_target = decode_time(tw, target.flops_per_token(), machine, dtype=target_dtype)
    t_draft = decode_time(dw, draft.flops_per_token(), machine, dtype=draft_dtype)
    return {
        "target": target.name,
        "draft": draft.name,
        "target_gb": round(tw / 1e9, 3),
        "draft_gb": round(dw / 1e9, 3),
        "kv_gb": round(kv / 1e9, 3),
        "total_gb": round(total / 1e9, 3),
        "ram_gb": round(machine.ram_bytes / 1e9, 3),
        "fits": total <= machine.ram_bytes,
        "draft_relative_cost": round(t_draft / t_target, 4) if t_target else None,
        "target_tokens_per_s": round(1.0 / t_target, 3),
    }


def max_params_in_budget(machine: Machine, bits_per_weight: float, kv_bytes: float = 0.0,
                         runtime_bytes: float = 0.6e9) -> float:
    """Largest parameter count that fits, at a given quantisation, in the RAM budget."""
    free = machine.ram_bytes - runtime_bytes - kv_bytes
    return free / (bits_per_weight / 8.0)
