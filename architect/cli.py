#!/usr/bin/env python3
"""architect -- answer the sizing and routing questions from the command line.

    python -m architect.cli size                      # what fits in the RAM budget
    python -m architect.cli throughput --dtype q4     # tokens/s for every candidate
    python -m architect.cli route --alpha 0.9         # tier + draft length
    python -m architect.cli spec --alpha 0.8          # speedup vs draft cost
    python -m architect.cli state                     # O(1) state vs KV memory
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import reference_specs
from .roofline import Machine, max_params_in_budget, plan_tiers, tokens_per_second
from .router import Foreman
from .spec_decode import speedup_table
from .state import kv_memory_bytes, state_memory_bytes

RESULTS = Path(__file__).resolve().parents[1] / "results" / "machine.json"


def load_machine(ram_gb: float) -> Machine:
    if RESULTS.exists():
        r = json.loads(RESULTS.read_text())
        k = r["kernel"]
        return Machine(read_gbs=k["roof_read_gbs"], copy_gbs=k["roof_copy_gbs"],
                       fma_gflops=k["roof_fma_gflops"], compute_gflops=k["ternary_gflops"],
                       disk_direct_gbs=k["disk_direct_gbs"], disk_buffered_gbs=k["disk_buffered_gbs"],
                       ram_bytes=ram_gb * 1e9)
    return Machine(ram_bytes=ram_gb * 1e9)


def main() -> int:
    ap = argparse.ArgumentParser(prog="architect")
    ap.add_argument("command", choices=["size", "throughput", "route", "spec", "state"])
    ap.add_argument("--ram-gb", type=float, default=8.0)
    ap.add_argument("--dtype", default="q4")
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--seq-len", type=int, default=4096)
    args = ap.parse_args()

    m = load_machine(args.ram_gb)
    specs = reference_specs()

    if args.command == "size":
        kv = specs["qwen2.5-coder-7b"].kv_bytes(args.seq_len)
        print(f"RAM budget {args.ram_gb} GB, runtime 0.6 GB, 7B KV at {args.seq_len}: {kv/1e9:.3f} GB")
        for b in (8, 4, 2):
            print(f"  {b}-bit: {max_params_in_budget(m, b, kv)/1e9:.2f} B params fit")
        print("\nPlans:")
        for name, (t, d, td, dd) in {
            "resident 7B q4 + 1.1B q2": ("qwen2.5-coder-7b", "tinyllama-1.1b", "q4", "q2"),
            "resident Kimi-Linear q4 + 1.1B q2": ("kimi-linear-48b-a3b", "tinyllama-1.1b", "q4", "q2"),
            "resident ternary 27B q2 + 1.1B q2": ("ternary-bonsai-27b", "tinyllama-1.1b", "q2", "q2"),
            "three-tier 33B architect (SSD)": ("deepseek-coder-33b", "qwen2.5-coder-7b", "q4", "q4"),
        }.items():
            p = plan_tiers(specs[t], specs[d], m, td, dd, args.seq_len)
            print(f"  {name}: {p['total_gb']:.2f} GB fits={p['fits']}")

    elif args.command == "throughput":
        for key, spec in specs.items():
            r = tokens_per_second(spec, args.dtype, m, args.seq_len)
            print(f"{key:22s} {r['weight_gb']:7.2f} GB  {r['bound']:8s}  {r['tokens_per_s']:7.2f} tok/s")

    elif args.command == "route":
        f = Foreman(specs["qwen2.5-coder-7b"], specs["tinyllama-1.1b"], m, "q4", "q2")
        r = f.route(args.alpha, args.seq_len)
        print(json.dumps(vars(r), indent=2))

    elif args.command == "spec":
        for row in speedup_table(args.alpha):
            print(json.dumps(row))

    elif args.command == "state":
        s = specs["kimi-linear-48b-a3b"]
        print(f"linear state: {state_memory_bytes(s.n_layer, s.n_head, s.d_head):,} bytes (flat in T)")
        for T in (4096, 8192, 32768, 131072):
            print(f"  KV at {T:6d}: {kv_memory_bytes(s.n_layer, s.n_head, s.d_head, T):>14,} bytes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
