#!/usr/bin/env python3
"""Measure this machine, size the models against it, and write results/machine.json.

Nothing here is hard-coded: the roofs come from the C kernel, the model geometry
comes from the HuggingFace hub (cached), and the throughput and budget come from
the arithmetic in `architect`.

    python bench/run.py            # full: build the kernel, probe the hub, write results
    python bench/run.py --no-net   # reuse the cached hub data
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from architect.model import reference_specs, spec_as_dict  # noqa: E402
from architect.roofline import (  # noqa: E402
    Machine, decode_time, max_params_in_budget, plan_tiers, tokens_per_second,
)
from architect.spec_decode import optimal_gamma, speedup, speedup_table  # noqa: E402
from architect.router import Foreman  # noqa: E402
from architect.jepa import make_rank_r_series, train_latent  # noqa: E402
from architect.state import kv_memory_bytes, state_memory_bytes  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Models worth sizing, and the hub ids for the ones we probe live.
HF_REPOS = [
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen3-32B",
    "deepseek-ai/deepseek-coder-33b-instruct",
    "moonshotai/Kimi-Linear-48B-A3B-Instruct",
    "microsoft/bitnet-b1.58-2B-4T",
    "BlinkDL/rwkv7-g1",
]


def build_kernel() -> Path:
    binary = ROOT / "bench" / "architect_kernels"
    src = ROOT / "kernel" / "architect_kernels.c"
    if not binary.exists() or binary.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(
            ["gcc", "-O3", "-march=native", "-o", str(binary), str(src), "-lm", "-lpthread"],
            check=True,
        )
    return binary


def run_kernel(binary: Path) -> dict:
    out = subprocess.run([str(binary), str(ROOT)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def probe_hub(use_net: bool) -> list[dict]:
    cache = RESULTS / "hf_models.json"
    if not use_net and cache.exists():
        return json.loads(cache.read_text())
    try:
        out = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "hf_probe.py"), *HF_REPOS],
            capture_output=True, text=True, check=True, timeout=300,
        )
        data = json.loads(out.stdout)
        cache.write_text(json.dumps(data, indent=2))
        return data
    except Exception as e:  # noqa: BLE001
        print(f"[bench] hub probe failed ({e}); using cache if present", file=sys.stderr)
        return json.loads(cache.read_text()) if cache.exists() else []


def machine_from_kernel(k: dict, ram_bytes: float) -> Machine:
    return Machine(
        read_gbs=k["roof_read_gbs"],
        copy_gbs=k["roof_copy_gbs"],
        fma_gflops=k["roof_fma_gflops"],
        compute_gflops=k["ternary_gflops"],
        disk_direct_gbs=k["disk_direct_gbs"],
        disk_buffered_gbs=k["disk_buffered_gbs"],
        ram_bytes=ram_bytes,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-net", action="store_true", help="reuse cached hub data")
    ap.add_argument("--ram-gb", type=float, default=8.0, help="RAM budget to plan against")
    args = ap.parse_args()

    binary = build_kernel()
    kern = run_kernel(binary)
    (RESULTS / "kernels_raw.json").write_text(json.dumps(kern, indent=2))

    ram = args.ram_gb * 1e9
    m = machine_from_kernel(kern, ram)

    hub = probe_hub(not args.no_net)
    specs = reference_specs()

    # 1. throughput of every candidate at every quantisation
    throughput = []
    for key, spec in specs.items():
        for dtype in ("fp16", "q4", "q2"):
            row = tokens_per_second(spec, dtype, m, seq_len=4096)
            row["key"] = key
            throughput.append(row)

    # 2. the RAM ceiling
    kv7b = specs["qwen2.5-coder-7b"].kv_bytes(4096)
    ceilings = {
        "no_kv": {b: max_params_in_budget(m, b, 0.0) for b in (8, 4, 2)},
        "with_7b_kv_4k": {b: max_params_in_budget(m, b, kv7b) for b in (8, 4, 2)},
    }

    # 3. does the three-tier plan fit, and what is the resident alternative
    plans = {
        "three_tier_ssd_architect_33b": plan_tiers(
            specs["deepseek-coder-33b"], specs["qwen2.5-coder-7b"], m, "q4", "q4", 4096),
        "three_tier_ssd_architect_32b": plan_tiers(
            specs["qwen3-32b"], specs["qwen2.5-coder-7b"], m, "q4", "q4", 4096),
        "resident_7b_q4_plus_1b_q2": plan_tiers(
            specs["qwen2.5-coder-7b"], specs["tinyllama-1.1b"], m, "q4", "q2", 4096),
        "resident_kimi_linear_q4": plan_tiers(
            specs["kimi-linear-48b-a3b"], specs["tinyllama-1.1b"], m, "q4", "q2", 4096),
        "resident_ternary_27b_q2": plan_tiers(
            specs["ternary-bonsai-27b"], specs["tinyllama-1.1b"], m, "q2", "q2", 4096),
    }

    # 4. speculation tables for a few agreement levels
    spec_tables = {str(a): speedup_table(a) for a in (0.6, 0.7, 0.8, 0.9)}

    # 5. the routing decision for a realistic pair
    foreman = Foreman(specs["qwen2.5-coder-7b"], specs["tinyllama-1.1b"], m, "q4", "q2")
    routes = {
        "alpha_0.9_len_4096": vars(foreman.route(0.9, 4096)),
        "alpha_0.7_len_4096": vars(foreman.route(0.7, 4096)),
        "alpha_0.9_len_60000": vars(foreman.route(0.9, 60000)),
    }

    # 6. the collapse experiment, several seeds
    jepa_rows = []
    X = make_rank_r_series(T=2000, m=8, freqs=(0.20, 0.35), noise=0.05, seed=0)
    for seed in range(3):
        for mode in ("none", "vicreg", "koopman_dim"):
            _, _, _, rep = train_latent(X, d_lat=8, mode=mode, r_star=4, steps=1500, seed=seed)
            jepa_rows.append({
                "mode": mode, "seed": seed,
                "latent_energy": round(rep.latent_energy, 5),
                "participation_ratio": round(rep.participation_ratio, 3),
                "next_step_r2": round(rep.next_step_r2, 4),
                "hankel_rank": rep.hankel_rank,
                "dmd_abs_lambda": rep.dmd_abs_lambda,
            })

    # 7. state memory, the two regimes
    L, H, dh = specs["kimi-linear-48b-a3b"].n_layer, specs["kimi-linear-48b-a3b"].n_head, specs["kimi-linear-48b-a3b"].d_head
    state_mem = {
        "linear_state_bytes": state_memory_bytes(L, H, dh),
        "kv_bytes_4k": kv_memory_bytes(L, H, dh, 4096),
        "kv_bytes_32k": kv_memory_bytes(L, H, dh, 32768),
        "kv_bytes_128k": kv_memory_bytes(L, H, dh, 131072),
    }

    report = {
        "generated_by": "bench/run.py",
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "kernel": kern,
        "machine": {
            "read_gbs": m.read_gbs, "copy_gbs": m.copy_gbs, "fma_gflops": m.fma_gflops,
            "disk_direct_gbs": m.disk_direct_gbs, "disk_buffered_gbs": m.disk_buffered_gbs,
            "ridge_flop_per_byte": round(m.ridge_flop_per_byte, 3),
            "ram_budget_gb": args.ram_gb,
        },
        "hub_models": [
            {"repo": h.get("repo"), "total_weight_gb": h.get("total_weight_gb"),
             "config": h.get("config"), "license": h.get("license"),
             "downloads": h.get("downloads")}
            for h in hub
        ],
        "specs": {k: spec_as_dict(v) for k, v in specs.items()},
        "throughput": throughput,
        "ram_ceiling_params": ceilings,
        "plans": plans,
        "spec_decode": spec_tables,
        "routes": routes,
        "jepa_collapse": jepa_rows,
        "state_memory": state_mem,
    }
    (RESULTS / "machine.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("machine", "ram_ceiling_params")}, indent=2))
    print(f"\nwrote {RESULTS / 'machine.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
