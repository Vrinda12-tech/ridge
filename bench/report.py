#!/usr/bin/env python3
"""Turn results/machine.json into results/MEASUREMENTS.md.

Every table is generated from the measurement, so the document cannot drift from
the numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = json.loads((ROOT / "results" / "machine.json").read_text())


def main() -> int:
    k = M["kernel"]
    m = M["machine"]
    out = []
    out.append("# Measured results\n")
    out.append("Generated from `results/machine.json` by `bench/report.py`. "
               "Host: " + M["host"]["platform"] + ".\n")

    out.append("## Machine roofs\n")
    out.append("| roof | measured |\n|---|---|")
    out.append(f"| read bandwidth | {m['read_gbs']:.2f} GB/s |")
    out.append(f"| copy bandwidth (read+write) | {m['copy_gbs']:.2f} GB/s |")
    out.append(f"| fp32 FMA peak | {m['fma_gflops']:.1f} GFLOP/s |")
    out.append(f"| bit-sliced ternary mat-vec | {k['ternary_gflops']:.2f} GFLOP/s |")
    out.append(f"| ternary packed stream | {k['ternary_packed_gbs']:.2f} GB/s |")
    out.append(f"| fp32 FMA ridge | {m['ridge_flop_per_byte']:.2f} flop/byte |")
    out.append(f"| ternary ridge | {k['ternary_ridge_flop_per_byte']:.2f} flop/byte |")
    out.append(f"| NVMe O_DIRECT read | {m['disk_direct_gbs']:.3f} GB/s |")
    out.append(f"| page-cache read | {m['disk_buffered_gbs']:.2f} GB/s |")
    out.append(f"| ternary kernel max abs error vs reference | {k['ternary_max_abs_err']:.2e} |\n")

    out.append("## Decode throughput, resident weights\n")
    out.append("| model | quant | weights GB | KV GB | flop/byte | bound | tok/s |")
    out.append("|---|---|---|---|---|---|---|")
    for t in M["throughput"]:
        out.append(f"| {t['key']} | {t['dtype']} | {t['weight_gb']:.2f} | {t['kv_total_gb']:.2f} | "
                   f"{t['intensity_flop_per_byte']:.2f} | {t['bound']} | {t['tokens_per_s']:.2f} |")
    out.append("")

    out.append("## RAM ceiling (params that fit an 8 GB budget)\n")
    out.append("| quantisation | no KV | with 7B/4k KV |\n|---|---|---|")
    for b in ("8", "4", "2"):
        out.append(f"| {b}-bit | {M['ram_ceiling_params']['no_kv'][b]/1e9:.2f} B | "
                   f"{M['ram_ceiling_params']['with_7b_kv_4k'][b]/1e9:.2f} B |")
    out.append("")

    out.append("## Tier plans\n")
    out.append("| plan | total GB | fits | draft cost c | target tok/s |\n|---|---|---|---|---|")
    for name, p in M["plans"].items():
        out.append(f"| {name} | {p['total_gb']:.2f} | {p['fits']} | {p['draft_relative_cost']} | "
                   f"{p['target_tokens_per_s']} |")
    out.append("")

    out.append("## Routing decisions (speculation)\n")
    out.append("| case | tier | gamma | predicted speedup | reason |\n|---|---|---|---|---|")
    for name, r in M["routes"].items():
        out.append(f"| {name} | {r['tier']} | {r['gamma']} | {r['predicted_speedup']} | {r['reason']} |")
    out.append("")

    out.append("## JEPA latent: collapse control\n")
    out.append("| mode | seed | latent energy | participation ratio | next-step R2 | Hankel rank | |lambda| |")
    out.append("|---|---|---|---|---|---|---|")
    for j in M["jepa_collapse"]:
        out.append(f"| {j['mode']} | {j['seed']} | {j['latent_energy']} | {j['participation_ratio']} | "
                   f"{j['next_step_r2']} | {j['hankel_rank']} | {j['dmd_abs_lambda']} |")
    out.append("")

    out.append("## Attention state: constant vs KV\n")
    sm = M["state_memory"]
    out.append("| quantity | bytes |\n|---|---|")
    for key, val in sm.items():
        out.append(f"| {key} | {val:,} |")
    out.append("")

    out.append("## HuggingFace models probed (real bytes)\n")
    out.append("| repo | weight GB | license | downloads |\n|---|---|---|---|")
    for h in M["hub_models"]:
        out.append(f"| {h['repo']} | {h.get('total_weight_gb')} | {h.get('license')} | {h.get('downloads')} |")
    out.append("")

    (ROOT / "results" / "MEASUREMENTS.md").write_text("\n".join(out))
    print(f"wrote {ROOT / 'results' / 'MEASUREMENTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
