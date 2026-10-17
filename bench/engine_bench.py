#!/usr/bin/env python3
"""Measure the engine: state cost vs context, spill bandwidth, layer throughput.

Writes results/engine.json. Every number here is measured on this machine, not
assumed. Run it, then read results/ENGINE_MEASUREMENTS.md.

    python bench/engine_bench.py
"""

from __future__ import annotations

import ctypes
import json
import os
import time
from pathlib import Path

import numpy as np

from ridge import engine_core as ec

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def bench_state_memory() -> dict:
    """The point of the delta rule: state bytes flat in T, KV linear in T.

    Geometry: a 27-layer, 32-head, head_dim 128 model, fp16 KV, delta state in
    fp32. Both are per layer, scaled to the whole model.
    """
    n_layer, n_head, head_dim = 27, 32, 128
    kv_per_token = n_layer * 2 * n_head * head_dim * 2      # fp16
    delta_per_layer = n_head * head_dim * head_dim * 4      # fp32
    delta_total = n_layer * delta_per_layer
    out = {"n_layer": n_layer, "n_head": n_head, "head_dim": head_dim,
           "kv_bytes_per_token": kv_per_token, "delta_state_bytes": delta_total}
    for T in (1024, 4096, 8192, 32768, 131072):
        kv = kv_per_token * T
        out[f"kv_bytes_at_{T}"] = kv
        out[f"ratio_kv_over_delta_at_{T}"] = kv / delta_total
    return out


def bench_spill_bandwidth(tmp_path: Path) -> dict:
    """How fast can the pool push blocks to the spill file and pull them back?

    This is the number that decides whether spilling is worth it at all. We fill a
    pool of 64 blocks, mark them dirty, then force recall of all of them.
    """
    blk = 1 << 20          # 1 MiB blocks
    pool_blocks = 64
    pool = blk * pool_blocks
    nlog = 4096
    spill = str(tmp_path / "spill.bin")
    m = ec.Memory(pool, blk, nlog, spill)

    ids = []
    t0 = time.perf_counter()
    for _ in range(nlog):
        bid = m.alloc()
        p = m.touch(bid)
        ctypes.memset(p, 0xAB, blk)
        m.mark_dirty(bid)
        ids.append(bid)
    t_fill = time.perf_counter() - t0

    # now touch all of them again, in order; each is a recall from the spill file
    t0 = time.perf_counter()
    for bid in ids:
        p = m.touch(bid)
        assert p
    t_recall = time.perf_counter() - t0

    total_mb = nlog * blk / 1e6
    # Each block is written once (spill) and read once (recall). With 1 MiB blocks
    # these writes are large enough that the kernel write-back thread, not the
    # calling thread, absorbs most of the cost -- which is the point of big blocks.
    spill_write_mbs = total_mb / t_fill if t_fill else 0.0
    recall_mbs = total_mb / t_recall if t_recall else 0.0

    del m
    return {
        "block_bytes": blk,
        "pool_bytes": pool,
        "logical_blocks": nlog,
        "fill_seconds": t_fill,
        "recall_seconds": t_recall,
        "spill_write_mbs": spill_write_mbs,
        "recall_read_mbs": recall_mbs,
    }


def bench_pool_policy(tmp_path: Path) -> dict:
    """Measure the cache model from ENGINE.md 3.1c, and check the access order.

    A cyclic pass over W blocks with a pool of P is the worst case for LRU: the
    block evicted is exactly the one needed next, so the miss rate is 100% for any
    P < W, not the 1 - P/W a random-access model predicts. We sweep P and measure.
    """
    blk = 4096
    W = 256                      # working set of blocks
    passes = 8
    out = {"working_set_blocks": W, "passes": passes,
           "access_order": "cyclic", "points": []}
    for pool_blocks in (16, 32, 64, 128, 256):
        pool = blk * pool_blocks
        m = ec.Memory(pool, blk, W, str(tmp_path / f"s{pool_blocks}.bin"))
        ids = [m.alloc() for _ in range(W)]
        for bid in ids:
            m.touch(bid)
            m.mark_dirty(bid)
        recalls = 0
        for _ in range(passes):
            for bid in ids:
                if not m.resident(bid):
                    recalls += 1
                m.touch(bid)
        touches = W * passes
        measured = recalls / touches
        # two models, so the gap between them is visible in the output
        random_model = max(0.0, 1.0 - pool_blocks / W)
        cyclic_model = 0.0 if pool_blocks >= W else 1.0
        out["points"].append({
            "pool_blocks": pool_blocks,
            "measured_miss_rate": measured,
            "predicted_random_model": random_model,
            "predicted_cyclic_model": cyclic_model,
        })
        del m
    return out


def bench_layer_throughput() -> dict:
    """Layer time: full attention grows with T, the delta state does not.

    Realistic small geometry so it runs quickly: d=1024, 8 heads, d_ff=2816.
    """
    rng = np.random.default_rng(7)
    d, n_heads, dff = 1024, 8, 2816

    def w(shape):
        return rng.normal(scale=0.02, size=shape).astype(np.float32)

    L = ec.Layer(w((d, d)), w((d, d)), w((d, d)), w((d, d)), w((dff, d)),
                 w((dff, d)), w((d, dff)), np.ones(d, np.float32),
                 np.ones(d, np.float32))
    L.set_heads(n_heads)

    out = {"d": d, "n_heads": n_heads, "d_ff": dff, "full_attention": [],
           "delta_state": []}

    # delta path: flat in T
    dh = d // n_heads
    st = np.zeros((n_heads, dh, dh), np.float32)
    x = rng.normal(size=d).astype(np.float32)
    for T in (16, 64, 256):
        # warm up the first T tokens, then time a single decode step at length T
        for pos in range(T):
            x, _, _ = L.forward(x, state=st, use_delta=1, pos=pos)
        reps = 20
        t0 = time.perf_counter()
        for r in range(reps):
            x, _, _ = L.forward(x, state=st, use_delta=1, pos=T + r)
        dt = (time.perf_counter() - t0) / reps
        out["delta_state"].append({"T": T, "seconds_per_token": dt})

    # full attention: grows with T
    for T in (16, 64, 256):
        max_T = T + 32
        kv = np.zeros((max_T, 2, d), np.float32)
        x2 = rng.normal(size=d).astype(np.float32)
        kv_len = 0
        for pos in range(T):
            x2, _, kv_len = L.forward(x2, kv=kv, kv_len=kv_len, max_T=max_T, pos=pos)
        reps = 20
        t0 = time.perf_counter()
        for r in range(reps):
            x2, _, kv_len = L.forward(x2, kv=kv, kv_len=kv_len, max_T=max_T,
                                      pos=T + r)
        dt = (time.perf_counter() - t0) / reps
        out["full_attention"].append({"T": T, "seconds_per_token": dt})
    return out


def bench_layer_quantized() -> dict:
    """The M3 payoff: the same layer in float and in int8, one decode step.

    Real geometry (d=1024, 8 heads, d_ff=2816) so the weight stream is large
    enough to matter. Both layers run the identical maths; only the GEMV changes.
    """
    rng = np.random.default_rng(21)
    d, n_heads, dff = 1024, 8, 2816

    def w(shape):
        return rng.normal(scale=0.02, size=shape).astype(np.float32)

    args = (w((d, d)), w((d, d)), w((d, d)), w((d, d)), w((dff, d)),
            w((dff, d)), w((d, dff)), np.ones(d, np.float32), np.ones(d, np.float32))
    lf = ec.Layer(*args)
    lf.set_heads(n_heads)
    lq = ec.Layer(*args)
    lq.set_heads(n_heads)
    lq.quantize()

    max_T = 128
    kvf = np.zeros((max_T, 2, d), np.float32)
    kvq = np.zeros((max_T, 2, d), np.float32)
    xf = rng.normal(size=d).astype(np.float32)
    xq = xf.copy()

    def run(layer, x, kv, T, reps):
        # Full attention path: stable over many steps, unlike the delta path with
        # the hard-coded g=0 (which grows without decay). The GEMVs are identical,
        # so this isolates the kernel difference.
        kv_len = 0
        for pos in range(T):
            x, _, kv_len = layer.forward(x, kv=kv, kv_len=kv_len, max_T=max_T, pos=pos)
        t0 = time.perf_counter()
        for r in range(reps):
            x, _, kv_len = layer.forward(x, kv=kv, kv_len=kv_len, max_T=max_T, pos=T + r)
        return (time.perf_counter() - t0) / reps, x

    T, reps = 64, 30
    dt_float, x_float = run(lf, xf, kvf, T, reps)
    dt_int8, x_int8 = run(lq, xq, kvq, T, reps)
    lq.dequantize()

    # float weights per token: the seven matrices, 4 bytes each
    weight_bytes = (4 * d * d + 3 * d * dff) * 4
    # relative L2 is the meaningful error metric; a max over a residual stream is
    # dominated by tiny components. Single-scale activation quantisation is lossy
    # when the activation has outliers, which is why real engines use per-token or
    # per-group scales. This number says how much that costs.
    l2 = float(np.linalg.norm(x_int8 - x_float) / (np.linalg.norm(x_float) + 1e-9))
    mx = float(np.abs(x_int8 - x_float).max() / (np.abs(x_float).max() + 1e-9))
    return {
        "d": d, "n_heads": n_heads, "d_ff": dff,
        "weight_bytes_float": int(weight_bytes),
        "weight_bytes_int8": int(weight_bytes // 4),
        "seconds_per_token_float": dt_float,
        "seconds_per_token_int8": dt_int8,
        "speedup": dt_float / dt_int8 if dt_int8 else 0.0,
        "weight_gbs_float": weight_bytes / dt_float / 1e9,
        "weight_gbs_int8": (weight_bytes / 4) / dt_int8 / 1e9,
        "rel_l2_error": l2,
        "max_rel_diff": mx,
        "note": "int8 weights fit in L3 at this size; the DRAM-bound number is in "
                "the kernel section of results/machine.json",
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "ridge_engine_bench"
    tmp.mkdir(exist_ok=True)

    print("building engine...")
    ec.build()

    print("state memory...")
    state = bench_state_memory()
    print("spill bandwidth...")
    spill = bench_spill_bandwidth(tmp)
    print("pool policy...")
    policy = bench_pool_policy(tmp)
    print("layer throughput...")
    layer = bench_layer_throughput()
    print("quantized layer...")
    qlayer = bench_layer_quantized()

    report = {
        "machine": {"block_pool_note": "measured on this host"},
        "state_memory": state,
        "spill": spill,
        "pool_policy": policy,
        "layer": layer,
        "layer_quantized": qlayer,
        "c_selftests": {"mem": ec.c_selftests()[0], "state": ec.c_selftests()[1],
                        "layer": ec.layer_selftest(),
                        "quant": ec.lib().ridge_quant_selftest()},
    }
    out_path = RESULTS / "engine.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_path}")
    print(json.dumps({"spill_write_mbs": round(spill["spill_write_mbs"], 1),
                      "recall_read_mbs": round(spill["recall_read_mbs"], 1),
                      "delta_state_mb": round(state["delta_state_bytes"] / 1e6, 1),
                      "kv_at_32768_mb": round(state["kv_bytes_at_32768"] / 1e6, 1)},
                     indent=2))


if __name__ == "__main__":
    main()
