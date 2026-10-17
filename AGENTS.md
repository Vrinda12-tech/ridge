# AGENTS.md

Repository memory for `ridge`. Read this before changing anything.

## What this repository is

The arithmetic, the measurements, and a C engine core for running a coding model
on 8 GB of RAM with no CUDA. `ridge/` + `kernel/` size models, predict throughput,
and measure the machine. `engine/` is the actual runtime core: a block memory
manager, the attention state (full and constant), and one transformer layer in C.
It does **not** download weights, and it does not yet have a tokenizer or a runner
(M5 in `docs/ENGINE.md`). If a change would require downloading weights to test,
it is out of scope for a normal run.

## The one rule

**No number without a measurement or a derivation.** Every constant in the model
comes from either `kernel/ridge_kernels.c` (this machine) or the HuggingFace
hub (`tools/hf_probe.py`, real configs). If you add a quantity, measure it or
derive it in `docs/MATH.md`, and add a test.

## Layout

```
docs/MATH.md              the derivations. Sections are referenced by number from code.
kernel/ridge_kernels.c  roofs + ternary GEMV + state memory. One file, no deps.
ridge/model.py        geometry, param counts, bytes. spec_from_hf_config() builds from hub JSON.
ridge/roofline.py     Machine (rates), decode_time(), plan_tiers(), budget math.
ridge/spec_decode.py  acceptance identity, expected_tokens(), speedup(), optimal_gamma().
ridge/jepa.py         hankel/dmd/participation_ratio, train_latent() and the collapse control.
ridge/state.py        kalman(), Reservoir, state/kv memory.
ridge/router.py       Foreman: tier + gamma from measured alpha and c.
ridge/cli.py          command line.
ridge/engine_core.py  ctypes binding + build for engine/. NumPy arrays in as pointers.
engine/ridge_mem.{h,c}    logical-block memory manager: mmap pool, block table, spill.
engine/ridge_state.{h,c}  full attention + gated delta rule (constant state).
engine/ridge_layer.{h,c}  one pre-norm transformer layer in C.
engine/ridge_quant.{h,c}  int8 quantisation + AVX-512 VNNI GEMV (the fast path).
docs/DEVELOPERS.md        what it does/does not, build, test, where it fails, how to extend.
docs/ENGINE.md            the engine plan: inputs/outputs, design, milestones, results.
bench/run.py              measure -> results/machine.json (the source of truth).
bench/report.py           results/machine.json -> results/MEASUREMENTS.md.
bench/engine_bench.py     measure the engine -> results/engine.json.
bench/engine_report.py    results/engine.json -> results/ENGINE_MEASUREMENTS.md.
tools/hf_probe.py         HuggingFace hub metadata probe (cached to results/hf_models.json).
tests/                    pytest. The maths is checked against simulation, not mocks.
```

## Commands

```bash
pip install --user numpy pytest
python bench/run.py            # needs network for the hub probe; --no-net uses the cache
python bench/report.py
python bench/engine_bench.py   # builds engine/librridge_engine.so itself
python bench/engine_report.py
python -m pytest tests/ -q
python -m ridge.cli {size,throughput,route,spec,state}
gcc -O3 -march=native -o bench/ridge_kernels kernel/ridge_kernels.c -lm -lpthread
```

`bench/run.py` builds the kernel itself if the source is newer than the binary.
`ridge/engine_core.py` builds `engine/librridge_engine.so` the same way, and the
engine tests build it on first import.

## The engine's invariants

Three things are true by construction and are checked by self-tests; do not break
them:

1. `ridge_mem_touch` never returns a pointer to non-resident bytes.
2. A dirty block is written to the spill file before it is dropped.
3. Resident bytes never exceed the pool, at any moment.

A logical block is a stable handle; a physical slot is a RAM window. They are
different types of thing and must not be conflated — that mistake was made once
(ids recycled across slots) and the self-test caught it.

## Two traps that will waste your time

* **Nyquist.** A real sinusoid sampled at frequency `f > 0.5` aliases to `1-f`.
  DMD tests must use frequencies below 0.5 (`tests/test_state_and_jepa.py` uses
  0.20 and 0.35) or the recovered frequency is the alias, not the truth.
* **Reinstall the environment.** `pip install` sometimes lands in a user site
  that a fresh shell does not have on `sys.path`. If `import numpy` fails after a
  successful install, use `pip install --user`.

## Facts that are load-bearing (do not "fix" these without re-measuring)

* Quantised decode on this box is **compute**-bound at the bit-sliced rate
  (~18 GFLOP/s), not memory-bound. `decode_time()` takes `dtype` for this reason;
  fp16 uses the FMA roof, q4/q2 use `Machine.compute_gflops`. Removing that
  distinction makes the model wrong by ~7×.
* `ModelSpec.d_ff` for a MoE is the **per-expert** width (`moe_intermediate_size`),
  and `n_experts_active` is `num_experts_per_tok`. Kimi-Linear is 48B total / 3B
  active; using 9216 as the FFN width inflates the active size ~9×.
* `disk_direct_gbs` is ~0.177 GB/s and `disk_buffered_gbs` is ~11–15 GB/s. The
  buffered number is the page cache, i.e. RAM; it does not extend the budget.

## Style

Derive, then measure, then test. Prefer the simplest kernel that makes a gap
visible over a fast kernel that hides it. Comments explain why, not what.
