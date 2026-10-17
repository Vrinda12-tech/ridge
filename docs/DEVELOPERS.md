# Developers

Everything a developer needs: what this does, what it does not, how to build and
test it, where it breaks, and what to do next. If you read one file before
changing anything, read this one and `AGENTS.md`.

## What this is, in one paragraph

`ridge` is two halves. The first half (`ridge/`, `kernel/`) is a **cost model**:
given a model's geometry and a measured machine, it says what fits in 8 GB, which
roof binds (memory or compute), how many tokens per second you get, and how much
of the speedup from speculative decoding survives the draft cost. The second half
(`engine/`) is a **runtime core in C**: a block memory manager that spills to disk,
the two attention state forms, and one transformer layer. Neither half downloads
weights, and neither half runs a model end to end yet. Read the next section
before you assume otherwise.

## What it does

**`ridge/` — the cost model (pure Python, NumPy only).**

| module | what it computes |
|---|---|
| `model.py` | parameter counts and byte budgets from geometry; `spec_from_hf_config` builds a spec from a real HuggingFace config JSON |
| `roofline.py` | bytes and flops per token, which roof binds, tokens/s, tier planning |
| `spec_decode.py` | acceptance rate, expected tokens per draft, speedup, optimal draft length |
| `jepa.py` | Hankel/DMD readout, participation ratio, and a latent predictor with a collapse control |
| `state.py` | Kalman belief state, reservoir, and the state-vs-KV memory identity |
| `router.py` | the "Foreman": picks a tier and a draft length from measured agreement and context pressure |
| `cli.py` | `size`, `throughput`, `route`, `spec`, `state` |

**`engine/` — the runtime core (C, no dependencies).**

| file | what it is |
|---|---|
| `ridge_mem.c` | logical-block memory manager: one `mmap` pool, a block table, a free list, a spill file |
| `ridge_state.c` | full softmax attention, and the gated delta rule with a constant state |
| `ridge_layer.c` | one pre-norm transformer layer: RMSNorm, RoPE, both attention paths, SwiGLU |
| `engine_core.py` | the ctypes binding; builds the shared object itself |

**`kernel/ridge_kernels.c`** measures the machine: read bandwidth, copy bandwidth,
FMA throughput, and a bit-sliced ternary GEMV. It writes
`results/machine.json`, which every number in the cost model depends on.

## What it does NOT do

Be clear about this before you file a bug:

- **It does not download weights.** `tools/hf_probe.py` reads HuggingFace *metadata*
  (configs), never tensors.
- **It does not run a model.** There is no tokenizer, no embedding table, no layer
  stack, and no sampler. That is milestone M5 in `docs/ENGINE.md` and it is not
  built. `engine/ridge_layer.c` runs *one* layer on arrays you supply.
- **It does not generate text.** The engine's self-tests check that a layer is
  finite and stateful, not that it produces sensible tokens.
- **It is not portable as built.** The shared object is compiled with
  `-march=native`, so it only runs on the CPU that built it. Rebuild on the target.
- **It does not use your GPU, and it is not trying to.** The whole point is CPU.

## Build, test, run

Requirements: `gcc`, Python 3.10+, `numpy`. `pytest` for the tests.

```bash
pip install --user numpy pytest

# 1. measure the machine (needs network once for the HF probe; --no-net uses cache)
python bench/run.py
python bench/report.py            # -> results/MEASUREMENTS.md

# 2. measure the engine (builds engine/librridge_engine.so itself)
python bench/engine_bench.py
python bench/engine_report.py     # -> results/ENGINE_MEASUREMENTS.md

# 3. tests
python -m pytest tests/ -q        # 53 tests

# 4. the CLI
python -m ridge.cli size
python -m ridge.cli throughput --dtype q4
python -m ridge.cli route --alpha 0.9
python -m ridge.cli spec --alpha 0.8
python -m ridge.cli state
```

Building the engine by hand, if you are not going through Python:

```bash
gcc -O3 -march=native -fPIC -shared -I engine -o engine/librridge_engine.so \
    engine/ridge_mem.c engine/ridge_state.c engine/ridge_layer.c -lm

gcc -O3 -march=native -o bench/ridge_kernels kernel/ridge_kernels.c -lm -lpthread
./bench/ridge_kernels . > kernel_measurements.json
```

`ridge/engine_core.py` builds the shared object on import if the sources are
newer, so you rarely need the first command. The Python tests build it too.

## The repo map

```
docs/DEVELOPERS.md      this file
docs/ENGINE.md          the engine plan, milestones, and measured results
docs/MATH.md            the derivations. Sections are cited by number from code.

ridge/                  the cost model (Python)
  model.py  roofline.py  spec_decode.py  jepa.py  state.py  router.py
  cli.py                the command line
  engine_core.py        ctypes binding + build for engine/

engine/                 the runtime core (C)
  ridge_mem.{h,c}       memory manager
  ridge_state.{h,c}     attention state
  ridge_layer.{h,c}     one transformer layer

kernel/ridge_kernels.c  the machine measurement kernel

bench/
  run.py                measure -> results/machine.json (source of truth)
  report.py             results/machine.json -> results/MEASUREMENTS.md
  engine_bench.py       measure the engine -> results/engine.json
  engine_report.py      results/engine.json -> results/ENGINE_MEASUREMENTS.md

tools/hf_probe.py       HuggingFace config probe (cached)
tests/                  pytest; the maths is checked against simulation, not mocks
results/                measured outputs. Regenerate, never hand-edit.
```

## The one rule

**No number without a measurement or a derivation.** Every constant in the model
comes from either `kernel/ridge_kernels.c` (this machine) or a real HuggingFace
config. If you add a quantity, measure it or derive it in `docs/MATH.md`, and add
a test. The `results/*.md` files are generated from `results/*.json` so a
document cannot drift away from the machine it describes. Do not hand-edit them.

## The engine's invariants

These are true by construction and checked by self-tests. Do not break them:

1. `ridge_mem_touch` never returns a pointer to non-resident bytes.
2. A dirty block is written to the spill file before it is dropped.
3. Resident bytes never exceed the pool, at any moment.

And one conceptual rule that has already bitten once: **a logical block is a
stable handle; a physical slot is a RAM window.** They are different kinds of
thing. An early version reused a logical id as a physical slot id and blocks
collided; the self-test caught it. Keep them separate.

## Where it fails, and why — read this before trusting anything

This is the honest list. Each is a real limitation, not a TODO hand-wave.

**The pool policy thrashes on the real access pattern.** `docs/ENGINE.md` 3.1c
predicted a miss rate of `max(0, 1 - P/W)`. Measurement gives **1.000 for every
pool smaller than the working set**. A decode pass over the KV cache is *cyclic*,
and cyclic is the worst case for LRU: the block LRU evicts is the one needed next.
Correct model: `miss = 0 if P >= W else 1`. Consequence: an LRU pool must be sized
to the whole working set, or the policy must be scan-resistant, or the access must
be made sequential with `MADV_SEQUENTIAL`. **Plain LRU is not good enough and the
current code is plain LRU.**

**Spill recall is fast only while the file is in RAM.** The engine measured
1,895 MB/s recall because a just-written spill file is still in the page cache.
Once the file exceeds available RAM, reads fall to device speed. On this machine
that is 0.177 GB/s direct, 16.1 GB/s buffered (`results/machine.json`). At device
speed, spilling is a 100x slowdown, so the pool must genuinely hold the working
set for spilling to be worth it at all.

**The layer's matrix-vector products are naive.** `matvec` in `ridge_layer.c` is a
plain loop: no blocking, no SIMD, single-threaded, float32. It is a correct
reference, not a fast kernel. The measured ternary GEMV (17.96 GFLOP/s) and the
int8 path are not wired in yet — that is M3, and it is where the decode gap gets
attacked.

**The layer is not a real model layer yet.** The delta path hard-codes `beta=1`
and `g=0`; there are no learned gates from weights. Attention uses the same head
count for Q, K and V — no grouped-query attention. Both are deliberate: the maths
is right, the plumbing is not there.

**No tokenizer, no runner, no sampler.** You cannot give it text and get text.
M5.

**Float32 only.** The engine does not read quantized weights. The quant story
exists only in the cost model and in the standalone GEMV.

**`-march=native` means the committed build does not exist** — the `.so` is
gitignored and built locally. Cross-machine use requires a rebuild, and the
optimized kernel would need runtime dispatch to be portable.

**The CI workflow is not in the repository.** GitHub refuses a workflow file
without the token's `workflow` scope, and no token has had it. The workflow lives
on a local branch `workflow-pending`; add it via the web UI, or with a token that
has Workflows write. Until then CI does not run — run the tests locally.

**Python-side tests need `pytest` and `numpy`.** The C self-tests need only a
compiler, but the full suite does not.

## How to extend it

**Add a kernel (M3).** `ridge_layer.c` calls `matvec(y, W, x, n_out, n_in)` in
three places. Replace the body with your kernel, keeping the signature. The maths
does not change; only the speed does. Check against the NumPy reference in
`tests/test_engine.py::test_layer_full_attention_matches_reference` at 1e-4.

**Add a pool policy.** `choose_victim` in `ridge_mem.c` returns the block to
evict. It is currently LRU by a monotonic counter. A scan-resistant policy
(CLOCK, or LRU-K) goes here, and `bench/engine_bench.py::bench_pool_policy`
already measures the miss rate so you can prove the improvement.

**Add a model to the cost model.** Add a spec in `ridge/model.py` and, if it is
from the hub, a config so `spec_from_hf_config` can read it. Add the geometry to
`reference_specs`. Then `python -m ridge.cli throughput` picks it up.

**Add a derivation.** Put it in `docs/MATH.md` as a numbered section, cite the
number from the code, and add a test that checks the code against the derivation.

## Definition of done for a change

- The maths is derived in `docs/MATH.md` or measured in `results/`.
- A test checks it. The maths is checked against a simulation or an independent
  reference, never a mock.
- `python -m pytest tests/ -q` is green.
- If it changes a measured quantity, `bench/*_report.py` is rerun so the generated
  documents match.

## Status of the milestones

From `docs/ENGINE.md`: M0 (plan), M1 (memory manager), M2 (state engine), M4
(layer forward), M6 (policy model, corrected) are **done**. M3 (kernels) and M5
(runner) are **next**. The engine can run one layer on arrays; it cannot run a
model. That is the gap a new contributor should close first.
