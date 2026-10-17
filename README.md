# ridge

[![CI](https://github.com/Vrinda12-tech/ridge/actions/workflows/ci.yml/badge.svg)](https://github.com/Vrinda12-tech/ridge/actions/workflows/ci.yml)

Run a coding model on **8 GB of RAM, no CUDA**, and be able to say *why* it does or
does not work. This repository is the arithmetic and the measurements behind that
question, plus the two kernels the answer depends on.

It is not a wrapper around anything. It downloads no weights. It sizes models,
predicts throughput, checks the memory budget, proves the speculation identity,
and measures this machine's real roofs — then says where the design has to land
and where the original plan was wrong.

```
docs/MATH.md          the derivations, in order
kernel/*.c            the two kernels that decide everything (AVX-512)
ridge/            the cost model, spec-decode math, JEPA readout, router
bench/run.py          measure the box, size the models, write results/
results/              machine.json and MEASUREMENTS.md, both generated
tests/                the derivations checked against simulation
```

## The three findings

Measured on an AMD EPYC 9B14 (2 cores, AVX-512 + VNNI + VBMI, 8 GB budget, NVMe):

1. **The SSD cannot be in the token path.** Direct reads run at **0.177 GB/s**.
   Streaming a 3 GB working set once per token is 17 s/token. The "Architect on
   the SSD, DMA its layers per token" design is off by ~100×, and no amount of
   `IoCompletionPort` engineering fixes arithmetic.

2. **Quantisation buys residency, not speed — the kernel decides the speed.** A
   4-bit 7B target plus a 2-bit 1B draft fits in **4.92 GB**, comfortably. The
   first kernel measured 18 GFLOP/s, which made q4 decode *compute*-bound at
   1.17 tok/s against a memory roof of 8.4 tok/s — a 7.2x gap. That gap is now
   closed: the int8 VNNI kernel runs at **57 GFLOP/s** and the layer at **28.4x**
   the float path. The 2-bit format was the wrong bet — see point 2b.

2b. **Narrower weights are not faster.** Measured per weight: 2-bit ternary
   **0.112 ns**, int8 **0.035 ns**. Int8 is 3.2x faster while moving four times
   the bytes, because batch-1 decode is latency-bound, not bandwidth-bound. The
   derivation is `docs/MATH.md` §9; the engine ships int8.

3. **Constant state is the difference between fitting and not.** At 32k context a
   27-layer full-attention KV is **8.15 GB** — the entire budget. A linear-attention
   or recurrent state is **17.9 MB**, flat in sequence length, and it is also the
   belief state the router can read for free.

The full derivation, with the speculation identity, the quantisation error bound,
the Hankel-rank argument and the exact budget, is in
**[docs/MATH.md](docs/MATH.md)**.

## What it predicts

For the resident pair (7B target at q4, 1.1B draft at q2), with the draft's
relative cost `c = 0.144` measured from the kernel:

| agreement α | optimal draft length γ* | speedup S | pair throughput |
|---|---|---|---|
| 0.7 | 3 | 1.77 | 2.1 tok/s |
| 0.9 | 8 | 2.84 | 3.3 tok/s |

`S(γ, α, c) = (1 − α^(γ+1)) / ((1−α)(γc+1))`, derived in MATH.md §3 and verified
against a real rejection-sampling loop in `tests/test_spec_decode.py`.

## The RAM budget

| quantisation | params that fit (no KV) | with a 7B/4k KV |
|---|---|---|
| 8-bit | 7.17 B | 6.94 B |
| 4-bit | 14.33 B | 13.87 B |
| 2-bit ternary | 28.66 B | 27.75 B |

The honest ceiling is a ~7B 4-bit target + ~1B 2-bit draft at 4k–32k context, or a
single ~28B ternary model. A 33B Architect at 4-bit is 16.7 GB and does not fit.

## Run it

```bash
pip install numpy pytest
python bench/run.py            # build the kernel, measure the box, write results/
python bench/report.py         # results/machine.json -> results/MEASUREMENTS.md
python -m pytest tests/ -q     # the derivations, checked
python -m ridge.cli size   # what fits
python -m ridge.cli route --alpha 0.9
```

`bench/run.py` re-measures every constant it uses; nothing in the model is
hard-coded to a machine. `tools/hf_probe.py` pulls real model geometry from the
HuggingFace hub so the sizing is against actual configs, not guesses.

New here? Read **`docs/DEVELOPERS.md`** — what it does, what it does not, how to
build and test it, and an honest list of where it fails and why.

## The kernels

`kernel/ridge_kernels.c` is one file, no dependencies, and it measures the
things the whole design turns on:

* the read, copy and FMA roofs, in the stream mixes a decode step uses;
* NVMe throughput with `O_DIRECT` versus the page cache;
* a **2-bit ternary GEMV**, bit-sliced into two bitplanes and executed with
  AVX-512 masked add/sub (verified against a scalar reference to 1.5e-5);
* an **int8 VNNI GEMV** (`vpdpbusd`), which matches a scalar integer dot exactly;
* the per-token memory of a constant state versus a KV cache.

The measurement it produced overturned the design's first assumption. Ternary
processes a weight in **0.112 ns**; int8 processes one in **0.035 ns** — 3.2x
faster, while moving four times the bytes. Decode at batch 1 is latency-bound, not
bandwidth-bound, so the narrower weight format loses. The engine therefore ships
int8 (see `docs/MATH.md` §9). The ternary kernel stays as the comparison that
proved it.

## What this is not

It does not download or run a language model, and it does not claim a working IDE.
It is the layer underneath one: the arithmetic that tells you which models can
run, how fast, and which of your assumptions is wrong. Every claim has a test or a
measured number behind it, and `results/MEASUREMENTS.md` is regenerated from the
measurement so the two cannot drift apart.

## The engine

`engine/` is the second half: not the arithmetic but the thing that runs. Plain C,
no dependencies, built by `gcc` through a ctypes binding.

- `engine/ridge_mem.c` — a logical-block memory manager. One `mmap`'d pool, a
  block table, a free list, and a spill file. Logical blocks are stable handles
  (KV blocks, weight slabs); physical slots are the RAM window. `touch` guarantees
  residency, spilling and recalling as needed. This is paged attention's block
  table, and the same shape as MNN's CPU KV manager.
- `engine/ridge_state.c` — the two attention state forms: full softmax attention,
  and the gated delta rule with a constant `[H, d_k, d_v]` state.
- `engine/ridge_layer.c` — one pre-norm transformer layer, in C, using both.
- `engine/ridge_quant.c` — int8 quantisation and the AVX-512 VNNI GEMV the layer
  uses after `Layer.quantize()`. The layer keeps a float reference path.

```bash
python bench/engine_bench.py     # measure: state, spill, policy, throughput, int8
python bench/engine_report.py    # -> results/ENGINE_MEASUREMENTS.md
python -m pytest tests/test_engine.py
```

Measured, not claimed (`results/ENGINE_MEASUREMENTS.md`):

| | value |
|---|---|
| delta state, 27 layers x 32 heads x 128 dim | **56.6 MB, flat in T** |
| KV cache at 32k context, same geometry | **14,496 MB (256x)** |
| spill fill / recall, 1 MiB blocks | 294 / 1,895 MB/s |
| one decode layer, d=1024, d_ff=2816, float | ~11.2 ms |
| the same layer, int8 weights | **0.39 ms (28.4x), 2.6% rel L2** |

The plan predicted a pool miss rate of `1 - P/W`; the measurement is **1.000 for
every pool smaller than the working set**, because a decode pass is cyclic and
cyclic is the worst case for LRU. The model was corrected to
`miss = 0 if P >= W else 1`. That correction is in `docs/ENGINE.md` §3.1c.

## Lineage

The engineering patterns are taken from a set of repositories the brief pointed
at, and re-derived rather than copied: the Hankel/Koopman/DMD readout and the
anti-collapse floor (`homeo`), the belief-state and reservoir view of a sequence
model (`eeg-preictal-esn`), the constant-state recurrent arithmetic (`rwkv-lm`,
ROSA), the tiering idea (`epsilon-ide`), the CPU traffic and roofline discipline
(`cpu-performance-engineering`, `onnx-light-cpu`), and the refusal to call a
measurement a proof (`Asmodeus`). See `docs/MATH.md` §5 and §8.
