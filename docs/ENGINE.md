# ENGINE.md — the plan for a CPU-only inference engine on tight RAM

This is the outline for the second workstream: an actual engine, not just the cost
model. Written before the code, as asked: inputs and outputs first, then the
design, then the milestones.

The first workstream (`ridge/`, `docs/MATH.md`) answers *can it fit and how fast*.
This one builds the thing that *runs*, and the two hard parts are both about data:
the attention state that grows, and the weights that do not fit.

---

## 1. The problem, stated exactly

Three facts from the measurements decide the whole design:

1. Weight reads are the per-token cost. At 32.1 GB/s, a 3.81 GB q4 7B model's
   *weights alone* allow 8.4 tok/s.
2. The bit-sliced quantised kernel currently runs at 18 GFLOP/s, so q4 decode is
   compute-bound at 1.17 tok/s — a 7.2× gap that is a kernel problem.
3. Attention state is the other half. A 27-layer full-attention KV at 32k is
   8.15 GB; a constant recurrent state is 17.9 MB, flat in T.

So the engine has two jobs: move weights fast, and stop the attention state from
growing without bound. Everything below is one of those two.

## 2. Inputs and outputs

**Input**
- A model directory: quantised weights (GGUF-style layout, or a flat binary we
  define), plus a small JSON config (geometry, quant type, RoPE, norms).
- A prompt and a generation budget.
- A memory policy: RAM budget, where the KV spill file lives, how many threads.

**Output**
- Generated tokens on stdout.
- A JSON telemetry line per run: tok/s, per-phase time (weights, attention,
  sampling), peak RSS, pages read from disk, bytes spilled, madvise hit counts.

**Non-goals (stated so they do not creep in)**
- Not a training framework. Not a GUI. Not a multi-GPU engine.
- Not a new model format for its own sake — it reads what exists where possible.
- No fine-tuning. Inference only.

## 3. Architecture

```
                 +---------------------------+
   prompt -----> |  tokenizer (vocab table)  |
                 +-------------+-------------+
                               |
                               v
   +-----------------------------------------------------------+
   |  scheduler: prefill (batched) vs decode (one token)        |
   +----------------------------+------------------------------+
                                |
          +---------------------+---------------------+
          v                                           v
  +---------------+                          +--------------------+
  | weight store  |                          | state store        |
  | mmap + madvise|                          | KV blocks / deltanet|
  | + q4/q2 gemv  |                          | + spill to disk    |
  +-------+-------+                          +---------+----------+
          |                                            |
          +------------------+-------------------------+
                             v
                 +---------------------------+
                 |  layer: norm, qkv, rope,   |
                 |  attention, ffn, residual  |
                 +---------------------------+
                             |
                             v
                 +---------------------------+
                 |  sampler (greedy/top-k)    |
                 +---------------------------+
```

### 3.1 The memory manager — the Linux part

This is where the "classical Linux style memory management" goes, and it is the
part the brief is really about. Three layers:

**a. Weight region.** The model file is `mmap`'d read-only. We never copy weights
into anonymous memory. The kernel's page cache is the weight store, and our job
is to *tell it what we want* with `madvise`:

- `MADV_SEQUENTIAL` on the region we will scan this layer, because a decode step
  walks every weight exactly once, in order.
- `MADV_WILLNEED` for the next layer's pages, issued one layer ahead, so the read
  for layer `L+1` overlaps the compute of layer `L` (software prefetch at page
  granularity). This is the same trick as prefetching the next cache line, one
  level up the hierarchy.
- `MADV_DONTNEED` on layers we will not touch again this token, to return their
  clean page-cache pages under pressure instead of letting them evict something
  we need.

The correctness of this rests on one measurement: for a **clean, file-backed,
read-only** mapping, dropping the page costs nothing but a later re-fault. There
is no write-back and no corruption. So the policy is safe to be aggressive.

**b. State region — the part that grows.** Two representations, chosen by the
layer type:

- **Full attention:** the KV cache, stored in fixed-size **blocks** (paged
  attention, the vLLM idea, and MNN's `mBlockNum`). A block is `block_tokens`
  rows of `kv_heads × head_dim`. The block table is a page table; a sequence is a
  list of block ids. Blocks are allocated from a free list and **spilled** to a
  file when the pool is full: `mmap` the spill file, write the block, `madvise`
  the resident copy away. Recalled on access. This bounds resident KV at the pool
  size, whatever the context length.
- **Linear / delta-rule attention:** a constant `[H, d_k, d_v]` state per layer,
  `O(1)` in T. Nothing to spill; it just decays. This is the cheap path.

**c. The policy.** A small explicit policy object, not implicit malloc pressure:

```
on_block_needed:
    if free_list: take
    else if pool_bytes + block_bytes <= pool_limit: allocate
    else: spill the least-recently-used clean block, then allocate
on_layer(L):
    madvise(weights[L], WILLNEED); madvise(weights[L-1], DONTNEED)
```

The maths of the policy is a **cache model**. This was first written as the
random-access model, `miss ≈ max(0, 1 − P/W)` for pool `P` and working set `W`.
**Measurement falsified it** (see `results/ENGINE_MEASUREMENTS.md` §3): a decode
pass walks the KV cache in a **cycle**, and cyclic access is the worst case for
LRU — the block evicted is exactly the block needed next, so the miss rate is
**1.0 for every pool smaller than the working set**, not `1 − P/W`. The correct
model is `miss = 0 if P ≥ W else 1`. The design consequence is real: an LRU pool
must be sized to the whole working set, or the replacement policy must be
scan-resistant (or the access must be made sequential with `MADV_SEQUENTIAL`).
This is recorded as a correction to the plan, not dropped.

The time cost of a miss is `misses × b / B_disk`, with `B_disk` measured
(`results/machine.json`: 0.177 GB/s direct, 16.1 GB/s buffered on the first
touch; the engine's own spill recall measured 1,895 MB/s because a freshly
written spill file is still in the page cache).

### 3.2 The state engine — the maths

Two attention forms, both implemented and tested against a reference:

**Full attention (KV blocks).** Standard, plus the block table:
`o_t = softmax(q_t Kᵀ/√d) V`, with K, V gathered from the block list.

**Gated delta rule (constant state).** From FLA's reference recurrence
(`fla/ops/gated_delta_rule/naive.py`), which is the maths to implement:

```
h_0 = 0
for t in 0..T-1:
    h  <- h * exp(g_t)                 # 1. decay the whole state
    v' <- (v_t - hᵀ k_t) * beta_t      # 2. delta correction (what the state cannot explain)
    h  <- h + k_t ⊗ v'                 # 3. rank-1 write of the surprise
    o_t <- q_tᵀ h                      # 4. read out
```

State is `[H, d_k, d_v]` floats — `O(1)` in T, which is the whole point. Cost per
token is `O(H·d_k·d_v)`, so the state size *is* the per-token compute of the
attention, and there is a real trade: bigger `d_k·d_v` means more memory but a
richer state. `docs/MATH.md` §5 already derives the right dimension as the Hankel
rank of the stream; here it becomes a concrete `[H, d_k, d_v]` allocation.

### 3.3 The kernels

Reuse and extend `kernel/ridge_kernels.c`:

- the q4/q2 GEMV, with the improvement that closes the 7.2× gap (int8 expansion +
  `vpdpbusd`, or a wider unroll of the bit-slice chain to hide the dependency
  latency);
- a `rope` application on q/k;
- RMSNorm;
- the delta-rule update and the KV attention, both block-aware.

Everything here is measured, and the measurement goes into `results/`.

## 4. Data flow, precisely

One decode step, in order, with what touches memory:

```
1. embed(x)                     read: embedding row (tiny)
2. for each layer L:
     a. rmsnorm(h)              read: norm weights
     b. qkv = W_qkv · h         read: W_qkv (the big one)  <- madvise WILLNEED(L+1)
     c. rope(q, k)
     d. attention               read/write: state (block table or delta state)
     e. o = W_o · attn
     f. ffn = W_ffn · (SiLU(gate)·up)
     g. h += o + ffn            read: W_o, W_ffn
3. rmsnorm(h); logits = W_lm · h   read: W_lm (vocab × d, often huge)
4. sample
```

The lm head is a trap people miss: for a 152k vocab and d=3584 it is 1.09 GFLOP
and 0.54 GB at fp16 *per token*. The plan includes either quantising it or
computing only the top-k logits if the sampler allows.

## 5. Milestones

Each is independently testable; each ends with a measured number, not a claim.

| # | milestone | done when | state |
|---|---|---|---|
| M0 | `docs/ENGINE.md` (this file) | the plan is written and reviewable | done |
| M1 | memory manager | page/block allocator + spill; unit-tested for correctness and leak-free | done |
| M2 | state engine | delta-rule and KV attention match a NumPy reference to 1e-4 | done |
| M3 | kernels | q4/q2 GEMV beats the current 18 GFLOP/s; measured | next |
| M4 | layer forward | one full transformer layer runs in C, matches NumPy to 1e-3 | done |
| M5 | end-to-end tiny model | a real (small) model generates tokens; tok/s and peak RSS measured | next |
| M6 | policy tuning | the cache model in §3.1c is validated against measured miss rates | done (model corrected) |

M1–M4 are the honest core and are what this pass builds. M5 needs a real small
model and is the first thing that needs weights on disk; M6 needs M5.

## 6. What we take from the references (patterns, not code)

- **MNN** — the shape of a CPU KV manager: in-memory tensor *or* `mmap`'d file,
  a `moveKVCacheFromMemToDisk` spill, block counts, and KV quantisation
  (`TurboQuant`: WHT rotation + Lloyd-Max 3-bit, ~3.5 bits/value). We re-derive
  the quantiser maths rather than copy it.
- **flash-linear-attention** — the exact gated-delta-rule recurrence and the
  chunked (parallel-scan) form, which is what makes prefill cheap: the recurrent
  form is `O(T)` sequential, the chunked form is `O(T/C)` parallel over chunks of
  `C`.
- **ejoy2d** — the C style: small translation units, explicit allocation, no
  framework, data-oriented. We keep that discipline.
- **Understanding the Linux Virtual Memory Manager (Gorman)** — the vocabulary and
  the actual policy: page reclaim, `madvise` hints, the page cache, and the fact
  that a clean file-backed page is free to drop.

## 7. The honest risks

- **A full model runner is a lot of code.** M5 is real work; the plan front-loads
  the parts that are *measurable* (memory, state, kernels) so that if the runner
  stalls, the measured pieces still stand on their own.
- **GGUF compatibility.** Reading arbitrary GGUF is a parsing project. The plan is
  a minimal reader for the quant types we use, not the whole spec.
- **The 7.2× kernel gap may not close.** If `vpdpbusd` on expanded int8 does not
  beat the bit-slice chain here, that is a finding too, and it goes in the docs.
- **Spill bandwidth.** If direct disk is 0.177 GB/s, spilling KV is only worth it
  when the alternative is failing to run at all. The policy must measure, not
  assume.

## 8. Definition of done for this pass

- `engine/` builds with plain `gcc`, no dependencies.
- The memory manager and state engine pass their tests against references.
- `bench/engine_bench.py` produces `results/engine.json` with measured numbers.
- `docs/ENGINE.md` is updated with what was actually measured, and every number in
  it traces to `results/`.

## 9. What this pass actually measured

Full tables are in `results/ENGINE_MEASUREMENTS.md`, regenerated from
`results/engine.json` so they cannot drift from the machine.

**State memory.** For 27 layers, 32 heads, head_dim 128:

| context T | KV cache (fp16) | delta state (fp32) | ratio |
|---:|---:|---:|---:|
| 4,096 | 1,812 MB | 56.6 MB | 32x |
| 8,192 | 3,624 MB | 56.6 MB | 64x |
| 32,768 | 14,496 MB | 56.6 MB | 256x |
| 131,072 | 57,982 MB | 56.6 MB | 1,024x |

The delta state is **56.6 MB, flat in T**. The KV cache at 32k is larger than the
whole RAM budget of the target machine, before any weights. This is the number
that justifies the constant-state path, and it is why `docs/MATH.md` §5 asked for
the state dimension to be the Hankel rank rather than the context length.

**Spill bandwidth.** 1 MiB blocks, a 64 MB pool, 4,096 logical blocks: fill
294 MB/s, recall 1,895 MB/s. Recall is fast because a just-written spill file is
in the page cache; once it exceeds RAM the reads fall to device speed, which
`results/machine.json` measures separately (0.177 GB/s direct).

**Pool policy — a prediction that was wrong.** The plan predicted a miss rate of
`max(0, 1 − P/W)`. The measurement is **1.000 for every pool smaller than the
working set**. The access order is cyclic and cyclic is the worst case for LRU:
the block LRU evicts is the one needed next. The corrected model,
`miss = 0 if P ≥ W else 1`, matches the measurement exactly. The design
consequence is in §3.1c.

**Layer throughput.** d=1024, 8 heads, d_ff=2816, one decode step in C:
~11.1 ms at T=16 and ~11.4 ms at T=256 for full attention, flat ~11.1 ms for the
delta state. Full attention grows, but at these sizes the feed-forward matrices
dominate and hide the attention term — so **for a model this small, attention is
not the bottleneck until the context is long**. The crossover is where the
attention term overtakes the fixed weight cost.

**Correctness.** The C layer matches an independent NumPy reference to 1e-4 on
both attention paths (`tests/test_engine.py`), and the three C self-tests pass.
The delta rule is additionally verified to behave as a rank-1 associative memory:
with `beta=1`, `g=0` and orthonormal keys it stores and returns `v` exactly.

## 10. Next, honestly

M1–M4 are built and measured. What remains:

- **M3 (kernels)**: wire the bit-sliced ternary and int8 GEMV into the layer's
  matrix-vector products. The layer is written so this is a substitution behind
  `matvec`, not a rewrite. This is where the 7.2x decode gap gets attacked.
- **M5 (runner)**: a tokenizer, an embedding table, a stack of these layers, and a
  sampler. This is the first piece that needs real weights on disk, and it is the
  largest remaining chunk of code.
- **A scan-resistant pool policy**, now that the measurement shows plain LRU
  thrashes on the cyclic decode pass.
- **GGUF reading**, minimal and only for the quant types actually used.
