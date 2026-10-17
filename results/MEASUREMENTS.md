# Measured results

Generated from `results/machine.json` by `bench/report.py`. Host: Linux-6.8.0-1055-gke-x86_64-with-glibc2.41.

## Machine roofs

| roof | measured |
|---|---|
| read bandwidth | 33.40 GB/s |
| copy bandwidth (read+write) | 21.74 GB/s |
| fp32 FMA peak | 109.1 GFLOP/s |
| bit-sliced ternary mat-vec | 17.92 GFLOP/s |
| ternary packed stream | 2.24 GB/s |
| int8 VNNI mat-vec (L3-resident) | 119.77 GFLOP/s |
| int8 VNNI mat-vec (DRAM-bound) | 57.42 GFLOP/s |
| int8 weight stream (DRAM-bound) | 28.71 GB/s |
| int8 VNNI max abs error vs reference | 0.00e+00 (exact) |
| time per weight: ternary / int8 | 0.112 / 0.035 ns (3.2x) |
| fp32 FMA ridge | 3.27 flop/byte |
| ternary ridge | 8.00 flop/byte |
| NVMe O_DIRECT read | 0.177 GB/s |
| page-cache read | 15.77 GB/s |
| ternary kernel max abs error vs reference | 1.55e-05 |

## Decode throughput, resident weights

| model | quant | weights GB | KV GB | flop/byte | bound | tok/s |
|---|---|---|---|---|---|---|
| qwen2.5-coder-7b | fp16 | 15.23 | 0.23 | 1.00 | memory | 2.19 |
| qwen2.5-coder-7b | q4 | 3.81 | 0.23 | 4.00 | compute | 1.18 |
| qwen2.5-coder-7b | q2 | 1.90 | 0.23 | 8.00 | compute | 1.18 |
| qwen2.5-coder-1.5b | fp16 | 3.55 | 0.12 | 1.00 | memory | 9.40 |
| qwen2.5-coder-1.5b | q4 | 0.89 | 0.12 | 4.00 | compute | 5.04 |
| qwen2.5-coder-1.5b | q2 | 0.44 | 0.12 | 8.00 | compute | 5.04 |
| tinyllama-1.1b | fp16 | 2.20 | 0.09 | 1.00 | memory | 15.18 |
| tinyllama-1.1b | q4 | 0.55 | 0.09 | 4.00 | compute | 8.14 |
| tinyllama-1.1b | q2 | 0.28 | 0.09 | 8.00 | compute | 8.14 |
| deepseek-coder-33b | fp16 | 66.69 | 1.04 | 1.00 | memory | 0.50 |
| deepseek-coder-33b | q4 | 16.67 | 1.04 | 4.00 | compute | 0.27 |
| deepseek-coder-33b | q2 | 8.34 | 1.04 | 8.00 | compute | 0.27 |
| kimi-linear-48b-a3b | fp16 | 6.29 | 0.02 | 1.00 | memory | 5.31 |
| kimi-linear-48b-a3b | q4 | 1.57 | 0.02 | 4.00 | compute | 2.85 |
| kimi-linear-48b-a3b | q2 | 0.79 | 0.02 | 8.00 | compute | 2.85 |
| qwen3-32b | fp16 | 65.52 | 1.07 | 1.00 | memory | 0.51 |
| qwen3-32b | q4 | 16.38 | 1.07 | 4.00 | compute | 0.27 |
| qwen3-32b | q2 | 8.19 | 1.07 | 8.00 | compute | 0.27 |
| bitnet-b1.58-2b | fp16 | 5.48 | 0.32 | 1.00 | memory | 6.09 |
| bitnet-b1.58-2b | q4 | 1.37 | 0.32 | 4.00 | compute | 3.27 |
| bitnet-b1.58-2b | q2 | 0.69 | 0.32 | 8.00 | compute | 3.27 |
| ternary-bonsai-27b | fp16 | 38.35 | 1.07 | 1.00 | memory | 0.87 |
| ternary-bonsai-27b | q4 | 9.59 | 1.07 | 4.00 | compute | 0.47 |
| ternary-bonsai-27b | q2 | 4.79 | 1.07 | 8.00 | compute | 0.47 |
| rwkv7-1.5b | fp16 | 3.36 | 0.01 | 1.00 | memory | 9.95 |
| rwkv7-1.5b | q4 | 0.84 | 0.01 | 4.00 | compute | 5.34 |
| rwkv7-1.5b | q2 | 0.42 | 0.01 | 8.00 | compute | 5.34 |

## RAM ceiling (params that fit an 8 GB budget)

| quantisation | no KV | with 7B/4k KV |
|---|---|---|
| 8-bit | 7.40 B | 7.17 B |
| 4-bit | 14.80 B | 14.33 B |
| 2-bit | 29.60 B | 28.66 B |

## Tier plans

| plan | total GB | fits | draft cost c | target tok/s |
|---|---|---|---|---|
| three_tier_ssd_ridge_33b | 22.12 | False | 0.2284 | 0.269 |
| three_tier_ssd_ridge_32b | 21.86 | False | 0.2324 | 0.273 |
| resident_7b_q4_plus_1b_q2 | 4.92 | True | 0.1444 | 1.177 |
| resident_kimi_linear_q4 | 2.46 | True | 0.3499 | 2.85 |
| resident_ternary_27b_q2 | 6.74 | True | 0.0574 | 0.467 |

## Routing decisions (speculation)

| case | tier | gamma | predicted speedup | reason |
|---|---|---|---|---|
| alpha_0.9_len_4096 | speculative | 8 | 2.842 | alpha=0.90, c=0.144 -> gamma*=8, S=2.84 |
| alpha_0.7_len_4096 | speculative | 3 | 1.767 | alpha=0.70, c=0.144 -> gamma*=3, S=1.77 |
| alpha_0.9_len_60000 | speculative | 8 | 2.842 | alpha=0.90, c=0.144 -> gamma*=8, S=2.84 |

## JEPA latent: collapse control

| mode | seed | latent energy | participation ratio | next-step R2 | Hankel rank | |lambda| |
|---|---|---|---|---|---|---|
| none | 0 | 0.01168 | 1.981 | 0.9715 | 2 | [1.0, 1.0, 0.8905, 0.9253] |
| vicreg | 0 | 0.97101 | 1.245 | 0.9928 | 2 | [1.0, 1.0, 0.9831, 0.9831] |
| koopman_dim | 0 | 0.9732 | 3.494 | 0.9952 | 4 | [1.0, 1.0, 1.0, 1.0] |
| none | 1 | 0.0438 | 1.824 | 0.9872 | 2 | [1.0, 1.0, 0.9665, 0.9665] |
| vicreg | 1 | 0.99224 | 2.169 | 0.996 | 4 | [1.0, 1.0, 1.0, 1.0] |
| koopman_dim | 1 | 0.97885 | 3.493 | 0.9953 | 4 | [1.0, 1.0, 1.0, 1.0] |
| none | 2 | 0.02244 | 1.976 | 0.9803 | 2 | [1.0, 1.0, 0.7236, 0.7236] |
| vicreg | 2 | 0.9815 | 1.237 | 0.9944 | 2 | [1.0, 1.0, 0.9996, 0.9996] |
| koopman_dim | 2 | 0.97441 | 3.491 | 0.9951 | 4 | [1.0, 1.0, 1.0, 1.0] |

## Attention state: constant vs KV

| quantity | bytes |
|---|---|
| linear_state_bytes | 17,915,904 |
| kv_bytes_4k | 1,019,215,872 |
| kv_bytes_32k | 8,153,726,976 |
| kv_bytes_128k | 32,614,907,904 |

## HuggingFace models probed (real bytes)

| repo | weight GB | license | downloads |
|---|---|---|---|
| Qwen/Qwen2.5-Coder-7B-Instruct | 15.232 | apache-2.0 | 1536463 |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | 3.087 | apache-2.0 | 445144 |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 2.2 | apache-2.0 | 1067000 |
| Qwen/Qwen3-32B | 65.527 | apache-2.0 | 3041641 |
| deepseek-ai/deepseek-coder-33b-instruct | 133.37 | other | 4772 |
| moonshotai/Kimi-Linear-48B-A3B-Instruct | 98.247 | mit | 184775 |
| microsoft/bitnet-b1.58-2B-4T | 1.179 | mit | 17727 |
| BlinkDL/rwkv7-g1 | 56.842 | apache-2.0 | 34808 |
