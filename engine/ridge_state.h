/*
 * ridge_state.h -- the two attention state forms, and their per-token maths.
 *
 * Full attention keeps a KV cache that grows with T; the delta rule keeps a
 * constant [H, d_k, d_v] state. Both are here because the engine picks per layer,
 * and the maths for the second is the reason the first can be avoided.
 */
#ifndef RIDGE_STATE_H
#define RIDGE_STATE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- full attention: softmax(q K^T / sqrt(d)) V over a contiguous KV ---- */

/* kv layout: [T][2][H][d] (key then value per token). Returns 0 on success. */
int ridge_attn_full(const float *q,   /* [H][d]      */
                    const float *kv,  /* [T][2][H][d] */
                    int T, int H, int d,
                    float *out);      /* [H][d]       */

/* ---- gated delta rule: constant state, O(1) in T ------------------------- */

/*
 * Reference recurrence (from flash-linear-attention, fla/ops/gated_delta_rule):
 *   h <- h * exp(g_t)
 *   v' <- (v_t - h^T k_t) * beta_t
 *   h <- h + k_t (x) v'
 *   o_t <- q_t^T h
 * State h is [H][d_k][d_v]; scale folds into q.
 */
int ridge_delta_step(float *h,        /* [H][d_k][d_v] in/out */
                     const float *q,  /* [H][d_k] */
                     const float *k,  /* [H][d_k] */
                     const float *v,  /* [H][d_v] */
                     const float *beta, /* [H] */
                     const float *g,    /* [H] decay log-rate */
                     int H, int dk, int dv,
                     float scale,
                     float *out);     /* [H][d_v] */

/* Run the recurrence over T steps. Returns 0 on success. */
int ridge_delta_run(const float *q, const float *k, const float *v,
                    const float *beta, const float *g,
                    int T, int H, int dk, int dv, float scale,
                    float *h, float *out);

/* Number of bytes one layer's delta state costs, and one token of KV. */
size_t ridge_delta_state_bytes(int H, int dk, int dv);
size_t ridge_kv_bytes_per_token(int H, int d, int bytes_per_elem);

int ridge_state_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* RIDGE_STATE_H */
