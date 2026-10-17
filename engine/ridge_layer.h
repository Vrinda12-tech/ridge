/*
 * ridge_layer.h -- one pre-norm transformer layer, in C, on the data path the
 * engine actually uses.
 *
 * This is where the three pieces meet: the weights are read (later through the
 * mmap'd, madvise'd store), the attention state is written (KV blocks through the
 * memory manager, or a constant delta state), and the arithmetic is the maths
 * from docs/MATH.md and docs/ENGINE.md.
 *
 * Layouts, all row-major [out_features, in_features]:
 *   wq [d,d]  wk [d_kv,d]  wv [d_kv,d]  wo [d,d]
 *   w_gate [d_ff,d]  w_up [d_ff,d]  w_down [d,d_ff]
 * Attention is multi-head with n_heads query heads and the same number of KV
 * heads (GQA is a later refinement, the maths is unchanged).
 */
#ifndef RIDGE_LAYER_H
#define RIDGE_LAYER_H

#include <stdint.h>

#include "ridge_quant.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int d;          /* model width */
  int n_heads;    /* query heads */
  int d_head;     /* per-head width, d == n_heads * d_head */
  int d_ff;       /* feed-forward width */

  const float *wq;
  const float *wk;
  const float *wv;
  const float *wo;
  const float *w_gate;
  const float *w_up;
  const float *w_down;
  const float *attn_norm;
  const float *ffn_norm;
  float rope_theta;

  /* Optional int8 copy of the seven weight matrices. After
   * ridge_layer_quantize(), the forward pass uses these instead of the float
   * weights. Measured 3x faster per weight than the ternary path; see
   * ridge_quant.h. */
  ridge_qmat q_wq, q_wk, q_wv, q_wo, q_gate, q_up, q_down;
  int quantized;
} ridge_layer;

/* Quantise the layer's float weights into its int8 copy. Idempotent enough:
 * calling it twice leaks the first copy, so call it once. Returns 0 on success.
 * The forward pass is bit-identical in structure but int8-quantised in value. */
int ridge_layer_quantize(ridge_layer *L);

void ridge_layer_dequantize(ridge_layer *L);

/*
 * Forward one token. x is [d], updated in place with the layer output.
 * kv holds the cache as [max_T][2][d][1] contiguous floats (key, then value, all
 * heads flattened), with kv_len the number of tokens already cached; this call
 * appends one. Pass kv=NULL and use_delta=1 to use the constant delta state
 * instead, in which case state must be [n_heads][d_head][d_head].
 * Returns 0 on success.
 */
int ridge_layer_fwd(const ridge_layer *L, float *x, float *kv, int *kv_len,
                    int max_T, float *state, int use_delta, int pos);

/* Scratch a layer needs, so the caller can size a pool. */
size_t ridge_layer_scratch_bytes(const ridge_layer *L);

int ridge_layer_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* RIDGE_LAYER_H */
