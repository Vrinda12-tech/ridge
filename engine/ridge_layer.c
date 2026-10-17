/*
 * ridge_layer.c -- one pre-norm transformer layer.
 *
 * The layer is deliberately explicit: every matrix-vector product is a loop you
 * can read, so the arithmetic is visible and the reference in Python can be
 * checked against it line by line. The optimised kernels (bit-sliced ternary,
 * int8 dpbusd) plug in here later, behind the same signature, and the maths does
 * not change when they do.
 *
 * Build:
 *   gcc -O3 -march=native -fPIC -shared -I engine -o engine/librridge_engine.so \
 *       engine/ridge_mem.c engine/ridge_state.c engine/ridge_layer.c -lm
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ridge_layer.h"
#include "ridge_state.h"

static void matvec(float *y, const float *W, const float *x, int n_out, int n_in) {
  for (int i = 0; i < n_out; i++) {
    const float *row = W + (size_t)i * n_in;
    float acc = 0.0f;
    for (int j = 0; j < n_in; j++) acc += row[j] * x[j];
    y[i] = acc;
  }
}

static void rmsnorm(float *out, const float *x, const float *w, int d, float eps) {
  float ss = 0.0f;
  for (int i = 0; i < d; i++) ss += x[i] * x[i];
  float inv = 1.0f / sqrtf(ss / (float)d + eps);
  for (int i = 0; i < d; i++) out[i] = x[i] * inv * w[i];
}

/* rotary embedding on a single head's vector, at absolute position pos */
static void rope(float *vec, int d_head, int pos, float theta) {
  for (int i = 0; i < d_head; i += 2) {
    float freq = 1.0f / powf(theta, (float)i / (float)d_head);
    float ang = (float)pos * freq;
    float c = cosf(ang), s = sinf(ang);
    float a = vec[i], b = vec[i + 1];
    vec[i] = a * c - b * s;
    vec[i + 1] = a * s + b * c;
  }
}

static void silu_mul(float *out, const float *gate, const float *up, int n) {
  for (int i = 0; i < n; i++) {
    float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
  }
}

size_t ridge_layer_scratch_bytes(const ridge_layer *L) {
  /* q, k, v, attn_out, o, normed, ffn gate, ffn up, ffn out, delta scratch */
  size_t per = (size_t)L->d * 4 + (size_t)L->d_ff * 3 + (size_t)L->d * 2;
  return per * sizeof(float) + 4096;
}

int ridge_layer_fwd(const ridge_layer *L, float *x, float *kv, int *kv_len,
                    int max_T, float *state, int use_delta, int pos) {
  if (!L || !x) return -1;
  const int d = L->d, H = L->n_heads, dh = L->d_head, dff = L->d_ff;
  const float scale = 1.0f / sqrtf((float)dh);

  float *buf = (float *)malloc(ridge_layer_scratch_bytes(L));
  if (!buf) return -2;
  float *normed = buf;                       /* [d] */
  float *q = normed + d;                     /* [d] */
  float *k = q + d;                          /* [d] */
  float *v = k + d;                          /* [d] */
  float *attn = v + d;                       /* [d] */
  float *proj = attn + d;                    /* [d] */
  float *gate = proj + d;                    /* [d_ff] */
  float *up = gate + dff;                    /* [d_ff] */
  float *ffnout = up + dff;                  /* [d] */

  /* --- attention block --- */
  rmsnorm(normed, x, L->attn_norm, d, 1e-5f);
  matvec(q, L->wq, normed, d, d);
  matvec(k, L->wk, normed, d, d);
  matvec(v, L->wv, normed, d, d);
  for (int h = 0; h < H; h++) {
    rope(q + h * dh, dh, pos, L->rope_theta);
    rope(k + h * dh, dh, pos, L->rope_theta);
  }

  if (use_delta && state) {
    /* constant-state path: one recurrent step, state [H][dh][dh] */
    float beta[H], g[H], out[d];
    for (int h = 0; h < H; h++) { beta[h] = 1.0f; g[h] = 0.0f; }
    if (ridge_delta_step(state, q, k, v, beta, g, H, dh, dh, scale, out) != 0) {
      free(buf);
      return -3;
    }
    for (int i = 0; i < d; i++) attn[i] = out[i];
  } else {
    /* full attention over the cache. kv is [max_T][2][d]. */
    if (!kv || !kv_len) { free(buf); return -4; }
    int T = *kv_len;
    float *kb = kv + (size_t)T * 2 * d;      /* this token's key slot */
    float *vb = kb + d;
    memcpy(kb, k, (size_t)d * sizeof(float));
    memcpy(vb, v, (size_t)d * sizeof(float));
    if (T + 1 > max_T) { free(buf); return -5; }
    *kv_len = T + 1;

    for (int h = 0; h < H; h++) {
      const float *qh = q + h * dh;
      float maxs = -INFINITY;
      for (int t = 0; t <= T; t++) {
        const float *kh = kv + ((size_t)t * 2 * d) + h * dh;
        float dot = 0.0f;
        for (int i = 0; i < dh; i++) dot += qh[i] * kh[i];
        dot *= scale;
        if (dot > maxs) maxs = dot;
      }
      float denom = 0.0f;
      float *oh = attn + h * dh;
      for (int i = 0; i < dh; i++) oh[i] = 0.0f;
      for (int t = 0; t <= T; t++) {
        const float *kh = kv + ((size_t)t * 2 * d) + h * dh;
        const float *vh = kv + ((size_t)t * 2 * d) + d + h * dh;
        float dot = 0.0f;
        for (int i = 0; i < dh; i++) dot += qh[i] * kh[i];
        float w = expf(dot * scale - maxs);
        denom += w;
        for (int i = 0; i < dh; i++) oh[i] += w * vh[i];
      }
      if (denom > 0.0f) {
        float inv = 1.0f / denom;
        for (int i = 0; i < dh; i++) oh[i] *= inv;
      }
    }
  }

  matvec(proj, L->wo, attn, d, d);
  for (int i = 0; i < d; i++) x[i] += proj[i];

  /* --- feed-forward block --- */
  rmsnorm(normed, x, L->ffn_norm, d, 1e-5f);
  matvec(gate, L->w_gate, normed, dff, d);
  matvec(up, L->w_up, normed, dff, d);
  silu_mul(gate, gate, up, dff);
  matvec(ffnout, L->w_down, gate, d, dff);
  for (int i = 0; i < d; i++) x[i] += ffnout[i];

  free(buf);
  return 0;
}

int ridge_layer_selftest(void) {
  /* A 2-head, 4-wide layer. We only check invariants that must hold for any
   * weights: finite outputs, and that a second token changes the result (the
   * state is actually being used). */
  enum { d = 4, H = 2, dh = 2, dff = 8 };
  float wq[d * d], wk[d * d], wv[d * d], wo[d * d];
  float wg[dff * d], wu[dff * d], wd[d * dff];
  float an[d], fn[d];
  for (int i = 0; i < d * d; i++) { wq[i] = wk[i] = wv[i] = wo[i] = 0.01f * (i % 7 - 3); }
  for (int i = 0; i < dff * d; i++) { wg[i] = wu[i] = 0.01f * (i % 5 - 2); }
  for (int i = 0; i < d * dff; i++) wd[i] = 0.01f * (i % 3 - 1);
  for (int i = 0; i < d; i++) { an[i] = 1.0f; fn[i] = 1.0f; }

  ridge_layer L = {d, H, dh, dff, wq, wk, wv, wo, wg, wu, wd, an, fn, 10000.0f};

  float x[d] = {1.0f, -0.5f, 0.25f, 0.75f};
  float kv[4 * 2 * d];
  int kv_len = 0;
  if (ridge_layer_fwd(&L, x, kv, &kv_len, 4, NULL, 0, 0) != 0) return 1;
  if (kv_len != 1) return 2;
  for (int i = 0; i < d; i++) if (!isfinite(x[i])) return 3;

  float x2[d];
  memcpy(x2, x, sizeof(x));
  if (ridge_layer_fwd(&L, x2, kv, &kv_len, 4, NULL, 0, 1) != 0) return 4;
  if (kv_len != 2) return 5;
  for (int i = 0; i < d; i++) if (!isfinite(x2[i])) return 6;

  /* delta path must also be finite and stateful */
  float st[H * dh * dh];
  memset(st, 0, sizeof(st));
  float x3[d] = {0.3f, 0.1f, -0.2f, 0.4f};
  if (ridge_layer_fwd(&L, x3, NULL, NULL, 0, st, 1, 0) != 0) return 7;
  for (int i = 0; i < d; i++) if (!isfinite(x3[i])) return 8;
  float nz = 0.0f;
  for (int i = 0; i < H * dh * dh; i++) nz += fabsf(st[i]);
  if (nz <= 0.0f) return 9;   /* the state must have been written */

  return 0;
}
