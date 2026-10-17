/*
 * ridge_state.c -- attention state: the growing kind and the constant kind.
 *
 * The delta rule is the interesting one. It is a rank-1 update driven by the
 * *surprise*: at each step the state is decayed by exp(g), asked to predict v,
 * and then written with whatever it could not predict, scaled by beta. A state
 * that predicts well barely changes; a surprising token rewrites a slice of it.
 * That is why the state can be O(1) in T and still carry information -- the
 * information it carries is exactly the part of the past that is not yet
 * predictable from it.
 *
 * Build:
 *   gcc -O3 -march=native -c -o bench/ridge_state.o engine/ridge_state.c
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "ridge_state.h"

/* ---- full attention ------------------------------------------------------ */

int ridge_attn_full(const float *q, const float *kv, int T, int H, int d,
                    float *out) {
  if (!q || !kv || !out || T <= 0 || H <= 0 || d <= 0) return -1;
  const float scale = 1.0f / sqrtf((float)d);

  for (int h = 0; h < H; h++) {
    const float *qh = q + (size_t)h * d;

    /* softmax weights over T, computed with a running max for stability */
    float maxs = -INFINITY;
    for (int t = 0; t < T; t++) {
      const float *kh = kv + (((size_t)t * 2 + 0) * H + h) * d;
      float dot = 0.0f;
      for (int i = 0; i < d; i++) dot += qh[i] * kh[i];
      dot *= scale;
      if (dot > maxs) maxs = dot;
    }
    float denom = 0.0f;
    float *oh = out + (size_t)h * d;
    for (int i = 0; i < d; i++) oh[i] = 0.0f;
    for (int t = 0; t < T; t++) {
      const float *kh = kv + (((size_t)t * 2 + 0) * H + h) * d;
      const float *vh = kv + (((size_t)t * 2 + 1) * H + h) * d;
      float dot = 0.0f;
      for (int i = 0; i < d; i++) dot += qh[i] * kh[i];
      float w = expf(dot * scale - maxs);
      denom += w;
      for (int i = 0; i < d; i++) oh[i] += w * vh[i];
    }
    if (denom > 0.0f) {
      float inv = 1.0f / denom;
      for (int i = 0; i < d; i++) oh[i] *= inv;
    }
  }
  return 0;
}

/* ---- gated delta rule ---------------------------------------------------- */

int ridge_delta_step(float *h, const float *q, const float *k, const float *v,
                     const float *beta, const float *g, int H, int dk, int dv,
                     float scale, float *out) {
  if (!h || !q || !k || !v || !beta || !g || !out) return -1;

  for (int hh = 0; hh < H; hh++) {
    float *Hm = h + (size_t)hh * dk * dv;
    const float *qh = q + (size_t)hh * dk;
    const float *kh = k + (size_t)hh * dk;
    const float *vh = v + (size_t)hh * dv;
    const float decay = expf(g[hh]);
    const float b = beta[hh];
    float *oh = out + (size_t)hh * dv;

    /* 1. decay the whole state */
    for (int i = 0; i < dk * dv; i++) Hm[i] *= decay;

    /* 2. predict v from the state: pred[j] = sum_i k[i] * h[i][j] */
    float pred[512];
    if (dv > 512) return -2;
    for (int j = 0; j < dv; j++) pred[j] = 0.0f;
    for (int i = 0; i < dk; i++) {
      const float ki = kh[i];
      const float *row = Hm + (size_t)i * dv;
      for (int j = 0; j < dv; j++) pred[j] += ki * row[j];
    }

    /* 3. the surprise, scaled by beta: what the state could not explain */
    float delta[512];
    for (int j = 0; j < dv; j++) delta[j] = (vh[j] - pred[j]) * b;

    /* 4. rank-1 write of the surprise */
    for (int i = 0; i < dk; i++) {
      const float ki = kh[i];
      float *row = Hm + (size_t)i * dv;
      for (int j = 0; j < dv; j++) row[j] += ki * delta[j];
    }

    /* 5. read out: o[j] = sum_i q[i] * h[i][j] */
    for (int j = 0; j < dv; j++) oh[j] = 0.0f;
    for (int i = 0; i < dk; i++) {
      const float qi = qh[i] * scale;
      const float *row = Hm + (size_t)i * dv;
      for (int j = 0; j < dv; j++) oh[j] += qi * row[j];
    }
  }
  return 0;
}

/* Run the recurrence over T steps. The state h is carried in and out, so a long
 * sequence can be streamed one chunk at a time (prefill in chunks, then decode). */
int ridge_delta_run(const float *q, const float *k, const float *v,
                    const float *beta, const float *g, int T, int H, int dk,
                    int dv, float scale, float *h, float *out) {
  if (!q || !k || !v || !beta || !g || !h || !out) return -1;
  for (int t = 0; t < T; t++) {
    const float *qt = q + ((size_t)t * H) * dk;
    const float *kt = k + ((size_t)t * H) * dk;
    const float *vt = v + ((size_t)t * H) * dv;
    const float *bt = beta + (size_t)t * H;
    const float *gt = g + (size_t)t * H;
    float *ot = out + ((size_t)t * H) * dv;
    int rc = ridge_delta_step(h, qt, kt, vt, bt, gt, H, dk, dv, scale, ot);
    if (rc) return rc;
  }
  return 0;
}

/* ---- sizes --------------------------------------------------------------- */

size_t ridge_delta_state_bytes(int H, int dk, int dv) {
  return (size_t)H * dk * dv * sizeof(float);
}

size_t ridge_kv_bytes_per_token(int H, int d, int bytes_per_elem) {
  /* two tensors (key and value), each H heads of width d */
  return (size_t)2 * H * d * bytes_per_elem;
}

/* ---- selftest ------------------------------------------------------------ */

int ridge_state_selftest(void) {
  /* With beta=1, g=0 and orthogonal-ish keys, the delta rule should reproduce a
   * target sequence exactly once it has seen it: it is a rank-1 associative
   * memory. Feed k=v unit vectors and check the read-out after a few steps. */
  enum { H = 1, DK = 4, DV = 4, T = 4 };
  float h[H * DK * DV];
  float q[T * H * DK], k[T * H * DK], v[T * H * DV], beta[T * H], g[T * H];
  float out[T * H * DV];
  memset(q, 0, sizeof(q));
  memset(k, 0, sizeof(k));
  memset(v, 0, sizeof(v));
  memset(h, 0, sizeof(h));
  for (int t = 0; t < T; t++) {
    k[t * DK + t] = 1.0f;   /* k_t = e_t */
    v[t * DV + t] = (float)(t + 1);
    beta[t] = 1.0f;
    g[t] = 0.0f;
    q[t * DK + t] = 1.0f;   /* query e_t */
  }
  if (ridge_delta_run(q, k, v, beta, g, T, H, DK, DV, 1.0f, h, out) != 0)
    return 1;

  /* After step t, reading with e_t must return v_t exactly. */
  for (int t = 0; t < T; t++) {
    float got = out[t * DV + t];
    float want = (float)(t + 1);
    if (fabsf(got - want) > 1e-4f) return 2;
  }

  /* decay: with g<0 and no writes, the state must shrink monotonically */
  memset(k, 0, sizeof(k));
  memset(v, 0, sizeof(v));
  for (int t = 0; t < T; t++) { beta[t] = 0.0f; g[t] = -1.0f; }
  float h2[H * DK * DV];
  memset(h2, 0, sizeof(h2));
  h2[0] = 100.0f;
  float out2[T * H * DV];
  if (ridge_delta_run(q, k, v, beta, g, T, H, DK, DV, 1.0f, h2, out2) != 0)
    return 3;
  if (!(fabsf(h2[0] - 100.0f * expf(-4.0f)) < 1e-3f)) return 4;

  /* full attention: a single key must be selected */
  {
    float qa[H * DK];
    float kva[3 * 2 * H * DK];
    float oa[H * DK];
    for (int i = 0; i < H * DK; i++) qa[i] = 0.0f;
    qa[0] = 1.0f;
    memset(kva, 0, sizeof(kva));
    kva[(0 * 2 + 0) * H * DK + 0] = 10.0f;  /* token 0 key matches strongly */
    kva[(0 * 2 + 1) * H * DK + 0] = 3.0f;   /* its value */
    kva[(1 * 2 + 1) * H * DK + 0] = 7.0f;
    kva[(2 * 2 + 1) * H * DK + 0] = 7.0f;
    if (ridge_attn_full(qa, kva, 3, H, DK, oa) != 0) return 5;
    /* logits: 10/sqrt(4)=5 -> weight 148, and two zeros -> weight 1 each.
     * o = (148*3 + 7 + 7) / 150 = 458/150. */
    if (fabsf(oa[0] - 458.0f / 150.0f) > 1e-3f) return 6;
  }
  return 0;
}
