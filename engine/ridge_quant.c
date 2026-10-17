/*
 * ridge_quant.c -- int8 quantisation and the VNNI GEMV the layer uses.
 *
 * Build with -mavx512vnni (or -march=native on a VNNI machine) for the fast
 * path. Without it the scalar reference below runs and gives identical integer
 * results, so the code is correct everywhere and fast where the hardware allows.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ridge_quant.h"

#if defined(__x86_64__)
#include <immintrin.h>
#endif

static int kpad(int k) { return (k + 63) & ~63; }

int ridge_quantize(ridge_qmat *q, const float *w, int n, int k) {
  if (!q || !w || n <= 0 || k <= 0) return -1;
  q->n = n; q->k = k; q->k_pad = kpad(k);
  q->w = (int8_t *)calloc((size_t)n * q->k_pad, 1);
  q->rowsum = (int *)calloc((size_t)n, sizeof(int));
  q->scale = (float *)calloc((size_t)n, sizeof(float));
  if (!q->w || !q->rowsum || !q->scale) { ridge_quant_free(q); return -1; }

  for (int i = 0; i < n; i++) {
    const float *row = w + (size_t)i * k;
    float amax = 0.0f;
    for (int j = 0; j < k; j++) { float a = fabsf(row[j]); if (a > amax) amax = a; }
    float s = amax > 0.0f ? amax / 127.0f : 1.0f;
    q->scale[i] = s;
    long rs = 0;
    for (int j = 0; j < k; j++) {
      long v = lrintf(row[j] / s);
      if (v > 127) v = 127;
      if (v < -127) v = -127;
      q->w[(size_t)i * q->k_pad + j] = (int8_t)v;
      rs += v;
    }
    q->rowsum[i] = (int)rs;   /* the padding contributes 0, so this is the row sum */
  }
  return 0;
}

void ridge_quant_free(ridge_qmat *q) {
  if (!q) return;
  free(q->w); free(q->rowsum); free(q->scale);
  q->w = NULL; q->rowsum = NULL; q->scale = NULL;
}

float ridge_quant_pack_x(const float *x, int k, uint8_t *au, int8_t *xq, int k_pad) {
  float amax = 0.0f;
  for (int j = 0; j < k; j++) { float a = fabsf(x[j]); if (a > amax) amax = a; }
  float s = amax > 0.0f ? amax / 127.0f : 1.0f;
  for (int j = 0; j < k; j++) {
    long v = lrintf(x[j] / s);
    if (v > 127) v = 127;
    if (v < -127) v = -127;
    xq[j] = (int8_t)v;
    au[j] = (uint8_t)(v + 128);
  }
  for (int j = k; j < k_pad; j++) { xq[j] = 0; au[j] = 128; } /* 128 == xq 0 */
  return s;
}

#if defined(__AVX512VNNI__)
static int dot_vnni(const int8_t *row, const uint8_t *au, int k_pad, int rowsum) {
  __m512i acc = _mm512_setzero_si512();
  for (int j = 0; j < k_pad; j += 64) {
    __m512i a = _mm512_loadu_si512((const void *)(au + j));
    __m512i b = _mm512_loadu_si512((const void *)(row + j));
    acc = _mm512_dpbusd_epi32(acc, a, b);
  }
  return _mm512_reduce_add_epi32(acc) - 128 * rowsum;
}
#endif

static int dot_ref(const int8_t *row, const int8_t *xq, int k) {
  int acc = 0;
  for (int j = 0; j < k; j++) acc += (int)row[j] * (int)xq[j];
  return acc;
}

int ridge_quant_has_vnni(void) {
#if defined(__AVX512VNNI__)
  return __builtin_cpu_supports("avx512vnni");
#else
  return 0;
#endif
}

int ridge_gemv_i8(const ridge_qmat *q, const float *x, float *y) {
  if (!q || !x || !y) return -1;
  uint8_t *au = (uint8_t *)malloc((size_t)q->k_pad);
  int8_t *xq = (int8_t *)malloc((size_t)q->k_pad);
  if (!au || !xq) { free(au); free(xq); return -1; }
  float sx = ridge_quant_pack_x(x, q->k, au, xq, q->k_pad);

  for (int i = 0; i < q->n; i++) {
    const int8_t *row = q->w + (size_t)i * q->k_pad;
    long ip;
#if defined(__AVX512VNNI__)
    if (ridge_quant_has_vnni()) ip = dot_vnni(row, au, q->k_pad, q->rowsum[i]);
    else
#endif
    ip = dot_ref(row, xq, q->k);
    y[i] = (float)ip * q->scale[i] * sx;
  }
  free(au); free(xq);
  return 0;
}

int ridge_quant_selftest(void) {
  /* A small known case, checked by hand. W = [[1,2,3,4],[-1,0,1,0]] scaled, x a
   * simple vector; the quantised path must reproduce the float path closely and
   * the integer path exactly. */
  int n = 2, k = 4;
  float w[8] = {1, 2, 3, 4, -1, 0, 1, 0};
  ridge_qmat q;
  if (ridge_quantize(&q, w, n, k) != 0) return 1;
  if (q.k_pad != 64) return 2;

  float x[4] = {0.5f, -0.25f, 1.0f, 0.0f};
  float y[2];
  if (ridge_gemv_i8(&q, x, y) != 0) return 3;

  /* float reference */
  float want[2] = {0, 0};
  for (int i = 0; i < n; i++)
    for (int j = 0; j < k; j++) want[i] += w[i * k + j] * x[j];
  /* int8 has ~1% quantisation error; check the relative error is small */
  for (int i = 0; i < n; i++) {
    float denom = fabsf(want[i]) > 1e-6f ? fabsf(want[i]) : 1.0f;
    if (fabsf(y[i] - want[i]) / denom > 0.02f) return 4;
  }

  /* the integer path must be exact against a scalar reference */
  uint8_t au[64];
  int8_t xq[64];
  ridge_quant_pack_x(x, k, au, xq, q.k_pad);
  for (int i = 0; i < n; i++) {
    long ref = dot_ref(q.w + (size_t)i * q.k_pad, xq, k);
    long got;
#if defined(__AVX512VNNI__)
    if (ridge_quant_has_vnni())
      got = dot_vnni(q.w + (size_t)i * q.k_pad, au, q.k_pad, q.rowsum[i]);
    else
#endif
      got = ref;
    if (ref != got) return 5;
  }

  ridge_quant_free(&q);
  return 0;
}
