/*
 * ridge_quant.h -- int8 quantisation and a VNNI GEMV, as a reusable module.
 *
 * The measured result that motivates this file (results/MACHINE.md, kernel
 * section): on this machine the 2-bit ternary kernel processes a weight in
 * 0.112 ns, and this int8 kernel processes one in 0.037 ns -- three times
 * faster, despite moving four times the bytes. Ternary is latency-bound on its
 * serial masked-add chain; int8 VNNI is bandwidth-bound with independent
 * accumulator lanes. So the engine uses int8, not ternary, and this is the
 * kernel the layer's matvec calls.
 *
 * Layout. A quantised matrix is symmetric per row:
 *     w[i][j] ~= scale[i] * w8[i][j],   w8 in [-127,127]
 * and each row keeps its integer sum so the activation offset can be undone.
 * The activation is quantised once per GEMV: xq[j] = round(x[j]/sx), and the
 * VNNI instruction (unsigned x signed) needs the activation unsigned, so we
 * store au[j] = xq[j] + 128 and subtract 128*rowsum at the end. That correction
 * is exact, which is why the kernel matches a scalar integer dot bit for bit.
 */
#ifndef RIDGE_QUANT_H
#define RIDGE_QUANT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int8_t *w;      /* [n][k_pad] row-major, signed; k_pad is k rounded up to 64 */
  int *rowsum;    /* [n] integer sum of the signed weights */
  float *scale;   /* [n] per-row scale */
  int n;
  int k;          /* logical k */
  int k_pad;      /* padded k, multiple of 64 */
} ridge_qmat;

/* Quantise a row-major [n][k] float matrix. Returns 0 on success. */
int ridge_quantize(ridge_qmat *q, const float *w, int n, int k);

void ridge_quant_free(ridge_qmat *q);

/* Quantise the activation vector x[k] into unsigned-offset bytes au[k_pad] and
 * signed xq[k_pad] (zero padded). Returns the activation scale sx. */
float ridge_quant_pack_x(const float *x, int k, uint8_t *au, int8_t *xq, int k_pad);

/* y[n] = W @ x, using the quantised matrix. Returns 0 on success. */
int ridge_gemv_i8(const ridge_qmat *q, const float *x, float *y);

/* True when the machine has AVX-512 VNNI and the fast path is in use. */
int ridge_quant_has_vnni(void);

int ridge_quant_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* RIDGE_QUANT_H */
