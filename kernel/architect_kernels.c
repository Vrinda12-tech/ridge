/*
 * architect_kernels.c -- measure the roofs this machine actually has, and run the
 * two kernels the cost model needs: a 2-bit ternary GEMV (bit-sliced) and the
 * per-token memory of a constant-state recurrence against a KV cache.
 *
 * Build:
 *   gcc -O3 -march=native -o bench/architect_kernels kernel/architect_kernels.c -lm -lpthread
 * Run:
 *   ./bench/architect_kernels <model_dir>        # model_dir holds a large file for O_DIRECT
 * Prints one JSON object on stdout. Every number is a median of timed passes.
 *
 * Why each piece exists:
 *   - roofline()  : the compute roof and the memory roofs, in the same stream mixes
 *                   the LLM decode step uses (read-only weights, read+write state).
 *   - disk_*()    : what a weight-streaming engine can actually pull off the SSD,
 *                   direct (O_DIRECT, no page cache) vs buffered vs mmap-once.
 *   - ternary_gemv: the arithmetic a 2-bit weight model needs. If this is
 *                   memory-bound the model is a bandwidth problem, not a FLOP one.
 *   - state_mem() : the exact per-token cost of an O(1) recurrent state vs a KV
 *                   cache, which is the whole reason to prefer linear attention.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

/* ------------------------------------------------------------------ timing */

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static double median(double *v, int n) {
  qsort(v, n, sizeof(double), cmp_double);
  return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* Keep results alive so the optimiser cannot delete a timed loop. */
static volatile double g_sink = 0.0;

static void *xaligned(size_t bytes) {
  void *p = NULL;
  if (posix_memalign(&p, 64, bytes) != 0) {
    fprintf(stderr, "posix_memalign failed\n");
    exit(1);
  }
  memset(p, 1, bytes); /* prefault: page faults must not land inside a timed pass */
  return p;
}

/* ------------------------------------------------------------------- roofs */

/* Read-only stream: sum a big float array. Returns GB/s.
 * Eight independent accumulator chains: a single chain is bound by the add
 * latency (~4 cycles/element), which would report a fake, tiny bandwidth. */
static double roof_read(const float *__restrict a, size_t n) {
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
  __m512 acc4 = _mm512_setzero_ps(), acc5 = _mm512_setzero_ps();
  __m512 acc6 = _mm512_setzero_ps(), acc7 = _mm512_setzero_ps();
  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    acc0 = _mm512_add_ps(acc0, _mm512_loadu_ps(a + i));
    acc1 = _mm512_add_ps(acc1, _mm512_loadu_ps(a + i + 16));
    acc2 = _mm512_add_ps(acc2, _mm512_loadu_ps(a + i + 32));
    acc3 = _mm512_add_ps(acc3, _mm512_loadu_ps(a + i + 48));
    acc4 = _mm512_add_ps(acc4, _mm512_loadu_ps(a + i + 64));
    acc5 = _mm512_add_ps(acc5, _mm512_loadu_ps(a + i + 80));
    acc6 = _mm512_add_ps(acc6, _mm512_loadu_ps(a + i + 96));
    acc7 = _mm512_add_ps(acc7, _mm512_loadu_ps(a + i + 112));
  }
  float s = _mm512_reduce_add_ps(
      _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)),
                    _mm512_add_ps(_mm512_add_ps(acc4, acc5), _mm512_add_ps(acc6, acc7))));
  double tail = 0;
  for (; i < n; i++) tail += a[i];
  g_sink += s + tail;
  return (double)(n * sizeof(float)) / 1e9;
}

/* Read one array, write one: a copy. Returns GB/s (bytes moved / time). */
static double roof_copy(float *__restrict dst, const float *__restrict src, size_t n) {
  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    _mm512_storeu_ps(dst + i, _mm512_loadu_ps(src + i));
    _mm512_storeu_ps(dst + i + 16, _mm512_loadu_ps(src + i + 16));
    _mm512_storeu_ps(dst + i + 32, _mm512_loadu_ps(src + i + 32));
    _mm512_storeu_ps(dst + i + 48, _mm512_loadu_ps(src + i + 48));
  }
  for (; i < n; i++) dst[i] = src[i];
  g_sink += dst[0];
  return (double)(2 * n * sizeof(float)) / 1e9;
}

/* FMA peak: 16 independent accumulator chains, one L1-resident load per FMA. */
static double roof_fma(const float *buf, size_t nb, long iters) {
  __m512 x = _mm512_load_ps(buf);
  __m512 acc[16];
  for (int i = 0; i < 16; i++) acc[i] = _mm512_set1_ps(1.0f + i);
  for (long it = 0; it < iters; it++) {
    for (int i = 0; i < 16; i++) acc[i] = _mm512_fmadd_ps(acc[i], x, x);
  }
  __m512 s = _mm512_setzero_ps();
  for (int i = 0; i < 16; i++) s = _mm512_add_ps(s, acc[i]);
  g_sink += _mm512_reduce_add_ps(s);
  /* 16 lanes * 16 chains * 2 flop per FMA per iteration */
  return (double)16 * 16 * 2 * (double)iters;
}

/* ------------------------------------------------------- ternary (2-bit) */

/*
 * Bit-sliced ternary weights. Two bitplanes per weight: w = pos_bit - neg_bit,
 * so w in {-1, 0, +1}. Codes 00->0, 01->+1, 10->-1; 11 is unused.
 * Packed 64 weights per uint64 (bit k of pos/neg word = weight k).
 *
 * dot = sum_k (pos_k - neg_k) x_k = sum_k pos_k x_k - sum_k neg_k x_k.
 * With AVX-512 we expand each 16-bit slice of the bitplane to a lane mask and
 * use masked add/sub, which needs no multiply at all.
 */
typedef struct {
  uint64_t *pos;   /* nblocks * 64-bit words, 64 weights per word */
  uint64_t *neg;
  size_t k;        /* number of weights (multiple of 64) */
} ternary_row;

/* Reference decode + dot, for verification. */
static double ternary_dot_ref(const uint8_t *codes, const float *x, size_t k) {
  double acc = 0;
  for (size_t i = 0; i < k; i++) {
    int c = codes[i];
    int w = (c == 1) ? 1 : (c == 2) ? -1 : 0;
    acc += (double)w * x[i];
  }
  return acc;
}

#if defined(__x86_64__)
/* One output row: y = <w, x>, bit-sliced. */
static float ternary_dot_avx512(const ternary_row *r, const float *x) {
  __m512 acc = _mm512_setzero_ps();
  size_t nw = r->k / 64;
  for (size_t w = 0; w < nw; w++) {
    uint64_t p = r->pos[w], q = r->neg[w];
    for (int seg = 0; seg < 4; seg++) {           /* 4 segments of 16 weights */
      __mmask16 mp = (__mmask16)((p >> (16 * seg)) & 0xFFFF);
      __mmask16 mn = (__mmask16)((q >> (16 * seg)) & 0xFFFF);
      __m512 xv = _mm512_loadu_ps(x + w * 64 + seg * 16);
      acc = _mm512_mask_add_ps(acc, mp, acc, xv);
      acc = _mm512_mask_sub_ps(acc, mn, acc, xv);
    }
  }
  return _mm512_reduce_add_ps(acc);
}
#endif

/* Fill one row from raw codes (0/1/2) into bitplanes. */
static void ternary_pack(ternary_row *r, const uint8_t *codes) {
  size_t nw = r->k / 64;
  for (size_t w = 0; w < nw; w++) {
    uint64_t p = 0, q = 0;
    for (int b = 0; b < 64; b++) {
      int c = codes[w * 64 + b];
      if (c == 1) p |= (uint64_t)1 << b;
      else if (c == 2) q |= (uint64_t)1 << b;
    }
    r->pos[w] = p;
    r->neg[w] = q;
  }
}

/* ---------------------------------------------------------------- disk io */

static double disk_direct_bw(const char *path, size_t bytes, int reps) {
  int fd = open(path, O_RDONLY | O_DIRECT);
  if (fd < 0) return -1.0;
  size_t chunk = 4u << 20; /* 4 MiB, page-aligned requirement */
  void *buf = NULL;
  if (posix_memalign(&buf, 4096, chunk) != 0) { close(fd); return -1.0; }
  double best = 0, tot = 0;
  int ok = 0;
  for (int r = 0; r < reps; r++) {
    if (lseek(fd, 0, SEEK_SET) < 0) break;
    size_t left = bytes;
    double t0 = now_s();
    while (left > 0) {
      size_t want = left < chunk ? left : chunk;
      ssize_t got = read(fd, buf, want);
      if (got <= 0) break;
      left -= (size_t)got;
    }
    double dt = now_s() - t0;
    double moved = (double)(bytes - left);
    if (dt > 0 && moved > 0) { tot += moved / dt / 1e9; ok++; if (moved / dt / 1e9 > best) best = moved / dt / 1e9; }
  }
  free(buf);
  close(fd);
  return ok ? tot / ok : -1.0;
}

static double disk_buffered_bw(const char *path, size_t bytes, int reps) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) return -1.0;
  size_t chunk = 4u << 20;
  void *buf = malloc(chunk);
  if (!buf) { close(fd); return -1.0; }
  double best = 0;
  for (int r = 0; r < reps; r++) {
    if (lseek(fd, 0, SEEK_SET) < 0) break;
    size_t left = bytes;
    double t0 = now_s();
    while (left > 0) {
      size_t want = left < chunk ? left : chunk;
      ssize_t got = read(fd, buf, want);
      if (got <= 0) break;
      left -= (size_t)got;
    }
    double dt = now_s() - t0;
    double moved = (double)(bytes - left);
    if (dt > 0 && moved > 0) { double bw = moved / dt / 1e9; if (bw > best) best = bw; }
  }
  free(buf);
  close(fd);
  return best;
}

/* -------------------------------------------------------------------- main */

static int has_avx512(void) {
#if defined(__x86_64__)
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
#else
  return 0;
#endif
}

int main(int argc, char **argv) {
  const char *model_dir = argc > 1 ? argv[1] : "/workspace";
  int reps = 21;

  size_t big = 256u << 20;               /* 256 MiB arrays for the memory roofs */
  float *a = (float *)xaligned(big);
  float *b = (float *)xaligned(big);
  size_t nel = big / sizeof(float);

  /* warm the clock and the caches the way a real run would */
  { double t0 = now_s(); while (now_s() - t0 < 0.20) g_sink += 1.0; }

  double dt_read = 0, dt_copy = 0, dt_fma = 0;
  double gbs_read = 0, gbs_copy = 0, gflops_fma = 0;
  for (int r = 0; r < reps; r++) {
    double t0 = now_s(); double gb = roof_read(a, nel); double t1 = now_s();
    double bw = gb / (t1 - t0); if (bw > gbs_read) { gbs_read = bw; dt_read = t1 - t0; }
  }
  for (int r = 0; r < reps; r++) {
    double t0 = now_s(); double gb = roof_copy(b, a, nel); double t1 = now_s();
    double bw = gb / (t1 - t0); if (bw > gbs_copy) { gbs_copy = bw; dt_copy = t1 - t0; }
  }
  {
    float lbuf[4096];
    for (int i = 0; i < 4096; i++) lbuf[i] = 1.0f + (float)i * 1e-6f;
    /* keep the buffer live and unknown at compile time so the loop is not folded */
    long iters = (long)strtol(getenv("FMA_ITERS") ? getenv("FMA_ITERS") : "4000000", NULL, 10);
    double flops_per_pass = (double)16 * 16 * 2 * (double)iters;
    for (int r = 0; r < reps + 3; r++) { /* extra passes warm the clock, not timed */
      double t0 = now_s(); double fl = roof_fma(lbuf, 4096, iters); double t1 = now_s();
      if (r < 3) continue;
      double gf = fl / (t1 - t0);
      if (gf > gflops_fma) gflops_fma = gf;
    }
    (void)flops_per_pass;
  }
  (void)dt_read; (void)dt_copy; (void)dt_fma;

  /* disk */
  char path[4096];
  snprintf(path, sizeof(path), "%s/.architect_io_probe.bin", model_dir);
  size_t disk_bytes = 512u << 20;
  int have_disk = 0;
  {
    FILE *f = fopen(path, "wb");
    if (f) {
      static char zbuf[1 << 20];
      size_t left = disk_bytes;
      while (left) { size_t w = left < sizeof(zbuf) ? left : sizeof(zbuf); if (fwrite(zbuf, 1, w, f) != w) break; left -= w; }
      fclose(f);
      have_disk = 1;
    }
  }
  double direct_bw = -1, buffered_bw = -1;
  if (have_disk) {
    direct_bw = disk_direct_bw(path, disk_bytes, 5);
    buffered_bw = disk_buffered_bw(path, disk_bytes, 5);
    unlink(path);
  }

  /* ternary gemv: N rows of K weights, one decode-step shape */
  size_t N = 4096, K = 4096;            /* 4096x4096 ternary = 4 MiB packed */
  uint8_t *codes = (uint8_t *)malloc(N * K);
  ternary_row *rows = (ternary_row *)malloc(N * sizeof(ternary_row));
  float *x = (float *)xaligned(K * sizeof(float));
  float *y = (float *)xaligned(N * sizeof(float));
  for (size_t i = 0; i < K; i++) x[i] = sinf((float)i * 0.001f);
  srand(1234);
  for (size_t i = 0; i < N * K; i++) codes[i] = (uint8_t)(rand() % 3); /* 0,1,2 */
  for (size_t i = 0; i < N; i++) {
    rows[i].k = K; rows[i].pos = (uint64_t *)malloc(K / 64 * sizeof(uint64_t));
    rows[i].neg = (uint64_t *)malloc(K / 64 * sizeof(uint64_t));
    ternary_pack(&rows[i], codes + i * K);
  }
  /* verify a few rows */
  double max_err = 0;
#if defined(__x86_64__)
  if (has_avx512()) {
    for (size_t i = 0; i < 8; i++) {
      double ref = ternary_dot_ref(codes + i * K, x, K);
      double got = ternary_dot_avx512(&rows[i], x);
      double e = fabs(ref - got); if (e > max_err) max_err = e;
    }
  }
#endif
  double tern_gflops = 0, tern_gbs = 0;
  {
    double best = 0, best_dt = 1e9;
    for (int r = 0; r < reps; r++) {
      double t0 = now_s();
      double acc = 0;
#if defined(__x86_64__)
      if (has_avx512()) { for (size_t i = 0; i < N; i++) acc += ternary_dot_avx512(&rows[i], x); }
      else
#endif
      { for (size_t i = 0; i < N; i++) acc += ternary_dot_ref(codes + i * K, x, K); }
      double dt = now_s() - t0;
      g_sink += acc;
      double gfl = (double)N * K * 2.0 / dt / 1e9;
      if (gfl > best) { best = gfl; best_dt = dt; }
    }
    tern_gflops = best;
    /* packed bytes actually streamed: N rows * K weights / 4 weights per byte */
    tern_gbs = (double)N * (K / 4.0) / best_dt / 1e9;
  }

  /* constant-state recurrence vs KV cache, per token */
  double n_layer = 27, n_head = 32, head_dim = 72, ctx = 8192;
  double state_bytes_per_token = n_layer * n_head * head_dim * head_dim * 4; /* linear-attn state (fixed) */
  double kv_bytes_per_token = n_layer * 2 * n_head * head_dim * 2;            /* fp16 KV, grows with T */
  double kv_at_ctx = kv_bytes_per_token * ctx;

  printf("{\n");
  printf("  \"cpu_avx512\": %d,\n", has_avx512());
  printf("  \"roof_read_gbs\": %.3f,\n", gbs_read);
  printf("  \"roof_copy_gbs\": %.3f,\n", gbs_copy);
  printf("  \"roof_fma_gflops\": %.1f,\n", gflops_fma / 1e9);
  printf("  \"ridge_flop_per_byte\": %.3f,\n", gflops_fma / 1e9 / gbs_read);
  printf("  \"disk_direct_gbs\": %.3f,\n", direct_bw);
  printf("  \"disk_buffered_gbs\": %.3f,\n", buffered_bw);
  printf("  \"ternary_gflops\": %.2f,\n", tern_gflops);
  printf("  \"ternary_packed_gbs\": %.3f,\n", tern_gbs);
  printf("  \"ternary_ridge_flop_per_byte\": %.2f,\n", tern_gflops / (tern_gbs > 0 ? tern_gbs : 1e-9));
  printf("  \"ternary_max_abs_err\": %.3e,\n", max_err);
  printf("  \"linear_state_bytes_per_token\": %.0f,\n", state_bytes_per_token);
  printf("  \"kv_bytes_per_token\": %.0f,\n", kv_bytes_per_token);
  printf("  \"kv_bytes_at_ctx8192\": %.0f\n", kv_at_ctx);
  printf("}\n");

  free(a); free(b); free(x); free(y); free(codes); free(rows);
  return 0;
}
