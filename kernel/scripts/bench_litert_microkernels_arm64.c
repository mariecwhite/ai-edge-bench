// Repeatable Standalone Microkernel Benchmark Harness for LiteRT-LM on ARM64
// Apple Silicon (XNNPACK + KleidiAI SME2 vs. YNNPACK SME2/SME + NEON Baselines).
//
// Links directly against compiled upstream XNNPACK, KleidiAI, and YNNPACK
// microkernel object files (`libaeb_ukernels_arm64.a`) and benchmarks all 7
// canonical Gemma 4 31B microkernel pairs across 1T, 6T (Super cluster), and
// 18T (full Apple M5 Max chip) using pthreads, QOS_CLASS_USER_INTERACTIVE, and
// lock-free atomic dynamic tile dispatch for heterogeneous core scaling.
//
// Usage:
//   ./bench_ukernels_arm64 --threads 6,18 --trials 7 \
//       --xnnpack-commit <sha> --output ukernel_benchmarks_apple_m5_max.json

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef __APPLE__
#include <pthread/qos.h>
#endif

#define MAX_TRIALS 16
#define MAX_THREAD_CFGS 8
#define MAX_THREADS 64
#define ALIGN_BYTES 128

// -----------------------------------------------------------------------------
// Timing, Allocation, Synchronization & Statistics Helpers
// -----------------------------------------------------------------------------

typedef struct {
  double median_ms;
  double gops_median;
  double gops_mean;
  double gops_min;
  double gops_max;
  double gops_stdev;
  double cv_pct;
  double gbps_median;
} bench_stats_t;

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *alloc_aligned(size_t bytes) {
  size_t sz = (bytes + ALIGN_BYTES - 1) & ~(size_t)(ALIGN_BYTES - 1);
  if (sz == 0) sz = ALIGN_BYTES;
  void *ptr = NULL;
  if (posix_memalign(&ptr, ALIGN_BYTES, sz) != 0 || !ptr) {
    fprintf(stderr, "ERROR: posix_memalign(%zu) failed\n", sz);
    exit(1);
  }
  return ptr;
}

static void set_high_qos(void) {
#ifdef __APPLE__
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

typedef struct {
  int count;
  atomic_int arrived;
  atomic_int phase;
} spin_barrier_t;

static void sb_init(spin_barrier_t *b, int count) {
  b->count = count;
  atomic_init(&b->arrived, 0);
  atomic_init(&b->phase, 0);
}

static void sb_wait(spin_barrier_t *b) {
  int p = atomic_load_explicit(&b->phase, memory_order_relaxed);
  if (atomic_fetch_add_explicit(&b->arrived, 1, memory_order_acq_rel) ==
      b->count - 1) {
    atomic_store_explicit(&b->arrived, 0, memory_order_relaxed);
    atomic_store_explicit(&b->phase, p ^ 1, memory_order_release);
  } else {
    while (atomic_load_explicit(&b->phase, memory_order_acquire) == p) {
#if defined(__aarch64__)
      __asm__ volatile("yield" ::: "memory");
#endif
    }
  }
}

static int cmp_double(const void *a, const void *b) {
  double da = *(const double *)a;
  double db = *(const double *)b;
  return (da > db) - (da < db);
}

static bench_stats_t compute_bench_stats(const double *dt_samples, int n,
                                         double total_ops, double total_bytes) {
  double gops[MAX_TRIALS];
  double dts[MAX_TRIALS];
  for (int i = 0; i < n; ++i) {
    dts[i] = dt_samples[i];
    gops[i] = (total_ops > 0.0) ? (total_ops / dt_samples[i] / 1e9)
                                : (total_bytes / dt_samples[i] / 1e9);
  }
  qsort(dts, (size_t)n, sizeof(double), cmp_double);
  qsort(gops, (size_t)n, sizeof(double), cmp_double);

  bench_stats_t s;
  double med_dt =
      (n % 2 == 1) ? dts[n / 2] : 0.5 * (dts[n / 2 - 1] + dts[n / 2]);
  s.median_ms = med_dt * 1e3;
  s.gops_min = gops[0];
  s.gops_max = gops[n - 1];
  s.gops_median =
      (n % 2 == 1) ? gops[n / 2] : 0.5 * (gops[n / 2 - 1] + gops[n / 2]);

  double sum = 0.0;
  for (int i = 0; i < n; ++i) sum += gops[i];
  s.gops_mean = sum / n;

  double var = 0.0;
  for (int i = 0; i < n; ++i) {
    double d = gops[i] - s.gops_mean;
    var += d * d;
  }
  s.gops_stdev = (n > 1) ? sqrt(var / (n - 1)) : 0.0;
  s.cv_pct = (s.gops_mean > 0.0) ? (100.0 * s.gops_stdev / s.gops_mean) : 0.0;
  s.gbps_median = total_bytes / med_dt / 1e9;
  return s;
}

static int parse_thread_list(const char *arg, int *out, int max_count) {
  int count = 0;
  const char *p = arg;
  while (p && *p && count < max_count) {
    char *end = NULL;
    long v = strtol(p, &end, 10);
    if (end == p) break;
    if (v > 1) out[count++] = (int)v;
    if (*end == ',') {
      p = end + 1;
    } else {
      break;
    }
  }
  return count;
}

// Generic pthread dynamic work-stealing trial runner
typedef void (*work_item_fn)(int work_idx, void *ctx);

typedef struct {
  int total_work;
  int warmup_work;
  int chunk;
  int warmup_chunk;
  int reps;
  int num_trials;
  work_item_fn fn;
  void *ctx;
  spin_barrier_t bar;
  atomic_int next_work;
  double t0;
  double dt_samples[MAX_TRIALS];
} parallel_run_t;

typedef struct {
  int tid;
  parallel_run_t *run;
} worker_arg_t;

static void execute_dynamic_work(parallel_run_t *run, int limit, int chunk) {
  for (;;) {
    int start = atomic_fetch_add_explicit(&run->next_work, chunk,
                                          memory_order_relaxed);
    if (start >= limit) break;
    int end = start + chunk;
    if (end > limit) end = limit;
    for (int w = start; w < end; ++w) {
      run->fn(w, run->ctx);
    }
  }
}

static void *worker_thread_fn(void *arg) {
  worker_arg_t *wa = (worker_arg_t *)arg;
  parallel_run_t *run = wa->run;
  int tid = wa->tid;
  set_high_qos();

  // Warmup pass
  sb_wait(&run->bar);
  execute_dynamic_work(run, run->warmup_work, run->warmup_chunk);
  sb_wait(&run->bar);

  // Timed trials
  for (int trial = 0; trial < run->num_trials; ++trial) {
    if (tid == 0) {
      atomic_store_explicit(&run->next_work, 0, memory_order_relaxed);
      run->t0 = now_sec();
    }
    sb_wait(&run->bar);
    execute_dynamic_work(run, run->total_work, run->chunk);
    sb_wait(&run->bar);
    if (tid == 0) {
      double t1 = now_sec();
      run->dt_samples[trial] = (t1 - run->t0) / (double)run->reps;
    }
  }
  return NULL;
}

static bench_stats_t run_parallel_bench(int num_threads, int total_tiles,
                                        int reps, int chunk, int num_trials,
                                        work_item_fn fn, void *ctx,
                                        double total_ops, double total_bytes) {
  if (num_threads <= 1) {
    double dt_samples[MAX_TRIALS];
    for (int t = 0; t < total_tiles; ++t) fn(t, ctx);
    for (int trial = 0; trial < num_trials; ++trial) {
      double t0 = now_sec();
      int total_work = reps * total_tiles;
      for (int w = 0; w < total_work; ++w) fn(w, ctx);
      double t1 = now_sec();
      dt_samples[trial] = (t1 - t0) / (double)reps;
    }
    return compute_bench_stats(dt_samples, num_trials, total_ops, total_bytes);
  }

  parallel_run_t run;
  memset(&run, 0, sizeof(run));
  run.total_work = reps * total_tiles;
  run.warmup_work = total_tiles;
  int min_chunk = (chunk > 0) ? chunk : 1;
  // Target ~48 dynamic chunks per thread so heterogeneous Super/Perf cores
  // load-balance within ~2% while eliminating cross-cluster atomic contention
  // on small NEON tiles (e.g., 4x16x1152 or 6x8x72).
  int target_chunks = num_threads * 48;
  int auto_chunk = (run.total_work >= target_chunks)
                       ? (run.total_work / target_chunks)
                       : 1;
  run.chunk = (auto_chunk > min_chunk) ? auto_chunk : min_chunk;
  int w_chunk = total_tiles / (num_threads * 4);
  if (w_chunk < 1) w_chunk = 1;
  if (w_chunk > run.chunk) w_chunk = run.chunk;
  run.warmup_chunk = w_chunk;
  run.reps = reps;
  run.num_trials = num_trials;
  run.fn = fn;
  run.ctx = ctx;
  sb_init(&run.bar, num_threads);
  atomic_init(&run.next_work, 0);

  pthread_t th[MAX_THREADS];
  worker_arg_t wa[MAX_THREADS];
  for (int i = 0; i < num_threads; ++i) {
    wa[i].tid = i;
    wa[i].run = &run;
  }
  for (int i = 1; i < num_threads; ++i) {
    pthread_create(&th[i], NULL, worker_thread_fn, &wa[i]);
  }
  worker_thread_fn(&wa[0]);
  for (int i = 1; i < num_threads; ++i) {
    pthread_join(th[i], NULL);
  }
  return compute_bench_stats(run.dt_samples, num_trials, total_ops,
                             total_bytes);
}

// -----------------------------------------------------------------------------
// Direct-Linked Upstream XNNPACK, KleidiAI, and YNNPACK Symbols
// -----------------------------------------------------------------------------

typedef struct {
  struct {
    float min;
    float max;
  } scalar;
} aeb_xnn_f32_minmax_params_t;

typedef struct {
  struct {
    float min;
    float max;
    int32_t kernel_zero_point;
  } scalar;
} aeb_xnn_f32_qc4w_minmax_params_t;

typedef struct {
  int32_t zero_point;
  float inv_scale;
} aeb_xnn_qd8_quant_params_t;

typedef struct {
  int8_t input_zero_point;
  uint8_t kernel_zero_point;
} aeb_xnn_qs8_qc4w_packing_params_t;

typedef struct {
  int8_t input_zero_point;
} aeb_xnn_qs8_packing_params_t;

typedef struct {
  struct {
    int16_t output_zero_point;
    int16_t output_min;
    int16_t output_max;
  } fp32_scalar;
} aeb_xnn_qs8_qc8w_conv_minmax_params_t;

struct kai_rhs_pack_qs4cxs1s0_param {
  int8_t lhs_zero_point;
  uint8_t rhs_zero_point;
};

struct kai_rhs_pack_qsi8cx_params {
  int32_t lhs_zero_point;
  float scale_multiplier;
};

// KleidiAI packers (LHS quantize+pack, RHS int4/int8/int2 pack)
extern size_t kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr);
extern size_t kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(
    size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr);
extern void kai_run_lhs_quant_pack_qai8dxp_f32(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start,
    const float *lhs, size_t lhs_stride, void *lhs_packed);

extern size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon(
    size_t k, size_t nr, size_t kr, size_t sr);
extern void kai_run_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr,
    const uint8_t *rhs, const float *bias, const float *scale, void *rhs_packed,
    size_t extra_bytes, const struct kai_rhs_pack_qs4cxs1s0_param *params);

extern size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
    size_t k, size_t nr, size_t kr, size_t sr);
extern void kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr,
    const int8_t *rhs, const float *bias, const float *scale, void *rhs_packed,
    size_t extra_bytes, const struct kai_rhs_pack_qsi8cx_params *params);

extern size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(
    size_t k, size_t nr, size_t kr, size_t sr);
extern void kai_run_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr,
    const uint8_t *rhs, const float *bias, const float *scale, void *rhs_packed,
    size_t extra_bytes, const struct kai_rhs_pack_qs4cxs1s0_param *params,
    const int32_t *lut_arg);

// KleidiAI FP32 SME packing kernels (low-level ASM entry points used directly)
extern void kai_kernel_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme(
    size_t height, size_t width, const void *in, size_t row_offset, void *out);
extern void kai_kernel_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme(
    size_t height, size_t width, const void *in, size_t row_offset, void *out,
    const void *bias);

// XNNPACK SME2 (KleidiAI) microkernels
typedef void (*xnn_qp8_gemm_fn)(size_t m, size_t n, size_t k,
                                const void *lhs_packed, const void *rhs_packed,
                                float *dst, size_t dst_stride_row,
                                size_t dst_stride_col, const void *params);

extern void xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void *lhs_packed,
    const void *rhs_packed, float *dst, size_t dst_stride_row,
    size_t dst_stride_col, const void *params);
extern void xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void *lhs_packed,
    const void *rhs_packed, float *dst, size_t dst_stride_row,
    size_t dst_stride_col, const void *params);
extern void xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void *lhs_packed,
    const void *rhs_packed, float *dst, size_t dst_stride_row,
    size_t dst_stride_col, const void *params);
extern void xnn_qp8_f32_qc2w_gemm_minmax_ukernel_16x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void *lhs_packed,
    const void *rhs_packed, float *dst, size_t dst_stride_row,
    size_t dst_stride_col, const void *params);
extern void xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2(
    size_t m, size_t n, size_t k_bytes, const void *lhs_packed,
    const void *rhs_packed, float *dst, size_t dst_stride_row,
    size_t dst_stride_col, const void *params);

// XNNPACK NEON I8MM / DotProd / FMA / Softmax / PackW microkernels
typedef void (*xnn_qd8_gemm_fn)(size_t mr, size_t nc, size_t kc, const void *a,
                                size_t a_stride, const void *w, float *c,
                                size_t cm_stride, size_t cn_stride,
                                const void *params, const void *quant_params);

extern void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm(
    size_t mr, size_t nc, size_t kc, const void *a, size_t a_stride,
    const void *w, float *c, size_t cm_stride, size_t cn_stride,
    const void *params, const void *quant_params);
extern void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__neoni8mm(
    size_t mr, size_t nc, size_t kc, const void *a, size_t a_stride,
    const void *w, float *c, size_t cm_stride, size_t cn_stride,
    const void *params, const void *quant_params);
extern void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm(
    size_t mr, size_t nc, size_t kc, const void *a, size_t a_stride,
    const void *w, float *c, size_t cm_stride, size_t cn_stride,
    const void *params, const void *quant_params);
extern void xnn_qs8_qc4w_packw_gemm_goi_ukernel_x16c8__neon_prfm(
    size_t g, size_t nc, size_t kc, size_t nr, size_t kr, size_t sr,
    size_t n_stride, const uint8_t *weights, const int32_t *bias,
    const float *scale, void *packed_weights, size_t extra_bytes,
    const void *params);
extern void xnn_qs8_packw_gemm_goi_ukernel_x16c8__neon(
    size_t g, size_t nc, size_t kc, size_t nr, size_t kr, size_t sr,
    size_t n_stride, const int8_t *weights, const int32_t *bias,
    const void *scale, void *packed_weights, size_t extra_bytes,
    const void *params);
extern void xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128(
    size_t mr, size_t nc, size_t kc_bytes, const float *a,
    size_t a_stride_bytes, const void *w, float *c, size_t cm_stride_bytes,
    size_t cn_stride_bytes, const void *params);
extern void xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_8x8c4__neondot(
    size_t mr, size_t nc, size_t kc, const int8_t *a, size_t a_stride,
    const void *w, int8_t *c, size_t cm_stride, size_t cn_stride,
    const void *params);
extern void xnn_f32_rmax_ukernel__neon_u16_acc4(size_t batch_bytes,
                                                const float *input,
                                                float *output,
                                                const void *params);
extern void xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc2(
    size_t batch_bytes, const float *input, const float *max, float *output,
    float *sum, const void *params);
extern void xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm(
    size_t g, size_t nc, size_t kc, size_t nr, size_t kr, size_t sr,
    size_t k_stride, const uint32_t *weights, const uint32_t *bias,
    const void *scale, void *packed_weights, size_t extra_bytes,
    const void *params);

// YNNPACK ARM64 bridge wrappers
typedef void (*ynn_dot_fn)(size_t M, size_t N, size_t K3, size_t K2, size_t K1,
                           size_t A_stride_m, size_t A_stride_k3,
                           size_t A_stride_k2, const void *A,
                           size_t B_stride_k3, size_t B_stride_k2,
                           size_t B_stride_k1, const void *B,
                           size_t C_in_stride_m, const void *C_in,
                           size_t C_out_stride_m, void *C_out, void *state);

extern void aeb_ynn_dot_int8_int4_1x32_neondot(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_dot_int8_int4_sme(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_dot_int8_int4_10x8_neoni8mm(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_dot_int8_int4_6x16_neoni8mm(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_dot_fp32_sme2(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_dot_fp32_sme(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_dot_fp32_6x8_neon(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_dot_int8_int8_sme2(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_dot_int8_int8_sme(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_dot_int8_int8_10x8_neoni8mm(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_dot_int8_int8_6x16_neoni8mm(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_dot_int8_int2_sme(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_dot_int8_int2_8x8_neondot(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_max_k1_fp32_neon(size_t n, size_t k, size_t a_stride_n,
                                     const void *a, void *x0, void *x1);
extern void aeb_ynn_sum_k1_fp32_neon(size_t n, size_t k, size_t a_stride_n,
                                     const void *a, void *x0, void *x1);
extern void aeb_ynn_transpose_x32_neon(size_t m, size_t n, size_t n_bytes_a,
                                       size_t stride_a, const void *a,
                                       size_t stride_x, void *x);

// -----------------------------------------------------------------------------
// JSON Emission Helper
// -----------------------------------------------------------------------------

static void print_kernel_result(FILE *out, const char *pair_id,
                                const char *category, const char *backend,
                                const char *kernel_name, const char *isa,
                                int M, int N, int K, const char *mem_regime,
                                bench_stats_t s1, const bench_stats_t *smc,
                                const int *mc_threads, int num_mc, double ops,
                                double bytes, int comma) {
  double ai = (bytes > 0.0) ? (ops / bytes) : 0.0;
  fprintf(out, "    {\n");
  fprintf(out, "      \"pair_id\": \"%s\",\n", pair_id);
  fprintf(out, "      \"category\": \"%s\",\n", category);
  fprintf(out, "      \"backend\": \"%s\",\n", backend);
  fprintf(out, "      \"kernel\": \"%s\",\n", kernel_name);
  fprintf(out, "      \"isa\": \"%s\",\n", isa);
  fprintf(out, "      \"shape\": { \"M\": %d, \"N\": %d, \"K\": %d },\n", M, N,
          K);
  fprintf(out, "      \"memory_regime\": \"%s\",\n", mem_regime);
  fprintf(out, "      \"arithmetic_intensity_ops_per_byte\": %.3f,\n", ai);
  fprintf(out, "      \"single_core\": {\n");
  fprintf(out,
          "        \"latency_ms\": %.4f, \"gops\": %.2f, \"gbps\": %.2f, "
          "\"mean_gops\": %.2f, \"min_gops\": %.2f, \"max_gops\": %.2f, "
          "\"cv_pct\": %.2f\n",
          s1.median_ms, s1.gops_median, s1.gbps_median, s1.gops_mean,
          s1.gops_min, s1.gops_max, s1.cv_pct);
  fprintf(out, "      }%s\n", num_mc > 0 ? "," : "");
  for (int i = 0; i < num_mc; ++i) {
    int trailing = (i + 1 < num_mc);
    fprintf(out, "      \"threads_%dc\": {\n", mc_threads[i]);
    fprintf(out,
            "        \"latency_ms\": %.4f, \"gops\": %.2f, \"gbps\": %.2f, "
            "\"mean_gops\": %.2f, \"min_gops\": %.2f, \"max_gops\": %.2f, "
            "\"cv_pct\": %.2f\n",
            smc[i].median_ms, smc[i].gops_median, smc[i].gbps_median,
            smc[i].gops_mean, smc[i].gops_min, smc[i].gops_max, smc[i].cv_pct);
    fprintf(out, "      }%s\n", trailing ? "," : "");
  }
  fprintf(out, "    }%s\n", comma ? "," : "");
  fflush(out);
}

// -----------------------------------------------------------------------------
// 1. XNNPACK KleidiAI SME2 Quantized GEMM/GEMV Runner (4-bit, 8-bit, 2-bit)
// -----------------------------------------------------------------------------

typedef struct {
  xnn_qp8_gemm_fn fn;
  int M, N, K;
  int mr_tile, nr_tile;
  int num_m_tiles, num_n_panels, total_tiles, num_replicas;
  size_t lhs_tile_stride;
  size_t rhs_panel_stride;
  size_t single_rhs_bytes;
  const uint8_t *lhs_packed;
  const uint8_t *rhs_packed;
  float *C;
  aeb_xnn_f32_qc4w_minmax_params_t params;
} xnn_qp8_ctx_t;

static void xnn_qp8_work(int w, void *arg) {
  xnn_qp8_ctx_t *c = (xnn_qp8_ctx_t *)arg;
  int r = w / c->total_tiles;
  int t = w % c->total_tiles;
  int p = t / c->num_m_tiles;
  int mt = t % c->num_m_tiles;
  int m_idx = mt * c->mr_tile;
  int cur_mr = (c->M - m_idx < c->mr_tile) ? (c->M - m_idx) : c->mr_tile;
  int n_idx = p * c->nr_tile;
  int cur_nc = (c->N - n_idx < c->nr_tile) ? (c->N - n_idx) : c->nr_tile;
  const uint8_t *lhs_ptr = c->lhs_packed + (size_t)mt * c->lhs_tile_stride;
  const uint8_t *rhs_base =
      c->rhs_packed + (size_t)(r % c->num_replicas) * c->single_rhs_bytes;
  const uint8_t *rhs_ptr = rhs_base + (size_t)p * c->rhs_panel_stride;
  c->fn((size_t)cur_mr, (size_t)cur_nc, (size_t)c->K, lhs_ptr, rhs_ptr,
        c->C + (size_t)m_idx * (size_t)c->N + (size_t)n_idx,
        (size_t)c->N * sizeof(float), sizeof(float), &c->params);
}

static bench_stats_t bench_xnn_qp8_sme2(xnn_qp8_gemm_fn fn, int mr_tile,
                                        int nr_tile, int M, int N, int K,
                                        int bits_per_weight, int cold_dram,
                                        int num_threads, int reps,
                                        int num_trials) {
  const size_t kr = 4, sr = 1;
  int num_m_tiles = (M + mr_tile - 1) / mr_tile;
  int num_n_panels = (N + nr_tile - 1) / nr_tile;
  int total_tiles = num_m_tiles * num_n_panels;

  size_t lhs_tile_stride = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(
      (size_t)mr_tile, (size_t)K, (size_t)mr_tile, kr, sr);
  size_t lhs_total_bytes = (size_t)num_m_tiles * lhs_tile_stride;

  size_t rhs_panel_stride = 0;
  if (bits_per_weight == 4) {
    rhs_panel_stride =
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon(
            (size_t)K, (size_t)nr_tile, kr, sr);
  } else if (bits_per_weight == 8) {
    rhs_panel_stride =
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
            (size_t)K, (size_t)nr_tile, kr, sr);
  } else {
    rhs_panel_stride =
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(
            (size_t)K, (size_t)nr_tile, kr, sr);
  }
  size_t single_rhs_bytes = (size_t)num_n_panels * rhs_panel_stride;

  // For cold Unified Memory streaming, allocate >= 512 MiB pool (> 8x SLC)
  int num_replicas = 1;
  if (cold_dram) {
    size_t target_pool = 512ULL * 1024ULL * 1024ULL;
    num_replicas = (int)((target_pool + single_rhs_bytes - 1) / single_rhs_bytes);
    if (num_replicas < 2) num_replicas = 2;
    if (num_replicas > 64) num_replicas = 64;
  }

  // Pack 1 authentic LHS tile and replicate across num_m_tiles
  float *lhs_raw = (float *)alloc_aligned((size_t)mr_tile * (size_t)K * sizeof(float));
  for (size_t i = 0; i < (size_t)mr_tile * (size_t)K; ++i) lhs_raw[i] = 1.0f;
  uint8_t *lhs_packed = (uint8_t *)alloc_aligned(lhs_total_bytes + 4096);
  kai_run_lhs_quant_pack_qai8dxp_f32((size_t)mr_tile, (size_t)K, (size_t)mr_tile,
                                     kr, sr, 0, lhs_raw,
                                     (size_t)K * sizeof(float), lhs_packed);
  for (int mt = 1; mt < num_m_tiles; ++mt) {
    memcpy(lhs_packed + (size_t)mt * lhs_tile_stride, lhs_packed,
           lhs_tile_stride);
  }
  free(lhs_raw);

  // Pack 1 authentic RHS panel and replicate across num_n_panels & num_replicas
  float *bias = (float *)alloc_aligned((size_t)nr_tile * sizeof(float));
  float *scale = (float *)alloc_aligned((size_t)nr_tile * sizeof(float));
  for (int j = 0; j < nr_tile; ++j) {
    bias[j] = 0.0f;
    scale[j] = 0.01f;
  }
  uint8_t *one_panel = (uint8_t *)alloc_aligned(rhs_panel_stride + 4096);
  if (bits_per_weight == 4) {
    size_t raw_bytes = (size_t)nr_tile * (size_t)K / 2;
    uint8_t *rhs_raw = (uint8_t *)alloc_aligned(raw_bytes);
    memset(rhs_raw, 0x99, raw_bytes);  // u4=9 -> signed +1 (with zp=8)
    struct kai_rhs_pack_qs4cxs1s0_param kp = {1, 8};
    kai_run_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon(
        1, (size_t)nr_tile, (size_t)K, (size_t)nr_tile, kr, sr, rhs_raw, bias,
        scale, one_panel, 0, &kp);
    free(rhs_raw);
  } else if (bits_per_weight == 8) {
    size_t raw_bytes = (size_t)nr_tile * (size_t)K;
    int8_t *rhs_raw = (int8_t *)alloc_aligned(raw_bytes);
    memset(rhs_raw, 1, raw_bytes);
    struct kai_rhs_pack_qsi8cx_params kp = {1, 1.0f};
    kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        1, (size_t)nr_tile, (size_t)K, (size_t)nr_tile, kr, sr, rhs_raw, bias,
        scale, one_panel, 0, &kp);
    free(rhs_raw);
  } else {
    size_t raw_bytes = (size_t)nr_tile * (size_t)K / 4;
    uint8_t *rhs_raw = (uint8_t *)alloc_aligned(raw_bytes);
    memset(rhs_raw, 0x55, raw_bytes);  // 2-bit code 1 -> lut[1] = +1
    struct kai_rhs_pack_qs4cxs1s0_param kp = {1, 2};
    static const int32_t lut[4] = {0, 1, -2, -1};
    kai_run_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(
        1, (size_t)nr_tile, (size_t)K, (size_t)nr_tile, kr, sr, rhs_raw, bias,
        scale, one_panel, 0, &kp, lut);
    free(rhs_raw);
  }
  free(bias);
  free(scale);

  uint8_t *rhs_packed =
      (uint8_t *)alloc_aligned((size_t)num_replicas * single_rhs_bytes + 4096);
  for (int p = 0; p < num_n_panels; ++p) {
    memcpy(rhs_packed + (size_t)p * rhs_panel_stride, one_panel,
           rhs_panel_stride);
  }
  for (int rep = 1; rep < num_replicas; ++rep) {
    memcpy(rhs_packed + (size_t)rep * single_rhs_bytes, rhs_packed,
           single_rhs_bytes);
  }
  free(one_panel);

  float *C = (float *)alloc_aligned((size_t)M * (size_t)N * sizeof(float) + 4096);
  memset(C, 0, (size_t)M * (size_t)N * sizeof(float));

  xnn_qp8_ctx_t ctx;
  ctx.fn = fn;
  ctx.M = M;
  ctx.N = N;
  ctx.K = K;
  ctx.mr_tile = mr_tile;
  ctx.nr_tile = nr_tile;
  ctx.num_m_tiles = num_m_tiles;
  ctx.num_n_panels = num_n_panels;
  ctx.total_tiles = total_tiles;
  ctx.num_replicas = num_replicas;
  ctx.lhs_tile_stride = lhs_tile_stride;
  ctx.rhs_panel_stride = rhs_panel_stride;
  ctx.single_rhs_bytes = single_rhs_bytes;
  ctx.lhs_packed = lhs_packed;
  ctx.rhs_packed = rhs_packed;
  ctx.C = C;
  ctx.params.scalar.min = -1e6f;
  ctx.params.scalar.max = 1e6f;
  ctx.params.scalar.kernel_zero_point = 0;

  double total_ops = 2.0 * (double)M * (double)N * (double)K;
  double total_bytes =
      (double)M * (double)K +
      (double)N * (double)K * (double)bits_per_weight / 8.0 +
      (double)M * (double)N * sizeof(float);

  bench_stats_t st = run_parallel_bench(num_threads, total_tiles, reps, 1,
                                        num_trials, xnn_qp8_work, &ctx,
                                        total_ops, total_bytes);
  free(lhs_packed);
  free(rhs_packed);
  free(C);
  return st;
}

// -----------------------------------------------------------------------------
// 2. XNNPACK KleidiAI SME2 FP32 GEMM Runner (pf32 32x32__neonsme2)
// -----------------------------------------------------------------------------

typedef struct {
  int M, N, K;
  int mr_tile, nr_tile;
  int num_m_tiles, num_n_panels, total_tiles;
  size_t lhs_tile_stride;
  size_t rhs_panel_stride;
  const uint8_t *lhs_packed;
  const uint8_t *rhs_packed;
  float *C;
  aeb_xnn_f32_minmax_params_t params;
} xnn_pf32_ctx_t;

static void xnn_pf32_work(int w, void *arg) {
  xnn_pf32_ctx_t *c = (xnn_pf32_ctx_t *)arg;
  int t = w % c->total_tiles;
  int p = t / c->num_m_tiles;
  int mt = t % c->num_m_tiles;
  int m_idx = mt * c->mr_tile;
  int cur_mr = (c->M - m_idx < c->mr_tile) ? (c->M - m_idx) : c->mr_tile;
  int n_idx = p * c->nr_tile;
  int cur_nc = (c->N - n_idx < c->nr_tile) ? (c->N - n_idx) : c->nr_tile;
  const uint8_t *lhs_ptr = c->lhs_packed + (size_t)mt * c->lhs_tile_stride;
  const uint8_t *rhs_ptr = c->rhs_packed + (size_t)p * c->rhs_panel_stride;
  xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2(
      (size_t)cur_mr, (size_t)cur_nc, (size_t)c->K * sizeof(float), lhs_ptr,
      rhs_ptr, c->C + (size_t)m_idx * (size_t)c->N + (size_t)n_idx,
      (size_t)c->N * sizeof(float), sizeof(float), &c->params);
}

static bench_stats_t bench_xnn_pf32_sme2(int M, int N, int K, int num_threads,
                                         int reps, int num_trials) {
  // On SVL=64B (16 floats), 4vs=16 and 8vs=32: each 32-row LHS tile has 2x16
  // blocks, and each 32-col RHS panel has 2x16 blocks.
  const int mr_tile = 32, nr_tile = 32;
  const int block = 16;
  int num_m_tiles = (M + mr_tile - 1) / mr_tile;
  int num_n_panels = (N + nr_tile - 1) / nr_tile;
  int total_tiles = num_m_tiles * num_n_panels;

  size_t lhs_block_stride = (size_t)block * (size_t)K * sizeof(float);
  size_t lhs_tile_stride = 2 * lhs_block_stride;
  size_t rhs_block_stride = (size_t)block * ((size_t)K + 1) * sizeof(float);
  size_t rhs_panel_stride = 2 * rhs_block_stride;

  // Pack 1 authentic 32-row LHS tile and 1 authentic 32-col RHS panel via KleidiAI SME packers
  float *lhs_raw = (float *)alloc_aligned((size_t)mr_tile * (size_t)K * sizeof(float));
  float *rhs_raw = (float *)alloc_aligned((size_t)nr_tile * (size_t)K * sizeof(float));
  float *bias = (float *)alloc_aligned((size_t)nr_tile * sizeof(float));
  for (size_t i = 0; i < (size_t)mr_tile * (size_t)K; ++i) lhs_raw[i] = 0.01f;
  for (size_t i = 0; i < (size_t)nr_tile * (size_t)K; ++i) rhs_raw[i] = 0.01f;
  for (int j = 0; j < nr_tile; ++j) bias[j] = 0.0f;

  uint8_t *lhs_packed =
      (uint8_t *)alloc_aligned((size_t)num_m_tiles * lhs_tile_stride + 4096);
  uint8_t *rhs_packed =
      (uint8_t *)alloc_aligned((size_t)num_n_panels * rhs_panel_stride + 4096);

  const void *ptrs[16];
  for (int b = 0; b < 2; ++b) {
    for (int r = 0; r < block; ++r) {
      ptrs[r] = lhs_raw + (size_t)(b * block + r) * (size_t)K;
    }
    kai_kernel_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme(
        (size_t)block, (size_t)K, ptrs, 0, lhs_packed + (size_t)b * lhs_block_stride);
    for (int r = 0; r < block; ++r) {
      ptrs[r] = rhs_raw + (size_t)(b * block + r) * (size_t)K;
    }
    kai_kernel_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme(
        (size_t)block, (size_t)K, ptrs, 0, rhs_packed + (size_t)b * rhs_block_stride,
        bias + b * block);
  }
  for (int mt = 1; mt < num_m_tiles; ++mt) {
    memcpy(lhs_packed + (size_t)mt * lhs_tile_stride, lhs_packed, lhs_tile_stride);
  }
  for (int p = 1; p < num_n_panels; ++p) {
    memcpy(rhs_packed + (size_t)p * rhs_panel_stride, rhs_packed, rhs_panel_stride);
  }
  free(lhs_raw);
  free(rhs_raw);
  free(bias);

  float *C = (float *)alloc_aligned((size_t)M * (size_t)N * sizeof(float) + 4096);
  memset(C, 0, (size_t)M * (size_t)N * sizeof(float));

  xnn_pf32_ctx_t ctx;
  ctx.M = M;
  ctx.N = N;
  ctx.K = K;
  ctx.mr_tile = mr_tile;
  ctx.nr_tile = nr_tile;
  ctx.num_m_tiles = num_m_tiles;
  ctx.num_n_panels = num_n_panels;
  ctx.total_tiles = total_tiles;
  ctx.lhs_tile_stride = lhs_tile_stride;
  ctx.rhs_panel_stride = rhs_panel_stride;
  ctx.lhs_packed = lhs_packed;
  ctx.rhs_packed = rhs_packed;
  ctx.C = C;
  ctx.params.scalar.min = -1e6f;
  ctx.params.scalar.max = 1e6f;

  double total_ops = 2.0 * (double)M * (double)N * (double)K;
  double total_bytes =
      ((double)M * K + (double)N * K + (double)M * N) * sizeof(float);

  bench_stats_t st = run_parallel_bench(num_threads, total_tiles, reps, 1,
                                        num_trials, xnn_pf32_work, &ctx,
                                        total_ops, total_bytes);
  free(lhs_packed);
  free(rhs_packed);
  free(C);
  return st;
}

// -----------------------------------------------------------------------------
// 3. XNNPACK NEON Baseline Runners (I8MM qd8, FMA f32, DotProd qc2w)
// -----------------------------------------------------------------------------

typedef struct {
  xnn_qd8_gemm_fn fn;
  int M, N, K;
  int mr_tile, nr_tile;
  int num_m_tiles, num_n_panels, total_tiles, num_replicas;
  size_t panel_w_bytes, single_w_bytes;
  const int8_t *A;
  const uint8_t *W;
  float *C;
  aeb_xnn_f32_qc4w_minmax_params_t params;
  aeb_xnn_qd8_quant_params_t qparams[16];
} xnn_qd8_ctx_t;

static void xnn_qd8_work(int w, void *arg) {
  xnn_qd8_ctx_t *c = (xnn_qd8_ctx_t *)arg;
  int r = w / c->total_tiles;
  int t = w % c->total_tiles;
  int p = t / c->num_m_tiles;
  int m_idx = (t % c->num_m_tiles) * c->mr_tile;
  int cur_mr = (c->M - m_idx < c->mr_tile) ? (c->M - m_idx) : c->mr_tile;
  int n_idx = p * c->nr_tile;
  int cur_nc = (c->N - n_idx < c->nr_tile) ? (c->N - n_idx) : c->nr_tile;
  const uint8_t *w_base =
      c->W + (size_t)(r % c->num_replicas) * c->single_w_bytes;
  c->fn((size_t)cur_mr, (size_t)cur_nc, (size_t)c->K,
        c->A + (size_t)m_idx * (size_t)c->K, (size_t)c->K,
        w_base + (size_t)p * c->panel_w_bytes,
        c->C + (size_t)m_idx * (size_t)c->N + (size_t)n_idx,
        (size_t)c->N * sizeof(float), (size_t)c->nr_tile * sizeof(float),
        &c->params, c->qparams);
}

static bench_stats_t bench_xnn_qd8_neon(xnn_qd8_gemm_fn fn, int mr_tile,
                                        int nr_tile, int M, int N, int K,
                                        int bits_per_weight, int cold_dram,
                                        int num_threads, int reps,
                                        int num_trials) {
  size_t ksum_bytes = (size_t)nr_tile * sizeof(int32_t);
  size_t w_data_bytes = (size_t)nr_tile * (size_t)K * (size_t)bits_per_weight / 8;
  size_t extra_bytes = 2 * (size_t)nr_tile * sizeof(float);  // scale + bias
  size_t panel_w_bytes =
      (ksum_bytes + w_data_bytes + extra_bytes + 63) & ~(size_t)63;

  int num_n_panels = (N + nr_tile - 1) / nr_tile;
  int num_m_tiles = (M + mr_tile - 1) / mr_tile;
  int total_tiles = num_n_panels * num_m_tiles;
  size_t single_w_bytes = (size_t)num_n_panels * panel_w_bytes;

  int num_replicas = 1;
  if (cold_dram) {
    size_t target_pool = 512ULL * 1024ULL * 1024ULL;
    num_replicas = (int)((target_pool + single_w_bytes - 1) / single_w_bytes);
    if (num_replicas < 2) num_replicas = 2;
    if (num_replicas > 64) num_replicas = 64;
  }

  int8_t *A = (int8_t *)alloc_aligned((size_t)M * (size_t)K + 4096);
  memset(A, 1, (size_t)M * (size_t)K);

  uint8_t *one_panel = (uint8_t *)alloc_aligned(panel_w_bytes + 4096);
  memset(one_panel, 0, panel_w_bytes);
  if (bits_per_weight == 4) {
    uint8_t *raw = (uint8_t *)alloc_aligned((size_t)nr_tile * (size_t)K / 2);
    memset(raw, 0x99, (size_t)nr_tile * (size_t)K / 2);
    aeb_xnn_qs8_qc4w_packing_params_t pp = {1, 8};
    xnn_qs8_qc4w_packw_gemm_goi_ukernel_x16c8__neon_prfm(
        1, (size_t)nr_tile, (size_t)K, (size_t)nr_tile, 8, 1, (size_t)K, raw,
        NULL, NULL, one_panel, extra_bytes, &pp);
    free(raw);
  } else {
    int8_t *raw = (int8_t *)alloc_aligned((size_t)nr_tile * (size_t)K);
    memset(raw, 1, (size_t)nr_tile * (size_t)K);
    aeb_xnn_qs8_packing_params_t pp = {1};
    xnn_qs8_packw_gemm_goi_ukernel_x16c8__neon(
        1, (size_t)nr_tile, (size_t)K, (size_t)nr_tile, 8, 1, (size_t)K, raw,
        NULL, NULL, one_panel, extra_bytes, &pp);
    free(raw);
  }
  float *scales = (float *)(one_panel + ksum_bytes + w_data_bytes);
  float *biases = scales + nr_tile;
  for (int j = 0; j < nr_tile; ++j) {
    scales[j] = 0.01f;
    biases[j] = 0.0f;
  }

  uint8_t *W =
      (uint8_t *)alloc_aligned((size_t)num_replicas * single_w_bytes + 4096);
  for (int p = 0; p < num_n_panels; ++p) {
    memcpy(W + (size_t)p * panel_w_bytes, one_panel, panel_w_bytes);
  }
  for (int rep = 1; rep < num_replicas; ++rep) {
    memcpy(W + (size_t)rep * single_w_bytes, W, single_w_bytes);
  }
  free(one_panel);

  float *C = (float *)alloc_aligned((size_t)M * (size_t)N * sizeof(float) + 4096);
  memset(C, 0, (size_t)M * (size_t)N * sizeof(float));

  xnn_qd8_ctx_t ctx;
  ctx.fn = fn;
  ctx.M = M;
  ctx.N = N;
  ctx.K = K;
  ctx.mr_tile = mr_tile;
  ctx.nr_tile = nr_tile;
  ctx.num_m_tiles = num_m_tiles;
  ctx.num_n_panels = num_n_panels;
  ctx.total_tiles = total_tiles;
  ctx.num_replicas = num_replicas;
  ctx.panel_w_bytes = panel_w_bytes;
  ctx.single_w_bytes = single_w_bytes;
  ctx.A = A;
  ctx.W = W;
  ctx.C = C;
  ctx.params.scalar.min = -1e6f;
  ctx.params.scalar.max = 1e6f;
  ctx.params.scalar.kernel_zero_point = 0;
  for (int i = 0; i < 16; ++i) {
    ctx.qparams[i].zero_point = 0;
    ctx.qparams[i].inv_scale = 0.01f;
  }

  double total_ops = 2.0 * (double)M * (double)N * (double)K;
  double total_bytes =
      (double)M * (double)K +
      (double)N * (double)K * (double)bits_per_weight / 8.0 +
      (double)M * (double)N * sizeof(float);

  bench_stats_t st = run_parallel_bench(num_threads, total_tiles, reps, 1,
                                        num_trials, xnn_qd8_work, &ctx,
                                        total_ops, total_bytes);
  free(A);
  free(W);
  free(C);
  return st;
}

typedef struct {
  int M, N, K;
  int mr_tile, nr_tile;
  int num_m_tiles, num_n_panels, total_tiles;
  size_t panel_w_floats;
  const float *A;
  const float *W;
  float *C;
  aeb_xnn_f32_minmax_params_t params;
} xnn_f32_ctx_t;

static void xnn_f32_work(int w, void *arg) {
  xnn_f32_ctx_t *c = (xnn_f32_ctx_t *)arg;
  int t = w % c->total_tiles;
  int p = t / c->num_m_tiles;
  int m_idx = (t % c->num_m_tiles) * c->mr_tile;
  int cur_mr = (c->M - m_idx < c->mr_tile) ? (c->M - m_idx) : c->mr_tile;
  int n_idx = p * c->nr_tile;
  int cur_nc = (c->N - n_idx < c->nr_tile) ? (c->N - n_idx) : c->nr_tile;
  xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128(
      (size_t)cur_mr, (size_t)cur_nc, (size_t)c->K * sizeof(float),
      c->A + (size_t)m_idx * (size_t)c->K, (size_t)c->K * sizeof(float),
      c->W + (size_t)p * c->panel_w_floats,
      c->C + (size_t)m_idx * (size_t)c->N + (size_t)n_idx,
      (size_t)c->N * sizeof(float), (size_t)c->nr_tile * sizeof(float),
      &c->params);
}

static bench_stats_t bench_xnn_f32_neon(int M, int N, int K, int num_threads,
                                        int reps, int num_trials) {
  const int mr_tile = 6, nr_tile = 8;
  int num_n_panels = (N + nr_tile - 1) / nr_tile;
  int num_m_tiles = (M + mr_tile - 1) / mr_tile;
  int total_tiles = num_n_panels * num_m_tiles;
  size_t panel_w_floats = (size_t)nr_tile * ((size_t)K + 1);

  float *A = (float *)alloc_aligned((size_t)M * (size_t)K * sizeof(float) + 4096);
  float *W = (float *)alloc_aligned((size_t)num_n_panels * panel_w_floats *
                                        sizeof(float) + 4096);
  float *C = (float *)alloc_aligned((size_t)M * (size_t)N * sizeof(float) + 4096);
  for (size_t i = 0; i < (size_t)M * (size_t)K; ++i) A[i] = 0.01f;
  for (size_t i = 0; i < (size_t)num_n_panels * panel_w_floats; ++i) W[i] = 0.01f;
  memset(C, 0, (size_t)M * (size_t)N * sizeof(float));

  xnn_f32_ctx_t ctx = {M, N, K, mr_tile, nr_tile, num_m_tiles, num_n_panels,
                       total_tiles, panel_w_floats, A, W, C, {{-1e6f, 1e6f}}};
  double total_ops = 2.0 * (double)M * (double)N * (double)K;
  double total_bytes =
      ((double)M * K + (double)N * K + (double)M * N) * sizeof(float);
  bench_stats_t st = run_parallel_bench(num_threads, total_tiles, reps, 2,
                                        num_trials, xnn_f32_work, &ctx,
                                        total_ops, total_bytes);
  free(A);
  free(W);
  free(C);
  return st;
}

typedef struct {
  int M, N, K;
  int mr_tile, nr_tile;
  int num_m_tiles, num_n_panels, total_tiles;
  size_t panel_w_bytes;
  const int8_t *A;
  const uint8_t *W;
  int8_t *C;
  aeb_xnn_qs8_qc8w_conv_minmax_params_t params;
} xnn_qc2w_ctx_t;

static void xnn_qc2w_work(int w, void *arg) {
  xnn_qc2w_ctx_t *c = (xnn_qc2w_ctx_t *)arg;
  int t = w % c->total_tiles;
  int p = t / c->num_m_tiles;
  int m_idx = (t % c->num_m_tiles) * c->mr_tile;
  int cur_mr = (c->M - m_idx < c->mr_tile) ? (c->M - m_idx) : c->mr_tile;
  int n_idx = p * c->nr_tile;
  int cur_nc = (c->N - n_idx < c->nr_tile) ? (c->N - n_idx) : c->nr_tile;
  xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_8x8c4__neondot(
      (size_t)cur_mr, (size_t)cur_nc, (size_t)c->K,
      c->A + (size_t)m_idx * (size_t)c->K, (size_t)c->K,
      c->W + (size_t)p * c->panel_w_bytes,
      c->C + (size_t)m_idx * (size_t)c->N + (size_t)n_idx, (size_t)c->N,
      (size_t)c->nr_tile, &c->params);
}

static bench_stats_t bench_xnn_qc2w_neondot(int M, int N, int K, int num_threads,
                                            int reps, int num_trials) {
  const int mr_tile = 8, nr_tile = 8;
  int num_n_panels = (N + nr_tile - 1) / nr_tile;
  int num_m_tiles = (M + mr_tile - 1) / mr_tile;
  int total_tiles = num_n_panels * num_m_tiles;
  size_t bias_bytes = (size_t)nr_tile * sizeof(int32_t);
  size_t w_data_bytes = (size_t)nr_tile * (size_t)K / 4;
  size_t scale_bytes = (size_t)nr_tile * sizeof(float);
  size_t panel_w_bytes =
      (bias_bytes + w_data_bytes + scale_bytes + 63) & ~(size_t)63;

  int8_t *A = (int8_t *)alloc_aligned((size_t)M * (size_t)K + 4096);
  uint8_t *W = (uint8_t *)alloc_aligned((size_t)num_n_panels * panel_w_bytes + 4096);
  int8_t *C = (int8_t *)alloc_aligned((size_t)M * (size_t)N + 4096);
  memset(A, 1, (size_t)M * (size_t)K);
  memset(W, 0x55, (size_t)num_n_panels * panel_w_bytes);
  memset(C, 0, (size_t)M * (size_t)N);
  for (int p = 0; p < num_n_panels; ++p) {
    uint8_t *panel = W + (size_t)p * panel_w_bytes;
    memset(panel, 0, bias_bytes);
    float *sc = (float *)(panel + bias_bytes + w_data_bytes);
    for (int j = 0; j < nr_tile; ++j) sc[j] = 0.01f;
  }

  xnn_qc2w_ctx_t ctx = {M, N, K, mr_tile, nr_tile, num_m_tiles, num_n_panels,
                        total_tiles, panel_w_bytes, A, W, C, {{0, -128, 127}}};
  double total_ops = 2.0 * (double)M * (double)N * (double)K;
  double total_bytes =
      (double)M * K + (double)N * K * 0.25 + (double)M * N;
  bench_stats_t st = run_parallel_bench(num_threads, total_tiles, reps, 2,
                                        num_trials, xnn_qc2w_work, &ctx,
                                        total_ops, total_bytes);
  free(A);
  free(W);
  free(C);
  return st;
}

// -----------------------------------------------------------------------------
// 4. Unified YNNPACK ARM64 Dot Runner (SME2, SME, NEON I8MM, DotProd, FMA)
// -----------------------------------------------------------------------------

typedef struct {
  ynn_dot_fn fn;
  int M, N, K;
  int mr_tile, nr_tile;
  int transpose_a;
  size_t a_elem_bytes;
  size_t a_tile_bytes;
  size_t a_stride_m;
  size_t b_stride_k1;
  size_t panel_w_bytes;
  size_t single_w_bytes;
  int num_m_tiles, num_n_panels, total_tiles, num_replicas;
  const uint8_t *A;
  const uint8_t *B;
  uint8_t *C;
} ynn_dot_ctx_t;

static void ynn_dot_work(int w, void *arg) {
  ynn_dot_ctx_t *c = (ynn_dot_ctx_t *)arg;
  int r = w / c->total_tiles;
  int t = w % c->total_tiles;
  int p = t / c->num_m_tiles;
  int mt = t % c->num_m_tiles;
  int m_idx = mt * c->mr_tile;
  int cur_mr = (c->M - m_idx < c->mr_tile) ? (c->M - m_idx) : c->mr_tile;
  int n_idx = p * c->nr_tile;
  int cur_nc = (c->N - n_idx < c->nr_tile) ? (c->N - n_idx) : c->nr_tile;
  const uint8_t *a_ptr = c->transpose_a
                             ? (c->A + (size_t)mt * c->a_tile_bytes)
                             : (c->A + (size_t)m_idx * c->a_stride_m);
  const uint8_t *b_base =
      c->B + (size_t)(r % c->num_replicas) * c->single_w_bytes;
  const uint8_t *b_ptr = b_base + (size_t)p * c->panel_w_bytes;
  size_t c_stride_m = (size_t)c->N * sizeof(int32_t);
  uint8_t *c_ptr =
      c->C + ((size_t)m_idx * (size_t)c->N + (size_t)n_idx) * sizeof(int32_t);
  c->fn((size_t)cur_mr, (size_t)cur_nc, 1, 1, (size_t)c->K, c->a_stride_m, 0, 0,
        a_ptr, 0, 0, c->b_stride_k1, b_ptr, 0, NULL, c_stride_m, c_ptr, NULL);
}

static bench_stats_t bench_ynn_dot(ynn_dot_fn fn, int mr_tile, int nr_tile,
                                   int M, int N, int K, int bits_per_weight,
                                   size_t b_stride_k1, size_t a_elem_bytes,
                                   int transpose_a, int cold_dram,
                                   int num_threads, int reps, int num_trials) {
  int num_n_panels = (N + nr_tile - 1) / nr_tile;
  int num_m_tiles = (M + mr_tile - 1) / mr_tile;
  int total_tiles = num_n_panels * num_m_tiles;

  size_t panel_w_bytes =
      ((size_t)nr_tile * (size_t)K * (size_t)bits_per_weight / 8 + 127) &
      ~(size_t)127;
  size_t single_w_bytes = (size_t)num_n_panels * panel_w_bytes;

  int num_replicas = 1;
  if (cold_dram) {
    size_t target_pool = 512ULL * 1024ULL * 1024ULL;
    num_replicas = (int)((target_pool + single_w_bytes - 1) / single_w_bytes);
    if (num_replicas < 2) num_replicas = 2;
    if (num_replicas > 64) num_replicas = 64;
  }

  size_t a_tile_bytes =
      ((size_t)mr_tile * (size_t)K * a_elem_bytes + 127) & ~(size_t)127;
  size_t a_bytes = transpose_a ? ((size_t)num_m_tiles * a_tile_bytes)
                               : ((size_t)M * (size_t)K * a_elem_bytes);
  size_t a_stride_m = transpose_a ? ((size_t)mr_tile * a_elem_bytes)
                                  : ((size_t)K * a_elem_bytes);

  uint8_t *A = (uint8_t *)alloc_aligned(a_bytes + 4096);
  uint8_t *B =
      (uint8_t *)alloc_aligned((size_t)num_replicas * single_w_bytes + 4096);
  uint8_t *C =
      (uint8_t *)alloc_aligned((size_t)M * (size_t)N * sizeof(int32_t) + 4096);

  if (a_elem_bytes == sizeof(float)) {
    float *af = (float *)A;
    float *bf = (float *)B;
    for (size_t i = 0; i < a_bytes / sizeof(float); ++i) af[i] = 0.01f;
    for (size_t i = 0;
         i < ((size_t)num_replicas * single_w_bytes) / sizeof(float); ++i) {
      bf[i] = 0.01f;
    }
  } else {
    memset(A, 1, a_bytes);
    memset(B, (bits_per_weight == 2) ? 0x55 : (bits_per_weight == 4 ? 0x11 : 1),
           (size_t)num_replicas * single_w_bytes);
  }
  memset(C, 0, (size_t)M * (size_t)N * sizeof(int32_t));

  ynn_dot_ctx_t ctx = {fn, M, N, K, mr_tile, nr_tile, transpose_a, a_elem_bytes,
                       a_tile_bytes, a_stride_m, b_stride_k1, panel_w_bytes,
                       single_w_bytes, num_m_tiles, num_n_panels, total_tiles,
                       num_replicas, A, B, C};

  int chunk = (total_tiles >= 512) ? 2 : 1;
  double total_ops = 2.0 * (double)M * (double)N * (double)K;
  double total_bytes =
      (double)M * (double)K * (double)a_elem_bytes +
      (double)N * (double)K * (double)bits_per_weight / 8.0 +
      (double)M * (double)N * sizeof(float);

  bench_stats_t st = run_parallel_bench(num_threads, total_tiles, reps, chunk,
                                        num_trials, ynn_dot_work, &ctx,
                                        total_ops, total_bytes);
  free(A);
  free(B);
  free(C);
  return st;
}

// -----------------------------------------------------------------------------
// 5. Pair 4: FP32 Attention Softmax (XNNPACK Fused vs. YNNPACK Reductions)
// -----------------------------------------------------------------------------

typedef struct {
  int num_rows, row_len, row_tile, num_row_tiles;
  size_t row_bytes;
  const float *in;
  float *out;
} softmax_ctx_t;

static void xnn_softmax_work(int w, void *arg) {
  softmax_ctx_t *c = (softmax_ctx_t *)arg;
  int m = w % c->num_rows;
  float rmax = -1e30f, rsum = 0.0f;
  xnn_f32_rmax_ukernel__neon_u16_acc4(c->row_bytes,
                                      c->in + (size_t)m * (size_t)c->row_len,
                                      &rmax, NULL);
  xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc2(
      c->row_bytes, c->in + (size_t)m * (size_t)c->row_len, &rmax,
      c->out + (size_t)m * (size_t)c->row_len, &rsum, NULL);
}

static void ynn_softmax_work(int w, void *arg) {
  softmax_ctx_t *c = (softmax_ctx_t *)arg;
  int rt = w % c->num_row_tiles;
  int m = rt * c->row_tile;
  int cur_n = (c->num_rows - m < c->row_tile) ? (c->num_rows - m) : c->row_tile;
  float max_out[8], sum_out[8];
  for (int j = 0; j < cur_n; ++j) {
    max_out[j] = -1e30f;
    sum_out[j] = 0.0f;
  }
  aeb_ynn_max_k1_fp32_neon((size_t)cur_n, (size_t)c->row_len, c->row_bytes,
                           c->in + (size_t)m * (size_t)c->row_len, max_out,
                           NULL);
  aeb_ynn_sum_k1_fp32_neon((size_t)cur_n, (size_t)c->row_len, c->row_bytes,
                           c->out + (size_t)m * (size_t)c->row_len, sum_out,
                           NULL);
}

static bench_stats_t bench_xnn_softmax(int num_rows, int row_len,
                                       int num_threads, int reps,
                                       int num_trials) {
  size_t n = (size_t)num_rows * (size_t)row_len;
  float *in = (float *)alloc_aligned(n * sizeof(float) + 4096);
  float *out = (float *)alloc_aligned(n * sizeof(float) + 4096);
  for (size_t i = 0; i < n; ++i) in[i] = 0.5f;
  memset(out, 0, n * sizeof(float));

  softmax_ctx_t ctx = {num_rows, row_len, 1, num_rows,
                       (size_t)row_len * sizeof(float), in, out};
  double total_bytes = 3.0 * (double)n * sizeof(float);
  double total_ops = 12.0 * (double)n;
  bench_stats_t st = run_parallel_bench(num_threads, num_rows, reps, 8,
                                        num_trials, xnn_softmax_work, &ctx,
                                        total_ops, total_bytes);
  free(in);
  free(out);
  return st;
}

static bench_stats_t bench_ynn_softmax_red(int num_rows, int row_len,
                                           int num_threads, int reps,
                                           int num_trials) {
  size_t n = (size_t)num_rows * (size_t)row_len;
  float *in = (float *)alloc_aligned(n * sizeof(float) + 4096);
  float *out = (float *)alloc_aligned(n * sizeof(float) + 4096);
  for (size_t i = 0; i < n; ++i) in[i] = 0.5f;
  for (size_t i = 0; i < n; ++i) out[i] = 0.5f;

  int row_tile = 4;
  int num_row_tiles = (num_rows + row_tile - 1) / row_tile;
  softmax_ctx_t ctx = {num_rows, row_len, row_tile, num_row_tiles,
                       (size_t)row_len * sizeof(float), in, out};
  double total_bytes = 2.0 * (double)n * sizeof(float);
  double total_ops = 2.0 * (double)n;
  bench_stats_t st = run_parallel_bench(num_threads, num_row_tiles, reps, 4,
                                        num_trials, ynn_softmax_work, &ctx,
                                        total_ops, total_bytes);
  free(in);
  free(out);
  return st;
}

// -----------------------------------------------------------------------------
// 6. Pair 7: Matrix Packing / 32-Bit Transpose
// -----------------------------------------------------------------------------

typedef struct {
  int N, K, tile, num_panels;
  size_t padded_N;
  const uint32_t *in;
  uint32_t *out;
} pack_ctx_t;

static void xnn_packw_work(int w, void *arg) {
  pack_ctx_t *c = (pack_ctx_t *)arg;
  int p = w % c->num_panels;
  int n_idx = p * c->tile;
  int cur_nr = (c->N - n_idx < c->tile) ? (c->N - n_idx) : c->tile;
  xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm(
      1, (size_t)cur_nr, (size_t)c->K, (size_t)c->tile, 1, 1, (size_t)c->K,
      c->in + (size_t)n_idx * (size_t)c->K, NULL, NULL,
      c->out + (size_t)p * (size_t)c->tile * ((size_t)c->K + 1), 0, NULL);
}

static void ynn_transpose_work(int w, void *arg) {
  pack_ctx_t *c = (pack_ctx_t *)arg;
  int p = w % c->num_panels;
  int m_idx = p * c->tile;
  int cur_m = (c->N - m_idx < c->tile) ? (c->N - m_idx) : c->tile;
  aeb_ynn_transpose_x32_neon((size_t)c->K, (size_t)cur_m,
                             (size_t)c->K * sizeof(uint32_t),
                             (size_t)c->K * sizeof(uint32_t),
                             c->in + (size_t)m_idx * (size_t)c->K,
                             c->padded_N * sizeof(uint32_t),
                             c->out + (size_t)m_idx);
}

static bench_stats_t bench_xnn_packw(int N, int K, int num_threads, int reps,
                                     int num_trials) {
  const int nr = 8;
  int num_panels = (N + nr - 1) / nr;
  size_t padded_N = (size_t)num_panels * (size_t)nr;
  size_t buf_bytes = padded_N * (size_t)(K + 16) * sizeof(uint32_t) + 65536;
  uint32_t *in = (uint32_t *)alloc_aligned(buf_bytes);
  uint32_t *out = (uint32_t *)alloc_aligned(buf_bytes);
  memset(in, 1, buf_bytes);
  memset(out, 0, buf_bytes);

  pack_ctx_t ctx = {N, K, nr, num_panels, padded_N, in, out};
  double total_bytes = 2.0 * (double)N * (double)K * sizeof(uint32_t);
  bench_stats_t st = run_parallel_bench(num_threads, num_panels, reps, 4,
                                        num_trials, xnn_packw_work, &ctx, 0.0,
                                        total_bytes);
  free(in);
  free(out);
  return st;
}

static bench_stats_t bench_ynn_transpose(int N, int K, int num_threads,
                                         int reps, int num_trials) {
  const int tile_m = 32;
  int num_panels = (N + tile_m - 1) / tile_m;
  size_t padded_N = (size_t)num_panels * (size_t)tile_m;
  size_t buf_bytes = padded_N * (size_t)(K + 16) * sizeof(uint32_t) + 65536;
  uint32_t *in = (uint32_t *)alloc_aligned(buf_bytes);
  uint32_t *out = (uint32_t *)alloc_aligned(buf_bytes);
  memset(in, 1, buf_bytes);
  memset(out, 0, buf_bytes);

  pack_ctx_t ctx = {N, K, tile_m, num_panels, padded_N, in, out};
  double total_bytes = 2.0 * (double)N * (double)K * sizeof(uint32_t);
  bench_stats_t st = run_parallel_bench(num_threads, num_panels, reps, 2,
                                        num_trials, ynn_transpose_work, &ctx,
                                        0.0, total_bytes);
  free(in);
  free(out);
  return st;
}

// -----------------------------------------------------------------------------
// Comprehensive Pre-Timing Numerical Smoke Checks (All Microkernels)
// -----------------------------------------------------------------------------

static void check_ynn_int_smoke(const char *name, ynn_dot_fn fn, int M, int N,
                                int K, int bits, size_t b_stride_k1,
                                int transpose_a) {
  int8_t *A = (int8_t *)alloc_aligned((size_t)M * (size_t)K + 128);
  uint8_t *B = (uint8_t *)alloc_aligned((size_t)N * (size_t)K + 128);
  int32_t *C = (int32_t *)alloc_aligned((size_t)M * (size_t)N * sizeof(int32_t) + 128);
  memset(A, 1, (size_t)M * (size_t)K);
  memset(B, (bits == 2) ? 0x55 : (bits == 4 ? 0x11 : 1), (size_t)N * (size_t)K);
  memset(C, 0, (size_t)M * (size_t)N * sizeof(int32_t));
  size_t a_stride_m = transpose_a ? (size_t)M : (size_t)K;
  fn((size_t)M, (size_t)N, 1, 1, (size_t)K, a_stride_m, 0, 0, A, 0, 0,
     b_stride_k1, B, 0, NULL, (size_t)N * sizeof(int32_t), C, NULL);
  if (C[0] != K || C[(M - 1) * N + (N - 1)] != K) {
    fprintf(stderr, "[smoke] FAIL %s: expected %d, got C[0]=%d C[last]=%d\n",
            name, K, C[0], C[(M - 1) * N + (N - 1)]);
    exit(1);
  }
  free(A);
  free(B);
  free(C);
}

static void check_ynn_fp32_smoke(const char *name, ynn_dot_fn fn, int M, int N,
                                 int K, int transpose_a) {
  float *A = (float *)alloc_aligned((size_t)M * (size_t)K * sizeof(float) + 128);
  float *B = (float *)alloc_aligned((size_t)N * (size_t)K * sizeof(float) + 128);
  float *C = (float *)alloc_aligned((size_t)M * (size_t)N * sizeof(float) + 128);
  for (int i = 0; i < M * K; ++i) A[i] = 1.0f;
  for (int i = 0; i < N * K; ++i) B[i] = 1.0f;
  memset(C, 0, (size_t)M * (size_t)N * sizeof(float));
  size_t a_stride_m = transpose_a ? ((size_t)M * sizeof(float))
                                  : ((size_t)K * sizeof(float));
  fn((size_t)M, (size_t)N, 1, 1, (size_t)K, a_stride_m, 0, 0, A, 0, 0,
     (size_t)N * sizeof(float), B, 0, NULL, (size_t)N * sizeof(float), C, NULL);
  if (fabsf(C[0] - (float)K) > 1e-2f ||
      fabsf(C[(M - 1) * N + (N - 1)] - (float)K) > 1e-2f) {
    fprintf(stderr, "[smoke] FAIL %s: expected %.1f, got C[0]=%.4f C[last]=%.4f\n",
            name, (float)K, C[0], C[(M - 1) * N + (N - 1)]);
    exit(1);
  }
  free(A);
  free(B);
  free(C);
}

static void run_smoke_checks(void) {
  fprintf(stderr, "[smoke] Verifying all ARM64 XNNPACK/KleidiAI/YNNPACK kernels...\n");

  // YNNPACK integer & FP32 dot kernels (A=1, B=1 -> C=K=64)
  check_ynn_int_smoke("ynn_int4_1x32_neondot", aeb_ynn_dot_int8_int4_1x32_neondot, 1, 32, 64, 4, 16, 0);
  check_ynn_int_smoke("ynn_int4_sme_16x64", aeb_ynn_dot_int8_int4_sme, 16, 64, 64, 4, 32, 1);
  check_ynn_int_smoke("ynn_int4_sme_1x64", aeb_ynn_dot_int8_int4_sme, 1, 64, 64, 4, 32, 1);
  check_ynn_int_smoke("ynn_int4_10x8_neoni8mm", aeb_ynn_dot_int8_int4_10x8_neoni8mm, 10, 8, 64, 4, 4, 1);
  check_ynn_int_smoke("ynn_int4_6x16_neoni8mm", aeb_ynn_dot_int8_int4_6x16_neoni8mm, 6, 16, 64, 4, 8, 1);
  check_ynn_int_smoke("ynn_int8_sme2", aeb_ynn_dot_int8_int8_sme2, 16, 64, 64, 8, 64, 1);
  check_ynn_int_smoke("ynn_int8_sme", aeb_ynn_dot_int8_int8_sme, 16, 64, 64, 8, 64, 1);
  check_ynn_int_smoke("ynn_int8_10x8_neoni8mm", aeb_ynn_dot_int8_int8_10x8_neoni8mm, 10, 8, 64, 8, 8, 1);
  check_ynn_int_smoke("ynn_int8_6x16_neoni8mm", aeb_ynn_dot_int8_int8_6x16_neoni8mm, 6, 16, 64, 8, 16, 1);
  check_ynn_int_smoke("ynn_int2_sme", aeb_ynn_dot_int8_int2_sme, 16, 64, 64, 2, 16, 1);
  check_ynn_int_smoke("ynn_int2_8x8_neondot", aeb_ynn_dot_int8_int2_8x8_neondot, 8, 8, 64, 2, 2, 0);
  check_ynn_fp32_smoke("ynn_fp32_sme2", aeb_ynn_dot_fp32_sme2, 16, 64, 64, 1);
  check_ynn_fp32_smoke("ynn_fp32_sme", aeb_ynn_dot_fp32_sme, 16, 64, 64, 1);
  check_ynn_fp32_smoke("ynn_fp32_6x8_neon", aeb_ynn_dot_fp32_6x8_neon, 6, 8, 64, 0);

  // YNNPACK Softmax reductions
  {
    float in[4 * 256];
    for (int i = 0; i < 4 * 256; ++i) in[i] = 0.5f;
    float max_o[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    float sum_o[4] = {0, 0, 0, 0};
    aeb_ynn_max_k1_fp32_neon(4, 256, 256 * sizeof(float), in, max_o, NULL);
    aeb_ynn_sum_k1_fp32_neon(4, 256, 256 * sizeof(float), in, sum_o, NULL);
    if (fabsf(max_o[0] - 0.5f) > 1e-5f || fabsf(sum_o[3] - 128.0f) > 1e-2f) {
      fprintf(stderr, "[smoke] FAIL ynn reduce: max=%.4f sum=%.4f\n", max_o[0], sum_o[3]);
      exit(1);
    }
  }

  // XNNPACK KleidiAI SME2 4-bit (1x64 & 16x64), 8-bit (16x64), 2-bit (16x64)
  {
    const size_t K = 64;
    float lhs_f[16 * 64], bias[64], scale[64], dst[16 * 64];
    for (int i = 0; i < 16 * 64; ++i) lhs_f[i] = 1.0f;
    for (int i = 0; i < 64; ++i) { bias[i] = 0.0f; scale[i] = 1.0f; }
    aeb_xnn_f32_qc4w_minmax_params_t p = {{-1e6f, 1e6f, 0}};

    uint8_t *lhs1 = (uint8_t *)alloc_aligned(kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(1, K, 1, 4, 1) + 256);
    uint8_t *lhs16 = (uint8_t *)alloc_aligned(kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(16, K, 16, 4, 1) + 256);
    kai_run_lhs_quant_pack_qai8dxp_f32(1, K, 1, 4, 1, 0, lhs_f, K * sizeof(float), lhs1);
    kai_run_lhs_quant_pack_qai8dxp_f32(16, K, 16, 4, 1, 0, lhs_f, K * sizeof(float), lhs16);

    // QC4W SME2
    uint8_t raw4[64 * 32];
    memset(raw4, 0x99, sizeof(raw4));
    uint8_t *rhs4 = (uint8_t *)alloc_aligned(kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon(K, 64, 4, 1) + 256);
    struct kai_rhs_pack_qs4cxs1s0_param kp4 = {1, 8};
    kai_run_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon(1, 64, K, 64, 4, 1, raw4, bias, scale, rhs4, 0, &kp4);
    memset(dst, 0, sizeof(dst));
    xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2(1, 64, K, lhs1, rhs4, dst, 64 * sizeof(float), sizeof(float), &p);
    if (fabsf(dst[0] - 64.0f) > 1.0f || fabsf(dst[63] - 64.0f) > 1.0f) {
      fprintf(stderr, "[smoke] FAIL xnn qp8 qc4w 1x64 sme2: got %.4f %.4f\n", dst[0], dst[63]);
      exit(1);
    }
    memset(dst, 0, sizeof(dst));
    xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x64c4__neonsme2(16, 64, K, lhs16, rhs4, dst, 64 * sizeof(float), sizeof(float), &p);
    if (fabsf(dst[0] - 64.0f) > 1.0f || fabsf(dst[15 * 64 + 63] - 64.0f) > 1.0f) {
      fprintf(stderr, "[smoke] FAIL xnn qp8 qc4w 16x64 sme2: got %.4f %.4f\n", dst[0], dst[15 * 64 + 63]);
      exit(1);
    }

    // QC8W SME2
    int8_t raw8[64 * 64];
    memset(raw8, 1, sizeof(raw8));
    uint8_t *rhs8 = (uint8_t *)alloc_aligned(kai_get_rhs_packed_stride_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(K, 64, 4, 1) + 256);
    struct kai_rhs_pack_qsi8cx_params kp8 = {1, 1.0f};
    kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(1, 64, K, 64, 4, 1, raw8, bias, scale, rhs8, 0, &kp8);
    memset(dst, 0, sizeof(dst));
    xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x64c4__neonsme2(16, 64, K, lhs16, rhs8, dst, 64 * sizeof(float), sizeof(float), &p);
    if (fabsf(dst[0] - 64.0f) > 1.0f || fabsf(dst[15 * 64 + 63] - 64.0f) > 1.0f) {
      fprintf(stderr, "[smoke] FAIL xnn qp8 qc8w 16x64 sme2: got %.4f %.4f\n", dst[0], dst[15 * 64 + 63]);
      exit(1);
    }

    // QC2W SME2
    uint8_t raw2[64 * 16];
    memset(raw2, 0x55, sizeof(raw2));
    uint8_t *rhs2 = (uint8_t *)alloc_aligned(kai_get_rhs_packed_stride_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(K, 64, 4, 1) + 256);
    struct kai_rhs_pack_qs4cxs1s0_param kp2 = {1, 2};
    static const int32_t lut[4] = {0, 1, -2, -1};
    kai_run_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(1, 64, K, 64, 4, 1, raw2, bias, scale, rhs2, 0, &kp2, lut);
    memset(dst, 0, sizeof(dst));
    xnn_qp8_f32_qc2w_gemm_minmax_ukernel_16x64c4__neonsme2(16, 64, K, lhs16, rhs2, dst, 64 * sizeof(float), sizeof(float), &p);
    if (fabsf(dst[0] - 64.0f) > 1.0f || fabsf(dst[15 * 64 + 63] - 64.0f) > 1.0f) {
      fprintf(stderr, "[smoke] FAIL xnn qp8 qc2w 16x64 sme2: got %.4f %.4f\n", dst[0], dst[15 * 64 + 63]);
      exit(1);
    }

    // PF32 SME2
    float lhs32[32 * 64], rhs32[32 * 64], dst32[32 * 32];
    for (int i = 0; i < 32 * 64; ++i) { lhs32[i] = 1.0f; rhs32[i] = 1.0f; }
    uint8_t *plhs32 = (uint8_t *)alloc_aligned(32 * 64 * sizeof(float) + 256);
    uint8_t *prhs32 = (uint8_t *)alloc_aligned(32 * 65 * sizeof(float) + 256);
    const void *ptrs[16];
    for (int b = 0; b < 2; ++b) {
      for (int r = 0; r < 16; ++r) ptrs[r] = lhs32 + (b * 16 + r) * 64;
      kai_kernel_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme(16, 64, ptrs, 0, plhs32 + b * 16 * 64 * sizeof(float));
      for (int r = 0; r < 16; ++r) ptrs[r] = rhs32 + (b * 16 + r) * 64;
      kai_kernel_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme(16, 64, ptrs, 0, prhs32 + b * 16 * 65 * sizeof(float), bias);
    }
    aeb_xnn_f32_minmax_params_t p32 = {{-1e6f, 1e6f}};
    memset(dst32, 0, sizeof(dst32));
    xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2(32, 32, 64 * sizeof(float), plhs32, prhs32, dst32, 32 * sizeof(float), sizeof(float), &p32);
    if (fabsf(dst32[0] - 64.0f) > 1e-2f || fabsf(dst32[31 * 32 + 31] - 64.0f) > 1e-2f) {
      fprintf(stderr, "[smoke] FAIL xnn pf32 32x32 sme2: got %.4f %.4f\n", dst32[0], dst32[31 * 32 + 31]);
      exit(1);
    }

    free(lhs1); free(lhs16); free(rhs4); free(rhs8); free(rhs2);
    free(plhs32); free(prhs32);
  }

  fprintf(stderr, "[smoke] All ARM64 microkernel sanity checks PASSED.\n");
}

// -----------------------------------------------------------------------------
// Benchmark Suite Driver
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  int mc_threads[MAX_THREAD_CFGS] = {6, 18};
  int num_mc = 2;
  int num_trials = 7;
  const char *out_path = NULL;
  const char *xnnpack_commit = "";
  const char *kleidiai_commit = "";

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      int parsed = parse_thread_list(argv[++i], mc_threads, MAX_THREAD_CFGS);
      if (parsed > 0) num_mc = parsed;
    } else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc) {
      num_trials = atoi(argv[++i]);
      if (num_trials < 1) num_trials = 1;
      if (num_trials > MAX_TRIALS) num_trials = MAX_TRIALS;
    } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      out_path = argv[++i];
    } else if (strcmp(argv[i], "--xnnpack-commit") == 0 && i + 1 < argc) {
      xnnpack_commit = argv[++i];
    } else if (strcmp(argv[i], "--kleidiai-commit") == 0 && i + 1 < argc) {
      kleidiai_commit = argv[++i];
    }
  }

  set_high_qos();
  run_smoke_checks();

  int max_mc = mc_threads[0];
  for (int i = 1; i < num_mc; ++i) {
    if (mc_threads[i] > max_mc) max_mc = mc_threads[i];
  }

  FILE *out = stdout;
  if (out_path && out_path[0]) {
    out = fopen(out_path, "w");
    if (!out) {
      perror("fopen output");
      return 1;
    }
  }

  fprintf(out, "{\n");
  fprintf(out, "  \"model\": \"Gemma 4 31B\",\n");
  fprintf(out, "  \"arch\": \"arm64\",\n");
  fprintf(out, "  \"microarchitecture\": \"Apple M5 Max\",\n");
  if (xnnpack_commit[0]) {
    fprintf(out, "  \"xnnpack_commit\": \"%s\",\n", xnnpack_commit);
  }
  if (kleidiai_commit[0]) {
    fprintf(out, "  \"kleidiai_commit\": \"%s\",\n", kleidiai_commit);
  }
  fprintf(out, "  \"threads_single_core\": 1,\n");
  fprintf(out, "  \"threads_max\": %d,\n", max_mc);
  fprintf(out, "  \"thread_configs\": [1");
  for (int t = 0; t < num_mc; ++t) fprintf(out, ", %d", mc_threads[t]);
  fprintf(out, "],\n");
  fprintf(out, "  \"num_trials\": %d,\n", num_trials);
  fprintf(out, "  \"benchmarks\": [\n");

  // ---------------------------------------------------------------------------
  // Pair 1: 4-Bit Quantized Linear GEMV (M = 1, Decode)
  // ---------------------------------------------------------------------------
  struct {
    int M, N, K;
    int cold_dram;
    const char *regime;
    int reps_1c, reps_mc;
  } p1_shapes[] = {
      {1, 21504, 5376, 1, "Unified_Memory_Streaming", 12, 24},
      {1, 5376, 21504, 1, "Unified_Memory_Streaming", 12, 24},
      {1, 8192, 5376, 1, "Unified_Memory_Streaming", 24, 48},
      {1, 8192, 5376, 0, "SLC_L2_Warm", 48, 96},
      {1, 4096, 5376, 0, "L2_Cluster_Warm", 64, 128},
  };
  fprintf(stderr, "[bench] Pair 1: 4-Bit Linear GEMV (M=1)...\n");
  for (size_t i = 0; i < sizeof(p1_shapes) / sizeof(p1_shapes[0]); ++i) {
    int M = p1_shapes[i].M, N = p1_shapes[i].N, K = p1_shapes[i].K;
    double ops = 2.0 * M * N * K;
    double bytes = (double)M * K + (double)N * K * 0.5 + (double)M * N * 4.0;

    bench_stats_t x1 = bench_xnn_qp8_sme2(
        xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2, 1, 64, M, N, K,
        4, p1_shapes[i].cold_dram, 1, p1_shapes[i].reps_1c, num_trials);
    bench_stats_t xmc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      xmc[t] = bench_xnn_qp8_sme2(
          xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2, 1, 64, M, N, K,
          4, p1_shapes[i].cold_dram, mc_threads[t], p1_shapes[i].reps_mc,
          num_trials);
    }
    print_kernel_result(
        out, "pair1_int4_gemv", "4-Bit Linear GEMV (Decode)", "XNNPACK",
        "xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2", "SME2-INT8", M,
        N, K, p1_shapes[i].regime, x1, xmc, mc_threads, num_mc, ops, bytes, 1);

    bench_stats_t y1 = bench_ynn_dot(
        aeb_ynn_dot_int8_int4_1x32_neondot, 1, 32, M, N, K, 4, 16, 1, 0,
        p1_shapes[i].cold_dram, 1, p1_shapes[i].reps_1c, num_trials);
    bench_stats_t ymc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      ymc[t] = bench_ynn_dot(
          aeb_ynn_dot_int8_int4_1x32_neondot, 1, 32, M, N, K, 4, 16, 1, 0,
          p1_shapes[i].cold_dram, mc_threads[t], p1_shapes[i].reps_mc,
          num_trials);
    }
    print_kernel_result(
        out, "pair1_int4_gemv", "4-Bit Linear GEMV (Decode)", "YNNPACK",
        "ynn::dot_int8_int4_int32_1x32x8_1x4x8_neondot", "NEON-DotProd", M, N,
        K, p1_shapes[i].regime, y1, ymc, mc_threads, num_mc, ops, bytes, 1);
  }

  // ---------------------------------------------------------------------------
  // Pair 2: 4-Bit Quantized Linear GEMM (M in {128, 1024}, Prefill)
  // ---------------------------------------------------------------------------
  struct {
    int M, N, K;
    const char *regime;
    int reps_1c, reps_mc;
  } p2_shapes[] = {
      {128, 21504, 5376, "Unified_Memory_to_L2_Tiled", 2, 6},
      {128, 5376, 21504, "Unified_Memory_to_L2_Tiled", 2, 6},
      {128, 8192, 5376, "SLC_to_L2_Tiled", 4, 10},
      {1024, 21504, 5376, "Compute_Bound_Tiled", 1, 2},
      {1024, 8192, 5376, "Compute_Bound_Tiled", 1, 3},
  };
  fprintf(stderr, "[bench] Pair 2: 4-Bit Linear GEMM (M=128, 1024)...\n");
  for (size_t i = 0; i < sizeof(p2_shapes) / sizeof(p2_shapes[0]); ++i) {
    int M = p2_shapes[i].M, N = p2_shapes[i].N, K = p2_shapes[i].K;
    double ops = 2.0 * M * N * K;
    double bytes = (double)M * K + (double)N * K * 0.5 + (double)M * N * 4.0;

    bench_stats_t x1 = bench_xnn_qp8_sme2(
        xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x64c4__neonsme2, 16, 64, M, N, K,
        4, 0, 1, p2_shapes[i].reps_1c, num_trials);
    bench_stats_t xmc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      xmc[t] = bench_xnn_qp8_sme2(
          xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x64c4__neonsme2, 16, 64, M, N,
          K, 4, 0, mc_threads[t], p2_shapes[i].reps_mc, num_trials);
    }
    print_kernel_result(
        out, "pair2_int4_gemm", "4-Bit Linear GEMM (Prefill)", "XNNPACK",
        "xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x64c4__neonsme2", "SME2-INT8",
        M, N, K, p2_shapes[i].regime, x1, xmc, mc_threads, num_mc, ops, bytes,
        1);

    bench_stats_t y1 = bench_ynn_dot(aeb_ynn_dot_int8_int4_sme, 16, 64, M, N, K,
                                     4, 32, 1, 1, 0, 1, p2_shapes[i].reps_1c,
                                     num_trials);
    bench_stats_t ymc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      ymc[t] = bench_ynn_dot(aeb_ynn_dot_int8_int4_sme, 16, 64, M, N, K, 4, 32,
                             1, 1, 0, mc_threads[t], p2_shapes[i].reps_mc,
                             num_trials);
    }
    print_kernel_result(
        out, "pair2_int4_gemm", "4-Bit Linear GEMM (Prefill)", "YNNPACK",
        "ynn::dot_int8_int4_int32_sme", "SME-INT8", M, N, K,
        p2_shapes[i].regime, y1, ymc, mc_threads, num_mc, ops, bytes, 1);
  }

  // ---------------------------------------------------------------------------
  // Pair 3: FP32 Attention BatchMatMul (Q*K^T and Scores*V)
  // ---------------------------------------------------------------------------
  struct {
    int M, N, K;
    const char *regime;
    int reps_1c, reps_mc;
  } p3_shapes[] = {
      {128, 1024, 256, "L2_Cluster_Resident", 30, 80},
      {1024, 1024, 256, "L2_Cluster_Resident", 8, 24},
      {1024, 4096, 512, "SLC_Resident", 2, 6},
      {2520, 2520, 72, "SLC_Resident", 6, 20},
  };
  fprintf(stderr, "[bench] Pair 3: FP32 Attention BatchMatMul...\n");
  for (size_t i = 0; i < sizeof(p3_shapes) / sizeof(p3_shapes[0]); ++i) {
    int M = p3_shapes[i].M, N = p3_shapes[i].N, K = p3_shapes[i].K;
    double ops = 2.0 * M * N * K;
    double bytes = ((double)M * K + (double)N * K + (double)M * N) * 4.0;

    bench_stats_t x1 =
        bench_xnn_pf32_sme2(M, N, K, 1, p3_shapes[i].reps_1c, num_trials);
    bench_stats_t xmc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      xmc[t] = bench_xnn_pf32_sme2(M, N, K, mc_threads[t], p3_shapes[i].reps_mc,
                                   num_trials);
    }
    print_kernel_result(
        out, "pair3_fp32_attn_bmm", "FP32 Attention BatchMatMul", "XNNPACK",
        "xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2", "SME2-FP32", M, N, K,
        p3_shapes[i].regime, x1, xmc, mc_threads, num_mc, ops, bytes, 1);

    bench_stats_t y1 = bench_ynn_dot(aeb_ynn_dot_fp32_sme2, 16, 64, M, N, K, 32,
                                     64 * sizeof(float), sizeof(float), 1, 0, 1,
                                     p3_shapes[i].reps_1c, num_trials);
    bench_stats_t ymc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      ymc[t] = bench_ynn_dot(aeb_ynn_dot_fp32_sme2, 16, 64, M, N, K, 32,
                             64 * sizeof(float), sizeof(float), 1, 0,
                             mc_threads[t], p3_shapes[i].reps_mc, num_trials);
    }
    print_kernel_result(
        out, "pair3_fp32_attn_bmm", "FP32 Attention BatchMatMul", "YNNPACK",
        "ynn::dot_fp32_sme2", "SME2-FP32", M, N, K, p3_shapes[i].regime, y1,
        ymc, mc_threads, num_mc, ops, bytes, 1);
  }

  // ---------------------------------------------------------------------------
  // Pair 4: FP32 Attention Softmax
  // ---------------------------------------------------------------------------
  struct {
    int rows, row_len;
    const char *regime;
    int reps_1c, reps_mc;
  } p4_shapes[] = {
      {1024, 1024, "L2_Cluster_Resident", 24, 80},
      {2520, 2520, "SLC_Resident", 8, 30},
      {1024, 4096, "SLC_Resident", 10, 40},
  };
  fprintf(stderr, "[bench] Pair 4: FP32 Attention Softmax...\n");
  for (size_t i = 0; i < sizeof(p4_shapes) / sizeof(p4_shapes[0]); ++i) {
    int R = p4_shapes[i].rows, L = p4_shapes[i].row_len;
    double x_ops = 12.0 * (double)R * (double)L;
    double x_bytes = 3.0 * (double)R * (double)L * sizeof(float);

    bench_stats_t x1 =
        bench_xnn_softmax(R, L, 1, p4_shapes[i].reps_1c, num_trials);
    bench_stats_t xmc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      xmc[t] = bench_xnn_softmax(R, L, mc_threads[t], p4_shapes[i].reps_mc,
                                 num_trials);
    }
    print_kernel_result(
        out, "pair4_fp32_softmax", "FP32 Attention Softmax", "XNNPACK",
        "xnn_f32_rmax + xnn_f32_raddstoreexpminusmax__neonfma_rr1_lut64_p2",
        "NEON-FP32", R, L, 1, p4_shapes[i].regime, x1, xmc, mc_threads, num_mc,
        x_ops, x_bytes, 1);

    double y_ops = 2.0 * (double)R * (double)L;
    double y_bytes = 2.0 * (double)R * (double)L * sizeof(float);
    bench_stats_t y1 =
        bench_ynn_softmax_red(R, L, 1, p4_shapes[i].reps_1c, num_trials);
    bench_stats_t ymc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      ymc[t] = bench_ynn_softmax_red(R, L, mc_threads[t], p4_shapes[i].reps_mc,
                                     num_trials);
    }
    print_kernel_result(
        out, "pair4_fp32_softmax", "FP32 Attention Softmax (Reductions Only)",
        "YNNPACK", "ynn::max_k1_fp32_neon + ynn::sum_k1_fp32_neon", "NEON-FP32",
        R, L, 1, p4_shapes[i].regime, y1, ymc, mc_threads, num_mc, y_ops,
        y_bytes, 1);
  }

  // ---------------------------------------------------------------------------
  // Pair 5: 8-Bit Dynamic-Quantized Linear GEMM (Vision ViT & Audio Conformer)
  // ---------------------------------------------------------------------------
  struct {
    int M, N, K;
    const char *regime;
    int reps_1c, reps_mc;
  } p5_shapes[] = {
      {280, 5376, 1152, "L2_Cluster_Resident", 12, 30},
      {2520, 1152, 1152, "SLC_Resident", 8, 20},
      {2520, 4304, 1152, "SLC_Resident", 4, 10},
  };
  fprintf(stderr, "[bench] Pair 5: 8-Bit Linear GEMM...\n");
  for (size_t i = 0; i < sizeof(p5_shapes) / sizeof(p5_shapes[0]); ++i) {
    int M = p5_shapes[i].M, N = p5_shapes[i].N, K = p5_shapes[i].K;
    double ops = 2.0 * M * N * K;
    double bytes = (double)M * K + (double)N * K + (double)M * N * 4.0;

    bench_stats_t x1 = bench_xnn_qp8_sme2(
        xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x64c4__neonsme2, 16, 64, M, N, K,
        8, 0, 1, p5_shapes[i].reps_1c, num_trials);
    bench_stats_t xmc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      xmc[t] = bench_xnn_qp8_sme2(
          xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x64c4__neonsme2, 16, 64, M, N,
          K, 8, 0, mc_threads[t], p5_shapes[i].reps_mc, num_trials);
    }
    print_kernel_result(
        out, "pair5_int8_gemm", "8-Bit Linear GEMM (Vision/Audio)", "XNNPACK",
        "xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x64c4__neonsme2", "SME2-INT8",
        M, N, K, p5_shapes[i].regime, x1, xmc, mc_threads, num_mc, ops, bytes,
        1);

    bench_stats_t y1 = bench_ynn_dot(aeb_ynn_dot_int8_int8_sme2, 16, 64, M, N, K,
                                     8, 64, 1, 1, 0, 1, p5_shapes[i].reps_1c,
                                     num_trials);
    bench_stats_t ymc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      ymc[t] = bench_ynn_dot(aeb_ynn_dot_int8_int8_sme2, 16, 64, M, N, K, 8, 64,
                             1, 1, 0, mc_threads[t], p5_shapes[i].reps_mc,
                             num_trials);
    }
    print_kernel_result(
        out, "pair5_int8_gemm", "8-Bit Linear GEMM (Vision/Audio)", "YNNPACK",
        "ynn::dot_int8_int8_int32_sme2", "SME2-INT8", M, N, K,
        p5_shapes[i].regime, y1, ymc, mc_threads, num_mc, ops, bytes, 1);
  }

  // ---------------------------------------------------------------------------
  // Pair 6: 2-Bit Weight-Only Linear GEMM (Audio Conformer)
  // ---------------------------------------------------------------------------
  struct {
    int M, N, K;
    const char *regime;
    int reps_1c, reps_mc;
  } p6_shapes[] = {
      {204, 1024, 1024, "L2_Cluster_Resident", 30, 80},
      {204, 4096, 1024, "L2_Cluster_Resident", 12, 30},
      {204, 1024, 4096, "L2_Cluster_Resident", 12, 30},
  };
  fprintf(stderr, "[bench] Pair 6: 2-Bit Linear GEMM...\n");
  for (size_t i = 0; i < sizeof(p6_shapes) / sizeof(p6_shapes[0]); ++i) {
    int M = p6_shapes[i].M, N = p6_shapes[i].N, K = p6_shapes[i].K;
    double ops = 2.0 * M * N * K;
    double bytes = (double)M * K + (double)N * K * 0.25 + (double)M * N * 4.0;

    bench_stats_t x1 = bench_xnn_qp8_sme2(
        xnn_qp8_f32_qc2w_gemm_minmax_ukernel_16x64c4__neonsme2, 16, 64, M, N, K,
        2, 0, 1, p6_shapes[i].reps_1c, num_trials);
    bench_stats_t xmc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      xmc[t] = bench_xnn_qp8_sme2(
          xnn_qp8_f32_qc2w_gemm_minmax_ukernel_16x64c4__neonsme2, 16, 64, M, N,
          K, 2, 0, mc_threads[t], p6_shapes[i].reps_mc, num_trials);
    }
    print_kernel_result(
        out, "pair6_int2_gemm", "2-Bit Linear GEMM (Audio)", "XNNPACK",
        "xnn_qp8_f32_qc2w_gemm_minmax_ukernel_16x64c4__neonsme2", "SME2-INT8",
        M, N, K, p6_shapes[i].regime, x1, xmc, mc_threads, num_mc, ops, bytes,
        1);

    bench_stats_t y1 = bench_ynn_dot(aeb_ynn_dot_int8_int2_sme, 16, 64, M, N, K,
                                     2, 16, 1, 1, 0, 1, p6_shapes[i].reps_1c,
                                     num_trials);
    bench_stats_t ymc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      ymc[t] = bench_ynn_dot(aeb_ynn_dot_int8_int2_sme, 16, 64, M, N, K, 2, 16,
                             1, 1, 0, mc_threads[t], p6_shapes[i].reps_mc,
                             num_trials);
    }
    print_kernel_result(
        out, "pair6_int2_gemm", "2-Bit Linear GEMM (Audio)", "YNNPACK",
        "ynn::dot_int8_int2_int32_sme", "SME-INT8", M, N, K,
        p6_shapes[i].regime, y1, ymc, mc_threads, num_mc, ops, bytes, 1);
  }

  // ---------------------------------------------------------------------------
  // Pair 7: Matrix Packing / 32-Bit Transpose
  // ---------------------------------------------------------------------------
  struct {
    int N, K;
    const char *regime;
    int reps_1c, reps_mc;
  } p7_shapes[] = {
      {1024, 256, "L2_Cluster_Copy", 80, 200},
      {4096, 512, "L2_Cluster_Copy", 24, 80},
      {2520, 1152, "SLC_Copy", 20, 60},
  };
  fprintf(stderr, "[bench] Pair 7: Matrix Packing / Transpose...\n");
  for (size_t i = 0; i < sizeof(p7_shapes) / sizeof(p7_shapes[0]); ++i) {
    int N = p7_shapes[i].N, K = p7_shapes[i].K;
    int is_last = (i + 1 == sizeof(p7_shapes) / sizeof(p7_shapes[0]));
    double bytes = 2.0 * (double)N * (double)K * sizeof(uint32_t);

    bench_stats_t x1 =
        bench_xnn_packw(N, K, 1, p7_shapes[i].reps_1c, num_trials);
    bench_stats_t xmc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      xmc[t] = bench_xnn_packw(N, K, mc_threads[t], p7_shapes[i].reps_mc,
                               num_trials);
    }
    print_kernel_result(
        out, "pair7_pack_transpose", "Matrix Packing / 32-Bit Transpose",
        "XNNPACK", "xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm",
        "NEON", 1, N, K, p7_shapes[i].regime, x1, xmc, mc_threads, num_mc, 0.0,
        bytes, 1);

    bench_stats_t y1 =
        bench_ynn_transpose(N, K, 1, p7_shapes[i].reps_1c, num_trials);
    bench_stats_t ymc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      ymc[t] = bench_ynn_transpose(N, K, mc_threads[t], p7_shapes[i].reps_mc,
                                   num_trials);
    }
    print_kernel_result(
        out, "pair7_pack_transpose", "Matrix Packing / 32-Bit Transpose",
        "YNNPACK", "ynn::transpose_x32_neon", "NEON", 1, N, K,
        p7_shapes[i].regime, y1, ymc, mc_threads, num_mc, 0.0, bytes, !is_last);
  }

  fprintf(out, "  ],\n");

  // ---------------------------------------------------------------------------
  // Controlled A/B Baselines: SME2/SME vs. NEON I8MM / DotProd / FMA
  // (Directly tests the M5 Max crossover where 18T NEON I8MM 8.13 TOPS exceeds
  //  18T shared-cluster SME2 6.47 TOPS!)
  // ---------------------------------------------------------------------------
  fprintf(stderr, "[bench] Controlled Baselines: SME2/SME vs NEON I8MM/DotProd/FMA...\n");
  fprintf(out, "  \"baseline_comparisons\": [\n");

  // B1: Pair 1 YNNPACK SME (1T production pick) + XNNPACK NEON I8MM (1x16c8)
  for (size_t i = 0; i < sizeof(p1_shapes) / sizeof(p1_shapes[0]); ++i) {
    int M = p1_shapes[i].M, N = p1_shapes[i].N, K = p1_shapes[i].K;
    double ops = 2.0 * M * N * K;
    double bytes = (double)M * K + (double)N * K * 0.5 + (double)M * N * 4.0;

    bench_stats_t y1 = bench_ynn_dot(aeb_ynn_dot_int8_int4_sme, 16, 64, M, N, K,
                                     4, 32, 1, 1, p1_shapes[i].cold_dram, 1,
                                     p1_shapes[i].reps_1c, num_trials);
    bench_stats_t ymc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      ymc[t] = bench_ynn_dot(aeb_ynn_dot_int8_int4_sme, 16, 64, M, N, K, 4, 32,
                             1, 1, p1_shapes[i].cold_dram, mc_threads[t],
                             p1_shapes[i].reps_mc, num_trials);
    }
    print_kernel_result(
        out, "pair1_int4_gemv", "4-Bit Linear GEMV (Decode)",
        "YNNPACK_SME_BASELINE", "ynn::dot_int8_int4_int32_sme", "SME-INT8", M,
        N, K, p1_shapes[i].regime, y1, ymc, mc_threads, num_mc, ops, bytes, 1);

    bench_stats_t x1 = bench_xnn_qd8_neon(
        xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm, 1, 16, M, N, K,
        4, p1_shapes[i].cold_dram, 1, p1_shapes[i].reps_1c, num_trials);
    bench_stats_t xmc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      xmc[t] = bench_xnn_qd8_neon(
          xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm, 1, 16, M, N, K,
          4, p1_shapes[i].cold_dram, mc_threads[t], p1_shapes[i].reps_mc,
          num_trials);
    }
    print_kernel_result(
        out, "pair1_int4_gemv", "4-Bit Linear GEMV (Decode)",
        "XNNPACK_NEON_BASELINE",
        "xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm", "NEON-I8MM", M,
        N, K, p1_shapes[i].regime, x1, xmc, mc_threads, num_mc, ops, bytes, 1);
  }

  // B2: Pair 2 NEON I8MM baselines (XNNPACK 4x16c8 + YNNPACK 10x8x8)
  for (size_t i = 0; i < sizeof(p2_shapes) / sizeof(p2_shapes[0]); ++i) {
    int M = p2_shapes[i].M, N = p2_shapes[i].N, K = p2_shapes[i].K;
    double ops = 2.0 * M * N * K;
    double bytes = (double)M * K + (double)N * K * 0.5 + (double)M * N * 4.0;
    int r1 = (p2_shapes[i].reps_1c > 1) ? 1 : 1;
    int rmc = (p2_shapes[i].reps_mc > 2) ? (p2_shapes[i].reps_mc / 2) : 2;

    bench_stats_t x1 = bench_xnn_qd8_neon(
        xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__neoni8mm, 4, 16, M, N, K,
        4, 0, 1, r1, num_trials);
    bench_stats_t xmc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      xmc[t] = bench_xnn_qd8_neon(
          xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__neoni8mm, 4, 16, M, N, K,
          4, 0, mc_threads[t], rmc, num_trials);
    }
    print_kernel_result(
        out, "pair2_int4_gemm", "4-Bit Linear GEMM (Prefill)",
        "XNNPACK_NEON_BASELINE",
        "xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__neoni8mm", "NEON-I8MM", M,
        N, K, p2_shapes[i].regime, x1, xmc, mc_threads, num_mc, ops, bytes, 1);

    bench_stats_t y1 = bench_ynn_dot(aeb_ynn_dot_int8_int4_10x8_neoni8mm, 10, 8,
                                     M, N, K, 4, 4, 1, 1, 0, 1, r1, num_trials);
    bench_stats_t ymc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      ymc[t] = bench_ynn_dot(aeb_ynn_dot_int8_int4_10x8_neoni8mm, 10, 8, M, N,
                             K, 4, 4, 1, 1, 0, mc_threads[t], rmc, num_trials);
    }
    print_kernel_result(
        out, "pair2_int4_gemm", "4-Bit Linear GEMM (Prefill)",
        "YNNPACK_NEON_BASELINE",
        "ynn::dot_int8_int4_int32_10x8x8_2x4x8_neoni8mm", "NEON-I8MM", M, N, K,
        p2_shapes[i].regime, y1, ymc, mc_threads, num_mc, ops, bytes, 1);
  }

  // B3: Pair 3 FP32 NEON FMA baselines
  for (size_t i = 0; i < sizeof(p3_shapes) / sizeof(p3_shapes[0]); ++i) {
    int M = p3_shapes[i].M, N = p3_shapes[i].N, K = p3_shapes[i].K;
    double ops = 2.0 * M * N * K;
    double bytes = ((double)M * K + (double)N * K + (double)M * N) * 4.0;
    int r1 = (p3_shapes[i].reps_1c > 2) ? (p3_shapes[i].reps_1c / 4) : 1;
    if (r1 < 1) r1 = 1;
    int rmc = (p3_shapes[i].reps_mc > 2) ? (p3_shapes[i].reps_mc / 2) : 2;

    bench_stats_t x1 = bench_xnn_f32_neon(M, N, K, 1, r1, num_trials);
    bench_stats_t xmc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      xmc[t] = bench_xnn_f32_neon(M, N, K, mc_threads[t], rmc, num_trials);
    }
    print_kernel_result(
        out, "pair3_fp32_attn_bmm", "FP32 Attention BatchMatMul",
        "XNNPACK_NEON_BASELINE",
        "xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128",
        "NEON-FP32", M, N, K, p3_shapes[i].regime, x1, xmc, mc_threads, num_mc,
        ops, bytes, 1);

    bench_stats_t y1 = bench_ynn_dot(aeb_ynn_dot_fp32_6x8_neon, 6, 8, M, N, K,
                                     32, 8 * sizeof(float), sizeof(float), 0, 0,
                                     1, r1, num_trials);
    bench_stats_t ymc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      ymc[t] = bench_ynn_dot(aeb_ynn_dot_fp32_6x8_neon, 6, 8, M, N, K, 32,
                             8 * sizeof(float), sizeof(float), 0, 0,
                             mc_threads[t], rmc, num_trials);
    }
    print_kernel_result(
        out, "pair3_fp32_attn_bmm", "FP32 Attention BatchMatMul",
        "YNNPACK_NEON_BASELINE", "ynn::dot_fp32_6x8x4_1x4x1_neon", "NEON-FP32",
        M, N, K, p3_shapes[i].regime, y1, ymc, mc_threads, num_mc, ops, bytes,
        1);
  }

  // B5: Pair 5 INT8 NEON I8MM baselines (tests 18T 8.13 TOPS NEON I8MM vs 6.47 TOPS SME2!)
  for (size_t i = 0; i < sizeof(p5_shapes) / sizeof(p5_shapes[0]); ++i) {
    int M = p5_shapes[i].M, N = p5_shapes[i].N, K = p5_shapes[i].K;
    double ops = 2.0 * M * N * K;
    double bytes = (double)M * K + (double)N * K + (double)M * N * 4.0;
    int r1 = (p5_shapes[i].reps_1c > 2) ? (p5_shapes[i].reps_1c / 2) : 1;
    int rmc = p5_shapes[i].reps_mc;

    bench_stats_t x1 = bench_xnn_qd8_neon(
        xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm, 4, 16, M, N, K,
        8, 0, 1, r1, num_trials);
    bench_stats_t xmc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      xmc[t] = bench_xnn_qd8_neon(
          xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm, 4, 16, M, N, K,
          8, 0, mc_threads[t], rmc, num_trials);
    }
    print_kernel_result(
        out, "pair5_int8_gemm", "8-Bit Linear GEMM (Vision/Audio)",
        "XNNPACK_NEON_BASELINE",
        "xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm", "NEON-I8MM", M,
        N, K, p5_shapes[i].regime, x1, xmc, mc_threads, num_mc, ops, bytes, 1);

    bench_stats_t y1 = bench_ynn_dot(aeb_ynn_dot_int8_int8_10x8_neoni8mm, 10, 8,
                                     M, N, K, 8, 8, 1, 1, 0, 1, r1, num_trials);
    bench_stats_t ymc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      ymc[t] = bench_ynn_dot(aeb_ynn_dot_int8_int8_10x8_neoni8mm, 10, 8, M, N,
                             K, 8, 8, 1, 1, 0, mc_threads[t], rmc, num_trials);
    }
    print_kernel_result(
        out, "pair5_int8_gemm", "8-Bit Linear GEMM (Vision/Audio)",
        "YNNPACK_NEON_BASELINE",
        "ynn::dot_int8_int8_int32_10x8x8_2x4x8_neoni8mm", "NEON-I8MM", M, N, K,
        p5_shapes[i].regime, y1, ymc, mc_threads, num_mc, ops, bytes, 1);
  }

  // B6: Pair 6 INT2 NEON DotProd baselines
  for (size_t i = 0; i < sizeof(p6_shapes) / sizeof(p6_shapes[0]); ++i) {
    int M = p6_shapes[i].M, N = p6_shapes[i].N, K = p6_shapes[i].K;
    int is_last = (i + 1 == sizeof(p6_shapes) / sizeof(p6_shapes[0]));
    double ops = 2.0 * M * N * K;
    double bytes_x = (double)M * K + (double)N * K * 0.25 + (double)M * N;
    double bytes_y = (double)M * K + (double)N * K * 0.25 + (double)M * N * 4.0;
    int r1 = (p6_shapes[i].reps_1c > 4) ? (p6_shapes[i].reps_1c / 4) : 2;
    int rmc = p6_shapes[i].reps_mc;

    bench_stats_t x1 = bench_xnn_qc2w_neondot(M, N, K, 1, r1, num_trials);
    bench_stats_t xmc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      xmc[t] = bench_xnn_qc2w_neondot(M, N, K, mc_threads[t], rmc, num_trials);
    }
    print_kernel_result(
        out, "pair6_int2_gemm", "2-Bit Linear GEMM (Audio)",
        "XNNPACK_NEON_BASELINE",
        "xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_8x8c4__neondot", "NEON-DotProd",
        M, N, K, p6_shapes[i].regime, x1, xmc, mc_threads, num_mc, ops, bytes_x,
        1);

    bench_stats_t y1 = bench_ynn_dot(aeb_ynn_dot_int8_int2_8x8_neondot, 8, 8, M,
                                     N, K, 2, 2, 1, 0, 0, 1, r1, num_trials);
    bench_stats_t ymc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      ymc[t] = bench_ynn_dot(aeb_ynn_dot_int8_int2_8x8_neondot, 8, 8, M, N, K,
                             2, 2, 1, 0, 0, mc_threads[t], rmc, num_trials);
    }
    print_kernel_result(
        out, "pair6_int2_gemm", "2-Bit Linear GEMM (Audio)",
        "YNNPACK_NEON_BASELINE",
        "ynn::dot_int8_int2_int32_8x8x16_1x4x16_neondot", "NEON-DotProd", M, N,
        K, p6_shapes[i].regime, y1, ymc, mc_threads, num_mc, ops, bytes_y,
        !is_last);
  }

  fprintf(out, "  ]\n");
  fprintf(out, "}\n");
  fflush(out);
  if (out != stdout) fclose(out);

  fprintf(stderr, "[bench] Apple M5 Max microkernel benchmark suite completed.\n");
  return 0;
}
