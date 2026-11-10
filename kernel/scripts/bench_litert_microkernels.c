// Repeatable Standalone Microkernel Benchmark Harness for LiteRT-LM
// (XNNPACK vs. YNNPACK on Intel Sapphire Rapids).
//
// Loaded via LD_PRELOAD into `/opt/litert-lm/bin/litert_lm_advanced_main` so
// that the exact production-compiled XNNPACK and YNNPACK microkernels are
// benchmarked directly in isolated single-core (1T) and NUMA Node 0 (28T)
// loops with zero framework overhead.
//
// Compile:
//   gcc -shared -fPIC -O3 -march=sapphirerapids -fopenmp \
//       kernel/scripts/bench_litert_microkernels.c -o libbench_ukernels.so -lm

#define _GNU_SOURCE
#include <immintrin.h>
#include <link.h>
#include <math.h>
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

#define MAX_TRIALS 16

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

static void pin_thread_to_core(int core_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
}

static void enable_amx_current_thread(void) {
  syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
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
  qsort(dts, n, sizeof(double), cmp_double);
  qsort(gops, n, sizeof(double), cmp_double);

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

static int phdr_cb(struct dl_phdr_info *info, size_t size, void *data) {
  (void)size;
  uintptr_t *base_out = (uintptr_t *)data;
  if (info->dlpi_name == NULL || info->dlpi_name[0] == '\0' ||
      strstr(info->dlpi_name, "litert_lm_advanced_main") != NULL) {
    *base_out = (uintptr_t)info->dlpi_addr;
    return 1;
  }
  return 0;
}

static uintptr_t find_binary_base(void) {
  uintptr_t base = 0;
  dl_iterate_phdr(phdr_cb, &base);
  return base;
}

// Microkernel function pointer typedefs
typedef void (*xnn_qd8_gemm_fn)(size_t mr, size_t nc, size_t kc, const void *a,
                                size_t a_stride, const void *w, void *c,
                                size_t cm_stride, size_t cn_stride,
                                const void *params, const void *quant_params);

typedef void (*xnn_f32_gemm_fn)(size_t mr, size_t nc, size_t kc_bytes,
                                const float *a, size_t a_stride_bytes,
                                const float *w, float *c,
                                size_t cm_stride_bytes, size_t cn_stride_bytes,
                                const void *params);

typedef void (*xnn_qs8_qc2w_gemm_fn)(size_t mr, size_t nc, size_t kc,
                                     const void *a, size_t a_stride,
                                     const void *w, void *c, size_t cm_stride,
                                     size_t cn_stride, const void *params);

typedef void (*xnn_f32_rmax_fn)(size_t batch_bytes, const float *input,
                                float *output, const void *params);

typedef void (*xnn_f32_raddstoreexpminusmax_fn)(size_t batch_bytes,
                                                const float *input,
                                                const float *max, float *output,
                                                float *sum, const void *params);

typedef void (*xnn_x32_packw_fn)(size_t g, size_t nc, size_t kc, size_t nr,
                                 size_t kr, size_t sr, size_t k_stride,
                                 const uint32_t *weights, const uint32_t *bias,
                                 const void *scale, void *packed_weights,
                                 size_t extra_bytes, const void *params);

typedef void (*ynn_dot_fn)(size_t M, size_t N, size_t K3, size_t K2, size_t K1,
                           size_t A_stride_m, size_t A_stride_k3,
                           size_t A_stride_k2, const void *A,
                           size_t B_stride_k3, size_t B_stride_k2,
                           size_t B_stride_k1, const void *B,
                           size_t C_in_stride_m, const void *C_in,
                           size_t C_out_stride_m, void *C_out, void *state);

typedef void (*ynn_reduce_fp32_fn)(size_t n, size_t k_bytes, size_t a_stride_n,
                                   const void *a, void *c, void *state);

typedef void (*ynn_transpose_x32_fn)(size_t m, size_t n, size_t n_bytes_a,
                                     size_t stride_a, const void *a,
                                     size_t stride_b, void *b);

// Global function pointers resolved at startup
static xnn_qd8_gemm_fn fn_xnn_qc4w_1x64_amx;
static xnn_qd8_gemm_fn fn_xnn_qc4w_16x64_amx;
static xnn_qd8_gemm_fn fn_xnn_qc8w_16x64_amx;
static xnn_f32_gemm_fn fn_xnn_f32_7x32_avx512;
static xnn_qs8_qc2w_gemm_fn fn_xnn_qc2w_5x8_avxvnni;
static xnn_f32_rmax_fn fn_xnn_rmax_avx512;
static xnn_f32_raddstoreexpminusmax_fn fn_xnn_raddexp_avx512;
static xnn_x32_packw_fn fn_xnn_packw_x32_avx512;

static ynn_dot_fn fn_ynn_dot_qc4_1x32_vnni;
static ynn_dot_fn fn_ynn_dot_qc4_12x16_vnni;
static ynn_dot_fn fn_ynn_dot_qc8_32x32_amx;
static ynn_dot_fn fn_ynn_dot_fp32_5x64_avx512;
static ynn_dot_fn fn_ynn_dot_qc2_4x64_vnni;
static ynn_reduce_fp32_fn fn_ynn_max_fp32_avx512;
static ynn_reduce_fp32_fn fn_ynn_sum_fp32_avx512;
static ynn_transpose_x32_fn fn_ynn_transpose_x32_avx512;

typedef struct {
  int32_t zero_point;
  float inv_scale;
} xnn_qd8_quant_params_t;

// -----------------------------------------------------------------------------
// Benchmark Helpers for GEMM / GEMV (XNNPACK vs YNNPACK)
// -----------------------------------------------------------------------------

static void print_kernel_result(FILE *out, const char *pair_id,
                                const char *category, const char *backend,
                                const char *kernel_name, const char *isa,
                                int M, int N, int K, const char *mem_regime,
                                bench_stats_t s1, bench_stats_t smc,
                                double ops, double bytes, int comma) {
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
  fprintf(out, "      },\n");
  fprintf(out, "      \"numa_node0_28c\": {\n");
  fprintf(out,
          "        \"latency_ms\": %.4f, \"gops\": %.2f, \"gbps\": %.2f, "
          "\"mean_gops\": %.2f, \"min_gops\": %.2f, \"max_gops\": %.2f, "
          "\"cv_pct\": %.2f\n",
          smc.median_ms, smc.gops_median, smc.gbps_median, smc.gops_mean,
          smc.gops_min, smc.gops_max, smc.cv_pct);
  fprintf(out, "      }\n");
  fprintf(out, "    }%s\n", comma ? "," : "");
  fflush(out);
}

// Generic GEMM runner for XNNPACK qd8_f32_qc4w / qc8w
static bench_stats_t bench_xnn_qd8_gemm(xnn_qd8_gemm_fn fn, int mr_tile,
                                        int nr_tile, int M, int N, int K,
                                        int bits_per_weight, int cold_dram,
                                        int num_threads, int reps,
                                        int num_trials) {
  size_t panel_w_bytes =
      (size_t)nr_tile * (size_t)K * (size_t)bits_per_weight / 8 + 1024;
  panel_w_bytes = (panel_w_bytes + 63) & ~(size_t)63;
  int num_n_panels = (N + nr_tile - 1) / nr_tile;
  size_t single_w_bytes = (size_t)num_n_panels * panel_w_bytes;

  // For cold DRAM streaming, allocate a 512 MB weight pool and cycle replicas
  int num_replicas = 1;
  if (cold_dram) {
    size_t target_pool = 512ULL * 1024ULL * 1024ULL;
    num_replicas = (int)(target_pool / single_w_bytes);
    if (num_replicas < 2) num_replicas = 2;
    if (num_replicas > 64) num_replicas = 64;
  }

  size_t total_w_bytes = (size_t)num_replicas * single_w_bytes;
  uint8_t *A = (uint8_t *)aligned_alloc(64, (size_t)M * (size_t)K + 4096);
  uint8_t *W = (uint8_t *)aligned_alloc(64, total_w_bytes + 4096);
  float *C = (float *)aligned_alloc(
      64, (size_t)M * (size_t)N * sizeof(float) + 4096);

  memset(A, 1, (size_t)M * (size_t)K + 4096);
  memset(W, 0x11, total_w_bytes + 4096);
  for (size_t i = 0; i < (size_t)M * (size_t)N; ++i) C[i] = 0.0f;

  // Initialize scale floats inside each packed weight panel so output isn't NaN
  float params[64];
  for (int i = 0; i < 64; ++i) params[i] = (i % 2 == 0) ? -1e6f : 1e6f;

  xnn_qd8_quant_params_t qparams[32];
  for (int i = 0; i < 32; ++i) {
    qparams[i].zero_point = 0;
    qparams[i].inv_scale = 0.01f;
  }

  double dt_samples[MAX_TRIALS];
  double t0 = 0.0, t1 = 0.0;
  enable_amx_current_thread();

  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    pin_thread_to_core(tid);
    enable_amx_current_thread();

    // Warmup
    #pragma omp for schedule(static)
    for (int p = 0; p < num_n_panels; ++p) {
      int n_idx = p * nr_tile;
      for (int m_idx = 0; m_idx < M; m_idx += mr_tile) {
        int cur_mr = (M - m_idx < mr_tile) ? (M - m_idx) : mr_tile;
        const uint8_t *w_ptr = W + (size_t)p * panel_w_bytes;
        fn(cur_mr, nr_tile, K, A + (size_t)m_idx * K, K, w_ptr,
           C + (size_t)m_idx * N + n_idx, (size_t)N * sizeof(float),
           (size_t)nr_tile * sizeof(float), params, qparams);
      }
    }

    for (int trial = 0; trial < num_trials; ++trial) {
      #pragma omp barrier
      #pragma omp master
      { t0 = now_sec(); }
      #pragma omp barrier

      for (int r = 0; r < reps; ++r) {
        const uint8_t *w_base =
            W + (size_t)(r % num_replicas) * single_w_bytes;
        #pragma omp for schedule(static) nowait
        for (int p = 0; p < num_n_panels; ++p) {
          int n_idx = p * nr_tile;
          const uint8_t *w_ptr = w_base + (size_t)p * panel_w_bytes;
          for (int m_idx = 0; m_idx < M; m_idx += mr_tile) {
            int cur_mr = (M - m_idx < mr_tile) ? (M - m_idx) : mr_tile;
            fn(cur_mr, nr_tile, K, A + (size_t)m_idx * K, K, w_ptr,
               C + (size_t)m_idx * N + n_idx, (size_t)N * sizeof(float),
               (size_t)nr_tile * sizeof(float), params, qparams);
          }
        }
      }

      #pragma omp barrier
      #pragma omp master
      {
        t1 = now_sec();
        dt_samples[trial] = (t1 - t0) / (double)reps;
      }
      #pragma omp barrier
    }
    _tile_release();
  }

  free(A);
  free(W);
  free(C);

  double total_ops = 2.0 * (double)M * (double)N * (double)K;
  double total_bytes =
      (double)M * (double)K +
      (double)N * (double)K * (double)bits_per_weight / 8.0 +
      (double)M * (double)N * sizeof(float);
  return compute_bench_stats(dt_samples, num_trials, total_ops, total_bytes);
}

// Generic GEMM runner for YNNPACK dot_*
static bench_stats_t bench_ynn_dot_gemm(ynn_dot_fn fn, int mr_tile, int nr_tile,
                                        int M, int N, int K,
                                        int bits_per_weight, size_t b_stride_k1,
                                        size_t a_elem_bytes, int cold_dram,
                                        int num_threads, int reps,
                                        int num_trials) {
  size_t panel_w_bytes =
      (size_t)nr_tile * (size_t)K * (size_t)bits_per_weight / 8 + 1024;
  panel_w_bytes = (panel_w_bytes + 63) & ~(size_t)63;
  int num_n_panels = (N + nr_tile - 1) / nr_tile;
  size_t single_w_bytes = (size_t)num_n_panels * panel_w_bytes;

  int num_replicas = 1;
  if (cold_dram) {
    size_t target_pool = 512ULL * 1024ULL * 1024ULL;
    num_replicas = (int)(target_pool / single_w_bytes);
    if (num_replicas < 2) num_replicas = 2;
    if (num_replicas > 64) num_replicas = 64;
  }

  size_t total_w_bytes = (size_t)num_replicas * single_w_bytes;
  uint8_t *A =
      (uint8_t *)aligned_alloc(64, (size_t)M * (size_t)K * a_elem_bytes + 4096);
  uint8_t *B = (uint8_t *)aligned_alloc(64, total_w_bytes + 4096);
  uint8_t *C = (uint8_t *)aligned_alloc(
      64, (size_t)M * (size_t)N * sizeof(int32_t) + 4096);

  memset(A, 1, (size_t)M * (size_t)K * a_elem_bytes + 4096);
  memset(B, 0x11, total_w_bytes + 4096);
  memset(C, 0, (size_t)M * (size_t)N * sizeof(int32_t) + 4096);

  double dt_samples[MAX_TRIALS];
  double t0 = 0.0, t1 = 0.0;
  enable_amx_current_thread();

  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    pin_thread_to_core(tid);
    enable_amx_current_thread();
    uint8_t state[256] __attribute__((aligned(64)));
    memset(state, 0, sizeof(state));

    size_t a_stride_m = (size_t)K * a_elem_bytes;
    size_t c_stride_m = (size_t)N * sizeof(int32_t);

    // Warmup
    #pragma omp for schedule(static)
    for (int p = 0; p < num_n_panels; ++p) {
      int n_idx = p * nr_tile;
      const uint8_t *b_ptr = B + (size_t)p * panel_w_bytes;
      for (int m_idx = 0; m_idx < M; m_idx += mr_tile) {
        int cur_mr = (M - m_idx < mr_tile) ? (M - m_idx) : mr_tile;
        uint8_t *c_ptr =
            C + ((size_t)m_idx * (size_t)N + (size_t)n_idx) * sizeof(int32_t);
        fn(cur_mr, nr_tile, 1, 1, K, a_stride_m, 0, 0,
           A + (size_t)m_idx * a_stride_m, 0, 0, b_stride_k1, b_ptr, c_stride_m,
           c_ptr, c_stride_m, c_ptr, state);
      }
    }

    for (int trial = 0; trial < num_trials; ++trial) {
      #pragma omp barrier
      #pragma omp master
      { t0 = now_sec(); }
      #pragma omp barrier

      for (int r = 0; r < reps; ++r) {
        const uint8_t *b_base =
            B + (size_t)(r % num_replicas) * single_w_bytes;
        #pragma omp for schedule(static) nowait
        for (int p = 0; p < num_n_panels; ++p) {
          int n_idx = p * nr_tile;
          const uint8_t *b_ptr = b_base + (size_t)p * panel_w_bytes;
          for (int m_idx = 0; m_idx < M; m_idx += mr_tile) {
            int cur_mr = (M - m_idx < mr_tile) ? (M - m_idx) : mr_tile;
            uint8_t *c_ptr = C + ((size_t)m_idx * (size_t)N + (size_t)n_idx) *
                                     sizeof(int32_t);
            fn(cur_mr, nr_tile, 1, 1, K, a_stride_m, 0, 0,
               A + (size_t)m_idx * a_stride_m, 0, 0, b_stride_k1, b_ptr,
               c_stride_m, c_ptr, c_stride_m, c_ptr, state);
          }
        }
      }

      #pragma omp barrier
      #pragma omp master
      {
        t1 = now_sec();
        dt_samples[trial] = (t1 - t0) / (double)reps;
      }
      #pragma omp barrier
    }
    _tile_release();
  }

  free(A);
  free(B);
  free(C);

  double total_ops = 2.0 * (double)M * (double)N * (double)K;
  double total_bytes =
      (double)M * (double)K * (double)a_elem_bytes +
      (double)N * (double)K * (double)bits_per_weight / 8.0 +
      (double)M * (double)N * sizeof(float);
  return compute_bench_stats(dt_samples, num_trials, total_ops, total_bytes);
}

// XNNPACK FP32 GEMM runner (7x32__avx512f_broadcast)
static bench_stats_t bench_xnn_f32_gemm(int M, int N, int K, int num_threads,
                                        int reps, int num_trials) {
  int mr_tile = 7, nr_tile = 32;
  int num_n_panels = (N + nr_tile - 1) / nr_tile;
  size_t panel_w_floats = (size_t)nr_tile * ((size_t)K + 1);
  float *A =
      (float *)aligned_alloc(64, (size_t)M * (size_t)K * sizeof(float) + 4096);
  float *W = (float *)aligned_alloc(
      64, (size_t)num_n_panels * panel_w_floats * sizeof(float) + 4096);
  float *C =
      (float *)aligned_alloc(64, (size_t)M * (size_t)N * sizeof(float) + 4096);

  for (size_t i = 0; i < (size_t)M * (size_t)K; ++i) A[i] = 0.01f;
  for (size_t i = 0; i < (size_t)num_n_panels * panel_w_floats; ++i)
    W[i] = 0.01f;
  for (size_t i = 0; i < (size_t)M * (size_t)N; ++i) C[i] = 0.0f;

  float minmax_params[32];
  for (int i = 0; i < 16; ++i) minmax_params[i] = -1e6f;
  for (int i = 16; i < 32; ++i) minmax_params[i] = 1e6f;

  double dt_samples[MAX_TRIALS];
  double t0 = 0.0, t1 = 0.0;

  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    pin_thread_to_core(tid);

    #pragma omp for schedule(static)
    for (int p = 0; p < num_n_panels; ++p) {
      int n_idx = p * nr_tile;
      const float *w_ptr = W + (size_t)p * panel_w_floats;
      for (int m_idx = 0; m_idx < M; m_idx += mr_tile) {
        int cur_mr = (M - m_idx < mr_tile) ? (M - m_idx) : mr_tile;
        fn_xnn_f32_7x32_avx512(cur_mr, nr_tile, (size_t)K * sizeof(float),
                               A + (size_t)m_idx * K, (size_t)K * sizeof(float),
                               w_ptr, C + (size_t)m_idx * N + n_idx,
                               (size_t)N * sizeof(float),
                               (size_t)nr_tile * sizeof(float), minmax_params);
      }
    }

    for (int trial = 0; trial < num_trials; ++trial) {
      #pragma omp barrier
      #pragma omp master
      { t0 = now_sec(); }
      #pragma omp barrier

      for (int r = 0; r < reps; ++r) {
        #pragma omp for schedule(static) nowait
        for (int p = 0; p < num_n_panels; ++p) {
          int n_idx = p * nr_tile;
          const float *w_ptr = W + (size_t)p * panel_w_floats;
          for (int m_idx = 0; m_idx < M; m_idx += mr_tile) {
            int cur_mr = (M - m_idx < mr_tile) ? (M - m_idx) : mr_tile;
            fn_xnn_f32_7x32_avx512(
                cur_mr, nr_tile, (size_t)K * sizeof(float),
                A + (size_t)m_idx * K, (size_t)K * sizeof(float), w_ptr,
                C + (size_t)m_idx * N + n_idx, (size_t)N * sizeof(float),
                (size_t)nr_tile * sizeof(float), minmax_params);
          }
        }
      }

      #pragma omp barrier
      #pragma omp master
      {
        t1 = now_sec();
        dt_samples[trial] = (t1 - t0) / (double)reps;
      }
      #pragma omp barrier
    }
  }

  free(A);
  free(W);
  free(C);

  double total_ops = 2.0 * (double)M * (double)N * (double)K;
  double total_bytes =
      ((double)M * K + (double)N * K + (double)M * N) * sizeof(float);
  return compute_bench_stats(dt_samples, num_trials, total_ops, total_bytes);
}

// XNNPACK 2-bit QC2W GEMM runner (5x8c8__avxvnni)
static bench_stats_t bench_xnn_qc2w_gemm(int M, int N, int K, int num_threads,
                                         int reps, int num_trials) {
  int mr_tile = 5, nr_tile = 8;
  int num_n_panels = (N + nr_tile - 1) / nr_tile;
  size_t panel_w_bytes = (size_t)nr_tile * (size_t)K / 4 + 256;
  panel_w_bytes = (panel_w_bytes + 63) & ~(size_t)63;

  uint8_t *A = (uint8_t *)aligned_alloc(64, (size_t)M * (size_t)K + 4096);
  uint8_t *W = (uint8_t *)aligned_alloc(
      64, (size_t)num_n_panels * panel_w_bytes + 4096);
  uint8_t *C = (uint8_t *)aligned_alloc(64, (size_t)M * (size_t)N + 4096);
  memset(A, 1, (size_t)M * (size_t)K + 4096);
  memset(W, 0x55, (size_t)num_n_panels * panel_w_bytes + 4096);
  memset(C, 0, (size_t)M * (size_t)N + 4096);

  uint8_t params[256] __attribute__((aligned(64)));
  memset(params, 0, sizeof(params));

  double dt_samples[MAX_TRIALS];
  double t0 = 0.0, t1 = 0.0;

  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    pin_thread_to_core(tid);

    #pragma omp for schedule(static)
    for (int p = 0; p < num_n_panels; ++p) {
      int n_idx = p * nr_tile;
      const uint8_t *w_ptr = W + (size_t)p * panel_w_bytes;
      for (int m_idx = 0; m_idx < M; m_idx += mr_tile) {
        int cur_mr = (M - m_idx < mr_tile) ? (M - m_idx) : mr_tile;
        fn_xnn_qc2w_5x8_avxvnni(cur_mr, nr_tile, K, A + (size_t)m_idx * K, K,
                                w_ptr, C + (size_t)m_idx * N + n_idx, N,
                                nr_tile, params);
      }
    }

    for (int trial = 0; trial < num_trials; ++trial) {
      #pragma omp barrier
      #pragma omp master
      { t0 = now_sec(); }
      #pragma omp barrier

      for (int r = 0; r < reps; ++r) {
        #pragma omp for schedule(static) nowait
        for (int p = 0; p < num_n_panels; ++p) {
          int n_idx = p * nr_tile;
          const uint8_t *w_ptr = W + (size_t)p * panel_w_bytes;
          for (int m_idx = 0; m_idx < M; m_idx += mr_tile) {
            int cur_mr = (M - m_idx < mr_tile) ? (M - m_idx) : mr_tile;
            fn_xnn_qc2w_5x8_avxvnni(cur_mr, nr_tile, K, A + (size_t)m_idx * K,
                                    K, w_ptr, C + (size_t)m_idx * N + n_idx, N,
                                    nr_tile, params);
          }
        }
      }

      #pragma omp barrier
      #pragma omp master
      {
        t1 = now_sec();
        dt_samples[trial] = (t1 - t0) / (double)reps;
      }
      #pragma omp barrier
    }
  }

  free(A);
  free(W);
  free(C);

  double total_ops = 2.0 * (double)M * (double)N * (double)K;
  double total_bytes =
      (double)M * K + (double)N * K * 0.25 + (double)M * N;
  return compute_bench_stats(dt_samples, num_trials, total_ops, total_bytes);
}

// Softmax Runners (Pair 4)
static bench_stats_t bench_xnn_softmax(int num_rows, int row_len,
                                       int num_threads, int reps,
                                       int num_trials) {
  size_t total_floats = (size_t)num_rows * (size_t)row_len;
  float *in = (float *)aligned_alloc(64, total_floats * sizeof(float) + 4096);
  float *out = (float *)aligned_alloc(64, total_floats * sizeof(float) + 4096);
  for (size_t i = 0; i < total_floats; ++i) in[i] = 0.5f;
  for (size_t i = 0; i < total_floats; ++i) out[i] = 0.0f;

  double dt_samples[MAX_TRIALS];
  double t0 = 0.0, t1 = 0.0;
  size_t row_bytes = (size_t)row_len * sizeof(float);

  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    pin_thread_to_core(tid);
    float max_val = -1e30f, sum_val = 0.0f;

    #pragma omp for schedule(static)
    for (int m = 0; m < num_rows; ++m) {
      max_val = -1e30f;
      fn_xnn_rmax_avx512(row_bytes, in + (size_t)m * row_len, &max_val, NULL);
      sum_val = 0.0f;
      fn_xnn_raddexp_avx512(row_bytes, in + (size_t)m * row_len, &max_val,
                            out + (size_t)m * row_len, &sum_val, NULL);
    }

    for (int trial = 0; trial < num_trials; ++trial) {
      #pragma omp barrier
      #pragma omp master
      { t0 = now_sec(); }
      #pragma omp barrier

      for (int r = 0; r < reps; ++r) {
        #pragma omp for schedule(static) nowait
        for (int m = 0; m < num_rows; ++m) {
          float rmax = -1e30f, rsum = 0.0f;
          fn_xnn_rmax_avx512(row_bytes, in + (size_t)m * row_len, &rmax, NULL);
          fn_xnn_raddexp_avx512(row_bytes, in + (size_t)m * row_len, &rmax,
                                out + (size_t)m * row_len, &rsum, NULL);
        }
      }

      #pragma omp barrier
      #pragma omp master
      {
        t1 = now_sec();
        dt_samples[trial] = (t1 - t0) / (double)reps;
      }
      #pragma omp barrier
    }
  }

  free(in);
  free(out);

  // Bytes: 1 pass read for rmax + 1 pass read + 1 pass write for raddstoreexpminusmax = 3 * N * 4B
  double total_bytes = 3.0 * (double)total_floats * sizeof(float);
  // Effective ops: ~12 FP32 ops per element (rmax + sub + range-red + deg-5 Horner poly + radd)
  double total_ops = 12.0 * (double)total_floats;
  return compute_bench_stats(dt_samples, num_trials, total_ops, total_bytes);
}

static bench_stats_t bench_ynn_softmax_red(int num_rows, int row_len,
                                           int num_threads, int reps,
                                           int num_trials) {
  size_t total_floats = (size_t)num_rows * (size_t)row_len;
  float *in = (float *)aligned_alloc(64, total_floats * sizeof(float) + 4096);
  float *out = (float *)aligned_alloc(64, total_floats * sizeof(float) + 4096);
  for (size_t i = 0; i < total_floats; ++i) in[i] = 0.5f;
  for (size_t i = 0; i < total_floats; ++i) out[i] = 0.5f;

  double dt_samples[MAX_TRIALS];
  double t0 = 0.0, t1 = 0.0;
  size_t row_bytes = (size_t)row_len * sizeof(float);

  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    pin_thread_to_core(tid);
    float max_out[16] __attribute__((aligned(64)));
    float sum_out[16] __attribute__((aligned(64)));

    #pragma omp for schedule(static)
    for (int m = 0; m < num_rows; ++m) {
      max_out[0] = -1e30f;
      fn_ynn_max_fp32_avx512(1, row_bytes, row_bytes, in + (size_t)m * row_len,
                             max_out, NULL);
      sum_out[0] = 0.0f;
      fn_ynn_sum_fp32_avx512(1, row_bytes, row_bytes, out + (size_t)m * row_len,
                             sum_out, NULL);
    }

    for (int trial = 0; trial < num_trials; ++trial) {
      #pragma omp barrier
      #pragma omp master
      { t0 = now_sec(); }
      #pragma omp barrier

      for (int r = 0; r < reps; ++r) {
        #pragma omp for schedule(static) nowait
        for (int m = 0; m < num_rows; ++m) {
          max_out[0] = -1e30f;
          fn_ynn_max_fp32_avx512(1, row_bytes, row_bytes,
                                 in + (size_t)m * row_len, max_out, NULL);
          sum_out[0] = 0.0f;
          fn_ynn_sum_fp32_avx512(1, row_bytes, row_bytes,
                                 out + (size_t)m * row_len, sum_out, NULL);
        }
      }

      #pragma omp barrier
      #pragma omp master
      {
        t1 = now_sec();
        dt_samples[trial] = (t1 - t0) / (double)reps;
      }
      #pragma omp barrier
    }
  }

  free(in);
  free(out);

  double total_bytes = 2.0 * (double)total_floats * sizeof(float);
  double total_ops = 2.0 * (double)total_floats;
  return compute_bench_stats(dt_samples, num_trials, total_ops, total_bytes);
}

// Pair 7: Matrix Packing / Transpose (packw_gemm_goi vs transpose_x32_avx512)
static bench_stats_t bench_xnn_packw(int N, int K, int num_threads, int reps,
                                     int num_trials) {
  int nr = 32;
  int num_panels = (N + nr - 1) / nr;
  size_t padded_N = (size_t)num_panels * (size_t)nr;
  size_t elem_count = padded_N * (size_t)(K + 16);
  uint32_t *in =
      (uint32_t *)aligned_alloc(64, elem_count * sizeof(uint32_t) + 65536);
  uint32_t *out =
      (uint32_t *)aligned_alloc(64, elem_count * sizeof(uint32_t) + 65536);
  memset(in, 1, elem_count * sizeof(uint32_t));
  memset(out, 0, elem_count * sizeof(uint32_t));

  double dt_samples[MAX_TRIALS];
  double t0 = 0.0, t1 = 0.0;

  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    pin_thread_to_core(tid);

    #pragma omp for schedule(static)
    for (int p = 0; p < num_panels; ++p) {
      int n_idx = p * nr;
      int cur_nr = (N - n_idx < nr) ? (N - n_idx) : nr;
      fn_xnn_packw_x32_avx512(1, cur_nr, K, nr, 1, 1, K,
                              in + (size_t)n_idx * K, NULL, NULL,
                              out + (size_t)p * nr * (K + 1), 0, NULL);
    }

    for (int trial = 0; trial < num_trials; ++trial) {
      #pragma omp barrier
      #pragma omp master
      { t0 = now_sec(); }
      #pragma omp barrier

      for (int r = 0; r < reps; ++r) {
        #pragma omp for schedule(static) nowait
        for (int p = 0; p < num_panels; ++p) {
          int n_idx = p * nr;
          int cur_nr = (N - n_idx < nr) ? (N - n_idx) : nr;
          fn_xnn_packw_x32_avx512(1, cur_nr, K, nr, 1, 1, K,
                                  in + (size_t)n_idx * K, NULL, NULL,
                                  out + (size_t)p * nr * (K + 1), 0, NULL);
        }
      }

      #pragma omp barrier
      #pragma omp master
      {
        t1 = now_sec();
        dt_samples[trial] = (t1 - t0) / (double)reps;
      }
      #pragma omp barrier
    }
  }

  free(in);
  free(out);

  double total_bytes = 2.0 * (double)N * (double)K * sizeof(uint32_t);
  return compute_bench_stats(dt_samples, num_trials, 0.0, total_bytes);
}

static bench_stats_t bench_ynn_transpose(int N, int K, int num_threads,
                                         int reps, int num_trials) {
  int tile_m = 32;
  int num_panels = (N + tile_m - 1) / tile_m;
  size_t padded_N = (size_t)num_panels * (size_t)tile_m;
  size_t elem_count = padded_N * (size_t)(K + 16);
  uint32_t *in =
      (uint32_t *)aligned_alloc(64, elem_count * sizeof(uint32_t) + 65536);
  uint32_t *out =
      (uint32_t *)aligned_alloc(64, elem_count * sizeof(uint32_t) + 65536);
  memset(in, 1, elem_count * sizeof(uint32_t));
  memset(out, 0, elem_count * sizeof(uint32_t));

  double dt_samples[MAX_TRIALS];
  double t0 = 0.0, t1 = 0.0;

  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    pin_thread_to_core(tid);

    #pragma omp for schedule(static)
    for (int p = 0; p < num_panels; ++p) {
      int m_idx = p * tile_m;
      int cur_m = (N - m_idx < tile_m) ? (N - m_idx) : tile_m;
      fn_ynn_transpose_x32_avx512(
          K, cur_m, (size_t)K * sizeof(uint32_t),
          (size_t)K * sizeof(uint32_t), in + (size_t)m_idx * K,
          padded_N * sizeof(uint32_t), out + (size_t)m_idx);
    }

    for (int trial = 0; trial < num_trials; ++trial) {
      #pragma omp barrier
      #pragma omp master
      { t0 = now_sec(); }
      #pragma omp barrier

      for (int r = 0; r < reps; ++r) {
        #pragma omp for schedule(static) nowait
        for (int p = 0; p < num_panels; ++p) {
          int m_idx = p * tile_m;
          int cur_m = (N - m_idx < tile_m) ? (N - m_idx) : tile_m;
          fn_ynn_transpose_x32_avx512(
              K, cur_m, (size_t)K * sizeof(uint32_t),
              (size_t)K * sizeof(uint32_t), in + (size_t)m_idx * K,
              padded_N * sizeof(uint32_t), out + (size_t)m_idx);
        }
      }

      #pragma omp barrier
      #pragma omp master
      {
        t1 = now_sec();
        dt_samples[trial] = (t1 - t0) / (double)reps;
      }
      #pragma omp barrier
    }
  }

  free(in);
  free(out);

  double total_bytes = 2.0 * (double)N * (double)K * sizeof(uint32_t);
  return compute_bench_stats(dt_samples, num_trials, 0.0, total_bytes);
}

__attribute__((constructor)) static void run_all_microkernel_benchmarks(void) {
  uintptr_t base = find_binary_base();
  if (!base) {
    fprintf(stderr, "ERROR: Could not find litert_lm_advanced_main base\n");
    _exit(1);
  }

  // Resolve exact microkernel symbols from litert_lm_advanced_main
  fn_xnn_qc4w_1x64_amx = (xnn_qd8_gemm_fn)(base + 0x01b53dc0);
  fn_xnn_qc4w_16x64_amx = (xnn_qd8_gemm_fn)(base + 0x01b512c0);
  fn_xnn_qc8w_16x64_amx = (xnn_qd8_gemm_fn)(base + 0x01b4ec20);
  fn_xnn_f32_7x32_avx512 = (xnn_f32_gemm_fn)(base + 0x01b07690);
  fn_xnn_qc2w_5x8_avxvnni = (xnn_qs8_qc2w_gemm_fn)(base + 0x01ab30b0);
  fn_xnn_rmax_avx512 = (xnn_f32_rmax_fn)(base + 0x01b050a0);
  fn_xnn_raddexp_avx512 =
      (xnn_f32_raddstoreexpminusmax_fn)(base + 0x01b06880);
  fn_xnn_packw_x32_avx512 = (xnn_x32_packw_fn)(base + 0x01afb150);

  fn_ynn_dot_qc4_1x32_vnni = (ynn_dot_fn)(base + 0x011f85a0);
  fn_ynn_dot_qc4_12x16_vnni = (ynn_dot_fn)(base + 0x011fb750);
  fn_ynn_dot_qc8_32x32_amx = (ynn_dot_fn)(base + 0x0120ab60);
  fn_ynn_dot_fp32_5x64_avx512 = (ynn_dot_fn)(base + 0x011d6ff0);
  fn_ynn_dot_qc2_4x64_vnni = (ynn_dot_fn)(base + 0x011ed490);
  fn_ynn_max_fp32_avx512 = (ynn_reduce_fp32_fn)(base + 0x0125dd00);
  fn_ynn_sum_fp32_avx512 = (ynn_reduce_fp32_fn)(base + 0x012637c0);
  fn_ynn_transpose_x32_avx512 = (ynn_transpose_x32_fn)(base + 0x012e68e0);

  int mc_threads = 28;
  int num_trials = 7;
  const char *out_path = getenv("AEB_BENCH_OUTPUT");
  FILE *out = stdout;
  if (out_path && out_path[0]) {
    FILE *f = fopen(out_path, "w");
    if (f) out = f;
  }

  fprintf(out, "{\n");
  fprintf(out, "  \"model\": \"Gemma 4 31B\",\n");
  fprintf(out, "  \"microarchitecture\": \"Intel Sapphire Rapids\",\n");
  fprintf(out, "  \"threads_single_core\": 1,\n");
  fprintf(out, "  \"threads_numa_node0\": %d,\n", mc_threads);
  fprintf(out, "  \"num_trials\": %d,\n", num_trials);
  fprintf(out, "  \"benchmarks\": [\n");

  // ---------------------------------------------------------------------------
  // Pair 1: 4-Bit Quantized Linear GEMV (M = 1, Decode)
  // ---------------------------------------------------------------------------
  struct {
    int M, N, K;
    int cold_dram;
    const char *regime;
    int reps;
  } p1_shapes[] = {
      {1, 21504, 5376, 1, "DDR5_DRAM_Streaming", 16},
      {1, 5376, 21504, 1, "DDR5_DRAM_Streaming", 16},
      {1, 8192, 5376, 1, "DDR5_DRAM_Streaming", 32},
      {1, 8192, 5376, 0, "L3_LLC_Warm", 64},
      {1, 4096, 5376, 0, "L3_LLC_Warm", 64},
  };
  for (size_t i = 0; i < sizeof(p1_shapes) / sizeof(p1_shapes[0]); ++i) {
    int M = p1_shapes[i].M, N = p1_shapes[i].N, K = p1_shapes[i].K;
    double ops = 2.0 * M * N * K;
    double bytes = (double)M * K + (double)N * K * 0.5 + (double)M * N * 4.0;
    bench_stats_t x1 =
        bench_xnn_qd8_gemm(fn_xnn_qc4w_1x64_amx, 1, 64, M, N, K, 4,
                           p1_shapes[i].cold_dram, 1, p1_shapes[i].reps,
                           num_trials);
    bench_stats_t xmc =
        bench_xnn_qd8_gemm(fn_xnn_qc4w_1x64_amx, 1, 64, M, N, K, 4,
                           p1_shapes[i].cold_dram, mc_threads,
                           p1_shapes[i].reps, num_trials);
    print_kernel_result(
        out, "pair1_int4_gemv", "4-Bit Linear GEMV (Decode)", "XNNPACK",
        "xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx", "AMX-INT8", M,
        N, K, p1_shapes[i].regime, x1, xmc, ops, bytes, 1);

    bench_stats_t y1 =
        bench_ynn_dot_gemm(fn_ynn_dot_qc4_1x32_vnni, 1, 32, M, N, K, 4, 16, 1,
                           p1_shapes[i].cold_dram, 1, p1_shapes[i].reps,
                           num_trials);
    bench_stats_t ymc =
        bench_ynn_dot_gemm(fn_ynn_dot_qc4_1x32_vnni, 1, 32, M, N, K, 4, 16, 1,
                           p1_shapes[i].cold_dram, mc_threads,
                           p1_shapes[i].reps, num_trials);
    print_kernel_result(
        out, "pair1_int4_gemv", "4-Bit Linear GEMV (Decode)", "YNNPACK",
        "ynn::dot_uint8_int4_int32_1x32x8_1x16x8_avx512vnni", "AVX512-VNNI", M,
        N, K, p1_shapes[i].regime, y1, ymc, ops, bytes, 1);
  }

  // ---------------------------------------------------------------------------
  // Pair 2: 4-Bit Quantized Linear Batched GEMM (M in {128, 1024}, Prefill)
  // ---------------------------------------------------------------------------
  struct {
    int M, N, K;
    const char *regime;
    int reps_1c, reps_mc;
  } p2_shapes[] = {
      {128, 21504, 5376, "DDR5_DRAM_to_L2_Tiled", 2, 10},
      {128, 5376, 21504, "DDR5_DRAM_to_L2_Tiled", 2, 10},
      {128, 8192, 5376, "L3_LLC_to_L2_Tiled", 4, 20},
      {1024, 21504, 5376, "Compute_Bound_Tiled", 1, 4},
      {1024, 8192, 5376, "Compute_Bound_Tiled", 1, 8},
  };
  for (size_t i = 0; i < sizeof(p2_shapes) / sizeof(p2_shapes[0]); ++i) {
    int M = p2_shapes[i].M, N = p2_shapes[i].N, K = p2_shapes[i].K;
    double ops = 2.0 * M * N * K;
    double bytes = (double)M * K + (double)N * K * 0.5 + (double)M * N * 4.0;
    bench_stats_t x1 = bench_xnn_qd8_gemm(
        fn_xnn_qc4w_16x64_amx, 16, 64, M, N, K, 4, 0, 1, p2_shapes[i].reps_1c,
        num_trials);
    bench_stats_t xmc = bench_xnn_qd8_gemm(
        fn_xnn_qc4w_16x64_amx, 16, 64, M, N, K, 4, 0, mc_threads,
        p2_shapes[i].reps_mc, num_trials);
    print_kernel_result(
        out, "pair2_int4_gemm", "4-Bit Linear GEMM (Prefill)", "XNNPACK",
        "xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx", "AMX-INT8",
        M, N, K, p2_shapes[i].regime, x1, xmc, ops, bytes, 1);

    bench_stats_t y1 = bench_ynn_dot_gemm(
        fn_ynn_dot_qc4_12x16_vnni, 12, 16, M, N, K, 4, 8, 1, 0, 1,
        p2_shapes[i].reps_1c, num_trials);
    bench_stats_t ymc = bench_ynn_dot_gemm(
        fn_ynn_dot_qc4_12x16_vnni, 12, 16, M, N, K, 4, 8, 1, 0, mc_threads,
        p2_shapes[i].reps_mc, num_trials);
    print_kernel_result(
        out, "pair2_int4_gemm", "4-Bit Linear GEMM (Prefill)", "YNNPACK",
        "ynn::dot_uint8_int4_int32_12x16x8_1x16x8_avx512vnni", "AVX512-VNNI", M,
        N, K, p2_shapes[i].regime, y1, ymc, ops, bytes, 1);
  }

  // ---------------------------------------------------------------------------
  // Pair 3: FP32 Attention BatchMatMul (Q*K^T and Scores*V)
  // ---------------------------------------------------------------------------
  struct {
    int M, N, K;
    const char *regime;
    int reps_1c, reps_mc;
  } p3_shapes[] = {
      {128, 1024, 256, "L2_Cache_Resident", 20, 80},
      {1024, 1024, 256, "L3_LLC_Resident", 4, 30},
      {1024, 4096, 512, "L3_LLC_Resident", 1, 10},
      {2520, 2520, 72, "L3_LLC_Resident", 4, 30},
  };
  for (size_t i = 0; i < sizeof(p3_shapes) / sizeof(p3_shapes[0]); ++i) {
    int M = p3_shapes[i].M, N = p3_shapes[i].N, K = p3_shapes[i].K;
    double ops = 2.0 * M * N * K;
    double bytes = ((double)M * K + (double)N * K + (double)M * N) * 4.0;
    bench_stats_t x1 = bench_xnn_f32_gemm(M, N, K, 1, p3_shapes[i].reps_1c,
                                          num_trials);
    bench_stats_t xmc = bench_xnn_f32_gemm(
        M, N, K, mc_threads, p3_shapes[i].reps_mc, num_trials);
    print_kernel_result(
        out, "pair3_fp32_attn_bmm", "FP32 Attention BatchMatMul", "XNNPACK",
        "xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast", "AVX512-FP32", M,
        N, K, p3_shapes[i].regime, x1, xmc, ops, bytes, 1);

    bench_stats_t y1 = bench_ynn_dot_gemm(
        fn_ynn_dot_fp32_5x64_avx512, 5, 64, M, N, K, 32, 64 * sizeof(float),
        sizeof(float), 0, 1, p3_shapes[i].reps_1c, num_trials);
    bench_stats_t ymc = bench_ynn_dot_gemm(
        fn_ynn_dot_fp32_5x64_avx512, 5, 64, M, N, K, 32, 64 * sizeof(float),
        sizeof(float), 0, mc_threads, p3_shapes[i].reps_mc, num_trials);
    print_kernel_result(
        out, "pair3_fp32_attn_bmm", "FP32 Attention BatchMatMul", "YNNPACK",
        "ynn::dot_fp32_5x64x1_1x16x1_avx512", "AVX512-FP32", M, N, K,
        p3_shapes[i].regime, y1, ymc, ops, bytes, 1);
  }

  // ---------------------------------------------------------------------------
  // Pair 4: FP32 Attention Softmax (Fused rmax+raddstoreexpminusmax vs max+sum)
  // ---------------------------------------------------------------------------
  struct {
    int rows, row_len;
    const char *regime;
    int reps;
  } p4_shapes[] = {
      {1024, 1024, "L2_L3_Resident", 40},
      {2520, 2520, "L3_LLC_Resident", 20},
      {1024, 4096, "L3_LLC_Resident", 20},
  };
  for (size_t i = 0; i < sizeof(p4_shapes) / sizeof(p4_shapes[0]); ++i) {
    int R = p4_shapes[i].rows, L = p4_shapes[i].row_len;
    double x_ops = 12.0 * (double)R * (double)L;
    double x_bytes = 3.0 * (double)R * (double)L * sizeof(float);
    bench_stats_t x1 = bench_xnn_softmax(R, L, 1, p4_shapes[i].reps, num_trials);
    bench_stats_t xmc =
        bench_xnn_softmax(R, L, mc_threads, p4_shapes[i].reps, num_trials);
    print_kernel_result(
        out, "pair4_fp32_softmax", "FP32 Attention Softmax", "XNNPACK",
        "xnn_f32_rmax + xnn_f32_raddstoreexpminusmax__avx512f_rr2_p5",
        "AVX512-FP32", R, L, 1, p4_shapes[i].regime, x1, xmc, x_ops, x_bytes, 1);

    double y_ops = 2.0 * (double)R * (double)L;
    double y_bytes = 2.0 * (double)R * (double)L * sizeof(float);
    bench_stats_t y1 =
        bench_ynn_softmax_red(R, L, 1, p4_shapes[i].reps, num_trials);
    bench_stats_t ymc =
        bench_ynn_softmax_red(R, L, mc_threads, p4_shapes[i].reps, num_trials);
    print_kernel_result(
        out, "pair4_fp32_softmax", "FP32 Attention Softmax (Reductions Only)",
        "YNNPACK", "ynn::max_k1_fp32_avx512 + ynn::sum_k1_fp32_avx512",
        "AVX512-FP32", R, L, 1, p4_shapes[i].regime, y1, ymc, y_ops, y_bytes, 1);
  }

  // ---------------------------------------------------------------------------
  // Pair 5: 8-Bit Dynamic-Quantized Linear GEMM (Vision ViT & Audio Conformer)
  // ---------------------------------------------------------------------------
  struct {
    int M, N, K;
    const char *regime;
    int reps_1c, reps_mc;
  } p5_shapes[] = {
      {280, 5376, 1152, "L3_LLC_Resident", 8, 40},
      {2520, 1152, 1152, "L3_LLC_Resident", 4, 30},
      {2520, 4304, 1152, "L3_LLC_Resident", 2, 15},
  };
  for (size_t i = 0; i < sizeof(p5_shapes) / sizeof(p5_shapes[0]); ++i) {
    int M = p5_shapes[i].M, N = p5_shapes[i].N, K = p5_shapes[i].K;
    double ops = 2.0 * M * N * K;
    double bytes = (double)M * K + (double)N * K + (double)M * N * 4.0;
    bench_stats_t x1 = bench_xnn_qd8_gemm(
        fn_xnn_qc8w_16x64_amx, 16, 64, M, N, K, 8, 0, 1, p5_shapes[i].reps_1c,
        num_trials);
    bench_stats_t xmc = bench_xnn_qd8_gemm(
        fn_xnn_qc8w_16x64_amx, 16, 64, M, N, K, 8, 0, mc_threads,
        p5_shapes[i].reps_mc, num_trials);
    print_kernel_result(
        out, "pair5_int8_gemm", "8-Bit Linear GEMM (Vision/Audio)", "XNNPACK",
        "xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx", "AMX-INT8",
        M, N, K, p5_shapes[i].regime, x1, xmc, ops, bytes, 1);

    bench_stats_t y1 = bench_ynn_dot_gemm(
        fn_ynn_dot_qc8_32x32_amx, 32, 32, M, N, K, 8, 32, 1, 0, 1,
        p5_shapes[i].reps_1c, num_trials);
    bench_stats_t ymc = bench_ynn_dot_gemm(
        fn_ynn_dot_qc8_32x32_amx, 32, 32, M, N, K, 8, 32, 1, 0, mc_threads,
        p5_shapes[i].reps_mc, num_trials);
    print_kernel_result(
        out, "pair5_int8_gemm", "8-Bit Linear GEMM (Vision/Audio)", "YNNPACK",
        "ynn::dot_int8_int8_int32_32x32x64_16x16x4_amxint8", "AMX-INT8", M, N,
        K, p5_shapes[i].regime, y1, ymc, ops, bytes, 1);
  }

  // ---------------------------------------------------------------------------
  // Pair 6: 2-Bit Weight-Only Linear GEMM (Audio Conformer)
  // ---------------------------------------------------------------------------
  struct {
    int M, N, K;
    const char *regime;
    int reps_1c, reps_mc;
  } p6_shapes[] = {
      {204, 1024, 1024, "L2_Cache_Resident", 20, 80},
      {204, 4096, 1024, "L2_L3_Resident", 8, 40},
      {204, 1024, 4096, "L2_L3_Resident", 8, 40},
  };
  for (size_t i = 0; i < sizeof(p6_shapes) / sizeof(p6_shapes[0]); ++i) {
    int M = p6_shapes[i].M, N = p6_shapes[i].N, K = p6_shapes[i].K;
    double ops = 2.0 * M * N * K;
    double bytes = (double)M * K + (double)N * K * 0.25 + (double)M * N;
    bench_stats_t x1 = bench_xnn_qc2w_gemm(M, N, K, 1, p6_shapes[i].reps_1c,
                                           num_trials);
    bench_stats_t xmc = bench_xnn_qc2w_gemm(
        M, N, K, mc_threads, p6_shapes[i].reps_mc, num_trials);
    print_kernel_result(
        out, "pair6_int2_gemm", "2-Bit Linear GEMM (Audio)", "XNNPACK",
        "xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni", "AVX-VNNI-256",
        M, N, K, p6_shapes[i].regime, x1, xmc, ops, bytes, 1);

    bench_stats_t y1 = bench_ynn_dot_gemm(
        fn_ynn_dot_qc2_4x64_vnni, 4, 64, M, N, K, 2, 16, 1, 0, 1,
        p6_shapes[i].reps_1c, num_trials);
    bench_stats_t ymc = bench_ynn_dot_gemm(
        fn_ynn_dot_qc2_4x64_vnni, 4, 64, M, N, K, 2, 16, 1, 0, mc_threads,
        p6_shapes[i].reps_mc, num_trials);
    print_kernel_result(
        out, "pair6_int2_gemm", "2-Bit Linear GEMM (Audio)", "YNNPACK",
        "ynn::dot_uint8_int2_int32_4x64x16_1x16x16_avx512vnni", "AVX512-VNNI",
        M, N, K, p6_shapes[i].regime, y1, ymc, ops, bytes, 1);
  }

  // ---------------------------------------------------------------------------
  // Pair 7: Matrix Packing / 32-Bit Transpose
  // ---------------------------------------------------------------------------
  struct {
    int N, K;
    const char *regime;
    int reps;
  } p7_shapes[] = {
      {1024, 256, "L2_Cache_Copy", 100},
      {4096, 512, "L3_LLC_Copy", 40},
      {2520, 1152, "L3_LLC_Copy", 30},
  };
  for (size_t i = 0; i < sizeof(p7_shapes) / sizeof(p7_shapes[0]); ++i) {
    int N = p7_shapes[i].N, K = p7_shapes[i].K;
    int is_last = (i + 1 == sizeof(p7_shapes) / sizeof(p7_shapes[0]));
    double bytes = 2.0 * (double)N * (double)K * sizeof(uint32_t);
    bench_stats_t x1 = bench_xnn_packw(N, K, 1, p7_shapes[i].reps, num_trials);
    bench_stats_t xmc =
        bench_xnn_packw(N, K, mc_threads, p7_shapes[i].reps, num_trials);
    print_kernel_result(
        out, "pair7_pack_transpose", "Matrix Packing / 32-Bit Transpose",
        "XNNPACK", "xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4_prfm",
        "AVX512F", 1, N, K, p7_shapes[i].regime, x1, xmc, 0.0, bytes, 1);

    bench_stats_t y1 =
        bench_ynn_transpose(N, K, 1, p7_shapes[i].reps, num_trials);
    bench_stats_t ymc =
        bench_ynn_transpose(N, K, mc_threads, p7_shapes[i].reps, num_trials);
    print_kernel_result(
        out, "pair7_pack_transpose", "Matrix Packing / 32-Bit Transpose",
        "YNNPACK", "ynn::transpose_x32_avx512", "AVX512F", 1, N, K,
        p7_shapes[i].regime, y1, ymc, 0.0, bytes, !is_last);
  }

  fprintf(out, "  ]\n");
  fprintf(out, "}\n");
  if (out != stdout) fclose(out);
  _exit(0);
}
