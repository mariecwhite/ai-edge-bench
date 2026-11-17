// Repeatable Standalone Microkernel Benchmark Harness for LiteRT-LM
// (XNNPACK vs. YNNPACK on Intel Sapphire Rapids).
//
// Supports two build modes:
//   1. Direct-linked standalone executable (`-DAEB_DIRECT_LINK`): links
//      directly against compiled upstream XNNPACK and YNNPACK microkernel
//      object files and runs from `main()` with CLI flags
//      (`--threads 28,56 --trials 7 --output <path>`).
//   2. LD_PRELOAD shared library (default): loaded into
//      `/opt/litert-lm/bin/litert_lm_advanced_main` so that the exact
//      production-compiled microkernels are benchmarked directly in isolated
//      single-core (1T) and NUMA Node 0 multi-core loops with zero framework
//      overhead.
//
// Compile (LD_PRELOAD mode):
//   gcc -shared -fPIC -O3 -march=sapphirerapids -fopenmp \
//       kernel/scripts/bench_litert_microkernels.c -o libbench_ukernels.so -lm
//
// Compile (Direct-link standalone mode):
//   gcc -DAEB_DIRECT_LINK -O3 -march=sapphirerapids -fopenmp -c \
//       kernel/scripts/bench_litert_microkernels.c -o bench_ukernels.o

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
#define MAX_THREAD_CFGS 8

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

#ifndef AEB_DIRECT_LINK
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
#endif

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

// YNNPACK reduce kernels take `k` in ELEMENTS and `a_stride_n` in BYTES.
typedef void (*ynn_reduce_fp32_fn)(size_t n, size_t k, size_t a_stride_n,
                                   const void *a, void *x0, void *x1);

typedef void (*ynn_transpose_x32_fn)(size_t m, size_t n, size_t n_bytes_a,
                                     size_t stride_a, const void *a,
                                     size_t stride_b, void *b);

#ifdef AEB_DIRECT_LINK
// Direct-linked XNNPACK microkernel symbols (C linkage)
extern void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx(
    size_t mr, size_t nc, size_t kc, const void *a, size_t a_stride,
    const void *w, void *c, size_t cm_stride, size_t cn_stride,
    const void *params, const void *quant_params);
extern void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx(
    size_t mr, size_t nc, size_t kc, const void *a, size_t a_stride,
    const void *w, void *c, size_t cm_stride, size_t cn_stride,
    const void *params, const void *quant_params);
extern void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx(
    size_t mr, size_t nc, size_t kc, const void *a, size_t a_stride,
    const void *w, void *c, size_t cm_stride, size_t cn_stride,
    const void *params, const void *quant_params);
extern void xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast(
    size_t mr, size_t nc, size_t kc_bytes, const float *a,
    size_t a_stride_bytes, const float *w, float *c, size_t cm_stride_bytes,
    size_t cn_stride_bytes, const void *params);
extern void xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni(
    size_t mr, size_t nc, size_t kc, const void *a, size_t a_stride,
    const void *w, void *c, size_t cm_stride, size_t cn_stride,
    const void *params);
extern void xnn_f32_rmax_ukernel__avx512f_u64_acc4(size_t batch_bytes,
                                                   const float *input,
                                                   float *output,
                                                   const void *params);
extern void xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc2(
    size_t batch_bytes, const float *input, const float *max, float *output,
    float *sum, const void *params);
extern void xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4_prfm(
    size_t g, size_t nc, size_t kc, size_t nr, size_t kr, size_t sr,
    size_t k_stride, const uint32_t *weights, const uint32_t *bias,
    const void *scale, void *packed_weights, size_t extra_bytes,
    const void *params);

// Direct-linked YNNPACK C bridge wrappers (provided by ynn_bridge.cc)
extern void aeb_ynn_dot_qc4_32x32_amx(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_dot_qc4_1x32_vnni(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_dot_qc4_12x16_vnni(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_dot_qc8_32x32_amx(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_dot_fp32_5x64_avx512(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_dot_qc2_4x64_vnni(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_max_fp32_avx512(size_t n, size_t k, size_t a_stride_n,
                                    const void *a, void *x0, void *x1);
extern void aeb_ynn_sum_fp32_avx512(size_t n, size_t k, size_t a_stride_n,
                                    const void *a, void *x0, void *x1);
extern void aeb_ynn_transpose_x32_avx512(size_t m, size_t n, size_t n_bytes_a,
                                         size_t stride_a, const void *a,
                                         size_t stride_b, void *b);
#endif

// Global function pointers resolved at startup
static xnn_qd8_gemm_fn fn_xnn_qc4w_1x64_amx;
static xnn_qd8_gemm_fn fn_xnn_qc4w_16x64_amx;
static xnn_qd8_gemm_fn fn_xnn_qc8w_16x64_amx;
static xnn_f32_gemm_fn fn_xnn_f32_7x32_avx512;
static xnn_qs8_qc2w_gemm_fn fn_xnn_qc2w_5x8_avxvnni;
static xnn_f32_rmax_fn fn_xnn_rmax_avx512;
static xnn_f32_raddstoreexpminusmax_fn fn_xnn_raddexp_avx512;
static xnn_x32_packw_fn fn_xnn_packw_x32_avx512;

static ynn_dot_fn fn_ynn_dot_qc4_32x32_amx;
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
  struct {
    int16_t output_zero_point;
    int16_t output_min;
    int16_t output_max;
  } fp32_scalar;
} aeb_xnn_qs8_qc8w_conv_minmax_params_t;

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
    fprintf(out, "      \"numa_node0_%dc\": {\n", mc_threads[i]);
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
// Benchmark Helpers for GEMM / GEMV (XNNPACK vs YNNPACK)
// -----------------------------------------------------------------------------

// Generic GEMM runner for XNNPACK qd8_f32_qc4w / qc8w
// Packed panel layout (nr = 64):
//   [0 .. 255]                 : int32_t ksum[64]
//   [256 .. 256 + w_data - 1]  : packed weights (nr * K * bits / 8)
//   [256 + w_data .. +255]     : float filter_output_scale[64]
//   [512 + w_data .. +255]     : float bias[64]
static bench_stats_t bench_xnn_qd8_gemm(xnn_qd8_gemm_fn fn, int mr_tile,
                                        int nr_tile, int M, int N, int K,
                                        int bits_per_weight, int cold_dram,
                                        int num_threads, int reps,
                                        int num_trials) {
  size_t ksum_bytes = (size_t)nr_tile * sizeof(int32_t);
  size_t w_data_bytes =
      (size_t)nr_tile * (size_t)K * (size_t)bits_per_weight / 8;
  size_t scale_bytes = (size_t)nr_tile * sizeof(float);
  size_t bias_bytes = (size_t)nr_tile * sizeof(float);
  size_t panel_w_bytes = ksum_bytes + w_data_bytes + scale_bytes + bias_bytes;
  panel_w_bytes = (panel_w_bytes + 63) & ~(size_t)63;

  int num_n_panels = (N + nr_tile - 1) / nr_tile;
  int num_m_tiles = (M + mr_tile - 1) / mr_tile;
  int total_tiles = num_n_panels * num_m_tiles;
  size_t single_w_bytes = (size_t)num_n_panels * panel_w_bytes;

  // For cold DRAM streaming, allocate a 1 GiB weight pool (~9x 105 MiB L3)
  int num_replicas = 1;
  if (cold_dram) {
    size_t target_pool = 1024ULL * 1024ULL * 1024ULL;
    num_replicas = (int)(target_pool / single_w_bytes);
    if (num_replicas < 2) num_replicas = 2;
    if (num_replicas > 128) num_replicas = 128;
  }

  size_t total_w_bytes = (size_t)num_replicas * single_w_bytes;
  size_t a_bytes = ((size_t)M * (size_t)K + 4096 + 63) & ~(size_t)63;
  size_t w_alloc = (total_w_bytes + 4096 + 63) & ~(size_t)63;
  size_t c_bytes =
      ((size_t)M * (size_t)N * sizeof(float) + 4096 + 63) & ~(size_t)63;

  uint8_t *A = (uint8_t *)aligned_alloc(64, a_bytes);
  uint8_t *W = (uint8_t *)aligned_alloc(64, w_alloc);
  float *C = (float *)aligned_alloc(64, c_bytes);

  memset(A, 1, a_bytes);
  memset(W, 0x11, w_alloc);
  for (size_t i = 0; i < (size_t)M * (size_t)N; ++i) C[i] = 0.0f;

  // Initialize ksum, scale, and bias inside every packed panel (zero denormals)
  for (int rep = 0; rep < num_replicas; ++rep) {
    uint8_t *w_rep = W + (size_t)rep * single_w_bytes;
    for (int p = 0; p < num_n_panels; ++p) {
      uint8_t *panel = w_rep + (size_t)p * panel_w_bytes;
      memset(panel, 0, ksum_bytes);
      float *scales = (float *)(panel + ksum_bytes + w_data_bytes);
      float *biases = (float *)(panel + ksum_bytes + w_data_bytes + scale_bytes);
      for (int j = 0; j < nr_tile; ++j) {
        scales[j] = 0.01f;
        biases[j] = 0.0f;
      }
    }
  }

  aeb_xnn_f32_qc4w_minmax_params_t params;
  params.scalar.min = -1e6f;
  params.scalar.max = 1e6f;
  params.scalar.kernel_zero_point = 0;

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

    // Warmup (N-major 2D tiling: adjacent tiles share packed weight panel p)
    #pragma omp for schedule(static)
    for (int t = 0; t < total_tiles; ++t) {
      int p = t / num_m_tiles;
      int m_idx = (t % num_m_tiles) * mr_tile;
      int cur_mr = (M - m_idx < mr_tile) ? (M - m_idx) : mr_tile;
      int n_idx = p * nr_tile;
      int cur_nc = (N - n_idx < nr_tile) ? (N - n_idx) : nr_tile;
      const uint8_t *w_ptr = W + (size_t)p * panel_w_bytes;
      fn(cur_mr, cur_nc, K, A + (size_t)m_idx * K, K, w_ptr,
         C + (size_t)m_idx * N + n_idx, (size_t)N * sizeof(float),
         (size_t)nr_tile * sizeof(float), &params, qparams);
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
        for (int t = 0; t < total_tiles; ++t) {
          int p = t / num_m_tiles;
          int m_idx = (t % num_m_tiles) * mr_tile;
          int cur_mr = (M - m_idx < mr_tile) ? (M - m_idx) : mr_tile;
          int n_idx = p * nr_tile;
          int cur_nc = (N - n_idx < nr_tile) ? (N - n_idx) : nr_tile;
          const uint8_t *w_ptr = w_base + (size_t)p * panel_w_bytes;
          fn(cur_mr, cur_nc, K, A + (size_t)m_idx * K, K, w_ptr,
             C + (size_t)m_idx * N + n_idx, (size_t)N * sizeof(float),
             (size_t)nr_tile * sizeof(float), &params, qparams);
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
  int num_m_tiles = (M + mr_tile - 1) / mr_tile;
  int total_tiles = num_n_panels * num_m_tiles;
  size_t single_w_bytes = (size_t)num_n_panels * panel_w_bytes;

  int num_replicas = 1;
  if (cold_dram) {
    size_t target_pool = 1024ULL * 1024ULL * 1024ULL;
    num_replicas = (int)(target_pool / single_w_bytes);
    if (num_replicas < 2) num_replicas = 2;
    if (num_replicas > 128) num_replicas = 128;
  }

  size_t total_w_bytes = (size_t)num_replicas * single_w_bytes;
  size_t a_bytes =
      ((size_t)M * (size_t)K * a_elem_bytes + 4096 + 63) & ~(size_t)63;
  size_t b_alloc = (total_w_bytes + 4096 + 63) & ~(size_t)63;
  size_t c_bytes =
      ((size_t)M * (size_t)N * sizeof(int32_t) + 4096 + 63) & ~(size_t)63;

  uint8_t *A = (uint8_t *)aligned_alloc(64, a_bytes);
  uint8_t *B = (uint8_t *)aligned_alloc(64, b_alloc);
  uint8_t *C = (uint8_t *)aligned_alloc(64, c_bytes);

  if (a_elem_bytes == sizeof(float)) {
    float *af = (float *)A;
    float *bf = (float *)B;
    for (size_t i = 0; i < a_bytes / sizeof(float); ++i) af[i] = 0.01f;
    for (size_t i = 0; i < b_alloc / sizeof(float); ++i) bf[i] = 0.01f;
  } else {
    memset(A, 1, a_bytes);
    memset(B, 0x11, b_alloc);
  }
  memset(C, 0, c_bytes);

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

    // Warmup (N-major 2D tiling; C_in = NULL for single-pass C = A * B)
    #pragma omp for schedule(static)
    for (int t = 0; t < total_tiles; ++t) {
      int p = t / num_m_tiles;
      int m_idx = (t % num_m_tiles) * mr_tile;
      int cur_mr = (M - m_idx < mr_tile) ? (M - m_idx) : mr_tile;
      int n_idx = p * nr_tile;
      int cur_nc = (N - n_idx < nr_tile) ? (N - n_idx) : nr_tile;
      const uint8_t *b_ptr = B + (size_t)p * panel_w_bytes;
      uint8_t *c_ptr =
          C + ((size_t)m_idx * (size_t)N + (size_t)n_idx) * sizeof(int32_t);
      fn(cur_mr, cur_nc, 1, 1, K, a_stride_m, 0, 0,
         A + (size_t)m_idx * a_stride_m, 0, 0, b_stride_k1, b_ptr, 0, NULL,
         c_stride_m, c_ptr, state);
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
        for (int t = 0; t < total_tiles; ++t) {
          int p = t / num_m_tiles;
          int m_idx = (t % num_m_tiles) * mr_tile;
          int cur_mr = (M - m_idx < mr_tile) ? (M - m_idx) : mr_tile;
          int n_idx = p * nr_tile;
          int cur_nc = (N - n_idx < nr_tile) ? (N - n_idx) : nr_tile;
          const uint8_t *b_ptr = b_base + (size_t)p * panel_w_bytes;
          uint8_t *c_ptr = C + ((size_t)m_idx * (size_t)N + (size_t)n_idx) *
                                   sizeof(int32_t);
          fn(cur_mr, cur_nc, 1, 1, K, a_stride_m, 0, 0,
             A + (size_t)m_idx * a_stride_m, 0, 0, b_stride_k1, b_ptr, 0, NULL,
             c_stride_m, c_ptr, state);
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
  int num_m_tiles = (M + mr_tile - 1) / mr_tile;
  int total_tiles = num_n_panels * num_m_tiles;
  size_t panel_w_floats = (size_t)nr_tile * ((size_t)K + 1);
  size_t a_bytes =
      ((size_t)M * (size_t)K * sizeof(float) + 4096 + 63) & ~(size_t)63;
  size_t w_bytes =
      ((size_t)num_n_panels * panel_w_floats * sizeof(float) + 4096 + 63) &
      ~(size_t)63;
  size_t c_bytes =
      ((size_t)M * (size_t)N * sizeof(float) + 4096 + 63) & ~(size_t)63;

  float *A = (float *)aligned_alloc(64, a_bytes);
  float *W = (float *)aligned_alloc(64, w_bytes);
  float *C = (float *)aligned_alloc(64, c_bytes);

  for (size_t i = 0; i < (size_t)M * (size_t)K; ++i) A[i] = 0.01f;
  for (size_t i = 0; i < (size_t)num_n_panels * panel_w_floats; ++i)
    W[i] = 0.01f;
  for (size_t i = 0; i < (size_t)M * (size_t)N; ++i) C[i] = 0.0f;

  aeb_xnn_f32_minmax_params_t minmax_params;
  minmax_params.scalar.min = -1e6f;
  minmax_params.scalar.max = 1e6f;

  double dt_samples[MAX_TRIALS];
  double t0 = 0.0, t1 = 0.0;

  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    pin_thread_to_core(tid);

    #pragma omp for schedule(static)
    for (int t = 0; t < total_tiles; ++t) {
      int p = t / num_m_tiles;
      int m_idx = (t % num_m_tiles) * mr_tile;
      int cur_mr = (M - m_idx < mr_tile) ? (M - m_idx) : mr_tile;
      int n_idx = p * nr_tile;
      int cur_nc = (N - n_idx < nr_tile) ? (N - n_idx) : nr_tile;
      const float *w_ptr = W + (size_t)p * panel_w_floats;
      fn_xnn_f32_7x32_avx512(cur_mr, cur_nc, (size_t)K * sizeof(float),
                             A + (size_t)m_idx * K, (size_t)K * sizeof(float),
                             w_ptr, C + (size_t)m_idx * N + n_idx,
                             (size_t)N * sizeof(float),
                             (size_t)nr_tile * sizeof(float), &minmax_params);
    }

    for (int trial = 0; trial < num_trials; ++trial) {
      #pragma omp barrier
      #pragma omp master
      { t0 = now_sec(); }
      #pragma omp barrier

      for (int r = 0; r < reps; ++r) {
        #pragma omp for schedule(static) nowait
        for (int t = 0; t < total_tiles; ++t) {
          int p = t / num_m_tiles;
          int m_idx = (t % num_m_tiles) * mr_tile;
          int cur_mr = (M - m_idx < mr_tile) ? (M - m_idx) : mr_tile;
          int n_idx = p * nr_tile;
          int cur_nc = (N - n_idx < nr_tile) ? (N - n_idx) : nr_tile;
          const float *w_ptr = W + (size_t)p * panel_w_floats;
          fn_xnn_f32_7x32_avx512(
              cur_mr, cur_nc, (size_t)K * sizeof(float), A + (size_t)m_idx * K,
              (size_t)K * sizeof(float), w_ptr, C + (size_t)m_idx * N + n_idx,
              (size_t)N * sizeof(float), (size_t)nr_tile * sizeof(float),
              &minmax_params);
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
  int num_m_tiles = (M + mr_tile - 1) / mr_tile;
  int total_tiles = num_n_panels * num_m_tiles;
  size_t bias_bytes = (size_t)nr_tile * sizeof(int32_t);
  size_t w_data_bytes = (size_t)nr_tile * (size_t)K / 4;
  size_t scale_bytes = (size_t)nr_tile * sizeof(float);
  size_t panel_w_bytes = (bias_bytes + w_data_bytes + scale_bytes + 63) & ~(size_t)63;

  size_t a_bytes = ((size_t)M * (size_t)K + 4096 + 63) & ~(size_t)63;
  size_t w_bytes =
      ((size_t)num_n_panels * panel_w_bytes + 4096 + 63) & ~(size_t)63;
  size_t c_bytes = ((size_t)M * (size_t)N + 4096 + 63) & ~(size_t)63;

  uint8_t *A = (uint8_t *)aligned_alloc(64, a_bytes);
  uint8_t *W = (uint8_t *)aligned_alloc(64, w_bytes);
  uint8_t *C = (uint8_t *)aligned_alloc(64, c_bytes);
  memset(A, 1, a_bytes);
  memset(W, 0x55, w_bytes);
  memset(C, 0, c_bytes);

  for (int p = 0; p < num_n_panels; ++p) {
    uint8_t *panel = W + (size_t)p * panel_w_bytes;
    memset(panel, 0, bias_bytes);
    float *scales = (float *)(panel + bias_bytes + w_data_bytes);
    for (int j = 0; j < nr_tile; ++j) scales[j] = 0.01f;
  }

  aeb_xnn_qs8_qc8w_conv_minmax_params_t params;
  params.fp32_scalar.output_zero_point = 0;
  params.fp32_scalar.output_min = -128;
  params.fp32_scalar.output_max = 127;

  double dt_samples[MAX_TRIALS];
  double t0 = 0.0, t1 = 0.0;

  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    pin_thread_to_core(tid);

    #pragma omp for schedule(static)
    for (int t = 0; t < total_tiles; ++t) {
      int p = t / num_m_tiles;
      int m_idx = (t % num_m_tiles) * mr_tile;
      int cur_mr = (M - m_idx < mr_tile) ? (M - m_idx) : mr_tile;
      int n_idx = p * nr_tile;
      int cur_nc = (N - n_idx < nr_tile) ? (N - n_idx) : nr_tile;
      const uint8_t *w_ptr = W + (size_t)p * panel_w_bytes;
      fn_xnn_qc2w_5x8_avxvnni(cur_mr, cur_nc, K, A + (size_t)m_idx * K, K,
                              w_ptr, C + (size_t)m_idx * N + n_idx, N, nr_tile,
                              &params);
    }

    for (int trial = 0; trial < num_trials; ++trial) {
      #pragma omp barrier
      #pragma omp master
      { t0 = now_sec(); }
      #pragma omp barrier

      for (int r = 0; r < reps; ++r) {
        #pragma omp for schedule(static) nowait
        for (int t = 0; t < total_tiles; ++t) {
          int p = t / num_m_tiles;
          int m_idx = (t % num_m_tiles) * mr_tile;
          int cur_mr = (M - m_idx < mr_tile) ? (M - m_idx) : mr_tile;
          int n_idx = p * nr_tile;
          int cur_nc = (N - n_idx < nr_tile) ? (N - n_idx) : nr_tile;
          const uint8_t *w_ptr = W + (size_t)p * panel_w_bytes;
          fn_xnn_qc2w_5x8_avxvnni(cur_mr, cur_nc, K, A + (size_t)m_idx * K, K,
                                  w_ptr, C + (size_t)m_idx * N + n_idx, N,
                                  nr_tile, &params);
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
  size_t buf_bytes = (total_floats * sizeof(float) + 4096 + 63) & ~(size_t)63;
  float *in = (float *)aligned_alloc(64, buf_bytes);
  float *out = (float *)aligned_alloc(64, buf_bytes);
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

  // Bytes: 1 pass read for rmax + 1 pass read + 1 pass write for raddstoreexpminusmax
  double total_bytes = 3.0 * (double)total_floats * sizeof(float);
  // Effective ops: ~12 FP32 ops per element (rmax + sub + range-red + deg-5 Horner poly + radd)
  double total_ops = 12.0 * (double)total_floats;
  return compute_bench_stats(dt_samples, num_trials, total_ops, total_bytes);
}

static bench_stats_t bench_ynn_softmax_red(int num_rows, int row_len,
                                           int num_threads, int reps,
                                           int num_trials) {
  size_t total_floats = (size_t)num_rows * (size_t)row_len;
  size_t buf_bytes = (total_floats * sizeof(float) + 4096 + 63) & ~(size_t)63;
  float *in = (float *)aligned_alloc(64, buf_bytes);
  float *out = (float *)aligned_alloc(64, buf_bytes);
  for (size_t i = 0; i < total_floats; ++i) in[i] = 0.5f;
  for (size_t i = 0; i < total_floats; ++i) out[i] = 0.5f;

  double dt_samples[MAX_TRIALS];
  double t0 = 0.0, t1 = 0.0;
  size_t row_bytes = (size_t)row_len * sizeof(float);
  int row_tile = 4;  // Matches min_max_accumulator<N=4> multi-row tiling
  int num_row_tiles = (num_rows + row_tile - 1) / row_tile;

  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    pin_thread_to_core(tid);
    float max_out[16] __attribute__((aligned(64)));
    float sum_out[16] __attribute__((aligned(64)));

    #pragma omp for schedule(static)
    for (int rt = 0; rt < num_row_tiles; ++rt) {
      int m = rt * row_tile;
      int cur_n = (num_rows - m < row_tile) ? (num_rows - m) : row_tile;
      for (int j = 0; j < cur_n; ++j) {
        max_out[j] = -1e30f;
        sum_out[j] = 0.0f;
      }
      // YNNPACK k1 reductions take k in ELEMENTS (`row_len`) and stride in BYTES (`row_bytes`)
      fn_ynn_max_fp32_avx512((size_t)cur_n, (size_t)row_len, row_bytes,
                             in + (size_t)m * row_len, max_out, NULL);
      fn_ynn_sum_fp32_avx512((size_t)cur_n, (size_t)row_len, row_bytes,
                             out + (size_t)m * row_len, sum_out, NULL);
    }

    for (int trial = 0; trial < num_trials; ++trial) {
      #pragma omp barrier
      #pragma omp master
      { t0 = now_sec(); }
      #pragma omp barrier

      for (int r = 0; r < reps; ++r) {
        #pragma omp for schedule(static) nowait
        for (int rt = 0; rt < num_row_tiles; ++rt) {
          int m = rt * row_tile;
          int cur_n = (num_rows - m < row_tile) ? (num_rows - m) : row_tile;
          for (int j = 0; j < cur_n; ++j) {
            max_out[j] = -1e30f;
            sum_out[j] = 0.0f;
          }
          fn_ynn_max_fp32_avx512((size_t)cur_n, (size_t)row_len, row_bytes,
                                 in + (size_t)m * row_len, max_out, NULL);
          fn_ynn_sum_fp32_avx512((size_t)cur_n, (size_t)row_len, row_bytes,
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
  size_t buf_bytes = (elem_count * sizeof(uint32_t) + 65536 + 63) & ~(size_t)63;
  uint32_t *in = (uint32_t *)aligned_alloc(64, buf_bytes);
  uint32_t *out = (uint32_t *)aligned_alloc(64, buf_bytes);
  memset(in, 1, buf_bytes);
  memset(out, 0, buf_bytes);

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
  size_t buf_bytes = (elem_count * sizeof(uint32_t) + 65536 + 63) & ~(size_t)63;
  uint32_t *in = (uint32_t *)aligned_alloc(64, buf_bytes);
  uint32_t *out = (uint32_t *)aligned_alloc(64, buf_bytes);
  memset(in, 1, buf_bytes);
  memset(out, 0, buf_bytes);

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

// -----------------------------------------------------------------------------
// Fast Sanity Check (verifies finite, expected non-zero outputs before timing)
// -----------------------------------------------------------------------------

static void run_smoke_checks(void) {
  enable_amx_current_thread();
  uint8_t state[256] __attribute__((aligned(64)));
  memset(state, 0, sizeof(state));

  // 1. Check YNNPACK AMX-INT8 4-bit dot (M=32, N=32, K=64; A=1, B_nibbles=1 -> dot=64)
  if (fn_ynn_dot_qc4_32x32_amx) {
    uint8_t A[32 * 64] __attribute__((aligned(64)));
    uint8_t B[32 * 64 / 2] __attribute__((aligned(64)));
    int32_t C[32 * 32] __attribute__((aligned(64)));
    memset(A, 1, sizeof(A));
    memset(B, 0x11, sizeof(B));
    memset(C, 0, sizeof(C));
    fn_ynn_dot_qc4_32x32_amx(32, 32, 1, 1, 64, 64, 0, 0, A, 0, 0, 16, B, 0,
                             NULL, 32 * sizeof(int32_t), C, state);
    if (C[0] != 64 || C[31 * 32 + 31] != 64) {
      fprintf(stderr,
              "[smoke] ERROR: ynn dot_uint8_int4_32x32_amx expected 64, got "
              "C[0]=%d C[last]=%d\n",
              C[0], C[31 * 32 + 31]);
      exit(1);
    }
    // Also verify M=1 path
    memset(C, 0, sizeof(C));
    fn_ynn_dot_qc4_32x32_amx(1, 32, 1, 1, 64, 64, 0, 0, A, 0, 0, 16, B, 0, NULL,
                             32 * sizeof(int32_t), C, state);
    if (C[0] != 64 || C[31] != 64) {
      fprintf(stderr,
              "[smoke] ERROR: ynn dot_uint8_int4_32x32_amx M=1 expected 64, "
              "got C[0]=%d C[31]=%d\n",
              C[0], C[31]);
      exit(1);
    }
    _tile_release();
  }

  // 2. Check YNNPACK FP32 reductions (n=4, k=1024 elements of 0.5f -> max=0.5, sum=512.0)
  {
    float in[4 * 1024] __attribute__((aligned(64)));
    for (int i = 0; i < 4 * 1024; ++i) in[i] = 0.5f;
    float max_out[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    float sum_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    fn_ynn_max_fp32_avx512(4, 1024, 1024 * sizeof(float), in, max_out, NULL);
    fn_ynn_sum_fp32_avx512(4, 1024, 1024 * sizeof(float), in, sum_out, NULL);
    if (fabsf(max_out[0] - 0.5f) > 1e-5f || fabsf(sum_out[0] - 512.0f) > 1e-2f ||
        fabsf(sum_out[3] - 512.0f) > 1e-2f) {
      fprintf(stderr,
              "[smoke] ERROR: ynn max/sum expected (0.5, 512.0), got "
              "(%.4f, %.4f, %.4f)\n",
              max_out[0], sum_out[0], sum_out[3]);
      exit(1);
    }
  }

  // 3. Check XNNPACK 4-bit AMX 16x64c4 produces finite non-zero output
  {
    int K = 64;
    uint8_t A[16 * 64] __attribute__((aligned(64)));
    uint8_t W[256 + 64 * 64 / 2 + 512] __attribute__((aligned(64)));
    float C[16 * 64] __attribute__((aligned(64)));
    memset(A, 1, sizeof(A));
    memset(W, 0x11, sizeof(W));
    memset(W, 0, 256);
    float *sc = (float *)(W + 256 + 64 * 64 / 2);
    float *bi = sc + 64;
    for (int j = 0; j < 64; ++j) {
      sc[j] = 1.0f;
      bi[j] = 0.0f;
    }
    aeb_xnn_f32_qc4w_minmax_params_t p = {{-1e6f, 1e6f, 0}};
    xnn_qd8_quant_params_t qp[16];
    for (int i = 0; i < 16; ++i) {
      qp[i].zero_point = 0;
      qp[i].inv_scale = 1.0f;
    }
    fn_xnn_qc4w_16x64_amx(16, 64, K, A, K, W, C, 64 * sizeof(float),
                          64 * sizeof(float), &p, qp);
    if (!isfinite(C[0]) || fabsf(C[0]) < 1.0f) {
      fprintf(stderr,
              "[smoke] ERROR: xnn qc4w 16x64c4 unexpected C[0]=%.4f\n", C[0]);
      exit(1);
    }
  }

  fprintf(stderr, "[smoke] All microkernel sanity checks PASSED.\n");
}

// -----------------------------------------------------------------------------
// Benchmark Suite Driver
// -----------------------------------------------------------------------------

static void execute_benchmark_suite(const int *mc_threads, int num_mc,
                                    int num_trials, const char *out_path,
                                    const char *xnnpack_commit) {
  // Pin master thread to core 0 (NUMA Node 0) so all allocations first-touch locally
  pin_thread_to_core(0);
  run_smoke_checks();

  int max_mc = mc_threads[0];
  for (int i = 1; i < num_mc; ++i) {
    if (mc_threads[i] > max_mc) max_mc = mc_threads[i];
  }

  FILE *out = stdout;
  if (out_path && out_path[0]) {
    FILE *f = fopen(out_path, "w");
    if (f) out = f;
  }

  int use_ynn_amx_4bit = (fn_ynn_dot_qc4_32x32_amx != NULL);

  fprintf(out, "{\n");
  fprintf(out, "  \"model\": \"Gemma 4 31B\",\n");
  fprintf(out, "  \"microarchitecture\": \"Intel Sapphire Rapids\",\n");
  if (xnnpack_commit && xnnpack_commit[0]) {
    fprintf(out, "  \"xnnpack_commit\": \"%s\",\n", xnnpack_commit);
  }
  fprintf(out, "  \"threads_single_core\": 1,\n");
  fprintf(out, "  \"threads_numa_node0\": %d,\n", max_mc);
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
    int reps;
  } p1_shapes[] = {
      {1, 21504, 5376, 1, "DDR5_DRAM_Streaming", 16},
      {1, 5376, 21504, 1, "DDR5_DRAM_Streaming", 16},
      {1, 8192, 5376, 1, "DDR5_DRAM_Streaming", 32},
      {1, 8192, 5376, 0, "L3_LLC_Warm", 64},
      {1, 4096, 5376, 0, "L3_LLC_Warm", 64},
  };
  fprintf(stderr, "[bench] Pair 1: 4-Bit Linear GEMV (M=1)...\n");
  for (size_t i = 0; i < sizeof(p1_shapes) / sizeof(p1_shapes[0]); ++i) {
    int M = p1_shapes[i].M, N = p1_shapes[i].N, K = p1_shapes[i].K;
    double ops = 2.0 * M * N * K;
    double bytes = (double)M * K + (double)N * K * 0.5 + (double)M * N * 4.0;
    bench_stats_t x1 =
        bench_xnn_qd8_gemm(fn_xnn_qc4w_1x64_amx, 1, 64, M, N, K, 4,
                           p1_shapes[i].cold_dram, 1, p1_shapes[i].reps,
                           num_trials);
    bench_stats_t xmc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      xmc[t] = bench_xnn_qd8_gemm(fn_xnn_qc4w_1x64_amx, 1, 64, M, N, K, 4,
                                  p1_shapes[i].cold_dram, mc_threads[t],
                                  p1_shapes[i].reps, num_trials);
    }
    print_kernel_result(
        out, "pair1_int4_gemv", "4-Bit Linear GEMV (Decode)", "XNNPACK",
        "xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx", "AMX-INT8", M,
        N, K, p1_shapes[i].regime, x1, xmc, mc_threads, num_mc, ops, bytes, 1);

    ynn_dot_fn y_p1_fn =
        use_ynn_amx_4bit ? fn_ynn_dot_qc4_32x32_amx : fn_ynn_dot_qc4_1x32_vnni;
    const char *y_p1_name =
        use_ynn_amx_4bit
            ? "ynn::dot_uint8_int4_int32_32x32x64_16x16x8_amxint8"
            : "ynn::dot_uint8_int4_int32_1x32x8_1x16x8_avx512vnni";
    const char *y_p1_isa = use_ynn_amx_4bit ? "AMX-INT8" : "AVX512-VNNI";
    int y_p1_mr = use_ynn_amx_4bit ? 32 : 1;
    bench_stats_t y1 =
        bench_ynn_dot_gemm(y_p1_fn, y_p1_mr, 32, M, N, K, 4, 16, 1,
                           p1_shapes[i].cold_dram, 1, p1_shapes[i].reps,
                           num_trials);
    bench_stats_t ymc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      ymc[t] = bench_ynn_dot_gemm(y_p1_fn, y_p1_mr, 32, M, N, K, 4, 16, 1,
                                  p1_shapes[i].cold_dram, mc_threads[t],
                                  p1_shapes[i].reps, num_trials);
    }
    print_kernel_result(
        out, "pair1_int4_gemv", "4-Bit Linear GEMV (Decode)", "YNNPACK",
        y_p1_name, y_p1_isa, M, N, K, p1_shapes[i].regime, y1, ymc, mc_threads,
        num_mc, ops, bytes, 1);
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
  fprintf(stderr, "[bench] Pair 2: 4-Bit Linear GEMM (M=128, 1024)...\n");
  for (size_t i = 0; i < sizeof(p2_shapes) / sizeof(p2_shapes[0]); ++i) {
    int M = p2_shapes[i].M, N = p2_shapes[i].N, K = p2_shapes[i].K;
    double ops = 2.0 * M * N * K;
    double bytes = (double)M * K + (double)N * K * 0.5 + (double)M * N * 4.0;
    bench_stats_t x1 = bench_xnn_qd8_gemm(
        fn_xnn_qc4w_16x64_amx, 16, 64, M, N, K, 4, 0, 1, p2_shapes[i].reps_1c,
        num_trials);
    bench_stats_t xmc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      xmc[t] = bench_xnn_qd8_gemm(fn_xnn_qc4w_16x64_amx, 16, 64, M, N, K, 4, 0,
                                  mc_threads[t], p2_shapes[i].reps_mc,
                                  num_trials);
    }
    print_kernel_result(
        out, "pair2_int4_gemm", "4-Bit Linear GEMM (Prefill)", "XNNPACK",
        "xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx", "AMX-INT8",
        M, N, K, p2_shapes[i].regime, x1, xmc, mc_threads, num_mc, ops, bytes,
        1);

    ynn_dot_fn y_p2_fn =
        use_ynn_amx_4bit ? fn_ynn_dot_qc4_32x32_amx : fn_ynn_dot_qc4_12x16_vnni;
    const char *y_p2_name =
        use_ynn_amx_4bit
            ? "ynn::dot_uint8_int4_int32_32x32x64_16x16x8_amxint8"
            : "ynn::dot_uint8_int4_int32_12x16x8_1x16x8_avx512vnni";
    const char *y_p2_isa = use_ynn_amx_4bit ? "AMX-INT8" : "AVX512-VNNI";
    int y_p2_mr = use_ynn_amx_4bit ? 32 : 12;
    int y_p2_nr = use_ynn_amx_4bit ? 32 : 16;
    size_t y_p2_bsk1 = use_ynn_amx_4bit ? 16 : 8;
    bench_stats_t y1 = bench_ynn_dot_gemm(
        y_p2_fn, y_p2_mr, y_p2_nr, M, N, K, 4, y_p2_bsk1, 1, 0, 1,
        p2_shapes[i].reps_1c, num_trials);
    bench_stats_t ymc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      ymc[t] = bench_ynn_dot_gemm(y_p2_fn, y_p2_mr, y_p2_nr, M, N, K, 4,
                                  y_p2_bsk1, 1, 0, mc_threads[t],
                                  p2_shapes[i].reps_mc, num_trials);
    }
    print_kernel_result(
        out, "pair2_int4_gemm", "4-Bit Linear GEMM (Prefill)", "YNNPACK",
        y_p2_name, y_p2_isa, M, N, K, p2_shapes[i].regime, y1, ymc, mc_threads,
        num_mc, ops, bytes, 1);
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
  fprintf(stderr, "[bench] Pair 3: FP32 Attention BatchMatMul...\n");
  for (size_t i = 0; i < sizeof(p3_shapes) / sizeof(p3_shapes[0]); ++i) {
    int M = p3_shapes[i].M, N = p3_shapes[i].N, K = p3_shapes[i].K;
    double ops = 2.0 * M * N * K;
    double bytes = ((double)M * K + (double)N * K + (double)M * N) * 4.0;
    bench_stats_t x1 = bench_xnn_f32_gemm(M, N, K, 1, p3_shapes[i].reps_1c,
                                          num_trials);
    bench_stats_t xmc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      xmc[t] = bench_xnn_f32_gemm(M, N, K, mc_threads[t], p3_shapes[i].reps_mc,
                                  num_trials);
    }
    print_kernel_result(
        out, "pair3_fp32_attn_bmm", "FP32 Attention BatchMatMul", "XNNPACK",
        "xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast", "AVX512-FP32", M,
        N, K, p3_shapes[i].regime, x1, xmc, mc_threads, num_mc, ops, bytes, 1);

    bench_stats_t y1 = bench_ynn_dot_gemm(
        fn_ynn_dot_fp32_5x64_avx512, 5, 64, M, N, K, 32, 64 * sizeof(float),
        sizeof(float), 0, 1, p3_shapes[i].reps_1c, num_trials);
    bench_stats_t ymc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      ymc[t] = bench_ynn_dot_gemm(
          fn_ynn_dot_fp32_5x64_avx512, 5, 64, M, N, K, 32, 64 * sizeof(float),
          sizeof(float), 0, mc_threads[t], p3_shapes[i].reps_mc, num_trials);
    }
    print_kernel_result(
        out, "pair3_fp32_attn_bmm", "FP32 Attention BatchMatMul", "YNNPACK",
        "ynn::dot_fp32_5x64x1_1x16x1_avx512", "AVX512-FP32", M, N, K,
        p3_shapes[i].regime, y1, ymc, mc_threads, num_mc, ops, bytes, 1);
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
  fprintf(stderr, "[bench] Pair 4: FP32 Attention Softmax...\n");
  for (size_t i = 0; i < sizeof(p4_shapes) / sizeof(p4_shapes[0]); ++i) {
    int R = p4_shapes[i].rows, L = p4_shapes[i].row_len;
    double x_ops = 12.0 * (double)R * (double)L;
    double x_bytes = 3.0 * (double)R * (double)L * sizeof(float);
    bench_stats_t x1 = bench_xnn_softmax(R, L, 1, p4_shapes[i].reps, num_trials);
    bench_stats_t xmc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      xmc[t] =
          bench_xnn_softmax(R, L, mc_threads[t], p4_shapes[i].reps, num_trials);
    }
    print_kernel_result(
        out, "pair4_fp32_softmax", "FP32 Attention Softmax", "XNNPACK",
        "xnn_f32_rmax + xnn_f32_raddstoreexpminusmax__avx512f_rr2_p5",
        "AVX512-FP32", R, L, 1, p4_shapes[i].regime, x1, xmc, mc_threads,
        num_mc, x_ops, x_bytes, 1);

    double y_ops = 2.0 * (double)R * (double)L;
    double y_bytes = 2.0 * (double)R * (double)L * sizeof(float);
    bench_stats_t y1 =
        bench_ynn_softmax_red(R, L, 1, p4_shapes[i].reps, num_trials);
    bench_stats_t ymc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      ymc[t] = bench_ynn_softmax_red(R, L, mc_threads[t], p4_shapes[i].reps,
                                     num_trials);
    }
    print_kernel_result(
        out, "pair4_fp32_softmax", "FP32 Attention Softmax (Reductions Only)",
        "YNNPACK", "ynn::max_k1_fp32_avx512 + ynn::sum_k1_fp32_avx512",
        "AVX512-FP32", R, L, 1, p4_shapes[i].regime, y1, ymc, mc_threads,
        num_mc, y_ops, y_bytes, 1);
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
  fprintf(stderr, "[bench] Pair 5: 8-Bit Linear GEMM...\n");
  for (size_t i = 0; i < sizeof(p5_shapes) / sizeof(p5_shapes[0]); ++i) {
    int M = p5_shapes[i].M, N = p5_shapes[i].N, K = p5_shapes[i].K;
    double ops = 2.0 * M * N * K;
    double bytes = (double)M * K + (double)N * K + (double)M * N * 4.0;
    bench_stats_t x1 = bench_xnn_qd8_gemm(
        fn_xnn_qc8w_16x64_amx, 16, 64, M, N, K, 8, 0, 1, p5_shapes[i].reps_1c,
        num_trials);
    bench_stats_t xmc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      xmc[t] = bench_xnn_qd8_gemm(fn_xnn_qc8w_16x64_amx, 16, 64, M, N, K, 8, 0,
                                  mc_threads[t], p5_shapes[i].reps_mc,
                                  num_trials);
    }
    print_kernel_result(
        out, "pair5_int8_gemm", "8-Bit Linear GEMM (Vision/Audio)", "XNNPACK",
        "xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx", "AMX-INT8",
        M, N, K, p5_shapes[i].regime, x1, xmc, mc_threads, num_mc, ops, bytes,
        1);

    bench_stats_t y1 = bench_ynn_dot_gemm(
        fn_ynn_dot_qc8_32x32_amx, 32, 32, M, N, K, 8, 32, 1, 0, 1,
        p5_shapes[i].reps_1c, num_trials);
    bench_stats_t ymc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      ymc[t] = bench_ynn_dot_gemm(fn_ynn_dot_qc8_32x32_amx, 32, 32, M, N, K, 8,
                                  32, 1, 0, mc_threads[t], p5_shapes[i].reps_mc,
                                  num_trials);
    }
    print_kernel_result(
        out, "pair5_int8_gemm", "8-Bit Linear GEMM (Vision/Audio)", "YNNPACK",
        "ynn::dot_int8_int8_int32_32x32x64_16x16x4_amxint8", "AMX-INT8", M, N,
        K, p5_shapes[i].regime, y1, ymc, mc_threads, num_mc, ops, bytes, 1);
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
  fprintf(stderr, "[bench] Pair 6: 2-Bit Linear GEMM...\n");
  for (size_t i = 0; i < sizeof(p6_shapes) / sizeof(p6_shapes[0]); ++i) {
    int M = p6_shapes[i].M, N = p6_shapes[i].N, K = p6_shapes[i].K;
    double ops = 2.0 * M * N * K;
    double bytes = (double)M * K + (double)N * K * 0.25 + (double)M * N;
    bench_stats_t x1 = bench_xnn_qc2w_gemm(M, N, K, 1, p6_shapes[i].reps_1c,
                                           num_trials);
    bench_stats_t xmc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      xmc[t] = bench_xnn_qc2w_gemm(M, N, K, mc_threads[t], p6_shapes[i].reps_mc,
                                   num_trials);
    }
    print_kernel_result(
        out, "pair6_int2_gemm", "2-Bit Linear GEMM (Audio)", "XNNPACK",
        "xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni", "AVX-VNNI-256",
        M, N, K, p6_shapes[i].regime, x1, xmc, mc_threads, num_mc, ops, bytes,
        1);

    bench_stats_t y1 = bench_ynn_dot_gemm(
        fn_ynn_dot_qc2_4x64_vnni, 4, 64, M, N, K, 2, 16, 1, 0, 1,
        p6_shapes[i].reps_1c, num_trials);
    bench_stats_t ymc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      ymc[t] = bench_ynn_dot_gemm(fn_ynn_dot_qc2_4x64_vnni, 4, 64, M, N, K, 2,
                                  16, 1, 0, mc_threads[t], p6_shapes[i].reps_mc,
                                  num_trials);
    }
    print_kernel_result(
        out, "pair6_int2_gemm", "2-Bit Linear GEMM (Audio)", "YNNPACK",
        "ynn::dot_uint8_int2_int32_4x64x16_1x16x16_avx512vnni", "AVX512-VNNI",
        M, N, K, p6_shapes[i].regime, y1, ymc, mc_threads, num_mc, ops, bytes,
        1);
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
  fprintf(stderr, "[bench] Pair 7: Matrix Packing / Transpose...\n");
  for (size_t i = 0; i < sizeof(p7_shapes) / sizeof(p7_shapes[0]); ++i) {
    int N = p7_shapes[i].N, K = p7_shapes[i].K;
    int is_last = (i + 1 == sizeof(p7_shapes) / sizeof(p7_shapes[0]));
    double bytes = 2.0 * (double)N * (double)K * sizeof(uint32_t);
    bench_stats_t x1 = bench_xnn_packw(N, K, 1, p7_shapes[i].reps, num_trials);
    bench_stats_t xmc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      xmc[t] =
          bench_xnn_packw(N, K, mc_threads[t], p7_shapes[i].reps, num_trials);
    }
    print_kernel_result(
        out, "pair7_pack_transpose", "Matrix Packing / 32-Bit Transpose",
        "XNNPACK", "xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4_prfm",
        "AVX512F", 1, N, K, p7_shapes[i].regime, x1, xmc, mc_threads, num_mc,
        0.0, bytes, 1);

    bench_stats_t y1 =
        bench_ynn_transpose(N, K, 1, p7_shapes[i].reps, num_trials);
    bench_stats_t ymc[MAX_THREAD_CFGS];
    for (int t = 0; t < num_mc; ++t) {
      ymc[t] = bench_ynn_transpose(N, K, mc_threads[t], p7_shapes[i].reps,
                                   num_trials);
    }
    print_kernel_result(
        out, "pair7_pack_transpose", "Matrix Packing / 32-Bit Transpose",
        "YNNPACK", "ynn::transpose_x32_avx512", "AVX512F", 1, N, K,
        p7_shapes[i].regime, y1, ymc, mc_threads, num_mc, 0.0, bytes, !is_last);
  }

  fprintf(out, "  ]");

  // ---------------------------------------------------------------------------
  // Controlled A/B Baselines: Prior YNNPACK AVX-512 VNNI 4-bit Kernels
  // ---------------------------------------------------------------------------
  if (use_ynn_amx_4bit && fn_ynn_dot_qc4_1x32_vnni &&
      fn_ynn_dot_qc4_12x16_vnni) {
    fprintf(stderr,
            "[bench] Baseline Comparisons: YNNPACK AVX-512 VNNI 4-bit...\n");
    fprintf(out, ",\n  \"baseline_comparisons\": [\n");
    for (size_t i = 0; i < sizeof(p1_shapes) / sizeof(p1_shapes[0]); ++i) {
      int M = p1_shapes[i].M, N = p1_shapes[i].N, K = p1_shapes[i].K;
      double ops = 2.0 * M * N * K;
      double bytes = (double)M * K + (double)N * K * 0.5 + (double)M * N * 4.0;
      bench_stats_t y1 =
          bench_ynn_dot_gemm(fn_ynn_dot_qc4_1x32_vnni, 1, 32, M, N, K, 4, 16, 1,
                             p1_shapes[i].cold_dram, 1, p1_shapes[i].reps,
                             num_trials);
      bench_stats_t ymc[MAX_THREAD_CFGS];
      for (int t = 0; t < num_mc; ++t) {
        ymc[t] = bench_ynn_dot_gemm(fn_ynn_dot_qc4_1x32_vnni, 1, 32, M, N, K, 4,
                                    16, 1, p1_shapes[i].cold_dram,
                                    mc_threads[t], p1_shapes[i].reps,
                                    num_trials);
      }
      print_kernel_result(
          out, "pair1_int4_gemv", "4-Bit Linear GEMV (Decode, Prior VNNI)",
          "YNNPACK_VNNI_BASELINE",
          "ynn::dot_uint8_int4_int32_1x32x8_1x16x8_avx512vnni", "AVX512-VNNI",
          M, N, K, p1_shapes[i].regime, y1, ymc, mc_threads, num_mc, ops, bytes,
          1);
    }
    for (size_t i = 0; i < sizeof(p2_shapes) / sizeof(p2_shapes[0]); ++i) {
      int M = p2_shapes[i].M, N = p2_shapes[i].N, K = p2_shapes[i].K;
      int is_last = (i + 1 == sizeof(p2_shapes) / sizeof(p2_shapes[0]));
      double ops = 2.0 * M * N * K;
      double bytes = (double)M * K + (double)N * K * 0.5 + (double)M * N * 4.0;
      bench_stats_t y1 = bench_ynn_dot_gemm(
          fn_ynn_dot_qc4_12x16_vnni, 12, 16, M, N, K, 4, 8, 1, 0, 1,
          p2_shapes[i].reps_1c, num_trials);
      bench_stats_t ymc[MAX_THREAD_CFGS];
      for (int t = 0; t < num_mc; ++t) {
        ymc[t] = bench_ynn_dot_gemm(fn_ynn_dot_qc4_12x16_vnni, 12, 16, M, N, K,
                                    4, 8, 1, 0, mc_threads[t],
                                    p2_shapes[i].reps_mc, num_trials);
      }
      print_kernel_result(
          out, "pair2_int4_gemm", "4-Bit Linear GEMM (Prefill, Prior VNNI)",
          "YNNPACK_VNNI_BASELINE",
          "ynn::dot_uint8_int4_int32_12x16x8_1x16x8_avx512vnni", "AVX512-VNNI",
          M, N, K, p2_shapes[i].regime, y1, ymc, mc_threads, num_mc, ops, bytes,
          !is_last);
    }
    fprintf(out, "  ]\n");
  } else {
    fprintf(out, "\n");
  }

  fprintf(out, "}\n");
  if (out != stdout) fclose(out);
  fprintf(stderr, "[bench] Benchmark suite completed successfully.\n");
}

#ifdef AEB_DIRECT_LINK
int main(int argc, char **argv) {
  int mc_threads[MAX_THREAD_CFGS] = {28, 56};
  int num_mc = 2;
  int num_trials = 7;
  const char *out_path = NULL;
  const char *xnnpack_commit = NULL;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      int n = parse_thread_list(argv[++i], mc_threads, MAX_THREAD_CFGS);
      if (n > 0) num_mc = n;
    } else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc) {
      num_trials = atoi(argv[++i]);
      if (num_trials < 1) num_trials = 1;
      if (num_trials > MAX_TRIALS) num_trials = MAX_TRIALS;
    } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      out_path = argv[++i];
    } else if (strcmp(argv[i], "--xnnpack-commit") == 0 && i + 1 < argc) {
      xnnpack_commit = argv[++i];
    }
  }

  fn_xnn_qc4w_1x64_amx = xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx;
  fn_xnn_qc4w_16x64_amx =
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx;
  fn_xnn_qc8w_16x64_amx =
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx;
  fn_xnn_f32_7x32_avx512 = xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast;
  fn_xnn_qc2w_5x8_avxvnni =
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni;
  fn_xnn_rmax_avx512 = xnn_f32_rmax_ukernel__avx512f_u64_acc4;
  fn_xnn_raddexp_avx512 =
      xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc2;
  fn_xnn_packw_x32_avx512 = xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4_prfm;

  fn_ynn_dot_qc4_32x32_amx = aeb_ynn_dot_qc4_32x32_amx;
  fn_ynn_dot_qc4_1x32_vnni = aeb_ynn_dot_qc4_1x32_vnni;
  fn_ynn_dot_qc4_12x16_vnni = aeb_ynn_dot_qc4_12x16_vnni;
  fn_ynn_dot_qc8_32x32_amx = aeb_ynn_dot_qc8_32x32_amx;
  fn_ynn_dot_fp32_5x64_avx512 = aeb_ynn_dot_fp32_5x64_avx512;
  fn_ynn_dot_qc2_4x64_vnni = aeb_ynn_dot_qc2_4x64_vnni;
  fn_ynn_max_fp32_avx512 = aeb_ynn_max_fp32_avx512;
  fn_ynn_sum_fp32_avx512 = aeb_ynn_sum_fp32_avx512;
  fn_ynn_transpose_x32_avx512 = aeb_ynn_transpose_x32_avx512;

  execute_benchmark_suite(mc_threads, num_mc, num_trials, out_path,
                          xnnpack_commit);
  return 0;
}
#else
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

  fn_ynn_dot_qc4_32x32_amx = NULL;
  fn_ynn_dot_qc4_1x32_vnni = (ynn_dot_fn)(base + 0x011f85a0);
  fn_ynn_dot_qc4_12x16_vnni = (ynn_dot_fn)(base + 0x011fb750);
  fn_ynn_dot_qc8_32x32_amx = (ynn_dot_fn)(base + 0x0120ab60);
  fn_ynn_dot_fp32_5x64_avx512 = (ynn_dot_fn)(base + 0x011d6ff0);
  fn_ynn_dot_qc2_4x64_vnni = (ynn_dot_fn)(base + 0x011ed490);
  fn_ynn_max_fp32_avx512 = (ynn_reduce_fp32_fn)(base + 0x0125dd00);
  fn_ynn_sum_fp32_avx512 = (ynn_reduce_fp32_fn)(base + 0x012637c0);
  fn_ynn_transpose_x32_avx512 = (ynn_transpose_x32_fn)(base + 0x012e68e0);

  int mc_threads[MAX_THREAD_CFGS] = {28};
  int num_mc = 1;
  const char *thr_env = getenv("AEB_BENCH_THREADS");
  if (thr_env && thr_env[0]) {
    int n = parse_thread_list(thr_env, mc_threads, MAX_THREAD_CFGS);
    if (n > 0) num_mc = n;
  }
  int num_trials = 7;
  const char *tr_env = getenv("AEB_BENCH_TRIALS");
  if (tr_env && tr_env[0]) {
    num_trials = atoi(tr_env);
    if (num_trials < 1) num_trials = 1;
    if (num_trials > MAX_TRIALS) num_trials = MAX_TRIALS;
  }
  const char *out_path = getenv("AEB_BENCH_OUTPUT");

  execute_benchmark_suite(mc_threads, num_mc, num_trials, out_path, NULL);
  _exit(0);
}
#endif
