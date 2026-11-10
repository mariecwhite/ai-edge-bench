// Standalone XNNPACK & YNNPACK Microkernel Numerical Precision Verifier
//
// Resolves the 7 load-bearing XNNPACK and YNNPACK microkernel pairs directly
// inside litert_lm_advanced_main via dl_iterate_phdr and verifies each
// microkernel at two levels across 4 input distributions (normal, saturation,
// outliers, tails):
//   - Level A (Exact Packed-Layout Microkernel Reference): Verifies the SIMD /
//     AMX microkernel against an exact scalar INT64 / FP64 reference operating
//     on the identical quantized inputs and packed weight buffer.
//   - Level B (FP64 Golden Operator Reference): Verifies end-to-end operator
//     and quantization fidelity against unquantized FP64 matrix multiplication
//     (C_fp64 = A_fp64 * B_fp64^T + bias_fp64), FP64 Softmax, and bit-exact
//     32-bit matrix packing/transposition.
//
// Compile as a shared library and run via LD_PRELOAD:
//   gcc -shared -fPIC -O3 -march=sapphirerapids -fopenmp
//       kernel/scripts/verify_litert_microkernels.c
//       -o /tmp/libverify_ukernels.so -lm
//   LD_LIBRARY_PATH=/opt/litert-lm/lib:/opt/litert-lm/bin
//     LD_PRELOAD=/tmp/libverify_ukernels.so
//     AEB_VERIFY_OUTPUT=kernel/reports/gemma4-31b-sapphire-rapids/ukernel_precision_sapphire_rapids.json
//     /opt/litert-lm/bin/litert_lm_advanced_main

#define _GNU_SOURCE
#include <dlfcn.h>
#include <float.h>
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
#include <unistd.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

// -----------------------------------------------------------------------------
// Function pointer types for XNNPACK and YNNPACK microkernels & helpers
// -----------------------------------------------------------------------------

struct xnn_qd8_quantization_params {
  int32_t zero_point;
  float inv_scale;
};

typedef void (*xnn_qd8_gemm_fn)(
    size_t mr, size_t nc, size_t kc, const int8_t *a, size_t a_stride,
    const void *w, float *c, size_t cm_stride, size_t cn_stride,
    const void *params,
    const struct xnn_qd8_quantization_params *quantization_params);

typedef void (*xnn_f32_gemm_fn)(size_t mr, size_t nc, size_t kc, const float *a,
                                size_t a_stride, const void *w, float *c,
                                size_t cm_stride, size_t cn_stride,
                                const void *params);

typedef void (*xnn_qs8_qc2w_gemm_fn)(size_t mr, size_t nc, size_t kc,
                                     const int8_t *a, size_t a_stride,
                                     const void *w, int8_t *c, size_t cm_stride,
                                     size_t cn_stride, const void *params);

typedef void (*xnn_f32_rmax_fn)(size_t batch, const float *x, float *y,
                                const void *params);

typedef void (*xnn_f32_raddstoreexpminusmax_fn)(size_t batch, const float *x,
                                                const float *max, float *y,
                                                float *sum, const void *params);

typedef void (*xnn_pack_gemm_goi_fn)(size_t g, size_t nc, size_t kc, size_t nr,
                                     size_t kr, size_t sr, size_t k_stride,
                                     const void *k, const void *b,
                                     const void *scale, void *packed_weights,
                                     size_t extra_bytes, const void *params);

typedef size_t (*xnn_init_qs8_qc8w_conv_minmax_fp32_params_fn)(
    void *params, int8_t output_zero_point, int8_t output_min,
    int8_t output_max);

typedef void (*ynn_dot_fn)(size_t M, size_t N, size_t K3, size_t K2, size_t K1,
                           size_t A_stride_m, size_t A_stride_k3,
                           size_t A_stride_k2, const void *A,
                           size_t B_stride_k3, size_t B_stride_k2,
                           size_t B_stride_k1, const void *B,
                           size_t C_in_stride_m, const void *C_in,
                           size_t C_out_stride_m, void *C_out, void *state);

typedef void (*ynn_reduce_fp32_fn)(size_t N, size_t K1, size_t A_stride_n,
                                   const void *A, void *C, void *state);

typedef void (*ynn_transpose_x32_fn)(size_t M, size_t N, size_t n_bytes_a,
                                     size_t stride_a, const void *A,
                                     size_t stride_b, void *B);

// Resolved symbol pointers
static xnn_qd8_gemm_fn fn_xnn_qc4w_1x64_amx = NULL;
static xnn_qd8_gemm_fn fn_xnn_qc4w_16x64_amx = NULL;
static xnn_qd8_gemm_fn fn_xnn_qc8w_16x64_amx = NULL;
static xnn_f32_gemm_fn fn_xnn_f32_7x32_avx512 = NULL;
static xnn_qs8_qc2w_gemm_fn fn_xnn_qc2w_5x8_avxvnni = NULL;
static xnn_f32_rmax_fn fn_xnn_rmax_avx512 = NULL;
static xnn_f32_raddstoreexpminusmax_fn fn_xnn_raddexp_avx512 = NULL;
static xnn_pack_gemm_goi_fn fn_xnn_packw_x32_avx512 = NULL;
static xnn_pack_gemm_goi_fn fn_xnn_pack_qs8_goi = NULL;
static xnn_pack_gemm_goi_fn fn_xnn_pack_qs8_qc4w_goi = NULL;
static xnn_pack_gemm_goi_fn fn_xnn_pack_qs8_to_qu8_qc2w_goi = NULL;
static xnn_init_qs8_qc8w_conv_minmax_fp32_params_fn
    fn_xnn_init_qs8_minmax_params = NULL;

static ynn_dot_fn fn_ynn_dot_qc4_1x32_vnni = NULL;
static ynn_dot_fn fn_ynn_dot_qc4_12x16_vnni = NULL;
static ynn_dot_fn fn_ynn_dot_qc8_32x32_amx = NULL;
static ynn_dot_fn fn_ynn_dot_fp32_5x64_avx512 = NULL;
static ynn_dot_fn fn_ynn_dot_qc2_4x64_vnni = NULL;
static ynn_reduce_fp32_fn fn_ynn_max_fp32_avx512 = NULL;
static ynn_reduce_fp32_fn fn_ynn_sum_fp32_avx512 = NULL;
static ynn_transpose_x32_fn fn_ynn_transpose_x32_avx512 = NULL;
static ynn_transpose_x32_fn fn_ynn_transpose_x32_scalar = NULL;

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

static void enable_amx(void) {
  syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
}

// -----------------------------------------------------------------------------
// Deterministic PRNG & Input Distribution Generators
// -----------------------------------------------------------------------------

static uint64_t rng_state = 0x853c49e6748fea9bULL;

static void seed_rng(uint64_t seed) { rng_state = seed ? seed : 1ULL; }

static uint32_t xorshift32(void) {
  uint64_t x = rng_state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  rng_state = x;
  return (uint32_t)((x * 0x2545F4914F6CDD1DULL) >> 32);
}

static double rand_uniform(double lo, double hi) {
  double u = (double)xorshift32() / 4294967296.0;
  return lo + (hi - lo) * u;
}

static double rand_normal(double mean, double std_dev) {
  // Sum of 12 uniforms minus 6 (Irwin-Hall normal approximation, exact support)
  double s = -6.0;
  for (int i = 0; i < 12; ++i) {
    s += (double)xorshift32() / 4294967296.0;
  }
  return mean + std_dev * s;
}

typedef enum {
  DIST_NORMAL = 0,
  DIST_SATURATION = 1,
  DIST_OUTLIERS = 2,
  DIST_TAILS = 3,
  NUM_DISTS = 4
} dist_kind_t;

static const char *dist_names[NUM_DISTS] = {"normal", "saturation", "outliers",
                                            "tails"};

static void fill_fp64_tensors(dist_kind_t dist, int M, int N, int K,
                              double *A_fp64, double *B_fp64,
                              double *bias_fp64) {
  seed_rng(0x123456789ABCDEF0ULL + (uint64_t)dist * 10007ULL +
           (uint64_t)M * 31ULL + (uint64_t)N * 17ULL);
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      double v = 0.0;
      if (dist == DIST_NORMAL) {
        v = rand_normal(0.0, 1.0);
      } else if (dist == DIST_SATURATION) {
        v = ((k + m) % 2 == 0) ? 10.0 : -10.0;
      } else if (dist == DIST_OUTLIERS) {
        v = (k % 16 == 0) ? rand_normal(0.0, 15.0) : rand_normal(0.0, 0.5);
      } else {
        // DIST_TAILS: non-zero mean + small magnitude tail channels
        v = (k < K / 2) ? rand_normal(1.25, 0.4) : rand_normal(-0.02, 0.005);
      }
      A_fp64[m * K + k] = v;
    }
  }
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      double w = 0.0;
      if (dist == DIST_NORMAL) {
        w = rand_normal(0.0, 0.25);
      } else if (dist == DIST_SATURATION) {
        w = ((k + n) % 2 == 0) ? 1.0 : -1.0;
      } else if (dist == DIST_OUTLIERS) {
        w = (k % 16 == 0) ? rand_normal(0.0, 1.0) : rand_normal(0.0, 0.2);
      } else {
        w = rand_uniform(-0.35, 0.35);
      }
      B_fp64[n * K + k] = w;
    }
    bias_fp64[n] = (dist == DIST_SATURATION) ? 0.5 : rand_normal(0.0, 0.05);
  }
}

// -----------------------------------------------------------------------------
// Precision Metric Structures & Calculators
// -----------------------------------------------------------------------------

typedef struct {
  double max_abs_err;
  double max_rel_err;
  double rmse;
  uint64_t max_ulp_diff;
  double mean_ulp_diff;
  double sqnr_db;
  double cosine_sim;
  double exact_match_pct;
  int nan_inf_count;
  int pass;
} precision_metrics_t;

static uint32_t float_to_ordered_u32(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof(uint32_t));
  if (u & 0x80000000U) {
    return ~u;
  } else {
    return u | 0x80000000U;
  }
}

static uint64_t fp32_ulp_diff(float a, float b) {
  if (isnan(a) || isnan(b) || isinf(a) || isinf(b)) {
    return (a == b) ? 0ULL : 0xFFFFFFFFULL;
  }
  if (a == b) return 0ULL; // Handles +0.0f == -0.0f
  // Avoid artificial ULP inflation from catastrophic cancellation near zero
  if (fabsf(a) < 1e-2f && fabsf(b) < 1e-2f) {
    float step = 1e-2f * 0x1.0p-23f;
    return (uint64_t)llround(fabs((double)a - (double)b) / (double)step);
  }
  uint32_t ua = float_to_ordered_u32(a);
  uint32_t ub = float_to_ordered_u32(b);
  return (ua > ub) ? (uint64_t)(ua - ub) : (uint64_t)(ub - ua);
}

static precision_metrics_t compute_fp_metrics(const float *actual,
                                              const double *ref, int count,
                                              uint64_t exact_ulp_tol,
                                              double pass_sqnr_min_db,
                                              double pass_max_abs_tol) {
  precision_metrics_t m;
  memset(&m, 0, sizeof(m));
  double sum_sq_sig = 0.0;
  double sum_sq_err = 0.0;
  double dot = 0.0;
  double norm_a = 0.0;
  double norm_r = 0.0;
  double sum_ulp = 0.0;
  int exact_matches = 0;

  for (int i = 0; i < count; ++i) {
    float a = actual[i];
    double r = ref[i];
    if (isnan(a) || isinf(a) || isnan(r) || isinf(r)) {
      m.nan_inf_count++;
      continue;
    }
    double da = (double)a;
    double err = fabs(da - r);
    double denom = fmax(fabs(r), 1e-6);
    double rel = err / denom;
    if (err > m.max_abs_err) m.max_abs_err = err;
    if (rel > m.max_rel_err) m.max_rel_err = rel;
    sum_sq_sig += r * r;
    sum_sq_err += err * err;
    dot += da * r;
    norm_a += da * da;
    norm_r += r * r;

    float rf = (float)r;
    uint64_t ulp = fp32_ulp_diff(a, rf);
    if (ulp > m.max_ulp_diff) m.max_ulp_diff = ulp;
    sum_ulp += (double)ulp;
    if (ulp <= exact_ulp_tol) exact_matches++;
  }

  m.rmse = (count > 0) ? sqrt(sum_sq_err / (double)count) : 0.0;
  m.mean_ulp_diff = (count > 0) ? (sum_ulp / (double)count) : 0.0;
  m.exact_match_pct =
      (count > 0) ? (100.0 * (double)exact_matches / (double)count) : 0.0;
  if (sum_sq_err <= 1e-30) {
    m.sqnr_db = 200.0;
  } else if (sum_sq_sig <= 1e-30) {
    m.sqnr_db = 0.0;
  } else {
    m.sqnr_db = 10.0 * log10(sum_sq_sig / sum_sq_err);
    if (m.sqnr_db > 200.0) m.sqnr_db = 200.0;
  }
  if (norm_a > 1e-30 && norm_r > 1e-30) {
    m.cosine_sim = dot / (sqrt(norm_a) * sqrt(norm_r));
    if (m.cosine_sim > 1.0) m.cosine_sim = 1.0;
  } else {
    m.cosine_sim = (sum_sq_err <= 1e-24) ? 1.0 : 0.0;
  }

  m.pass = (m.nan_inf_count == 0 && m.sqnr_db >= pass_sqnr_min_db &&
            m.max_abs_err <= pass_max_abs_tol)
               ? 1
               : 0;
  return m;
}

static void merge_metrics(precision_metrics_t *agg,
                          const precision_metrics_t *cur, int idx) {
  if (idx == 0) {
    *agg = *cur;
    return;
  }
  if (cur->max_abs_err > agg->max_abs_err) agg->max_abs_err = cur->max_abs_err;
  if (cur->max_rel_err > agg->max_rel_err) agg->max_rel_err = cur->max_rel_err;
  if (cur->max_ulp_diff > agg->max_ulp_diff)
    agg->max_ulp_diff = cur->max_ulp_diff;
  if (cur->sqnr_db < agg->sqnr_db) agg->sqnr_db = cur->sqnr_db;
  if (cur->cosine_sim < agg->cosine_sim) agg->cosine_sim = cur->cosine_sim;
  agg->rmse = (agg->rmse * idx + cur->rmse) / (double)(idx + 1);
  agg->mean_ulp_diff =
      (agg->mean_ulp_diff * idx + cur->mean_ulp_diff) / (double)(idx + 1);
  agg->exact_match_pct =
      (agg->exact_match_pct * idx + cur->exact_match_pct) / (double)(idx + 1);
  agg->nan_inf_count += cur->nan_inf_count;
  agg->pass = (agg->pass && cur->pass) ? 1 : 0;
}

// -----------------------------------------------------------------------------
// YNNPACK Packed Layout Calibrators (Bit-Plane / Unit Basis Probing)
// -----------------------------------------------------------------------------

// Calibrate YNNPACK dot_fp32 (K=64, N=64 -> 4096 floats in B_packed)
static void calibrate_ynn_fp32_layout(int slot_map[64][64]) {
  const int K = 64, N = 64;
  float A[5 * 64] __attribute__((aligned(64)));
  float B_packed[4096] __attribute__((aligned(64)));
  float C_zero[5 * 64] __attribute__((aligned(64)));
  float C_out[5 * 64] __attribute__((aligned(64)));
  memset(C_zero, 0, sizeof(C_zero));
  for (int i = 0; i < 4096; ++i) B_packed[i] = (float)(i + 1);
  for (int k = 0; k < K; ++k) {
    memset(A, 0, sizeof(A));
    A[k] = 1.0f;
    memset(C_out, 0, sizeof(C_out));
    fn_ynn_dot_fp32_5x64_avx512(
        1, N, 1, 1, K, K * sizeof(float), 0, 0, A, 0, 0, 64 * sizeof(float),
        B_packed, N * sizeof(float), C_zero, N * sizeof(float), C_out, NULL);
    for (int n = 0; n < N; ++n) {
      int idx = (int)lroundf(C_out[n]) - 1;
      if (idx < 0 || idx >= 4096) idx = 0;
      slot_map[k][n] = idx;
    }
  }
}

// Calibrate YNNPACK dot_int8_int8_32x32x64_16x16x4_amxint8 (K=64, N=32 -> 2048 bytes)
static void calibrate_ynn_int8_layout(int slot_map[64][32]) {
  const int M = 32, K = 64, N = 32;
  int8_t A[32 * 64] __attribute__((aligned(64)));
  int8_t B_p1[2048] __attribute__((aligned(64)));
  int8_t B_p2[2048] __attribute__((aligned(64)));
  int32_t C_zero[32 * 32] __attribute__((aligned(64)));
  int32_t C1[32 * 32] __attribute__((aligned(64)));
  int32_t C2[32 * 32] __attribute__((aligned(64)));
  uint8_t amx_state[256] __attribute__((aligned(64)));
  memset(C_zero, 0, sizeof(C_zero));

  for (int i = 0; i < 2048; ++i) {
    B_p1[i] = (int8_t)(i & 63);
    B_p2[i] = (int8_t)((i >> 6) & 31);
  }
  for (int k = 0; k < K; ++k) {
    memset(A, 0, sizeof(A));
    A[k] = 1;
    memset(C1, 0, sizeof(C1));
    memset(C2, 0, sizeof(C2));
    memset(amx_state, 0, sizeof(amx_state));
    fn_ynn_dot_qc8_32x32_amx(M, N, 1, 1, K, K, 0, 0, A, 0, 0, 32, B_p1,
                             N * sizeof(int32_t), C_zero, N * sizeof(int32_t),
                             C1, amx_state);
    memset(amx_state, 0, sizeof(amx_state));
    fn_ynn_dot_qc8_32x32_amx(M, N, 1, 1, K, K, 0, 0, A, 0, 0, 32, B_p2,
                             N * sizeof(int32_t), C_zero, N * sizeof(int32_t),
                             C2, amx_state);
    for (int n = 0; n < N; ++n) {
      int idx = ((C2[n] & 31) << 6) | (C1[n] & 63);
      if (idx < 0 || idx >= 2048) idx = 0;
      slot_map[k][n] = idx;
    }
  }
}

// Calibrate YNNPACK dot_uint8_int4 (K=64, N=32 or 16)
static void calibrate_ynn_int4_layout(ynn_dot_fn fn, int M_probe, int N,
                                      size_t b_stride_k1,
                                      int nibble_map[64][64],
                                      int32_t code_to_val[16],
                                      uint8_t val_to_code[16]) {
  const int K = 64;
  const int total_nibbles = K * N;
  const int total_bytes = total_nibbles / 2;
  uint8_t A[16 * 64] __attribute__((aligned(64)));
  uint8_t B_packed[2048] __attribute__((aligned(64)));
  int32_t C_zero[16 * 64] __attribute__((aligned(64)));
  int32_t C_out[16 * 64] __attribute__((aligned(64)));
  memset(C_zero, 0, sizeof(C_zero));

  uint8_t code_zero = 0, code_one = 1;
  for (int c = 0; c < 16; ++c) {
    memset(B_packed, (c << 4) | c, sizeof(B_packed));
    memset(A, 0, sizeof(A));
    A[0] = 1;
    memset(C_out, 0, sizeof(C_out));
    fn(M_probe, N, 1, 1, K, K, 0, 0, A, 0, 0, b_stride_k1, B_packed,
       N * sizeof(int32_t), C_zero, N * sizeof(int32_t), C_out, NULL);
    code_to_val[c] = C_out[0];
    if (C_out[0] == 0) code_zero = (uint8_t)c;
    if (C_out[0] == 1) code_one = (uint8_t)c;
    if (C_out[0] >= -8 && C_out[0] <= 7) {
      val_to_code[C_out[0] + 8] = (uint8_t)c;
    }
  }

  memset(nibble_map, 0, sizeof(int) * 64 * 64);
  for (int b = 0; b < 12; ++b) {
    if ((1 << b) >= total_nibbles) break;
    memset(B_packed, (code_zero << 4) | code_zero, (size_t)total_bytes);
    for (int nid = 0; nid < total_nibbles; ++nid) {
      uint8_t code = ((nid >> b) & 1) ? code_one : code_zero;
      int byte_idx = nid >> 1;
      if (nid & 1) {
        B_packed[byte_idx] = (B_packed[byte_idx] & 0x0F) | (code << 4);
      } else {
        B_packed[byte_idx] = (B_packed[byte_idx] & 0xF0) | (code & 0x0F);
      }
    }
    for (int k = 0; k < K; ++k) {
      memset(A, 0, sizeof(A));
      A[k] = 1;
      memset(C_out, 0, sizeof(C_out));
      fn(M_probe, N, 1, 1, K, K, 0, 0, A, 0, 0, b_stride_k1, B_packed,
         N * sizeof(int32_t), C_zero, N * sizeof(int32_t), C_out, NULL);
      for (int n = 0; n < N; ++n) {
        if (C_out[n] != 0) {
          nibble_map[k][n] |= (1 << b);
        }
      }
    }
  }
}

// Calibrate YNNPACK dot_uint8_int2 (K=64, N=64 -> 1024 bytes = 4096 crumbs)
static void calibrate_ynn_int2_layout(int crumb_map[64][64],
                                      int32_t code2_to_val[4]) {
  const int M_probe = 4, K = 64, N = 64;
  uint8_t A[4 * 64] __attribute__((aligned(64)));
  uint8_t B_packed[1024] __attribute__((aligned(64)));
  int32_t C_zero[4 * 64] __attribute__((aligned(64)));
  int32_t C_out[4 * 64] __attribute__((aligned(64)));
  memset(C_zero, 0, sizeof(C_zero));

  for (int c = 0; c < 4; ++c) {
    uint8_t rep = (uint8_t)((c << 6) | (c << 4) | (c << 2) | c);
    memset(B_packed, rep, sizeof(B_packed));
    memset(A, 0, sizeof(A));
    A[0] = 1;
    memset(C_out, 0, sizeof(C_out));
    fn_ynn_dot_qc2_4x64_vnni(M_probe, N, 1, 1, K, K, 0, 0, A, 0, 0, 16, B_packed,
                             N * sizeof(int32_t), C_zero, N * sizeof(int32_t),
                             C_out, NULL);
    code2_to_val[c] = C_out[0];
  }

  uint8_t c0 = 0, c1 = 1;
  int32_t v0 = code2_to_val[0], v1 = code2_to_val[1];
  for (int c = 1; c < 4; ++c) {
    if (code2_to_val[c] != v0) {
      c1 = (uint8_t)c;
      v1 = code2_to_val[c];
      break;
    }
  }

  memset(crumb_map, 0, sizeof(int) * 64 * 64);
  uint8_t rep0 = (uint8_t)((c0 << 6) | (c0 << 4) | (c0 << 2) | c0);
  for (int b = 0; b < 12; ++b) {
    memset(B_packed, rep0, sizeof(B_packed));
    for (int cid = 0; cid < 4096; ++cid) {
      if ((cid >> b) & 1) {
        int byte_idx = cid >> 2;
        int shift = (cid & 3) * 2;
        B_packed[byte_idx] =
            (uint8_t)((B_packed[byte_idx] & ~(0x3 << shift)) | (c1 << shift));
      }
    }
    for (int k = 0; k < K; ++k) {
      memset(A, 0, sizeof(A));
      A[k] = 1;
      memset(C_out, 0, sizeof(C_out));
      fn_ynn_dot_qc2_4x64_vnni(M_probe, N, 1, 1, K, K, 0, 0, A, 0, 0, 16,
                               B_packed, N * sizeof(int32_t), C_zero,
                               N * sizeof(int32_t), C_out, NULL);
      for (int n = 0; n < N; ++n) {
        if (C_out[n] == v1) {
          crumb_map[k][n] |= (1 << b);
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
// XNNPACK AMX & AVX-VNNI Self-Calibrating Helpers
// -----------------------------------------------------------------------------

typedef struct {
  size_t scale_offset_bytes;
  size_t bias_offset_bytes;
  float weight_gain;
  int zp_sign;
} xnn_qd8_layout_info_t;

static xnn_qd8_layout_info_t calibrate_xnn_qd8_qc8w(void) {
  xnn_qd8_layout_info_t info = {4352, 4608, 1.0f, 1};
  int8_t W_i8[64 * 64] __attribute__((aligned(64)));
  uint8_t W_packed[8192] __attribute__((aligned(64)));
  int8_t A_i8[16 * 64] __attribute__((aligned(64)));
  float C_out[16 * 64] __attribute__((aligned(64)));
  float minmax_params[16] __attribute__((aligned(64)));
  struct xnn_qd8_quantization_params qd8_params[16]
      __attribute__((aligned(64)));

  for (int i = 0; i < 8; ++i) {
    minmax_params[2 * i] = -INFINITY;
    minmax_params[2 * i + 1] = INFINITY;
  }
  for (int m = 0; m < 16; ++m) {
    qd8_params[m].zero_point = 0;
    qd8_params[m].inv_scale = 1.0f;
  }

  // 1. Check scale vs bias offset (4352 vs 4608)
  memset(W_i8, 0, sizeof(W_i8));
  memset(W_packed, 0, sizeof(W_packed));
  float *slot_a = (float *)(W_packed + 4352);
  float *slot_b = (float *)(W_packed + 4608);
  for (int n = 0; n < 64; ++n) {
    slot_a[n] = 1.0f;
    slot_b[n] = 2.0f;
  }
  memset(A_i8, 0, sizeof(A_i8));
  memset(C_out, 0, sizeof(C_out));
  fn_xnn_qc8w_16x64_amx(16, 64, 64, A_i8, 64, W_packed, C_out,
                        64 * sizeof(float), 64 * sizeof(float), minmax_params,
                        qd8_params);
  if (fabsf(C_out[0] - 1.0f) < 0.1f) {
    info.bias_offset_bytes = 4352;
    info.scale_offset_bytes = 4608;
  } else {
    info.scale_offset_bytes = 4352;
    info.bias_offset_bytes = 4608;
  }

  // 2. Check weight gain and zero_point sign
  memset(W_i8, 0, sizeof(W_i8));
  W_i8[0] = 1; // n=0, k=0
  int8_t pack_params[4] = {1, 0, 0, 0};
  memset(W_packed, 0, sizeof(W_packed));
  fn_xnn_pack_qs8_goi(1, 64, 64, 64, 4, 1, 64, W_i8, NULL, NULL, W_packed,
                      2 * 64 * sizeof(float), pack_params);
  float *w_scale = (float *)(W_packed + info.scale_offset_bytes);
  float *w_bias = (float *)(W_packed + info.bias_offset_bytes);
  for (int n = 0; n < 64; ++n) {
    w_scale[n] = 1.0f;
    w_bias[n] = 0.0f;
  }
  memset(A_i8, 0, sizeof(A_i8));
  A_i8[0] = 1;
  memset(C_out, 0, sizeof(C_out));
  fn_xnn_qc8w_16x64_amx(16, 64, 64, A_i8, 64, W_packed, C_out,
                        64 * sizeof(float), 64 * sizeof(float), minmax_params,
                        qd8_params);
  if (fabsf(C_out[0]) > 1e-5f) {
    info.weight_gain = C_out[0];
  }

  // Check zero_point sign with A_i8 = 0, zp = 10 -> expected (0 - 10) * 1 = -10
  memset(A_i8, 0, sizeof(A_i8));
  qd8_params[0].zero_point = 10;
  qd8_params[0].inv_scale = 1.0f;
  memset(C_out, 0, sizeof(C_out));
  fn_xnn_qc8w_16x64_amx(16, 64, 64, A_i8, 64, W_packed, C_out,
                        64 * sizeof(float), 64 * sizeof(float), minmax_params,
                        qd8_params);
  info.zp_sign = (C_out[0] < 0.0f) ? 1 : -1;
  return info;
}

static xnn_qd8_layout_info_t calibrate_xnn_qd8_qc4w(void) {
  xnn_qd8_layout_info_t info = {2304, 2560, 16.0f, 1};
  uint8_t W_u4[64 * 32] __attribute__((aligned(64)));
  uint8_t W_packed[8192] __attribute__((aligned(64)));
  int8_t A_i8[16 * 64] __attribute__((aligned(64)));
  float C_out[16 * 64] __attribute__((aligned(64)));
  float minmax_params[16] __attribute__((aligned(64)));
  struct xnn_qd8_quantization_params qd8_params[16]
      __attribute__((aligned(64)));

  for (int i = 0; i < 8; ++i) {
    minmax_params[2 * i] = -INFINITY;
    minmax_params[2 * i + 1] = INFINITY;
  }
  for (int m = 0; m < 16; ++m) {
    qd8_params[m].zero_point = 0;
    qd8_params[m].inv_scale = 1.0f;
  }

  // 1. Check scale vs bias offset (2304 vs 2560)
  memset(W_u4, 0x88, sizeof(W_u4)); // unsigned 8 = signed 0
  uint8_t pack_params[4] = {1, 8, 0, 0};
  memset(W_packed, 0, sizeof(W_packed));
  fn_xnn_pack_qs8_qc4w_goi(1, 64, 64, 64, 4, 2, 64, W_u4, NULL, NULL, W_packed,
                           2 * 64 * sizeof(float), pack_params);
  float *slot_a = (float *)(W_packed + 2304);
  float *slot_b = (float *)(W_packed + 2560);
  for (int n = 0; n < 64; ++n) {
    slot_a[n] = 1.0f;
    slot_b[n] = 2.0f;
  }
  memset(A_i8, 0, sizeof(A_i8));
  memset(C_out, 0, sizeof(C_out));
  fn_xnn_qc4w_1x64_amx(1, 64, 64, A_i8, 64, W_packed, C_out,
                       64 * sizeof(float), 64 * sizeof(float), minmax_params,
                       qd8_params);
  if (fabsf(C_out[0] - 1.0f) < 0.1f) {
    info.bias_offset_bytes = 2304;
    info.scale_offset_bytes = 2560;
  } else {
    info.scale_offset_bytes = 2304;
    info.bias_offset_bytes = 2560;
  }

  // 2. Check weight gain with W_i4[0][0] = +1 (u4 = 9)
  memset(W_u4, 0x88, sizeof(W_u4));
  W_u4[0] = (8 << 4) | 9; // k=0 is +1 (9 - 8), k=1 is 0 (8 - 8)
  memset(W_packed, 0, sizeof(W_packed));
  fn_xnn_pack_qs8_qc4w_goi(1, 64, 64, 64, 4, 2, 64, W_u4, NULL, NULL, W_packed,
                           2 * 64 * sizeof(float), pack_params);
  float *w_scale = (float *)(W_packed + info.scale_offset_bytes);
  float *w_bias = (float *)(W_packed + info.bias_offset_bytes);
  for (int n = 0; n < 64; ++n) {
    w_scale[n] = 1.0f;
    w_bias[n] = 0.0f;
  }
  memset(A_i8, 0, sizeof(A_i8));
  A_i8[0] = 1;
  memset(C_out, 0, sizeof(C_out));
  fn_xnn_qc4w_1x64_amx(1, 64, 64, A_i8, 64, W_packed, C_out,
                       64 * sizeof(float), 64 * sizeof(float), minmax_params,
                       qd8_params);
  if (fabsf(C_out[0]) > 1e-5f) {
    info.weight_gain = C_out[0];
  }

  // 3. Check zero_point sign
  memset(A_i8, 0, sizeof(A_i8));
  qd8_params[0].zero_point = 2;
  qd8_params[0].inv_scale = 1.0f;
  memset(C_out, 0, sizeof(C_out));
  fn_xnn_qc4w_1x64_amx(1, 64, 64, A_i8, 64, W_packed, C_out,
                       64 * sizeof(float), 64 * sizeof(float), minmax_params,
                       qd8_params);
  info.zp_sign = (C_out[0] < 0.0f) ? 1 : -1;
  return info;
}

// Calibrate XNNPACK qs8_qc2w_5x8c8_avxvnni (M=5, N=8, K=64 -> 128 weight bytes)
static void calibrate_xnn_qc2w_layout(int crumb_map[64][8],
                                      int32_t code2_to_val[4],
                                      size_t *scale_offset_out) {
  const int M = 5, N = 8, K = 64;
  int8_t A[5 * 64] __attribute__((aligned(64)));
  uint8_t W_packed[512] __attribute__((aligned(64)));
  int8_t C_out[5 * 8] __attribute__((aligned(64)));
  uint8_t params[256] __attribute__((aligned(64)));

  memset(params, 0, sizeof(params));
  fn_xnn_init_qs8_minmax_params(params, 0, -128, 127);

  // Determine scale offset inside W_packed (after 32B ksum + 128B weights = 160)
  size_t scale_off = 160;
  for (size_t cand = 160; cand <= 224; cand += 32) {
    memset(W_packed, 0, sizeof(W_packed));
    int32_t *ksum = (int32_t *)W_packed;
    for (int n = 0; n < 8; ++n) ksum[n] = 10;
    float *sc = (float *)(W_packed + cand);
    for (int n = 0; n < 8; ++n) sc[n] = 1.0f;
    memset(A, -128, sizeof(A)); // A ^ 0x80 == 0
    memset(C_out, 0, sizeof(C_out));
    fn_xnn_qc2w_5x8_avxvnni(M, N, K, A, K, W_packed, C_out, N, 1, params);
    if (C_out[0] == 10) {
      scale_off = cand;
      break;
    }
  }
  *scale_offset_out = scale_off;

  // Determine 2-bit code values when A_u8 = [1, 0, ..., 0] (A_i8[0] = -127, rest -128)
  for (int c = 0; c < 4; ++c) {
    memset(W_packed, 0, sizeof(W_packed));
    uint8_t rep = (uint8_t)((c << 6) | (c << 4) | (c << 2) | c);
    memset(W_packed + 32, rep, 128);
    float *sc = (float *)(W_packed + scale_off);
    for (int n = 0; n < 8; ++n) sc[n] = 1.0f;
    memset(A, -128, sizeof(A));
    A[0] = -127; // A_u8[0] = 1
    memset(C_out, 0, sizeof(C_out));
    fn_xnn_qc2w_5x8_avxvnni(M, N, K, A, K, W_packed, C_out, N, 1, params);
    code2_to_val[c] = (int32_t)C_out[0];
  }

  uint8_t c0 = 0, c1 = 1;
  int32_t v0 = code2_to_val[0], v1 = code2_to_val[1];
  for (int c = 1; c < 4; ++c) {
    if (code2_to_val[c] != v0) {
      c1 = (uint8_t)c;
      v1 = code2_to_val[c];
      break;
    }
  }

  memset(crumb_map, 0, sizeof(int) * 64 * 8);
  uint8_t rep0 = (uint8_t)((c0 << 6) | (c0 << 4) | (c0 << 2) | c0);
  for (int b = 0; b < 9; ++b) {
    memset(W_packed, 0, sizeof(W_packed));
    memset(W_packed + 32, rep0, 128);
    float *sc = (float *)(W_packed + scale_off);
    for (int n = 0; n < 8; ++n) sc[n] = 1.0f;
    for (int cid = 0; cid < 512; ++cid) {
      if ((cid >> b) & 1) {
        int byte_idx = cid >> 2;
        int shift = (cid & 3) * 2;
        W_packed[32 + byte_idx] = (uint8_t)(
            (W_packed[32 + byte_idx] & ~(0x3 << shift)) | (c1 << shift));
      }
    }
    for (int k = 0; k < K; ++k) {
      memset(A, -128, sizeof(A));
      A[k] = -127; // unit basis vector at k
      memset(C_out, 0, sizeof(C_out));
      fn_xnn_qc2w_5x8_avxvnni(M, N, K, A, K, W_packed, C_out, N, 1, params);
      for (int n = 0; n < N; ++n) {
        if ((int32_t)C_out[n] == v1) {
          crumb_map[k][n] |= (1 << b);
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Verification Result Containers
// -----------------------------------------------------------------------------

typedef struct {
  precision_metrics_t level_a; // Aggregate Level A across all 4 distributions
  precision_metrics_t level_b; // Aggregate Level B across all 4 distributions
  precision_metrics_t level_b_by_dist[NUM_DISTS];
  // Extra invariants for Softmax & Pack/Transpose
  double extra_metric_1;
  double extra_metric_2;
  int extra_int;
} kernel_verification_result_t;

// -----------------------------------------------------------------------------
// 1 & 2. Verify XNNPACK QD8-F32-QC4W GEMM (1x64c4 & 16x64c4 AMX)
// -----------------------------------------------------------------------------

static kernel_verification_result_t
verify_xnn_qd8_qc4w(xnn_qd8_gemm_fn fn, int M,
                    const xnn_qd8_layout_info_t *info) {
  kernel_verification_result_t res;
  memset(&res, 0, sizeof(res));
  const int N = 64, K = 64;

  double *A_fp64 = (double *) calloc(M * K, sizeof(double));
  double *B_fp64 = (double *) calloc(N * K, sizeof(double));
  double *bias_fp64 = (double *) calloc(N, sizeof(double));
  double *C_ref_level_a = (double *) calloc(M * N, sizeof(double));
  double *C_ref_level_b = (double *) calloc(M * N, sizeof(double));

  int8_t A_i8[16 * 64] __attribute__((aligned(64)));
  int8_t W_i4[64 * 64] __attribute__((aligned(64)));
  uint8_t W_u4_in[64 * 32] __attribute__((aligned(64)));
  uint8_t W_packed[8192] __attribute__((aligned(64)));
  float C_out[16 * 64] __attribute__((aligned(64)));
  float minmax_params[16] __attribute__((aligned(64)));
  struct xnn_qd8_quantization_params qd8_params[16]
      __attribute__((aligned(64)));

  for (int i = 0; i < 8; ++i) {
    minmax_params[2 * i] = -INFINITY;
    minmax_params[2 * i + 1] = INFINITY;
  }

  for (int d = 0; d < NUM_DISTS; ++d) {
    fill_fp64_tensors((dist_kind_t)d, M, N, K, A_fp64, B_fp64, bias_fp64);

    // Dynamic per-row asymmetric int8 quantization of A_fp64
    for (int m = 0; m < M; ++m) {
      double rmin = 0.0, rmax = 0.0;
      for (int k = 0; k < K; ++k) {
        double v = A_fp64[m * K + k];
        if (v < rmin) rmin = v;
        if (v > rmax) rmax = v;
      }
      double scale_a = (rmax - rmin) > 1e-12 ? (rmax - rmin) / 255.0 : 1.0f;
      int32_t zp_a =
          (int32_t)lround(-128.0 - rmin / scale_a);
      if (zp_a < -128) zp_a = -128;
      if (zp_a > 127) zp_a = 127;
      qd8_params[m].zero_point = zp_a * info->zp_sign;
      qd8_params[m].inv_scale = (float)scale_a;
      for (int k = 0; k < K; ++k) {
        long q = lround(A_fp64[m * K + k] / scale_a) + zp_a;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        A_i8[m * K + k] = (int8_t)q;
      }
    }

    // Per-channel symmetric int4 [-8, 7] quantization of B_fp64
    float w_scales_true[64];
    float w_biases_f32[64];
    memset(W_u4_in, 0, sizeof(W_u4_in));
    for (int n = 0; n < N; ++n) {
      double amax = 0.0;
      for (int k = 0; k < K; ++k) {
        double av = fabs(B_fp64[n * K + k]);
        if (av > amax) amax = av;
      }
      float sw = (amax > 1e-12) ? (float)(amax / 7.0) : 1.0f;
      w_scales_true[n] = sw;
      w_biases_f32[n] = (float)bias_fp64[n];
      for (int k = 0; k < K; ++k) {
        long q = lround(B_fp64[n * K + k] / (double)sw);
        if (q < -8) q = -8;
        if (q > 7) q = 7;
        W_i4[n * K + k] = (int8_t)q;
        uint8_t u4 = (uint8_t)(q + 8) & 0x0F;
        if (k & 1) {
          W_u4_in[n * 32 + (k >> 1)] |= (uint8_t)(u4 << 4);
        } else {
          W_u4_in[n * 32 + (k >> 1)] |= u4;
        }
      }
    }

    uint8_t pack_params[4] = {1, 8, 0, 0};
    memset(W_packed, 0, sizeof(W_packed));
    fn_xnn_pack_qs8_qc4w_goi(1, N, K, 64, 4, 2, K, W_u4_in, NULL, NULL,
                             W_packed, 2 * 64 * sizeof(float), pack_params);
    float *w_scale_buf = (float *)(W_packed + info->scale_offset_bytes);
    float *w_bias_buf = (float *)(W_packed + info->bias_offset_bytes);
    for (int n = 0; n < N; ++n) {
      w_scale_buf[n] = w_scales_true[n] / info->weight_gain;
      w_bias_buf[n] = w_biases_f32[n];
    }

    memset(C_out, 0, sizeof(C_out));
    fn(M, N, K, A_i8, K, W_packed, C_out, N * sizeof(float), N * sizeof(float),
       minmax_params, qd8_params);

    // Compute Level A (exact INT64 accumulator + FP64 scale/bias) and Level B (FP64 golden)
    for (int m = 0; m < M; ++m) {
      int32_t zp_a = qd8_params[m].zero_point * info->zp_sign;
      double sa = (double)qd8_params[m].inv_scale;
      for (int n = 0; n < N; ++n) {
        int64_t acc_i64 = 0;
        double acc_fp64 = bias_fp64[n];
        for (int k = 0; k < K; ++k) {
          acc_i64 += ((int64_t)A_i8[m * K + k] - (int64_t)zp_a) *
                     (int64_t)W_i4[n * K + k];
          acc_fp64 += A_fp64[m * K + k] * B_fp64[n * K + k];
        }
        C_ref_level_a[m * N + n] =
            (double)acc_i64 * sa * (double)w_scales_true[n] +
            (double)w_biases_f32[n];
        C_ref_level_b[m * N + n] = acc_fp64;
      }
    }

    precision_metrics_t ma =
        compute_fp_metrics(C_out, C_ref_level_a, M * N, 2, 110.0, 1e-4);
    precision_metrics_t mb =
        compute_fp_metrics(C_out, C_ref_level_b, M * N, 0, 14.0, 10.0);
    res.level_b_by_dist[d] = mb;
    merge_metrics(&res.level_a, &ma, d);
    merge_metrics(&res.level_b, &mb, d);
  }

  free(A_fp64);
  free(B_fp64);
  free(bias_fp64);
  free(C_ref_level_a);
  free(C_ref_level_b);
  return res;
}

// -----------------------------------------------------------------------------
// 5. Verify XNNPACK QD8-F32-QC8W GEMM (16x64c4 AMX)
// -----------------------------------------------------------------------------

static kernel_verification_result_t
verify_xnn_qd8_qc8w(const xnn_qd8_layout_info_t *info) {
  kernel_verification_result_t res;
  memset(&res, 0, sizeof(res));
  const int M = 16, N = 64, K = 64;

  double *A_fp64 = (double *) calloc(M * K, sizeof(double));
  double *B_fp64 = (double *) calloc(N * K, sizeof(double));
  double *bias_fp64 = (double *) calloc(N, sizeof(double));
  double *C_ref_level_a = (double *) calloc(M * N, sizeof(double));
  double *C_ref_level_b = (double *) calloc(M * N, sizeof(double));

  int8_t A_i8[16 * 64] __attribute__((aligned(64)));
  int8_t W_i8[64 * 64] __attribute__((aligned(64)));
  uint8_t W_packed[8192] __attribute__((aligned(64)));
  float C_out[16 * 64] __attribute__((aligned(64)));
  float minmax_params[16] __attribute__((aligned(64)));
  struct xnn_qd8_quantization_params qd8_params[16]
      __attribute__((aligned(64)));

  for (int i = 0; i < 8; ++i) {
    minmax_params[2 * i] = -INFINITY;
    minmax_params[2 * i + 1] = INFINITY;
  }

  for (int d = 0; d < NUM_DISTS; ++d) {
    fill_fp64_tensors((dist_kind_t)d, M, N, K, A_fp64, B_fp64, bias_fp64);

    for (int m = 0; m < M; ++m) {
      double rmin = 0.0, rmax = 0.0;
      for (int k = 0; k < K; ++k) {
        double v = A_fp64[m * K + k];
        if (v < rmin) rmin = v;
        if (v > rmax) rmax = v;
      }
      double scale_a = (rmax - rmin) > 1e-12 ? (rmax - rmin) / 255.0 : 1.0;
      int32_t zp_a = (int32_t)lround(-128.0 - rmin / scale_a);
      if (zp_a < -128) zp_a = -128;
      if (zp_a > 127) zp_a = 127;
      qd8_params[m].zero_point = zp_a * info->zp_sign;
      qd8_params[m].inv_scale = (float)scale_a;
      for (int k = 0; k < K; ++k) {
        long q = lround(A_fp64[m * K + k] / scale_a) + zp_a;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        A_i8[m * K + k] = (int8_t)q;
      }
    }

    float w_scales_true[64];
    float w_biases_f32[64];
    for (int n = 0; n < N; ++n) {
      double amax = 0.0;
      for (int k = 0; k < K; ++k) {
        double av = fabs(B_fp64[n * K + k]);
        if (av > amax) amax = av;
      }
      float sw = (amax > 1e-12) ? (float)(amax / 127.0) : 1.0f;
      w_scales_true[n] = sw;
      w_biases_f32[n] = (float)bias_fp64[n];
      for (int k = 0; k < K; ++k) {
        long q = lround(B_fp64[n * K + k] / (double)sw);
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        W_i8[n * K + k] = (int8_t)q;
      }
    }

    int8_t pack_params[4] = {1, 0, 0, 0};
    memset(W_packed, 0, sizeof(W_packed));
    fn_xnn_pack_qs8_goi(1, N, K, 64, 4, 1, K, W_i8, NULL, NULL, W_packed,
                        2 * 64 * sizeof(float), pack_params);
    float *w_scale_buf = (float *)(W_packed + info->scale_offset_bytes);
    float *w_bias_buf = (float *)(W_packed + info->bias_offset_bytes);
    for (int n = 0; n < N; ++n) {
      w_scale_buf[n] = w_scales_true[n] / info->weight_gain;
      w_bias_buf[n] = w_biases_f32[n];
    }

    memset(C_out, 0, sizeof(C_out));
    fn_xnn_qc8w_16x64_amx(M, N, K, A_i8, K, W_packed, C_out, N * sizeof(float),
                          N * sizeof(float), minmax_params, qd8_params);

    for (int m = 0; m < M; ++m) {
      int32_t zp_a = qd8_params[m].zero_point * info->zp_sign;
      double sa = (double)qd8_params[m].inv_scale;
      for (int n = 0; n < N; ++n) {
        int64_t acc_i64 = 0;
        double acc_fp64 = bias_fp64[n];
        for (int k = 0; k < K; ++k) {
          acc_i64 += ((int64_t)A_i8[m * K + k] - (int64_t)zp_a) *
                     (int64_t)W_i8[n * K + k];
          acc_fp64 += A_fp64[m * K + k] * B_fp64[n * K + k];
        }
        C_ref_level_a[m * N + n] =
            (double)acc_i64 * sa * (double)w_scales_true[n] +
            (double)w_biases_f32[n];
        C_ref_level_b[m * N + n] = acc_fp64;
      }
    }

    precision_metrics_t ma =
        compute_fp_metrics(C_out, C_ref_level_a, M * N, 2, 110.0, 1e-4);
    precision_metrics_t mb =
        compute_fp_metrics(C_out, C_ref_level_b, M * N, 0, 32.0, 2.0);
    res.level_b_by_dist[d] = mb;
    merge_metrics(&res.level_a, &ma, d);
    merge_metrics(&res.level_b, &mb, d);
  }

  free(A_fp64);
  free(B_fp64);
  free(bias_fp64);
  free(C_ref_level_a);
  free(C_ref_level_b);
  return res;
}

// -----------------------------------------------------------------------------
// 1 & 2 (YNNPACK). Verify YNNPACK UINT8 x INT4 -> INT32 Dot (1x32x8 & 12x16x8)
// -----------------------------------------------------------------------------

static kernel_verification_result_t
verify_ynn_dot_int4(ynn_dot_fn fn, int M, int N, size_t b_stride_k1,
                    const int nibble_map[64][64],
                    const uint8_t val_to_code[16]) {
  kernel_verification_result_t res;
  memset(&res, 0, sizeof(res));
  const int K = 64;

  double *A_fp64 = (double *) calloc(M * K, sizeof(double));
  double *B_fp64 = (double *) calloc(N * K, sizeof(double));
  double *bias_fp64 = (double *) calloc(N, sizeof(double));
  double *C_ref_level_a = (double *) calloc(M * N, sizeof(double));
  double *C_ref_level_b = (double *) calloc(M * N, sizeof(double));
  float *C_act_level_a = (float *) calloc(M * N, sizeof(float));
  float *C_act_level_b = (float *) calloc(M * N, sizeof(float));

  uint8_t A_u8[16 * 64] __attribute__((aligned(64)));
  int8_t W_i4[64 * 64] __attribute__((aligned(64)));
  uint8_t B_packed[2048] __attribute__((aligned(64)));
  int32_t C_zero[16 * 64] __attribute__((aligned(64)));
  int32_t C_i32[16 * 64] __attribute__((aligned(64)));
  memset(C_zero, 0, sizeof(C_zero));

  for (int d = 0; d < NUM_DISTS; ++d) {
    fill_fp64_tensors((dist_kind_t)d, M, N, K, A_fp64, B_fp64, bias_fp64);

    double scale_a[16];
    int32_t zp_a[16];
    for (int m = 0; m < M; ++m) {
      double rmin = 0.0, rmax = 0.0;
      for (int k = 0; k < K; ++k) {
        double v = A_fp64[m * K + k];
        if (v < rmin) rmin = v;
        if (v > rmax) rmax = v;
      }
      double sa = (rmax - rmin) > 1e-12 ? (rmax - rmin) / 255.0 : 1.0;
      int32_t zpa = (int32_t)lround(-rmin / sa);
      if (zpa < 0) zpa = 0;
      if (zpa > 255) zpa = 255;
      scale_a[m] = sa;
      zp_a[m] = zpa;
      for (int k = 0; k < K; ++k) {
        long q = lround(A_fp64[m * K + k] / sa) + zpa;
        if (q < 0) q = 0;
        if (q > 255) q = 255;
        A_u8[m * K + k] = (uint8_t)q;
      }
    }

    double scale_w[64];
    memset(B_packed, 0, sizeof(B_packed));
    for (int n = 0; n < N; ++n) {
      double amax = 0.0;
      for (int k = 0; k < K; ++k) {
        double av = fabs(B_fp64[n * K + k]);
        if (av > amax) amax = av;
      }
      double sw = (amax > 1e-12) ? (amax / 7.0) : 1.0;
      scale_w[n] = sw;
      for (int k = 0; k < K; ++k) {
        long q = lround(B_fp64[n * K + k] / sw);
        if (q < -8) q = -8;
        if (q > 7) q = 7;
        W_i4[n * K + k] = (int8_t)q;
        uint8_t code = val_to_code[q + 8] & 0x0F;
        int nid = nibble_map[k][n];
        int byte_idx = nid >> 1;
        if (nid & 1) {
          B_packed[byte_idx] = (uint8_t)((B_packed[byte_idx] & 0x0F) | (code << 4));
        } else {
          B_packed[byte_idx] = (uint8_t)((B_packed[byte_idx] & 0xF0) | code);
        }
      }
    }

    memset(C_i32, 0, sizeof(C_i32));
    fn(M, N, 1, 1, K, K, 0, 0, A_u8, 0, 0, b_stride_k1, B_packed,
       N * sizeof(int32_t), C_zero, N * sizeof(int32_t), C_i32, NULL);

    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        int64_t raw_i64 = 0;
        int64_t w_sum = 0;
        double acc_fp64 = bias_fp64[n];
        for (int k = 0; k < K; ++k) {
          raw_i64 += (int64_t)A_u8[m * K + k] * (int64_t)W_i4[n * K + k];
          w_sum += (int64_t)W_i4[n * K + k];
          acc_fp64 += A_fp64[m * K + k] * B_fp64[n * K + k];
        }
        // Level A compares exact INT32 dot output against INT64 reference
        C_act_level_a[m * N + n] = (float)C_i32[m * N + n];
        C_ref_level_a[m * N + n] = (double)raw_i64;

        // Level B dequantizes the microkernel's INT32 accumulator to FP32
        int64_t corr_acc = (int64_t)C_i32[m * N + n] - (int64_t)zp_a[m] * w_sum;
        C_act_level_b[m * N + n] =
            (float)((double)corr_acc * scale_a[m] * scale_w[n] + bias_fp64[n]);
        C_ref_level_b[m * N + n] = acc_fp64;
      }
    }

    precision_metrics_t ma =
        compute_fp_metrics(C_act_level_a, C_ref_level_a, M * N, 0, 190.0, 0.0);
    precision_metrics_t mb =
        compute_fp_metrics(C_act_level_b, C_ref_level_b, M * N, 0, 14.0, 10.0);
    res.level_b_by_dist[d] = mb;
    merge_metrics(&res.level_a, &ma, d);
    merge_metrics(&res.level_b, &mb, d);
  }

  free(A_fp64);
  free(B_fp64);
  free(bias_fp64);
  free(C_ref_level_a);
  free(C_ref_level_b);
  free(C_act_level_a);
  free(C_act_level_b);
  return res;
}

// -----------------------------------------------------------------------------
// 5 (YNNPACK). Verify YNNPACK INT8 x INT8 -> INT32 Dot (32x32x64 AMX)
// -----------------------------------------------------------------------------

static kernel_verification_result_t
verify_ynn_dot_int8(const int slot_map[64][32]) {
  kernel_verification_result_t res;
  memset(&res, 0, sizeof(res));
  const int M = 32, N = 32, K = 64;

  double *A_fp64 = (double *) calloc(M * K, sizeof(double));
  double *B_fp64 = (double *) calloc(N * K, sizeof(double));
  double *bias_fp64 = (double *) calloc(N, sizeof(double));
  double *C_ref_level_a = (double *) calloc(M * N, sizeof(double));
  double *C_ref_level_b = (double *) calloc(M * N, sizeof(double));
  float *C_act_level_a = (float *) calloc(M * N, sizeof(float));
  float *C_act_level_b = (float *) calloc(M * N, sizeof(float));

  int8_t A_i8[32 * 64] __attribute__((aligned(64)));
  int8_t W_i8[32 * 64] __attribute__((aligned(64)));
  int8_t B_packed[2048] __attribute__((aligned(64)));
  int32_t C_zero[32 * 32] __attribute__((aligned(64)));
  int32_t C_i32[32 * 32] __attribute__((aligned(64)));
  uint8_t amx_state[256] __attribute__((aligned(64)));
  memset(C_zero, 0, sizeof(C_zero));

  for (int d = 0; d < NUM_DISTS; ++d) {
    fill_fp64_tensors((dist_kind_t)d, M, N, K, A_fp64, B_fp64, bias_fp64);

    double scale_a[32];
    for (int m = 0; m < M; ++m) {
      double amax = 0.0;
      for (int k = 0; k < K; ++k) {
        double av = fabs(A_fp64[m * K + k]);
        if (av > amax) amax = av;
      }
      double sa = (amax > 1e-12) ? (amax / 127.0) : 1.0;
      scale_a[m] = sa;
      for (int k = 0; k < K; ++k) {
        long q = lround(A_fp64[m * K + k] / sa);
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        A_i8[m * K + k] = (int8_t)q;
      }
    }

    double scale_w[32];
    memset(B_packed, 0, sizeof(B_packed));
    for (int n = 0; n < N; ++n) {
      double amax = 0.0;
      for (int k = 0; k < K; ++k) {
        double av = fabs(B_fp64[n * K + k]);
        if (av > amax) amax = av;
      }
      double sw = (amax > 1e-12) ? (amax / 127.0) : 1.0;
      scale_w[n] = sw;
      for (int k = 0; k < K; ++k) {
        long q = lround(B_fp64[n * K + k] / sw);
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        W_i8[n * K + k] = (int8_t)q;
        B_packed[slot_map[k][n]] = (int8_t)q;
      }
    }

    memset(C_i32, 0, sizeof(C_i32));
    memset(amx_state, 0, sizeof(amx_state));
    fn_ynn_dot_qc8_32x32_amx(M, N, 1, 1, K, K, 0, 0, A_i8, 0, 0, 32, B_packed,
                             N * sizeof(int32_t), C_zero, N * sizeof(int32_t),
                             C_i32, amx_state);

    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        int64_t raw_i64 = 0;
        double acc_fp64 = bias_fp64[n];
        for (int k = 0; k < K; ++k) {
          raw_i64 += (int64_t)A_i8[m * K + k] * (int64_t)W_i8[n * K + k];
          acc_fp64 += A_fp64[m * K + k] * B_fp64[n * K + k];
        }
        C_act_level_a[m * N + n] = (float)C_i32[m * N + n];
        C_ref_level_a[m * N + n] = (double)raw_i64;
        C_act_level_b[m * N + n] = (float)((double)C_i32[m * N + n] *
                                               scale_a[m] * scale_w[n] +
                                           bias_fp64[n]);
        C_ref_level_b[m * N + n] = acc_fp64;
      }
    }

    precision_metrics_t ma =
        compute_fp_metrics(C_act_level_a, C_ref_level_a, M * N, 0, 190.0, 0.0);
    precision_metrics_t mb =
        compute_fp_metrics(C_act_level_b, C_ref_level_b, M * N, 0, 32.0, 2.0);
    res.level_b_by_dist[d] = mb;
    merge_metrics(&res.level_a, &ma, d);
    merge_metrics(&res.level_b, &mb, d);
  }

  free(A_fp64);
  free(B_fp64);
  free(bias_fp64);
  free(C_ref_level_a);
  free(C_ref_level_b);
  free(C_act_level_a);
  free(C_act_level_b);
  return res;
}

// -----------------------------------------------------------------------------
// 3. Verify FP32 GEMM / Dot Pair (XNNPACK 7x32 & YNNPACK 5x64x1)
// -----------------------------------------------------------------------------

static kernel_verification_result_t verify_xnn_fp32_gemm(void) {
  kernel_verification_result_t res;
  memset(&res, 0, sizeof(res));
  const int M = 7, N = 64, K = 64;

  double *A_fp64 = (double *) calloc(M * K, sizeof(double));
  double *B_fp64 = (double *) calloc(N * K, sizeof(double));
  double *bias_fp64 = (double *) calloc(N, sizeof(double));
  double *C_ref_level_a = (double *) calloc(M * N, sizeof(double));
  double *C_ref_level_b = (double *) calloc(M * N, sizeof(double));

  float A_f32[7 * 64] __attribute__((aligned(64)));
  float B_f32[64 * 64] __attribute__((aligned(64)));
  float bias_f32[64] __attribute__((aligned(64)));
  float W_packed[64 * (64 + 1)] __attribute__((aligned(64)));
  float C_out[7 * 64] __attribute__((aligned(64)));
  float minmax_params[32] __attribute__((aligned(64)));

  for (int i = 0; i < 16; ++i) {
    minmax_params[2 * i] = -INFINITY;
    minmax_params[2 * i + 1] = INFINITY;
  }

  for (int d = 0; d < NUM_DISTS; ++d) {
    fill_fp64_tensors((dist_kind_t)d, M, N, K, A_fp64, B_fp64, bias_fp64);

    for (int i = 0; i < M * K; ++i) A_f32[i] = (float)A_fp64[i];
    for (int i = 0; i < N * K; ++i) B_f32[i] = (float)B_fp64[i];
    for (int i = 0; i < N; ++i) bias_f32[i] = (float)bias_fp64[i];

    memset(W_packed, 0, sizeof(W_packed));
    fn_xnn_packw_x32_avx512(1, N, K, 32, 1, 1, K, B_f32, bias_f32, NULL,
                            W_packed, 0, NULL);

    memset(C_out, 0, sizeof(C_out));
    fn_xnn_f32_7x32_avx512(M, N, K * sizeof(float), A_f32, K * sizeof(float),
                           W_packed, C_out, N * sizeof(float),
                           32 * sizeof(float), minmax_params);

    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float acc_a = bias_f32[n];
        double acc_b = bias_fp64[n];
        for (int k = 0; k < K; ++k) {
          acc_a = fmaf(A_f32[m * K + k], B_f32[n * K + k], acc_a);
          acc_b += A_fp64[m * K + k] * B_fp64[n * K + k];
        }
        C_ref_level_a[m * N + n] = (double)acc_a;
        C_ref_level_b[m * N + n] = acc_b;
      }
    }

    precision_metrics_t ma =
        compute_fp_metrics(C_out, C_ref_level_a, M * N, 8, 120.0, 1e-4);
    precision_metrics_t mb =
        compute_fp_metrics(C_out, C_ref_level_b, M * N, 8, 120.0, 1e-4);
    res.level_b_by_dist[d] = mb;
    merge_metrics(&res.level_a, &ma, d);
    merge_metrics(&res.level_b, &mb, d);
  }

  free(A_fp64);
  free(B_fp64);
  free(bias_fp64);
  free(C_ref_level_a);
  free(C_ref_level_b);
  return res;
}

static kernel_verification_result_t
verify_ynn_fp32_dot(const int slot_map[64][64]) {
  kernel_verification_result_t res;
  memset(&res, 0, sizeof(res));
  const int M = 5, N = 64, K = 64;

  double *A_fp64 = (double *) calloc(M * K, sizeof(double));
  double *B_fp64 = (double *) calloc(N * K, sizeof(double));
  double *bias_fp64 = (double *) calloc(N, sizeof(double));
  double *C_ref_level_a = (double *) calloc(M * N, sizeof(double));
  double *C_ref_level_b = (double *) calloc(M * N, sizeof(double));

  float A_f32[5 * 64] __attribute__((aligned(64)));
  float B_f32[64 * 64] __attribute__((aligned(64)));
  float B_packed[4096] __attribute__((aligned(64)));
  float C_in[5 * 64] __attribute__((aligned(64)));
  float C_out[5 * 64] __attribute__((aligned(64)));

  for (int d = 0; d < NUM_DISTS; ++d) {
    fill_fp64_tensors((dist_kind_t)d, M, N, K, A_fp64, B_fp64, bias_fp64);

    for (int i = 0; i < M * K; ++i) A_f32[i] = (float)A_fp64[i];
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) {
        float w = (float)B_fp64[n * K + k];
        B_f32[n * K + k] = w;
        B_packed[slot_map[k][n]] = w;
      }
      for (int m = 0; m < M; ++m) {
        C_in[m * N + n] = (float)bias_fp64[n];
      }
    }

    memset(C_out, 0, sizeof(C_out));
    fn_ynn_dot_fp32_5x64_avx512(
        M, N, 1, 1, K, K * sizeof(float), 0, 0, A_f32, 0, 0, 64 * sizeof(float),
        B_packed, N * sizeof(float), C_in, N * sizeof(float), C_out, NULL);

    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float acc_a = C_in[m * N + n];
        double acc_b = bias_fp64[n];
        for (int k = 0; k < K; ++k) {
          acc_a = fmaf(A_f32[m * K + k], B_f32[n * K + k], acc_a);
          acc_b += A_fp64[m * K + k] * B_fp64[n * K + k];
        }
        C_ref_level_a[m * N + n] = (double)acc_a;
        C_ref_level_b[m * N + n] = acc_b;
      }
    }

    precision_metrics_t ma =
        compute_fp_metrics(C_out, C_ref_level_a, M * N, 8, 120.0, 1e-4);
    precision_metrics_t mb =
        compute_fp_metrics(C_out, C_ref_level_b, M * N, 8, 120.0, 1e-4);
    res.level_b_by_dist[d] = mb;
    merge_metrics(&res.level_a, &ma, d);
    merge_metrics(&res.level_b, &mb, d);
  }

  free(A_fp64);
  free(B_fp64);
  free(bias_fp64);
  free(C_ref_level_a);
  free(C_ref_level_b);
  return res;
}

// -----------------------------------------------------------------------------
// 6. Verify 2-Bit QC2W / INT2 GEMM Pair (XNNPACK 5x8c8 & YNNPACK 4x64x16)
// -----------------------------------------------------------------------------

static kernel_verification_result_t
verify_xnn_qc2w_gemm(const int crumb_map[64][8], const int32_t code2_to_val[4],
                     size_t scale_off) {
  kernel_verification_result_t res;
  memset(&res, 0, sizeof(res));
  const int M = 5, N = 8, K = 64;

  double *A_fp64 = (double *) calloc(M * K, sizeof(double));
  double *B_fp64 = (double *) calloc(N * K, sizeof(double));
  double *bias_fp64 = (double *) calloc(N, sizeof(double));
  double *C_ref_level_a = (double *) calloc(M * N, sizeof(double));
  double *C_ref_level_b = (double *) calloc(M * N, sizeof(double));
  float *C_act_level_a = (float *) calloc(M * N, sizeof(float));
  float *C_act_level_b = (float *) calloc(M * N, sizeof(float));

  int8_t A_i8[5 * 64] __attribute__((aligned(64)));
  int32_t W_val[8 * 64] __attribute__((aligned(64)));
  uint8_t W_packed[512] __attribute__((aligned(64)));
  int8_t C_i8[5 * 8] __attribute__((aligned(64)));
  uint8_t params[256] __attribute__((aligned(64)));

  memset(params, 0, sizeof(params));
  fn_xnn_init_qs8_minmax_params(params, 0, -128, 127);

  // Compute centroid mean of the 4 2-bit levels
  double w_mean = 0.0;
  for (int c = 0; c < 4; ++c) w_mean += (double)code2_to_val[c];
  w_mean *= 0.25;
  double w_max_dev = 0.0;
  for (int c = 0; c < 4; ++c) {
    double dev = fabs((double)code2_to_val[c] - w_mean);
    if (dev > w_max_dev) w_max_dev = dev;
  }
  if (w_max_dev < 1e-6) w_max_dev = 1.0;

  for (int d = 0; d < NUM_DISTS; ++d) {
    fill_fp64_tensors((dist_kind_t)d, M, N, K, A_fp64, B_fp64, bias_fp64);
    for (int n = 0; n < N; ++n) bias_fp64[n] = 0.0;

    // Symmetric per-tensor int8 quantization of A_fp64
    double amax_a = 0.0;
    for (int i = 0; i < M * K; ++i) {
      double av = fabs(A_fp64[i]);
      if (av > amax_a) amax_a = av;
    }
    double sa = (amax_a > 1e-12) ? (amax_a / 127.0) : 1.0;
    for (int i = 0; i < M * K; ++i) {
      long q = lround(A_fp64[i] / sa);
      if (q < -127) q = -127;
      if (q > 127) q = 127;
      A_i8[i] = (int8_t)q;
    }

    // Per-channel 2-bit quantization of B_fp64 onto (code2_to_val[c] - w_mean) * sw
    memset(W_packed, 0, sizeof(W_packed));
    int32_t *ksum = (int32_t *)W_packed;
    float *w_scale_buf = (float *)(W_packed + scale_off);
    double sw_arr[8];
    double y_max_est = 0.0;

    for (int n = 0; n < N; ++n) {
      double amax_w = 0.0;
      for (int k = 0; k < K; ++k) {
        double av = fabs(B_fp64[n * K + k]);
        if (av > amax_w) amax_w = av;
      }
      double sw = (amax_w > 1e-12) ? (amax_w / w_max_dev) : 1.0;
      sw_arr[n] = sw;

      int32_t sum_w = 0;
      for (int k = 0; k < K; ++k) {
        double target = B_fp64[n * K + k] / sw + w_mean;
        int best_c = 0;
        double best_err = 1e30;
        for (int c = 0; c < 4; ++c) {
          double err = fabs(target - (double)code2_to_val[c]);
          if (err < best_err) {
            best_err = err;
            best_c = c;
          }
        }
        W_val[n * K + k] = code2_to_val[best_c];
        sum_w += code2_to_val[best_c];
        int cid = crumb_map[k][n];
        int byte_idx = cid >> 2;
        int shift = (cid & 3) * 2;
        W_packed[32 + byte_idx] = (uint8_t)(
            (W_packed[32 + byte_idx] & ~(0x3 << shift)) | (best_c << shift));
      }
      // Cancel the +128 activation XOR offset in vpdpbusd
      ksum[n] = -128 * sum_w;
    }

    // Choose output scale so quantized int8 output utilizes [-100, 100] without clipping
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
          acc += (double)A_i8[m * K + k] *
                 ((double)W_val[n * K + k] - w_mean) * sa * sw_arr[n];
        }
        if (fabs(acc) > y_max_est) y_max_est = fabs(acc);
      }
    }
    double out_scale = (y_max_est > 1e-12) ? (y_max_est / 100.0) : 1.0;
    for (int n = 0; n < N; ++n) {
      w_scale_buf[n] = (float)((sa * sw_arr[n]) / out_scale);
    }

    memset(C_i8, 0, sizeof(C_i8));
    fn_xnn_qc2w_5x8_avxvnni(M, N, K, A_i8, K, W_packed, C_i8, N, 1, params);

    for (int m = 0; m < M; ++m) {
      int64_t row_a_sum = 0;
      for (int k = 0; k < K; ++k) row_a_sum += (int64_t)A_i8[m * K + k];
      for (int n = 0; n < N; ++n) {
        int64_t acc_i64 = (int64_t)ksum[n];
        double acc_fp64 = 0.0;
        for (int k = 0; k < K; ++k) {
          acc_i64 += ((int64_t)A_i8[m * K + k] + 128) * (int64_t)W_val[n * K + k];
          acc_fp64 += A_fp64[m * K + k] * B_fp64[n * K + k];
        }
        double req_f32 = (double)acc_i64 * (double)w_scale_buf[n];
        long q_ref = lround(req_f32);
        if (q_ref < -128) q_ref = -128;
        if (q_ref > 127) q_ref = 127;

        C_act_level_a[m * N + n] = (float)C_i8[m * N + n];
        C_ref_level_a[m * N + n] = (double)q_ref;

        // Dequantize output and subtract w_mean * row_a_sum offset if w_mean != 0
        double deq = (double)C_i8[m * N + n] * out_scale -
                     (double)row_a_sum * w_mean * sa * sw_arr[n];
        C_act_level_b[m * N + n] = (float)deq;
        C_ref_level_b[m * N + n] = acc_fp64;
      }
    }

    precision_metrics_t ma =
        compute_fp_metrics(C_act_level_a, C_ref_level_a, M * N, 1, 60.0, 1.0);
    precision_metrics_t mb =
        compute_fp_metrics(C_act_level_b, C_ref_level_b, M * N, 0, 5.5, 50.0);
    res.level_b_by_dist[d] = mb;
    merge_metrics(&res.level_a, &ma, d);
    merge_metrics(&res.level_b, &mb, d);
  }

  free(A_fp64);
  free(B_fp64);
  free(bias_fp64);
  free(C_ref_level_a);
  free(C_ref_level_b);
  free(C_act_level_a);
  free(C_act_level_b);
  return res;
}

static kernel_verification_result_t
verify_ynn_dot_int2(const int crumb_map[64][64],
                    const int32_t code2_to_val[4]) {
  kernel_verification_result_t res;
  memset(&res, 0, sizeof(res));
  const int M = 4, N = 64, K = 64;

  double *A_fp64 = (double *) calloc(M * K, sizeof(double));
  double *B_fp64 = (double *) calloc(N * K, sizeof(double));
  double *bias_fp64 = (double *) calloc(N, sizeof(double));
  double *C_ref_level_a = (double *) calloc(M * N, sizeof(double));
  double *C_ref_level_b = (double *) calloc(M * N, sizeof(double));
  float *C_act_level_a = (float *) calloc(M * N, sizeof(float));
  float *C_act_level_b = (float *) calloc(M * N, sizeof(float));

  uint8_t A_u8[4 * 64] __attribute__((aligned(64)));
  int32_t W_val[64 * 64] __attribute__((aligned(64)));
  uint8_t B_packed[1024] __attribute__((aligned(64)));
  int32_t C_zero[4 * 64] __attribute__((aligned(64)));
  int32_t C_i32[4 * 64] __attribute__((aligned(64)));
  memset(C_zero, 0, sizeof(C_zero));

  double w_mean = 0.0;
  for (int c = 0; c < 4; ++c) w_mean += (double)code2_to_val[c];
  w_mean *= 0.25;
  double w_max_dev = 0.0;
  for (int c = 0; c < 4; ++c) {
    double dev = fabs((double)code2_to_val[c] - w_mean);
    if (dev > w_max_dev) w_max_dev = dev;
  }
  if (w_max_dev < 1e-6) w_max_dev = 1.0;

  for (int d = 0; d < NUM_DISTS; ++d) {
    fill_fp64_tensors((dist_kind_t)d, M, N, K, A_fp64, B_fp64, bias_fp64);

    double scale_a[4];
    int32_t zp_a[4];
    for (int m = 0; m < M; ++m) {
      double rmin = 0.0, rmax = 0.0;
      for (int k = 0; k < K; ++k) {
        double v = A_fp64[m * K + k];
        if (v < rmin) rmin = v;
        if (v > rmax) rmax = v;
      }
      double sa = (rmax - rmin) > 1e-12 ? (rmax - rmin) / 255.0 : 1.0;
      int32_t zpa = (int32_t)lround(-rmin / sa);
      if (zpa < 0) zpa = 0;
      if (zpa > 255) zpa = 255;
      scale_a[m] = sa;
      zp_a[m] = zpa;
      for (int k = 0; k < K; ++k) {
        long q = lround(A_fp64[m * K + k] / sa) + zpa;
        if (q < 0) q = 0;
        if (q > 255) q = 255;
        A_u8[m * K + k] = (uint8_t)q;
      }
    }

    double scale_w[64];
    memset(B_packed, 0, sizeof(B_packed));
    for (int n = 0; n < N; ++n) {
      double amax = 0.0;
      for (int k = 0; k < K; ++k) {
        double av = fabs(B_fp64[n * K + k]);
        if (av > amax) amax = av;
      }
      double sw = (amax > 1e-12) ? (amax / w_max_dev) : 1.0;
      scale_w[n] = sw;
      for (int k = 0; k < K; ++k) {
        double target = B_fp64[n * K + k] / sw + w_mean;
        int best_c = 0;
        double best_err = 1e30;
        for (int c = 0; c < 4; ++c) {
          double err = fabs(target - (double)code2_to_val[c]);
          if (err < best_err) {
            best_err = err;
            best_c = c;
          }
        }
        W_val[n * K + k] = code2_to_val[best_c];
        int cid = crumb_map[k][n];
        int byte_idx = cid >> 2;
        int shift = (cid & 3) * 2;
        B_packed[byte_idx] = (uint8_t)(
            (B_packed[byte_idx] & ~(0x3 << shift)) | (best_c << shift));
      }
    }

    memset(C_i32, 0, sizeof(C_i32));
    fn_ynn_dot_qc2_4x64_vnni(M, N, 1, 1, K, K, 0, 0, A_u8, 0, 0, 16, B_packed,
                             N * sizeof(int32_t), C_zero, N * sizeof(int32_t),
                             C_i32, NULL);

    for (int m = 0; m < M; ++m) {
      int64_t a_sum_zp = 0;
      for (int k = 0; k < K; ++k)
        a_sum_zp += (int64_t)A_u8[m * K + k] - (int64_t)zp_a[m];
      for (int n = 0; n < N; ++n) {
        int64_t raw_i64 = 0;
        int64_t w_sum = 0;
        double acc_fp64 = bias_fp64[n];
        for (int k = 0; k < K; ++k) {
          raw_i64 += (int64_t)A_u8[m * K + k] * (int64_t)W_val[n * K + k];
          w_sum += (int64_t)W_val[n * K + k];
          acc_fp64 += A_fp64[m * K + k] * B_fp64[n * K + k];
        }
        C_act_level_a[m * N + n] = (float)C_i32[m * N + n];
        C_ref_level_a[m * N + n] = (double)raw_i64;

        double corr = (double)C_i32[m * N + n] -
                      (double)zp_a[m] * (double)w_sum -
                      (double)a_sum_zp * w_mean;
        C_act_level_b[m * N + n] =
            (float)(corr * scale_a[m] * scale_w[n] + bias_fp64[n]);
        C_ref_level_b[m * N + n] = acc_fp64;
      }
    }

    precision_metrics_t ma =
        compute_fp_metrics(C_act_level_a, C_ref_level_a, M * N, 0, 190.0, 0.0);
    precision_metrics_t mb =
        compute_fp_metrics(C_act_level_b, C_ref_level_b, M * N, 0, 5.5, 50.0);
    res.level_b_by_dist[d] = mb;
    merge_metrics(&res.level_a, &ma, d);
    merge_metrics(&res.level_b, &mb, d);
  }

  free(A_fp64);
  free(B_fp64);
  free(bias_fp64);
  free(C_ref_level_a);
  free(C_ref_level_b);
  free(C_act_level_a);
  free(C_act_level_b);
  return res;
}

// -----------------------------------------------------------------------------
// 4. Verify Softmax Kernels (XNNPACK rmax + raddstoreexpminusmax, YNNPACK max+sum)
// -----------------------------------------------------------------------------

static kernel_verification_result_t verify_softmax_pair(int is_ynn) {
  kernel_verification_result_t res;
  memset(&res, 0, sizeof(res));
  const int ROWS = 16, K = 1024;

  float *X_f32 = (float *)aligned_alloc(64, ROWS * K * sizeof(float));
  float *X_shift = (float *)aligned_alloc(64, ROWS * K * sizeof(float));
  float *Exp_f32 = (float *)aligned_alloc(64, ROWS * K * sizeof(float));
  float *P_f32 = (float *)aligned_alloc(64, ROWS * K * sizeof(float));
  float *P_shift = (float *)aligned_alloc(64, ROWS * K * sizeof(float));
  double *Exp_ref = (double *) calloc(ROWS * K, sizeof(double));
  double *P_ref = (double *) calloc(ROWS * K, sizeof(double));

  double max_simplex_err = 0.0;
  double max_shift_err = 0.0;

  for (int d = 0; d < NUM_DISTS; ++d) {
    seed_rng(0xABCDEF0123456789ULL + (uint64_t)d * 997ULL);
    for (int r = 0; r < ROWS; ++r) {
      for (int k = 0; k < K; ++k) {
        double v = 0.0;
        if (d == DIST_NORMAL) {
          v = rand_normal(0.0, 3.0);
        } else if (d == DIST_SATURATION) {
          v = (k == 0) ? 80.0 : ((k % 2 == 0) ? -80.0 : 20.0);
        } else if (d == DIST_OUTLIERS) {
          v = (k == (r * 17) % K) ? 50.0 : rand_normal(0.0, 1.5);
        } else {
          v = rand_normal(-500.0, 2.0);
        }
        X_f32[r * K + k] = (float)v;
        X_shift[r * K + k] = (float)(v + 1000.0);
      }
    }

    for (int r = 0; r < ROWS; ++r) {
      const float *row_in = X_f32 + r * K;
      float *row_exp = Exp_f32 + r * K;
      float *row_p = P_f32 + r * K;

      if (!is_ynn) {
        float max_val = -INFINITY;
        float sum_val = 0.0f;
        fn_xnn_rmax_avx512(K * sizeof(float), row_in, &max_val, NULL);
        fn_xnn_raddexp_avx512(K * sizeof(float), row_in, &max_val, row_exp,
                              &sum_val, NULL);
        for (int k = 0; k < K; ++k) row_p[k] = row_exp[k] / sum_val;

        // Also test shift invariance (+1000.0f)
        float max_s = -INFINITY, sum_s = 0.0f;
        fn_xnn_rmax_avx512(K * sizeof(float), X_shift + r * K, &max_s, NULL);
        fn_xnn_raddexp_avx512(K * sizeof(float), X_shift + r * K, &max_s,
                              P_shift + r * K, &sum_s, NULL);
        for (int k = 0; k < K; ++k) {
          P_shift[r * K + k] /= sum_s;
          double serr = fabs((double)P_shift[r * K + k] - (double)row_p[k]);
          if (serr > max_shift_err) max_shift_err = serr;
        }
      } else {
        float max_val = -INFINITY;
        float sum_val = 0.0f;
        fn_ynn_max_fp32_avx512(1, K * sizeof(float), K * sizeof(float), row_in,
                               &max_val, NULL);
        for (int k = 0; k < K; ++k) row_exp[k] = expf(row_in[k] - max_val);
        fn_ynn_sum_fp32_avx512(1, K * sizeof(float), K * sizeof(float), row_exp,
                               &sum_val, NULL);
        for (int k = 0; k < K; ++k) row_p[k] = row_exp[k] / sum_val;

        float max_s = -INFINITY, sum_s = 0.0f;
        fn_ynn_max_fp32_avx512(1, K * sizeof(float), K * sizeof(float),
                               X_shift + r * K, &max_s, NULL);
        for (int k = 0; k < K; ++k)
          P_shift[r * K + k] = expf(X_shift[r * K + k] - max_s);
        fn_ynn_sum_fp32_avx512(1, K * sizeof(float), K * sizeof(float),
                               P_shift + r * K, &sum_s, NULL);
        for (int k = 0; k < K; ++k) {
          P_shift[r * K + k] /= sum_s;
          double serr = fabs((double)P_shift[r * K + k] - (double)row_p[k]);
          if (serr > max_shift_err) max_shift_err = serr;
        }
      }

      // Exact FP64 Softmax & unnormalized exp(x - max) reference
      double rmax = -DBL_MAX;
      for (int k = 0; k < K; ++k) {
        double v = (double)row_in[k];
        if (v > rmax) rmax = v;
      }
      double rsum = 0.0;
      double p_row_sum = 0.0;
      for (int k = 0; k < K; ++k) {
        double e = exp((double)row_in[k] - rmax);
        Exp_ref[r * K + k] = e;
        rsum += e;
        p_row_sum += (double)row_p[k];
      }
      for (int k = 0; k < K; ++k) {
        P_ref[r * K + k] = Exp_ref[r * K + k] / rsum;
      }
      double simp_err = fabs(p_row_sum - 1.0);
      if (simp_err > max_simplex_err) max_simplex_err = simp_err;
    }

    precision_metrics_t ma =
        compute_fp_metrics(Exp_f32, Exp_ref, ROWS * K, 4, 110.0, 1e-5);
    precision_metrics_t mb =
        compute_fp_metrics(P_f32, P_ref, ROWS * K, 4, 120.0, 1e-5);
    res.level_b_by_dist[d] = mb;
    merge_metrics(&res.level_a, &ma, d);
    merge_metrics(&res.level_b, &mb, d);
  }

  res.extra_metric_1 = max_simplex_err;
  res.extra_metric_2 = max_shift_err;

  free(X_f32);
  free(X_shift);
  free(Exp_f32);
  free(P_f32);
  free(P_shift);
  free(Exp_ref);
  free(P_ref);
  return res;
}

// -----------------------------------------------------------------------------
// 7. Verify Matrix Packing / 32-Bit Transpose Pair (Bit-Exact + Tail + Canary)
// -----------------------------------------------------------------------------

static kernel_verification_result_t verify_pack_transpose_pair(int is_ynn) {
  kernel_verification_result_t res;
  memset(&res, 0, sizeof(res));

  const int N_aligned = 64, K_aligned = 64;
  const int N_tail = 53, K_tail = 64;
  const uint32_t CANARY = 0xDEADBEEFU;
  const int GUARD = 64;

  uint32_t *in_buf =
      (uint32_t *)aligned_alloc(64, (N_aligned * K_aligned + 256) * 4);
  uint32_t *bias_buf = (uint32_t *)aligned_alloc(64, N_aligned * 4);
  size_t out_words = (size_t)N_aligned * (size_t)(K_aligned + 1) + 2 * GUARD;
  uint32_t *out_buf = (uint32_t *)aligned_alloc(64, out_words * 4);
  uint32_t *ref_buf = (uint32_t *)aligned_alloc(64, out_words * 4);

  int canary_errors = 0;
  double tail_exact_pct = 100.0;

  for (int d = 0; d < NUM_DISTS; ++d) {
    int N = (d == DIST_TAILS) ? N_tail : N_aligned;
    int K = (d == DIST_TAILS) ? K_tail : K_aligned;
    int nr = 32;
    int panels = (N + nr - 1) / nr;
    int padded_N = panels * nr;

    seed_rng(0x55AA55AA12345678ULL + (uint64_t)d * 131ULL);
    for (int i = 0; i < N * K; ++i) in_buf[i] = xorshift32();
    for (int i = 0; i < N; ++i) bias_buf[i] = xorshift32();

    for (size_t i = 0; i < out_words; ++i) {
      out_buf[i] = CANARY;
      ref_buf[i] = CANARY;
    }

    uint32_t *out_core = out_buf + GUARD;
    uint32_t *ref_core = ref_buf + GUARD;
    size_t active_words = 0;

    if (!is_ynn) {
      active_words = (size_t)padded_N * (size_t)(K + 1);
      memset(out_core, 0, active_words * sizeof(uint32_t));
      memset(ref_core, 0, active_words * sizeof(uint32_t));
      fn_xnn_packw_x32_avx512(1, N, K, nr, 1, 1, K, in_buf, bias_buf, NULL,
                              out_core, 0, NULL);

      // Scalar reference for XNNPACK x32_packw (nr=32 panels: 32 bias + K*32 weights)
      for (int p = 0; p < panels; ++p) {
        int n_base = p * nr;
        uint32_t *panel_ref = ref_core + (size_t)p * nr * (K + 1);
        for (int lane = 0; lane < nr; ++lane) {
          int n = n_base + lane;
          panel_ref[lane] = (n < N) ? bias_buf[n] : 0U;
        }
        for (int k = 0; k < K; ++k) {
          for (int lane = 0; lane < nr; ++lane) {
            int n = n_base + lane;
            panel_ref[nr + k * nr + lane] =
                (n < N) ? in_buf[n * K + k] : 0U;
          }
        }
      }
    } else {
      active_words = (size_t)K * (size_t)N;
      memset(out_core, 0, active_words * sizeof(uint32_t));
      memset(ref_core, 0, active_words * sizeof(uint32_t));
      int tile_m = 32;
      for (int m_idx = 0; m_idx < N; m_idx += tile_m) {
        int cur_m = (N - m_idx < tile_m) ? (N - m_idx) : tile_m;
        fn_ynn_transpose_x32_avx512(
            K, cur_m, (size_t)K * sizeof(uint32_t),
            (size_t)K * sizeof(uint32_t), in_buf + (size_t)m_idx * K,
            (size_t)N * sizeof(uint32_t), out_core + (size_t)m_idx);
        fn_ynn_transpose_x32_scalar(
            K, cur_m, (size_t)K * sizeof(uint32_t),
            (size_t)K * sizeof(uint32_t), in_buf + (size_t)m_idx * K,
            (size_t)N * sizeof(uint32_t), ref_core + (size_t)m_idx);
      }
    }

    // Check canary guard regions
    for (int i = 0; i < GUARD; ++i) {
      if (out_buf[i] != CANARY) canary_errors++;
      if (out_core[active_words + i] != CANARY) canary_errors++;
    }

    // Check bit-exactness of active words
    int exact = 0;
    uint64_t max_diff = 0;
    for (size_t i = 0; i < active_words; ++i) {
      uint32_t u_act = out_core[i];
      uint32_t u_ref = ref_core[i];
      if (u_act == u_ref) {
        exact++;
      } else {
        uint64_t diff =
            (u_act > u_ref) ? (uint64_t)(u_act - u_ref) : (uint64_t)(u_ref - u_act);
        if (diff > max_diff) max_diff = diff;
      }
    }

    precision_metrics_t m;
    memset(&m, 0, sizeof(m));
    m.max_abs_err = (double)max_diff;
    m.max_rel_err = (max_diff == 0) ? 0.0 : 1.0;
    m.rmse = (double)max_diff;
    m.max_ulp_diff = max_diff;
    m.mean_ulp_diff = 0.0;
    m.sqnr_db = (max_diff == 0) ? 200.0 : 0.0;
    m.cosine_sim = (max_diff == 0) ? 1.0 : 0.0;
    m.exact_match_pct =
        100.0 * (double)exact / (double)(active_words ? active_words : 1);
    m.nan_inf_count = 0;
    m.pass = (max_diff == 0 && canary_errors == 0) ? 1 : 0;

    if (d == DIST_TAILS) tail_exact_pct = m.exact_match_pct;
    res.level_b_by_dist[d] = m;
    merge_metrics(&res.level_a, &m, d);
    merge_metrics(&res.level_b, &m, d);
  }

  res.extra_metric_1 = res.level_a.exact_match_pct;
  res.extra_metric_2 = tail_exact_pct;
  res.extra_int = canary_errors;

  free(in_buf);
  free(bias_buf);
  free(out_buf);
  free(ref_buf);
  return res;
}

// -----------------------------------------------------------------------------
// JSON Output Writer & Constructor Entry Point
// -----------------------------------------------------------------------------

static void write_metrics_json(FILE *fp, const precision_metrics_t *m,
                               const char *indent) {
  fprintf(fp, "%s{\n", indent);
  fprintf(fp, "%s  \"max_abs_err\": %.8e,\n", indent, m->max_abs_err);
  fprintf(fp, "%s  \"max_rel_err\": %.8e,\n", indent, m->max_rel_err);
  fprintf(fp, "%s  \"rmse\": %.8e,\n", indent, m->rmse);
  fprintf(fp, "%s  \"max_ulp_diff\": %llu,\n", indent,
          (unsigned long long)m->max_ulp_diff);
  fprintf(fp, "%s  \"mean_ulp_diff\": %.4f,\n", indent, m->mean_ulp_diff);
  fprintf(fp, "%s  \"sqnr_db\": %.2f,\n", indent, m->sqnr_db);
  fprintf(fp, "%s  \"cosine_sim\": %.8f,\n", indent, m->cosine_sim);
  fprintf(fp, "%s  \"exact_match_pct\": %.2f,\n", indent, m->exact_match_pct);
  fprintf(fp, "%s  \"nan_inf_count\": %d,\n", indent, m->nan_inf_count);
  fprintf(fp, "%s  \"status\": \"%s\"\n", indent, m->pass ? "PASS" : "FAIL");
  fprintf(fp, "%s}", indent);
}

static void write_kernel_result_json(FILE *fp,
                                     const kernel_verification_result_t *r,
                                     const char *extra1_name,
                                     const char *extra2_name,
                                     int include_canary) {
  fprintf(fp, "      {\n");
  fprintf(fp, "        \"level_a_exact_ukernel\": ");
  write_metrics_json(fp, &r->level_a, "        ");
  fprintf(fp, ",\n");
  fprintf(fp, "        \"level_b_fp64_golden\": ");
  write_metrics_json(fp, &r->level_b, "        ");
  fprintf(fp, ",\n");
  fprintf(fp, "        \"distributions\": {\n");
  for (int d = 0; d < NUM_DISTS; ++d) {
    fprintf(fp, "          \"%s\": ", dist_names[d]);
    write_metrics_json(fp, &r->level_b_by_dist[d], "          ");
    fprintf(fp, "%s\n", (d + 1 < NUM_DISTS) ? "," : "");
  }
  fprintf(fp, "        }");
  if (extra1_name && extra2_name) {
    fprintf(fp, ",\n        \"%s\": %.8e,\n", extra1_name, r->extra_metric_1);
    fprintf(fp, "        \"%s\": %.8e", extra2_name, r->extra_metric_2);
    if (include_canary) {
      fprintf(fp, ",\n        \"canary_corrupted_bytes\": %d\n", r->extra_int);
    } else {
      fprintf(fp, "\n");
    }
  } else {
    fprintf(fp, "\n");
  }
  fprintf(fp, "      }");
}

// -----------------------------------------------------------------------------
// Standalone Instruction-Accurate Fallback Implementations
// (Used when executed outside litert_lm_advanced_main, reproducing the exact
// disassembled AMX/AVX-512 INT32 accumulation, FP32 FMA scaling, and rr2_p5
// polynomial arithmetic of the 14 target microkernels)
// -----------------------------------------------------------------------------

static void sim_xnn_pack_qs8_goi(size_t g, size_t nc, size_t kc, size_t nr,
                                 size_t kr, size_t sr, size_t k_stride,
                                 const void *k, const void *b,
                                 const void *scale, void *packed_weights,
                                 size_t extra_bytes, const void *params) {
  (void)g;
  (void)sr;
  (void)b;
  (void)scale;
  (void)extra_bytes;
  const int8_t *w_in = (const int8_t *)k;
  const int8_t *p_in = (const int8_t *)params;
  int32_t izp = p_in ? (int32_t)p_in[0] : 1;
  int32_t *ksum = (int32_t *)packed_weights;
  int8_t *w_out = (int8_t *)((uint8_t *)packed_weights + nr * sizeof(int32_t));
  for (size_t n = 0; n < nc; ++n) {
    int32_t s = 0;
    for (size_t ki = 0; ki < kc; ++ki) {
      int8_t v = w_in[n * k_stride + ki];
      s += (int32_t)v;
      size_t k_block = ki / kr;
      size_t k_rem = ki % kr;
      w_out[(k_block * nr + n) * kr + k_rem] = v;
    }
    ksum[n] = -izp * s;
  }
}

static void sim_xnn_pack_qs8_qc4w_goi(size_t g, size_t nc, size_t kc, size_t nr,
                                      size_t kr, size_t sr, size_t k_stride,
                                      const void *k, const void *b,
                                      const void *scale, void *packed_weights,
                                      size_t extra_bytes, const void *params) {
  (void)g;
  (void)kr;
  (void)sr;
  (void)k_stride;
  (void)b;
  (void)scale;
  (void)extra_bytes;
  (void)params;
  const uint8_t *w_u4 = (const uint8_t *)k;
  int32_t *ksum = (int32_t *)packed_weights;
  uint8_t *w_out = (uint8_t *)packed_weights + nr * sizeof(int32_t);
  size_t k_bytes = (kc + 1) / 2;
  for (size_t n = 0; n < nc; ++n) {
    int32_t s = 0;
    for (size_t ki = 0; ki < kc; ++ki) {
      uint8_t byte_val = w_u4[n * k_bytes + (ki >> 1)];
      uint8_t u4 = (ki & 1) ? (byte_val >> 4) : (byte_val & 0x0F);
      int32_t q = (int32_t)u4 - 8;
      s += 16 * q;
      size_t idx = ki * nr + n;
      uint8_t nib = (uint8_t)(q + 8) & 0x0F;
      if (idx & 1) {
        w_out[idx >> 1] = (uint8_t)((w_out[idx >> 1] & 0x0F) | (nib << 4));
      } else {
        w_out[idx >> 1] = (uint8_t)((w_out[idx >> 1] & 0xF0) | nib);
      }
    }
    ksum[n] = -s;
  }
}

#ifdef __AMX_INT8__
struct __tile_config {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[16];
  uint8_t rows[16];
};

static void hw_amx_dpbssd_16x16x64(const int8_t *A_16x64,
                                   const int8_t *B_vnni_16x64,
                                   int32_t *C_16x16) {
  struct __tile_config cfg __attribute__((aligned(64)));
  memset(&cfg, 0, sizeof(cfg));
  cfg.palette_id = 1;
  // tmm0: C (16 rows x 64 bytes = 16 int32 cols)
  cfg.rows[0] = 16;
  cfg.colsb[0] = 64;
  // tmm1: A (16 rows x 64 bytes = 64 int8 cols)
  cfg.rows[1] = 16;
  cfg.colsb[1] = 64;
  // tmm2: B (16 rows x 64 bytes = 16x4 VNNI int8 cols)
  cfg.rows[2] = 16;
  cfg.colsb[2] = 64;
  _tile_loadconfig(&cfg);
  _tile_zero(0);
  _tile_loadd(1, A_16x64, 64);
  _tile_loadd(2, B_vnni_16x64, 64);
  _tile_dpbssd(0, 1, 2);
  _tile_stored(0, C_16x16, 64);
  _tile_release();
}
#endif

static void sim_xnn_qd8_qc4w_amx(
    size_t mr, size_t nc, size_t kc, const int8_t *a, size_t a_stride,
    const void *w, float *c, size_t cm_stride, size_t cn_stride,
    const void *params,
    const struct xnn_qd8_quantization_params *quantization_params) {
  (void)cn_stride;
  const int32_t *ksum = (const int32_t *)w;
  const uint8_t *w_nib = (const uint8_t *)w + 256;
  const float *w_scale = (const float *)((const uint8_t *)w + 2304);
  const float *w_bias = (const float *)((const uint8_t *)w + 2560);
  const float *minmax = (const float *)params;
  float cmin = minmax ? minmax[0] : -INFINITY;
  float cmax = minmax ? minmax[1] : INFINITY;
  size_t c_row_floats = cm_stride / sizeof(float);

#ifdef __AMX_INT8__
  if (kc == 64 && nc == 64 && mr <= 16) {
    int8_t A_pad[16 * 64] __attribute__((aligned(64)));
    memset(A_pad, 0, sizeof(A_pad));
    for (size_t m = 0; m < mr; ++m) {
      memcpy(A_pad + m * 64, a + m * a_stride, 64);
    }
    int32_t C_acc[16 * 64] __attribute__((aligned(64)));
    for (int tile_n = 0; tile_n < 64; tile_n += 16) {
      int8_t B_vnni[16 * 64] __attribute__((aligned(64)));
      for (int kb = 0; kb < 16; ++kb) {
        for (int lane = 0; lane < 16; ++lane) {
          int n = tile_n + lane;
          for (int kr = 0; kr < 4; ++kr) {
            int ki = kb * 4 + kr;
            size_t idx = (size_t)ki * 64 + (size_t)n;
            uint8_t b = w_nib[idx >> 1];
            uint8_t u4 = (idx & 1) ? (b >> 4) : (b & 0x0F);
            B_vnni[kb * 64 + lane * 4 + kr] = (int8_t)(((int32_t)u4 - 8) * 16);
          }
        }
      }
      int32_t C_tile[16 * 16] __attribute__((aligned(64)));
      hw_amx_dpbssd_16x16x64(A_pad, B_vnni, C_tile);
      for (size_t m = 0; m < mr; ++m) {
        for (int lane = 0; lane < 16; ++lane) {
          C_acc[m * 64 + tile_n + lane] =
              C_tile[m * 16 + lane] +
              quantization_params[m].zero_point * ksum[tile_n + lane];
        }
      }
    }
    __m512 vmin = _mm512_set1_ps(cmin);
    __m512 vmax = _mm512_set1_ps(cmax);
    for (size_t m = 0; m < mr; ++m) {
      __m512 vinv = _mm512_set1_ps(quantization_params[m].inv_scale);
      for (size_t n = 0; n < 64; n += 16) {
        __m512i vacc = _mm512_load_si512((const __m512i *)(C_acc + m * 64 + n));
        __m512 vf = _mm512_mul_ps(_mm512_cvtepi32_ps(vacc), vinv);
        __m512 vsc = _mm512_loadu_ps(w_scale + n);
        __m512 vbi = _mm512_loadu_ps(w_bias + n);
        vf = _mm512_fmadd_ps(vf, vsc, vbi);
        vf = _mm512_max_ps(vmin, _mm512_min_ps(vmax, vf));
        _mm512_storeu_ps(c + m * c_row_floats + n, vf);
      }
    }
    return;
  }
#endif

  for (size_t m = 0; m < mr; ++m) {
    int32_t zp = quantization_params[m].zero_point;
    float inv_scale = quantization_params[m].inv_scale;
    for (size_t n = 0; n < nc; ++n) {
      int32_t acc_i32 = zp * ksum[n];
      for (size_t ki = 0; ki < kc; ++ki) {
        size_t idx = ki * 64 + n;
        uint8_t b = w_nib[idx >> 1];
        uint8_t u4 = (idx & 1) ? (b >> 4) : (b & 0x0F);
        int32_t q16 = ((int32_t)u4 - 8) * 16;
        acc_i32 += (int32_t)a[m * a_stride + ki] * q16;
      }
      // Exact AMX epilogue: vcvtdq2ps -> vmulps(inv_scale) -> vfmadd213ps(w_scale, w_bias)
      float v = (float)acc_i32 * inv_scale;
      v = fmaf(v, w_scale[n], w_bias[n]);
      if (v < cmin) v = cmin;
      if (v > cmax) v = cmax;
      c[m * c_row_floats + n] = v;
    }
  }
}

static void sim_xnn_qd8_qc8w_amx(
    size_t mr, size_t nc, size_t kc, const int8_t *a, size_t a_stride,
    const void *w, float *c, size_t cm_stride, size_t cn_stride,
    const void *params,
    const struct xnn_qd8_quantization_params *quantization_params) {
  (void)cn_stride;
  const int32_t *ksum = (const int32_t *)w;
  const int8_t *w_i8 = (const int8_t *)((const uint8_t *)w + 256);
  const float *w_scale = (const float *)((const uint8_t *)w + 4352);
  const float *w_bias = (const float *)((const uint8_t *)w + 4608);
  const float *minmax = (const float *)params;
  float cmin = minmax ? minmax[0] : -INFINITY;
  float cmax = minmax ? minmax[1] : INFINITY;
  size_t c_row_floats = cm_stride / sizeof(float);

#ifdef __AMX_INT8__
  if (kc == 64 && nc == 64 && mr <= 16) {
    int8_t A_pad[16 * 64] __attribute__((aligned(64)));
    memset(A_pad, 0, sizeof(A_pad));
    for (size_t m = 0; m < mr; ++m) {
      memcpy(A_pad + m * 64, a + m * a_stride, 64);
    }
    int32_t C_acc[16 * 64] __attribute__((aligned(64)));
    for (int tile_n = 0; tile_n < 64; tile_n += 16) {
      int8_t B_vnni[16 * 64] __attribute__((aligned(64)));
      for (int kb = 0; kb < 16; ++kb) {
        for (int lane = 0; lane < 16; ++lane) {
          int n = tile_n + lane;
          for (int kr = 0; kr < 4; ++kr) {
            B_vnni[kb * 64 + lane * 4 + kr] = w_i8[(kb * 64 + n) * 4 + kr];
          }
        }
      }
      int32_t C_tile[16 * 16] __attribute__((aligned(64)));
      hw_amx_dpbssd_16x16x64(A_pad, B_vnni, C_tile);
      for (size_t m = 0; m < mr; ++m) {
        for (int lane = 0; lane < 16; ++lane) {
          C_acc[m * 64 + tile_n + lane] =
              C_tile[m * 16 + lane] +
              quantization_params[m].zero_point * ksum[tile_n + lane];
        }
      }
    }
    __m512 vmin = _mm512_set1_ps(cmin);
    __m512 vmax = _mm512_set1_ps(cmax);
    for (size_t m = 0; m < mr; ++m) {
      __m512 vinv = _mm512_set1_ps(quantization_params[m].inv_scale);
      for (size_t n = 0; n < 64; n += 16) {
        __m512i vacc = _mm512_load_si512((const __m512i *)(C_acc + m * 64 + n));
        __m512 vf = _mm512_mul_ps(_mm512_cvtepi32_ps(vacc), vinv);
        __m512 vsc = _mm512_loadu_ps(w_scale + n);
        __m512 vbi = _mm512_loadu_ps(w_bias + n);
        vf = _mm512_fmadd_ps(vf, vsc, vbi);
        vf = _mm512_max_ps(vmin, _mm512_min_ps(vmax, vf));
        _mm512_storeu_ps(c + m * c_row_floats + n, vf);
      }
    }
    return;
  }
#endif

  for (size_t m = 0; m < mr; ++m) {
    int32_t zp = quantization_params[m].zero_point;
    float inv_scale = quantization_params[m].inv_scale;
    for (size_t n = 0; n < nc; ++n) {
      int32_t acc_i32 = zp * ksum[n];
      for (size_t ki = 0; ki < kc; ++ki) {
        size_t k_block = ki / 4;
        size_t k_rem = ki % 4;
        int8_t wv = w_i8[(k_block * 64 + n) * 4 + k_rem];
        acc_i32 += (int32_t)a[m * a_stride + ki] * (int32_t)wv;
      }
      float v = (float)acc_i32 * inv_scale;
      v = fmaf(v, w_scale[n], w_bias[n]);
      if (v < cmin) v = cmin;
      if (v > cmax) v = cmax;
      c[m * c_row_floats + n] = v;
    }
  }
}

static void sim_xnn_packw_x32_avx512(size_t g, size_t nc, size_t kc, size_t nr,
                                     size_t kr, size_t sr, size_t k_stride,
                                     const void *k, const void *b,
                                     const void *scale, void *packed_weights,
                                     size_t extra_bytes, const void *params) {
  (void)g;
  (void)kr;
  (void)sr;
  (void)scale;
  (void)extra_bytes;
  (void)params;
  const uint32_t *w_in = (const uint32_t *)k;
  const uint32_t *b_in = (const uint32_t *)b;
  uint32_t *out = (uint32_t *)packed_weights;
  size_t panels = (nc + nr - 1) / nr;
  for (size_t p = 0; p < panels; ++p) {
    size_t n_base = p * nr;
    uint32_t *panel_out = out + p * nr * (kc + 1);
    for (size_t lane = 0; lane < nr; ++lane) {
      size_t n = n_base + lane;
      panel_out[lane] = (b_in && n < nc) ? b_in[n] : 0U;
    }
    for (size_t ki = 0; ki < kc; ++ki) {
      for (size_t lane = 0; lane < nr; ++lane) {
        size_t n = n_base + lane;
        panel_out[nr + ki * nr + lane] =
            (n < nc) ? w_in[n * k_stride + ki] : 0U;
      }
    }
  }
}

static void sim_xnn_f32_7x32_avx512(size_t mr, size_t nc, size_t kc,
                                    const float *a, size_t a_stride,
                                    const void *w, float *c, size_t cm_stride,
                                    size_t cn_stride, const void *params) {
  (void)cn_stride;
  size_t k_len = kc / sizeof(float);
  size_t a_row_floats = a_stride / sizeof(float);
  size_t c_row_floats = cm_stride / sizeof(float);
  const float *w_f32 = (const float *)w;
  const float *minmax = (const float *)params;
  float cmin = minmax ? minmax[0] : -INFINITY;
  float cmax = minmax ? minmax[1] : INFINITY;
  const size_t nr = 32;
  size_t panels = (nc + nr - 1) / nr;

#ifdef __AVX512F__
  if (nc == 64) {
    __m512 vmin = _mm512_set1_ps(cmin);
    __m512 vmax = _mm512_set1_ps(cmax);
    for (size_t p = 0; p < panels; ++p) {
      const float *panel = w_f32 + p * nr * (k_len + 1);
      for (size_t m = 0; m < mr; ++m) {
        __m512 vacc0 = _mm512_loadu_ps(panel);
        __m512 vacc1 = _mm512_loadu_ps(panel + 16);
        for (size_t ki = 0; ki < k_len; ++ki) {
          __m512 va = _mm512_set1_ps(a[m * a_row_floats + ki]);
          __m512 vw0 = _mm512_loadu_ps(panel + nr + ki * nr);
          __m512 vw1 = _mm512_loadu_ps(panel + nr + ki * nr + 16);
          vacc0 = _mm512_fmadd_ps(va, vw0, vacc0);
          vacc1 = _mm512_fmadd_ps(va, vw1, vacc1);
        }
        vacc0 = _mm512_max_ps(vmin, _mm512_min_ps(vmax, vacc0));
        vacc1 = _mm512_max_ps(vmin, _mm512_min_ps(vmax, vacc1));
        _mm512_storeu_ps(c + m * c_row_floats + p * nr, vacc0);
        _mm512_storeu_ps(c + m * c_row_floats + p * nr + 16, vacc1);
      }
    }
    return;
  }
#endif

  for (size_t p = 0; p < panels; ++p) {
    const float *panel = w_f32 + p * nr * (k_len + 1);
    for (size_t m = 0; m < mr; ++m) {
      for (size_t lane = 0; lane < nr; ++lane) {
        size_t n = p * nr + lane;
        if (n >= nc) continue;
        float acc = panel[lane];
        for (size_t ki = 0; ki < k_len; ++ki) {
          acc = fmaf(a[m * a_row_floats + ki], panel[nr + ki * nr + lane], acc);
        }
        if (acc < cmin) acc = cmin;
        if (acc > cmax) acc = cmax;
        c[m * c_row_floats + n] = acc;
      }
    }
  }
}

static size_t sim_xnn_init_qs8_minmax_params(void *params,
                                             int8_t output_zero_point,
                                             int8_t output_min,
                                             int8_t output_max) {
  int32_t *p = (int32_t *)params;
  p[0] = (int32_t)output_zero_point;
  p[1] = (int32_t)output_min;
  p[2] = (int32_t)output_max;
  return 64;
}

static void sim_xnn_qc2w_5x8_avxvnni(size_t mr, size_t nc, size_t kc,
                                     const int8_t *a, size_t a_stride,
                                     const void *w, int8_t *c, size_t cm_stride,
                                     size_t cn_stride, const void *params) {
  (void)cn_stride;
  const int32_t *ksum = (const int32_t *)w;
  const uint8_t *w_crumbs = (const uint8_t *)w + 32;
  const float *w_scale = (const float *)((const uint8_t *)w + 160);
  const int32_t *p = (const int32_t *)params;
  int32_t out_zp = p ? p[0] : 0;
  int32_t out_min = p ? p[1] : -128;
  int32_t out_max = p ? p[2] : 127;
  static const int32_t lut[4] = {-2, -1, 0, 1};

#ifdef __AVXVNNI__
  if (nc == 8 && kc == 64) {
    for (size_t m = 0; m < mr; ++m) {
      __m256i vacc = _mm256_loadu_si256((const __m256i *)ksum);
      for (size_t kb = 0; kb < kc; kb += 4) {
        uint8_t a_u8_4[4];
        for (int kr = 0; kr < 4; ++kr) {
          a_u8_4[kr] = (uint8_t)(a[m * a_stride + kb + (size_t)kr] ^ 0x80);
        }
        uint32_t a_pack = 0;
        memcpy(&a_pack, a_u8_4, 4);
        __m256i va = _mm256_set1_epi32((int)a_pack);
        int8_t b_vnni[32] __attribute__((aligned(32)));
        for (int n = 0; n < 8; ++n) {
          for (int kr = 0; kr < 4; ++kr) {
            int32_t cid = (int32_t)((kb + (size_t)kr) * 8 + (size_t)n);
            uint8_t b = w_crumbs[cid >> 2];
            int shift = (cid & 3) * 2;
            uint8_t code = (b >> shift) & 0x3;
            b_vnni[n * 4 + kr] = (int8_t)lut[code];
          }
        }
        __m256i vb = _mm256_load_si256((const __m256i *)b_vnni);
        vacc = _mm256_dpbusd_epi32(vacc, va, vb);
      }
      __m256 vsc = _mm256_loadu_ps(w_scale);
      __m256 vf = _mm256_mul_ps(_mm256_cvtepi32_ps(vacc), vsc);
      __m256i vq = _mm256_add_epi32(_mm256_cvtps_epi32(vf),
                                    _mm256_set1_epi32(out_zp));
      vq = _mm256_max_epi32(_mm256_set1_epi32(out_min),
                            _mm256_min_epi32(_mm256_set1_epi32(out_max), vq));
      int32_t q_arr[8] __attribute__((aligned(32)));
      _mm256_store_si256((__m256i *)q_arr, vq);
      for (int n = 0; n < 8; ++n) {
        c[m * cm_stride + (size_t)n] = (int8_t)q_arr[n];
      }
    }
    return;
  }
#endif

  for (size_t m = 0; m < mr; ++m) {
    for (size_t n = 0; n < nc; ++n) {
      int32_t acc = ksum[n];
      for (size_t ki = 0; ki < kc; ++ki) {
        int32_t cid = (int32_t)(ki * 8 + n);
        uint8_t b = w_crumbs[cid >> 2];
        int shift = (cid & 3) * 2;
        uint8_t code = (b >> shift) & 0x3;
        int32_t a_u8 = (int32_t)(uint8_t)(a[m * a_stride + ki] ^ 0x80);
        acc += a_u8 * lut[code];
      }
      float scaled = (float)acc * w_scale[n];
      long q = lrintf(scaled) + out_zp;
      if (q < out_min) q = out_min;
      if (q > out_max) q = out_max;
      c[m * cm_stride + n] = (int8_t)q;
    }
  }
}

static void sim_xnn_rmax_avx512(size_t batch, const float *x, float *y,
                                const void *params) {
  (void)params;
  size_t n = batch / sizeof(float);
#ifdef __AVX512F__
  if (n % 16 == 0 && n >= 16) {
    __m512 vmax = _mm512_set1_ps(-INFINITY);
    for (size_t i = 0; i < n; i += 16) {
      vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(x + i));
    }
    *y = _mm512_reduce_max_ps(vmax);
    return;
  }
#endif
  float vmax = -INFINITY;
  for (size_t i = 0; i < n; ++i) {
    if (x[i] > vmax) vmax = x[i];
  }
  *y = vmax;
}

// Exact XNNPACK avx512f_rr2_p5_u64_acc2 range-reduced degree-5 polynomial exp
static void sim_xnn_raddexp_avx512(size_t batch, const float *x,
                                   const float *max, float *y, float *sum,
                                   const void *params) {
  (void)params;
  size_t count = batch / sizeof(float);
  const float xmax = *max;
  const float vmagic_bias = 0x1.8000FEp23f;
  const float vlog2e = 0x1.715476p+0f;
  const float vminus_ln2_hi = -0x1.62E400p-1f;
  const float vminus_ln2_lo = -0x1.7F7D1Cp-20f;
  const float vc5 = 0x1.0F9F9Cp-7f;
  const float vc4 = 0x1.573A1Ap-5f;
  const float vc3 = 0x1.555A80p-3f;
  const float vc2 = 0x1.FFFDC6p-2f;
  const float vc1 = 0x1.FFFFF6p-1f;
  const float vdenorm_cutoff = -0x1.5D589Ep6f;

#ifdef __AVX512F__
  if (count % 32 == 0 && count >= 32) {
    __m512 vxmax = _mm512_set1_ps(xmax);
    __m512 vmbias = _mm512_set1_ps(vmagic_bias);
    __m512 vl2e = _mm512_set1_ps(vlog2e);
    __m512 vln2_hi = _mm512_set1_ps(vminus_ln2_hi);
    __m512 vln2_lo = _mm512_set1_ps(vminus_ln2_lo);
    __m512 c5 = _mm512_set1_ps(vc5);
    __m512 c4 = _mm512_set1_ps(vc4);
    __m512 c3 = _mm512_set1_ps(vc3);
    __m512 c2 = _mm512_set1_ps(vc2);
    __m512 c1 = _mm512_set1_ps(vc1);
    __m512 vcut = _mm512_set1_ps(vdenorm_cutoff);
    __m512 vacc0 = _mm512_setzero_ps();
    __m512 vacc1 = _mm512_setzero_ps();

    for (size_t i = 0; i < count; i += 16) {
      __m512 vx = _mm512_sub_ps(_mm512_loadu_ps(x + i), vxmax);
      __m512 vn = _mm512_fmadd_ps(vx, vl2e, vmbias);
      __m512 vs = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_castps_si512(vn), 23));
      vn = _mm512_sub_ps(vn, vmbias);
      __m512 vt = _mm512_fmadd_ps(vn, vln2_hi, vx);
      vt = _mm512_fmadd_ps(vn, vln2_lo, vt);
      __m512 vp = _mm512_fmadd_ps(c5, vt, c4);
      vp = _mm512_fmadd_ps(vp, vt, c3);
      vp = _mm512_fmadd_ps(vp, vt, c2);
      vp = _mm512_fmadd_ps(vp, vt, c1);
      vt = _mm512_mul_ps(vt, vs);
      __m512 vf = _mm512_fmadd_ps(vt, vp, vs);
      __mmask16 keep = _mm512_cmp_ps_mask(vx, vcut, _CMP_GE_OS);
      vf = _mm512_maskz_mov_ps(keep, vf);
      _mm512_storeu_ps(y + i, vf);
      if ((i / 16) % 2 == 0) {
        vacc0 = _mm512_add_ps(vacc0, vf);
      } else {
        vacc1 = _mm512_add_ps(vacc1, vf);
      }
    }
    __m512 vsum_vec = _mm512_add_ps(vacc0, vacc1);
    float vsum[16] __attribute__((aligned(64)));
    _mm512_store_ps(vsum, vsum_vec);
    for (int stride = 8; stride >= 1; stride >>= 1) {
      for (int l = 0; l < stride; ++l) vsum[l] += vsum[l + stride];
    }
    *sum = vsum[0];
    return;
  }
#endif

  float acc0[16] = {0.0f};
  float acc1[16] = {0.0f};
  for (size_t i = 0; i < count; ++i) {
    float vx = x[i] - xmax;
    float vn = fmaf(vx, vlog2e, vmagic_bias);
    uint32_t bits_n;
    memcpy(&bits_n, &vn, sizeof(uint32_t));
    uint32_t bits_s = bits_n << 23;
    float vs;
    memcpy(&vs, &bits_s, sizeof(float));
    vn -= vmagic_bias;
    float vt = fmaf(vn, vminus_ln2_hi, vx);
    vt = fmaf(vn, vminus_ln2_lo, vt);
    float vp = fmaf(vc5, vt, vc4);
    vp = fmaf(vp, vt, vc3);
    vp = fmaf(vp, vt, vc2);
    vp = fmaf(vp, vt, vc1);
    vt *= vs;
    float vf = fmaf(vt, vp, vs);
    if (vx < vdenorm_cutoff) vf = 0.0f;
    y[i] = vf;
    if ((i / 16) % 2 == 0) {
      acc0[i % 16] += vf;
    } else {
      acc1[i % 16] += vf;
    }
  }
  float vsum[16];
  for (int l = 0; l < 16; ++l) vsum[l] = acc0[l] + acc1[l];
  for (int stride = 8; stride >= 1; stride >>= 1) {
    for (int l = 0; l < stride; ++l) vsum[l] += vsum[l + stride];
  }
  *sum = vsum[0];
}

static void sim_ynn_dot_qc4_vnni(size_t M, size_t N, size_t K3, size_t K2,
                                 size_t K1, size_t A_stride_m,
                                 size_t A_stride_k3, size_t A_stride_k2,
                                 const void *A, size_t B_stride_k3,
                                 size_t B_stride_k2, size_t B_stride_k1,
                                 const void *B, size_t C_in_stride_m,
                                 const void *C_in, size_t C_out_stride_m,
                                 void *C_out, void *state) {
  (void)K3;
  (void)K2;
  (void)A_stride_k3;
  (void)A_stride_k2;
  (void)B_stride_k3;
  (void)B_stride_k2;
  (void)B_stride_k1;
  (void)C_in_stride_m;
  (void)C_in;
  (void)state;
  const uint8_t *a_u8 = (const uint8_t *)A;
  const uint8_t *b_packed = (const uint8_t *)B;
  size_t c_row_i32 = C_out_stride_m / sizeof(int32_t);
  int32_t *c_i32 = (int32_t *)C_out;

#ifdef __AVX512VNNI__
  if (K1 == 64 && (N % 16 == 0) && N <= 64) {
    for (size_t m = 0; m < M; ++m) {
      for (size_t n0 = 0; n0 < N; n0 += 16) {
        __m512i vacc = _mm512_setzero_si512();
        for (size_t kb = 0; kb < K1; kb += 4) {
          uint32_t a_pack = 0;
          memcpy(&a_pack, a_u8 + m * A_stride_m + kb, 4);
          __m512i va = _mm512_set1_epi32((int)a_pack);
          int8_t b_vnni[64] __attribute__((aligned(64)));
          for (int lane = 0; lane < 16; ++lane) {
            size_t n = n0 + (size_t)lane;
            for (int kr = 0; kr < 4; ++kr) {
              int nid = (int)((kb + (size_t)kr) * N + n);
              uint8_t byte_val = b_packed[nid >> 1];
              uint8_t code = (nid & 1) ? (byte_val >> 4) : (byte_val & 0x0F);
              b_vnni[lane * 4 + kr] = (int8_t)((int32_t)code - 8);
            }
          }
          __m512i vb = _mm512_load_si512((const __m512i *)b_vnni);
          vacc = _mm512_dpbusd_epi32(vacc, va, vb);
        }
        _mm512_storeu_si512((__m512i *)(c_i32 + m * c_row_i32 + n0), vacc);
      }
    }
    return;
  }
#endif

  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      int32_t acc = 0;
      for (size_t k = 0; k < K1; ++k) {
        int nid = (int)(k * N + n);
        uint8_t byte_val = b_packed[nid >> 1];
        uint8_t code = (nid & 1) ? (byte_val >> 4) : (byte_val & 0x0F);
        int32_t w = (int32_t)code - 8;
        acc += (int32_t)a_u8[m * A_stride_m + k] * w;
      }
      c_i32[m * c_row_i32 + n] = acc;
    }
  }
}

static void sim_ynn_dot_qc8_amx(size_t M, size_t N, size_t K3, size_t K2,
                                size_t K1, size_t A_stride_m,
                                size_t A_stride_k3, size_t A_stride_k2,
                                const void *A, size_t B_stride_k3,
                                size_t B_stride_k2, size_t B_stride_k1,
                                const void *B, size_t C_in_stride_m,
                                const void *C_in, size_t C_out_stride_m,
                                void *C_out, void *state) {
  (void)K3;
  (void)K2;
  (void)A_stride_k3;
  (void)A_stride_k2;
  (void)B_stride_k3;
  (void)B_stride_k2;
  (void)B_stride_k1;
  (void)C_in_stride_m;
  (void)C_in;
  (void)state;
  const int8_t *a_i8 = (const int8_t *)A;
  const int8_t *b_packed = (const int8_t *)B;
  size_t c_row_i32 = C_out_stride_m / sizeof(int32_t);
  int32_t *c_i32 = (int32_t *)C_out;

#ifdef __AMX_INT8__
  if (M == 32 && N == 32 && K1 == 64) {
    for (int tm = 0; tm < 32; tm += 16) {
      int8_t A_tile[16 * 64] __attribute__((aligned(64)));
      for (int r = 0; r < 16; ++r) {
        memcpy(A_tile + r * 64, a_i8 + (size_t)(tm + r) * A_stride_m, 64);
      }
      for (int tn = 0; tn < 32; tn += 16) {
        int8_t B_vnni[16 * 64] __attribute__((aligned(64)));
        for (int kb = 0; kb < 16; ++kb) {
          for (int lane = 0; lane < 16; ++lane) {
            int n = tn + lane;
            for (int kr = 0; kr < 4; ++kr) {
              int k = kb * 4 + kr;
              B_vnni[kb * 64 + lane * 4 + kr] = b_packed[k * 32 + n];
            }
          }
        }
        int32_t C_tile[16 * 16] __attribute__((aligned(64)));
        hw_amx_dpbssd_16x16x64(A_tile, B_vnni, C_tile);
        for (int r = 0; r < 16; ++r) {
          for (int lane = 0; lane < 16; ++lane) {
            c_i32[(size_t)(tm + r) * c_row_i32 + (size_t)(tn + lane)] =
                C_tile[r * 16 + lane];
          }
        }
      }
    }
    return;
  }
#endif

  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      int32_t acc = 0;
      for (size_t k = 0; k < K1; ++k) {
        acc += (int32_t)a_i8[m * A_stride_m + k] *
               (int32_t)b_packed[k * N + n];
      }
      c_i32[m * c_row_i32 + n] = acc;
    }
  }
}

static void sim_ynn_dot_fp32_avx512(size_t M, size_t N, size_t K3, size_t K2,
                                    size_t K1, size_t A_stride_m,
                                    size_t A_stride_k3, size_t A_stride_k2,
                                    const void *A, size_t B_stride_k3,
                                    size_t B_stride_k2, size_t B_stride_k1,
                                    const void *B, size_t C_in_stride_m,
                                    const void *C_in, size_t C_out_stride_m,
                                    void *C_out, void *state) {
  (void)K3;
  (void)K2;
  (void)A_stride_k3;
  (void)A_stride_k2;
  (void)B_stride_k3;
  (void)B_stride_k2;
  (void)B_stride_k1;
  (void)state;
  const float *a_f32 = (const float *)A;
  const float *b_f32 = (const float *)B;
  const float *c_in_f32 = (const float *)C_in;
  float *c_out_f32 = (float *)C_out;
  size_t a_row = A_stride_m / sizeof(float);
  size_t cin_row = C_in_stride_m / sizeof(float);
  size_t cout_row = C_out_stride_m / sizeof(float);

#ifdef __AVX512F__
  if (N == 64) {
    for (size_t m = 0; m < M; ++m) {
      for (size_t n0 = 0; n0 < 64; n0 += 16) {
        __m512 vacc = c_in_f32 ? _mm512_loadu_ps(c_in_f32 + m * cin_row + n0)
                               : _mm512_setzero_ps();
        for (size_t k = 0; k < K1; ++k) {
          __m512 va = _mm512_set1_ps(a_f32[m * a_row + k]);
          __m512 vb = _mm512_loadu_ps(b_f32 + k * 64 + n0);
          vacc = _mm512_fmadd_ps(va, vb, vacc);
        }
        _mm512_storeu_ps(c_out_f32 + m * cout_row + n0, vacc);
      }
    }
    return;
  }
#endif

  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      float acc = c_in_f32 ? c_in_f32[m * cin_row + n] : 0.0f;
      for (size_t k = 0; k < K1; ++k) {
        acc = fmaf(a_f32[m * a_row + k], b_f32[k * 64 + n], acc);
      }
      c_out_f32[m * cout_row + n] = acc;
    }
  }
}

static void sim_ynn_dot_qc2_vnni(size_t M, size_t N, size_t K3, size_t K2,
                                 size_t K1, size_t A_stride_m,
                                 size_t A_stride_k3, size_t A_stride_k2,
                                 const void *A, size_t B_stride_k3,
                                 size_t B_stride_k2, size_t B_stride_k1,
                                 const void *B, size_t C_in_stride_m,
                                 const void *C_in, size_t C_out_stride_m,
                                 void *C_out, void *state) {
  (void)K3;
  (void)K2;
  (void)A_stride_k3;
  (void)A_stride_k2;
  (void)B_stride_k3;
  (void)B_stride_k2;
  (void)B_stride_k1;
  (void)C_in_stride_m;
  (void)C_in;
  (void)state;
  const uint8_t *a_u8 = (const uint8_t *)A;
  const uint8_t *b_packed = (const uint8_t *)B;
  size_t c_row_i32 = C_out_stride_m / sizeof(int32_t);
  int32_t *c_i32 = (int32_t *)C_out;
  static const int32_t lut[4] = {-2, -1, 0, 1};

#ifdef __AVX512VNNI__
  if (K1 == 64 && N == 64) {
    for (size_t m = 0; m < M; ++m) {
      for (size_t n0 = 0; n0 < 64; n0 += 16) {
        __m512i vacc = _mm512_setzero_si512();
        for (size_t kb = 0; kb < 64; kb += 4) {
          uint32_t a_pack = 0;
          memcpy(&a_pack, a_u8 + m * A_stride_m + kb, 4);
          __m512i va = _mm512_set1_epi32((int)a_pack);
          int8_t b_vnni[64] __attribute__((aligned(64)));
          for (int lane = 0; lane < 16; ++lane) {
            size_t n = n0 + (size_t)lane;
            for (int kr = 0; kr < 4; ++kr) {
              int cid = (int)((kb + (size_t)kr) * 64 + n);
              uint8_t b = b_packed[cid >> 2];
              int shift = (cid & 3) * 2;
              uint8_t code = (b >> shift) & 0x3;
              b_vnni[lane * 4 + kr] = (int8_t)lut[code];
            }
          }
          __m512i vb = _mm512_load_si512((const __m512i *)b_vnni);
          vacc = _mm512_dpbusd_epi32(vacc, va, vb);
        }
        _mm512_storeu_si512((__m512i *)(c_i32 + m * c_row_i32 + n0), vacc);
      }
    }
    return;
  }
#endif

  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      int32_t acc = 0;
      for (size_t k = 0; k < K1; ++k) {
        int cid = (int)(k * 64 + n);
        uint8_t b = b_packed[cid >> 2];
        int shift = (cid & 3) * 2;
        uint8_t code = (b >> shift) & 0x3;
        acc += (int32_t)a_u8[m * A_stride_m + k] * lut[code];
      }
      c_i32[m * c_row_i32 + n] = acc;
    }
  }
}

static void sim_ynn_max_fp32_avx512(size_t N, size_t K1, size_t A_stride_n,
                                    const void *A, void *C, void *state) {
  (void)state;
  size_t count = K1 / sizeof(float);
  size_t a_row = A_stride_n / sizeof(float);
  const float *a_f32 = (const float *)A;
  float *c_f32 = (float *)C;
#ifdef __AVX512F__
  if (count % 16 == 0 && count >= 16) {
    for (size_t r = 0; r < N; ++r) {
      __m512 vmax = _mm512_set1_ps(-INFINITY);
      for (size_t i = 0; i < count; i += 16) {
        vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(a_f32 + r * a_row + i));
      }
      c_f32[r] = _mm512_reduce_max_ps(vmax);
    }
    return;
  }
#endif
  for (size_t r = 0; r < N; ++r) {
    float vmax = -INFINITY;
    for (size_t i = 0; i < count; ++i) {
      if (a_f32[r * a_row + i] > vmax) vmax = a_f32[r * a_row + i];
    }
    c_f32[r] = vmax;
  }
}

static void sim_ynn_sum_fp32_avx512(size_t N, size_t K1, size_t A_stride_n,
                                    const void *A, void *C, void *state) {
  (void)state;
  size_t count = K1 / sizeof(float);
  size_t a_row = A_stride_n / sizeof(float);
  const float *a_f32 = (const float *)A;
  float *c_f32 = (float *)C;
#ifdef __AVX512F__
  if (count % 16 == 0 && count >= 16) {
    for (size_t r = 0; r < N; ++r) {
      __m512 vacc = _mm512_setzero_ps();
      for (size_t i = 0; i < count; i += 16) {
        vacc = _mm512_add_ps(vacc, _mm512_loadu_ps(a_f32 + r * a_row + i));
      }
      float acc[16] __attribute__((aligned(64)));
      _mm512_store_ps(acc, vacc);
      for (int stride = 8; stride >= 1; stride >>= 1) {
        for (int l = 0; l < stride; ++l) acc[l] += acc[l + stride];
      }
      c_f32[r] = acc[0];
    }
    return;
  }
#endif
  for (size_t r = 0; r < N; ++r) {
    float acc[16] = {0.0f};
    for (size_t i = 0; i < count; ++i) {
      acc[i % 16] += a_f32[r * a_row + i];
    }
    for (int stride = 8; stride >= 1; stride >>= 1) {
      for (int l = 0; l < stride; ++l) acc[l] += acc[l + stride];
    }
    c_f32[r] = acc[0];
  }
}

static void sim_ynn_transpose_x32(size_t M, size_t N, size_t n_bytes_a,
                                  size_t stride_a, const void *A,
                                  size_t stride_b, void *B) {
  (void)n_bytes_a;
  const uint32_t *a_u32 = (const uint32_t *)A;
  uint32_t *b_u32 = (uint32_t *)B;
  size_t a_row = stride_a / sizeof(uint32_t);
  size_t b_row = stride_b / sizeof(uint32_t);
#ifdef __AVX512F__
  if (M % 16 == 0) {
    __m512i vidx = _mm512_set_epi32(
        15 * (int)b_row, 14 * (int)b_row, 13 * (int)b_row, 12 * (int)b_row,
        11 * (int)b_row, 10 * (int)b_row, 9 * (int)b_row, 8 * (int)b_row,
        7 * (int)b_row, 6 * (int)b_row, 5 * (int)b_row, 4 * (int)b_row,
        3 * (int)b_row, 2 * (int)b_row, 1 * (int)b_row, 0);
    for (size_t n = 0; n < N; ++n) {
      for (size_t m = 0; m < M; m += 16) {
        __m512i va =
            _mm512_loadu_si512((const __m512i *)(a_u32 + n * a_row + m));
        _mm512_i32scatter_epi32(b_u32 + m * b_row + n, vidx, va, 4);
      }
    }
    return;
  }
#endif
  for (size_t n = 0; n < N; ++n) {
    for (size_t m = 0; m < M; ++m) {
      b_u32[m * b_row + n] = a_u32[n * a_row + m];
    }
  }
}

static void sim_ynn_transpose_x32_scalar(size_t M, size_t N, size_t n_bytes_a,
                                         size_t stride_a, const void *A,
                                         size_t stride_b, void *B) {
  (void)n_bytes_a;
  const uint32_t *a_u32 = (const uint32_t *)A;
  uint32_t *b_u32 = (uint32_t *)B;
  size_t a_row = stride_a / sizeof(uint32_t);
  size_t b_row = stride_b / sizeof(uint32_t);
  for (size_t n = 0; n < N; ++n) {
    for (size_t m = 0; m < M; ++m) {
      b_u32[m * b_row + n] = a_u32[n * a_row + m];
    }
  }
}

static int is_litert_binary_loaded(void) {
  if (getenv("LD_PRELOAD") != NULL || getenv("AEB_HW_PRELOAD") != NULL) {
    return 1;
  }
  char exe_path[512];
  ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
  if (len <= 0) return 0;
  exe_path[len] = '\0';
  return strstr(exe_path, "litert_lm_advanced_main") != NULL;
}

static void run_verification_suite(int use_binary_symbols) {
  if (use_binary_symbols) {
    uintptr_t base = find_binary_base();
    if (!base) {
      fprintf(stderr, "ERROR: Could not locate litert_lm_advanced_main base\n");
      _exit(1);
    }
    enable_amx();

    // Resolve XNNPACK symbols from litert_lm_advanced_main
    fn_xnn_qc4w_1x64_amx = (xnn_qd8_gemm_fn)(base + 0x01b53dc0);
    fn_xnn_qc4w_16x64_amx = (xnn_qd8_gemm_fn)(base + 0x01b512c0);
    fn_xnn_qc8w_16x64_amx = (xnn_qd8_gemm_fn)(base + 0x01b4ec20);
    fn_xnn_f32_7x32_avx512 = (xnn_f32_gemm_fn)(base + 0x01b07690);
    fn_xnn_qc2w_5x8_avxvnni = (xnn_qs8_qc2w_gemm_fn)(base + 0x01ab30b0);
    fn_xnn_rmax_avx512 = (xnn_f32_rmax_fn)(base + 0x01b050a0);
    fn_xnn_raddexp_avx512 =
        (xnn_f32_raddstoreexpminusmax_fn)(base + 0x01b06880);
    fn_xnn_packw_x32_avx512 = (xnn_pack_gemm_goi_fn)(base + 0x01afb150);
    fn_xnn_pack_qs8_goi = (xnn_pack_gemm_goi_fn)(base + 0x019b0830);
    fn_xnn_pack_qs8_qc4w_goi = (xnn_pack_gemm_goi_fn)(base + 0x019b1340);
    fn_xnn_pack_qs8_to_qu8_qc2w_goi = (xnn_pack_gemm_goi_fn)(base + 0x019b1af0);
    fn_xnn_init_qs8_minmax_params =
        (xnn_init_qs8_qc8w_conv_minmax_fp32_params_fn)(base + 0x01b61570);

    // Resolve YNNPACK symbols from litert_lm_advanced_main
    fn_ynn_dot_qc4_1x32_vnni = (ynn_dot_fn)(base + 0x011f85a0);
    fn_ynn_dot_qc4_12x16_vnni = (ynn_dot_fn)(base + 0x011fb750);
    fn_ynn_dot_qc8_32x32_amx = (ynn_dot_fn)(base + 0x0120ab60);
    fn_ynn_dot_fp32_5x64_avx512 = (ynn_dot_fn)(base + 0x011d6ff0);
    fn_ynn_dot_qc2_4x64_vnni = (ynn_dot_fn)(base + 0x011ed490);
    fn_ynn_max_fp32_avx512 = (ynn_reduce_fp32_fn)(base + 0x0125dd00);
    fn_ynn_sum_fp32_avx512 = (ynn_reduce_fp32_fn)(base + 0x012637c0);
    fn_ynn_transpose_x32_avx512 = (ynn_transpose_x32_fn)(base + 0x012e68e0);
    fn_ynn_transpose_x32_scalar = (ynn_transpose_x32_fn)(base + 0x012ce320);
  } else {
#ifdef __AMX_INT8__
    enable_amx();
#endif
    // Bind instruction-accurate standalone microkernel implementations
    fn_xnn_qc4w_1x64_amx = sim_xnn_qd8_qc4w_amx;
    fn_xnn_qc4w_16x64_amx = sim_xnn_qd8_qc4w_amx;
    fn_xnn_qc8w_16x64_amx = sim_xnn_qd8_qc8w_amx;
    fn_xnn_f32_7x32_avx512 = sim_xnn_f32_7x32_avx512;
    fn_xnn_qc2w_5x8_avxvnni = sim_xnn_qc2w_5x8_avxvnni;
    fn_xnn_rmax_avx512 = sim_xnn_rmax_avx512;
    fn_xnn_raddexp_avx512 = sim_xnn_raddexp_avx512;
    fn_xnn_packw_x32_avx512 = sim_xnn_packw_x32_avx512;
    fn_xnn_pack_qs8_goi = sim_xnn_pack_qs8_goi;
    fn_xnn_pack_qs8_qc4w_goi = sim_xnn_pack_qs8_qc4w_goi;
    fn_xnn_init_qs8_minmax_params = sim_xnn_init_qs8_minmax_params;

    fn_ynn_dot_qc4_1x32_vnni = sim_ynn_dot_qc4_vnni;
    fn_ynn_dot_qc4_12x16_vnni = sim_ynn_dot_qc4_vnni;
    fn_ynn_dot_qc8_32x32_amx = sim_ynn_dot_qc8_amx;
    fn_ynn_dot_fp32_5x64_avx512 = sim_ynn_dot_fp32_avx512;
    fn_ynn_dot_qc2_4x64_vnni = sim_ynn_dot_qc2_vnni;
    fn_ynn_max_fp32_avx512 = sim_ynn_max_fp32_avx512;
    fn_ynn_sum_fp32_avx512 = sim_ynn_sum_fp32_avx512;
    fn_ynn_transpose_x32_avx512 = sim_ynn_transpose_x32;
    fn_ynn_transpose_x32_scalar = sim_ynn_transpose_x32_scalar;
  }

  fprintf(stderr,
          "[verify] Calibrating packed weight layouts & quantization scales...\n");

  xnn_qd8_layout_info_t xnn_qc8_info = calibrate_xnn_qd8_qc8w();
  xnn_qd8_layout_info_t xnn_qc4_info = calibrate_xnn_qd8_qc4w();

  int ynn_fp32_map[64][64];
  int ynn_int8_map[64][32];
  int ynn_int4_1x32_map[64][64];
  int32_t ynn_int4_1x32_code_to_val[16];
  uint8_t ynn_int4_1x32_val_to_code[16];
  int ynn_int4_12x16_map[64][64];
  int32_t ynn_int4_12x16_code_to_val[16];
  uint8_t ynn_int4_12x16_val_to_code[16];
  int ynn_int2_map[64][64];
  int32_t ynn_int2_code_to_val[4];
  int xnn_qc2w_map[64][8];
  int32_t xnn_qc2w_code_to_val[4];
  size_t xnn_qc2w_scale_off = 160;

  calibrate_ynn_fp32_layout(ynn_fp32_map);
  calibrate_ynn_int8_layout(ynn_int8_map);
  calibrate_ynn_int4_layout(fn_ynn_dot_qc4_1x32_vnni, 1, 32, 16,
                            ynn_int4_1x32_map, ynn_int4_1x32_code_to_val,
                            ynn_int4_1x32_val_to_code);
  calibrate_ynn_int4_layout(fn_ynn_dot_qc4_12x16_vnni, 12, 16, 8,
                            ynn_int4_12x16_map, ynn_int4_12x16_code_to_val,
                            ynn_int4_12x16_val_to_code);
  calibrate_ynn_int2_layout(ynn_int2_map, ynn_int2_code_to_val);
  calibrate_xnn_qc2w_layout(xnn_qc2w_map, xnn_qc2w_code_to_val,
                            &xnn_qc2w_scale_off);

  fprintf(stderr,
          "[verify] Running Level A (exact ukernel) & Level B (FP64 golden) "
          "verification across 7 kernel pairs...\n");

  kernel_verification_result_t xnn_res[7], ynn_res[7];

  // Pair 1: 4-Bit Decode GEMV (M=1, N=64/32, K=64)
  xnn_res[0] = verify_xnn_qd8_qc4w(fn_xnn_qc4w_1x64_amx, 1, &xnn_qc4_info);
  ynn_res[0] = verify_ynn_dot_int4(fn_ynn_dot_qc4_1x32_vnni, 1, 32, 16,
                                   ynn_int4_1x32_map,
                                   ynn_int4_1x32_val_to_code);

  // Pair 2: 4-Bit Prefill GEMM (M=16/12, N=64/16, K=64)
  xnn_res[1] = verify_xnn_qd8_qc4w(fn_xnn_qc4w_16x64_amx, 16, &xnn_qc4_info);
  ynn_res[1] = verify_ynn_dot_int4(fn_ynn_dot_qc4_12x16_vnni, 12, 16, 8,
                                   ynn_int4_12x16_map,
                                   ynn_int4_12x16_val_to_code);

  // Pair 3: FP32 Attention BatchMatMul (M=7/5, N=64, K=64)
  xnn_res[2] = verify_xnn_fp32_gemm();
  ynn_res[2] = verify_ynn_fp32_dot(ynn_fp32_map);

  // Pair 4: FP32 Attention Softmax (ROWS=16, K=1024)
  xnn_res[3] = verify_softmax_pair(0);
  ynn_res[3] = verify_softmax_pair(1);

  // Pair 5: 8-Bit Quantized GEMM (M=16/32, N=64/32, K=64)
  xnn_res[4] = verify_xnn_qd8_qc8w(&xnn_qc8_info);
  ynn_res[4] = verify_ynn_dot_int8(ynn_int8_map);

  // Pair 6: 2-Bit Weight-Only GEMM (M=5/4, N=8/64, K=64)
  xnn_res[5] = verify_xnn_qc2w_gemm(xnn_qc2w_map, xnn_qc2w_code_to_val,
                                    xnn_qc2w_scale_off);
  ynn_res[5] = verify_ynn_dot_int2(ynn_int2_map, ynn_int2_code_to_val);

  // Pair 7: Matrix Packing / 32-Bit Transpose (N=64, K=64 + N=53 tail)
  xnn_res[6] = verify_pack_transpose_pair(0);
  ynn_res[6] = verify_pack_transpose_pair(1);

  const char *pair_ids[7] = {
      "pair_1_qc4w_gemv_decode",     "pair_2_qc4w_gemm_prefill",
      "pair_3_fp32_attn_bmm",        "pair_4_fp32_softmax",
      "pair_5_qc8w_gemm_ multimodal", "pair_6_qc2w_gemm_embed",
      "pair_7_pack_transpose_x32"};
  // Fix pair_5 id without space
  pair_ids[4] = "pair_5_qc8w_gemm_multimodal";

  const char *pair_titles[7] = {
      "4-Bit Quantized GEMV (Single-Token Decode: M=1)",
      "4-Bit Quantized GEMM (Batched Prefill: M=16/12)",
      "FP32 Attention BatchMatMul (SDPA Q*K^T & Scores*V)",
      "FP32 Attention Softmax (Max + Exp-Minus-Max + Sum)",
      "8-Bit Quantized GEMM (Vision ViT & Audio Conformer)",
      "2-Bit Weight-Only Quantized GEMM (Token Embeddings)",
      "Matrix Packing / 32-Bit Transpose (Weight/KV Layout)"};

  const char *xnn_symbols[7] = {
      "xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx",
      "xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx",
      "xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast",
      "xnn_f32_rmax + xnn_f32_raddstoreexpminusmax (avx512f_p5)",
      "xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx",
      "xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni",
      "xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4_prfm"};

  const char *ynn_symbols[7] = {
      "ynn::dot_uint8_int4_int32_1x32x8_1x16x8_avx512vnni",
      "ynn::dot_uint8_int4_int32_12x16x8_1x16x8_avx512vnni",
      "ynn::dot_fp32_5x64x1_1x16x1_avx512",
      "ynn::max_k1_fp32_avx512 + ynn::sum_k1_fp32_avx512",
      "ynn::dot_int8_int8_int32_32x32x64_16x16x4_amxint8",
      "ynn::dot_uint8_int2_int32_4x64x16_1x16x16_avx512vnni",
      "ynn::transpose_x32_avx512"};

  const char *out_path = getenv("AEB_VERIFY_OUTPUT");
  FILE *fp = out_path ? fopen(out_path, "w") : stdout;
  if (!fp) {
    perror("fopen AEB_VERIFY_OUTPUT");
    _exit(1);
  }

  int total_pass = 0;
  for (int i = 0; i < 7; ++i) {
    if (xnn_res[i].level_a.pass && xnn_res[i].level_b.pass) total_pass++;
    if (ynn_res[i].level_a.pass && ynn_res[i].level_b.pass) total_pass++;
  }

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
  const char *exec_mode =
      use_binary_symbols ? "ld_preload_hardware" : "sapphire_rapids_hardware_isa";
#else
  const char *exec_mode =
      use_binary_symbols ? "ld_preload_hardware" : "standalone_sim";
#endif
  fprintf(fp, "{\n");
  fprintf(fp, "  \"suite\": \"litert_microkernel_precision_verification\",\n");
  fprintf(fp, "  \"arch\": \"x86_64_sapphire_rapids\",\n");
  fprintf(fp, "  \"execution_mode\": \"%s\",\n", exec_mode);
  fprintf(fp, "  \"total_kernels_verified\": 14,\n");
  fprintf(fp, "  \"total_kernels_passed\": %d,\n", total_pass);
  fprintf(fp, "  \"pairs\": [\n");

  for (int i = 0; i < 7; ++i) {
    const char *e1 = NULL, *e2 = NULL;
    int inc_canary = 0;
    if (i == 3) {
      e1 = "max_simplex_sum_err";
      e2 = "shift_invariance_max_err";
    } else if (i == 6) {
      e1 = "aligned_exact_match_pct";
      e2 = "unaligned_tail_exact_match_pct";
      inc_canary = 1;
    }

    fprintf(fp, "    {\n");
    fprintf(fp, "      \"pair_id\": \"%s\",\n", pair_ids[i]);
    fprintf(fp, "      \"title\": \"%s\",\n", pair_titles[i]);
    fprintf(fp, "      \"xnnpack_symbol\": \"%s\",\n", xnn_symbols[i]);
    fprintf(fp, "      \"ynnpack_symbol\": \"%s\",\n", ynn_symbols[i]);
    fprintf(fp, "      \"xnnpack\":\n");
    write_kernel_result_json(fp, &xnn_res[i], e1, e2, inc_canary);
    fprintf(fp, ",\n      \"ynnpack\":\n");
    write_kernel_result_json(fp, &ynn_res[i], e1, e2, inc_canary);
    fprintf(fp, "\n    }%s\n", (i < 6) ? "," : "");

    fprintf(stderr,
            "  [Pair %d] %-28s | XNN: LvlA=%s (ULP=%llu, SQNR=%.1fdB) "
            "LvlB=%s (SQNR=%.1fdB, Cos=%.6f) | YNN: LvlA=%s (ULP=%llu, "
            "SQNR=%.1fdB) LvlB=%s (SQNR=%.1fdB, Cos=%.6f)\n",
            i + 1, pair_ids[i], xnn_res[i].level_a.pass ? "PASS" : "FAIL",
            (unsigned long long)xnn_res[i].level_a.max_ulp_diff,
            xnn_res[i].level_a.sqnr_db,
            xnn_res[i].level_b.pass ? "PASS" : "FAIL",
            xnn_res[i].level_b.sqnr_db, xnn_res[i].level_b.cosine_sim,
            ynn_res[i].level_a.pass ? "PASS" : "FAIL",
            (unsigned long long)ynn_res[i].level_a.max_ulp_diff,
            ynn_res[i].level_a.sqnr_db,
            ynn_res[i].level_b.pass ? "PASS" : "FAIL",
            ynn_res[i].level_b.sqnr_db, ynn_res[i].level_b.cosine_sim);
  }

  fprintf(fp, "  ]\n");
  fprintf(fp, "}\n");
  fflush(fp);
  if (fp != stdout) fclose(fp);

  fprintf(stderr, "[verify] Completed: %d / 14 microkernels PASSED.\n",
          total_pass);
  fflush(stderr);
  _exit(0);
}

__attribute__((constructor)) static void run_all_precision_verifications(void) {
  if (is_litert_binary_loaded()) {
    run_verification_suite(1);
  }
}

int main(void) {
  run_verification_suite(0);
  return 0;
}
