// Standalone ARM64 XNNPACK (KleidiAI SME2) & YNNPACK (SME2/SME/NEON)
// Microkernel Numerical Precision & Quantization Fidelity Verifier.
//
// Links directly against `libaeb_ukernels_arm64.a` and verifies all 14 primary
// Gemma 4 31B microkernels (7 XNNPACK + 7 YNNPACK) at two levels across 4
// input distributions (normal, saturation, outliers, tails):
//   - Level A (Exact Packed-Layout Microkernel Reference): Verifies the
//     SME2/SME/NEON microkernel against an exact scalar INT64 / FP64 reference
//     operating on the identical quantized inputs and packed weight buffers.
//   - Level B (FP64 Golden Operator Reference): Verifies end-to-end operator
//     and quantization fidelity against unquantized FP64 matrix multiplication
//     (C_fp64 = A_fp64 * B_fp64^T + bias_fp64), FP64 Softmax, and bit-exact
//     32-bit matrix packing/transposition (including unaligned tails & canary
//     buffer overflow guards).
//
// Usage:
//   ./verify_ukernels_arm64 --output ukernel_precision_apple_m5_max.json

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ALIGN_BYTES 128

static void *alloc_aligned(size_t bytes) {
  size_t sz = (bytes + ALIGN_BYTES - 1) & ~(size_t)(ALIGN_BYTES - 1);
  if (sz == 0) sz = ALIGN_BYTES;
  void *p = NULL;
  if (posix_memalign(&p, ALIGN_BYTES, sz) != 0 || !p) {
    fprintf(stderr, "ERROR: posix_memalign(%zu) failed\n", sz);
    exit(1);
  }
  return p;
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

struct kai_rhs_pack_qs4cxs1s0_param {
  int8_t lhs_zero_point;
  uint8_t rhs_zero_point;
};

struct kai_rhs_pack_qsi8cx_params {
  int32_t lhs_zero_point;
  float scale_multiplier;
};

extern size_t kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr);
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

extern void kai_kernel_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme(
    size_t height, size_t width, const void *in, size_t row_offset, void *out);
extern void kai_kernel_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme(
    size_t height, size_t width, const void *in, size_t row_offset, void *out,
    const void *bias);

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
extern void aeb_ynn_dot_fp32_sme2(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_dot_int8_int8_sme2(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void *A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void *B, size_t C_in_stride_m,
    const void *C_in, size_t C_out_stride_m, void *C_out, void *state);
extern void aeb_ynn_dot_int8_int2_sme(
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
// Deterministic PRNG & Input Distribution Generators (identical to x86_64)
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
// Precision Metric Calculators (identical schema to x86_64 verifier)
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

typedef struct {
  precision_metrics_t level_a;
  precision_metrics_t level_b;
  precision_metrics_t level_b_by_dist[NUM_DISTS];
  double extra_metric_1;
  double extra_metric_2;
  int extra_int;
} kernel_verification_result_t;

static uint32_t float_to_ordered_u32(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof(uint32_t));
  return (u & 0x80000000U) ? ~u : (u | 0x80000000U);
}

static uint64_t fp32_ulp_diff(float a, float b) {
  if (isnan(a) || isnan(b) || isinf(a) || isinf(b)) {
    return (a == b) ? 0ULL : 0xFFFFFFFFULL;
  }
  if (a == b) return 0ULL;
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
  double sum_sq_sig = 0.0, sum_sq_err = 0.0, dot = 0.0, norm_a = 0.0,
         norm_r = 0.0, sum_ulp = 0.0;
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
    double rel = err / fmax(fabs(r), 1e-6);
    if (err > m.max_abs_err) m.max_abs_err = err;
    if (rel > m.max_rel_err) m.max_rel_err = rel;
    sum_sq_sig += r * r;
    sum_sq_err += err * err;
    dot += da * r;
    norm_a += da * da;
    norm_r += r * r;

    uint64_t ulp = fp32_ulp_diff(a, (float)r);
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
// 1. XNNPACK KleidiAI SME2 Quantized Verifiers (Pairs 1, 2, 5, 6)
// -----------------------------------------------------------------------------

static kernel_verification_result_t verify_xnn_qp8_sme2(xnn_qp8_gemm_fn fn,
                                                        int M, int N, int K,
                                                        int bits) {
  kernel_verification_result_t res;
  memset(&res, 0, sizeof(res));

  double *A_fp64 = (double *)calloc((size_t)M * (size_t)K, sizeof(double));
  double *B_fp64 = (double *)calloc((size_t)N * (size_t)K, sizeof(double));
  double *bias_fp64 = (double *)calloc((size_t)N, sizeof(double));
  double *C_ref_a = (double *)calloc((size_t)M * (size_t)N, sizeof(double));
  double *C_ref_b = (double *)calloc((size_t)M * (size_t)N, sizeof(double));

  float *A_f32 = (float *)alloc_aligned((size_t)M * (size_t)K * sizeof(float));
  int8_t *W_q = (int8_t *)alloc_aligned((size_t)N * (size_t)K);
  float *w_scale = (float *)alloc_aligned((size_t)N * sizeof(float));
  float *w_bias = (float *)alloc_aligned((size_t)N * sizeof(float));
  float *C_out = (float *)alloc_aligned((size_t)M * (size_t)N * sizeof(float));

  size_t lhs_sz = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(
      (size_t)M, (size_t)K, (size_t)M, 4, 1);
  uint8_t *lhs_packed = (uint8_t *)alloc_aligned(lhs_sz + 256);

  size_t rhs_sz = 0;
  if (bits == 4) {
    rhs_sz = kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon(
        (size_t)K, (size_t)N, 4, 1);
  } else if (bits == 8) {
    rhs_sz = kai_get_rhs_packed_stride_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        (size_t)K, (size_t)N, 4, 1);
  } else {
    rhs_sz = kai_get_rhs_packed_stride_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(
        (size_t)K, (size_t)N, 4, 1);
  }
  uint8_t *rhs_packed = (uint8_t *)alloc_aligned(rhs_sz + 256);

  aeb_xnn_f32_qc4w_minmax_params_t params = {{-FLT_MAX, FLT_MAX, 0}};
  static const int32_t lut2[4] = {0, 1, -2, -1};

  for (int d = 0; d < NUM_DISTS; ++d) {
    fill_fp64_tensors((dist_kind_t)d, M, N, K, A_fp64, B_fp64, bias_fp64);
    for (int i = 0; i < M * K; ++i) A_f32[i] = (float)A_fp64[i];
    for (int n = 0; n < N; ++n) w_bias[n] = (float)bias_fp64[n];

    // Quantize & pack LHS via genuine KleidiAI lhs_quant_pack_qai8dxp_f32
    memset(lhs_packed, 0, lhs_sz);
    kai_run_lhs_quant_pack_qai8dxp_f32((size_t)M, (size_t)K, (size_t)M, 4, 1, 0,
                                       A_f32, (size_t)K * sizeof(float),
                                       lhs_packed);

    // Quantize & pack RHS via genuine KleidiAI rhs_pack_nxk_*
    memset(rhs_packed, 0, rhs_sz);
    if (bits == 4) {
      uint8_t *raw4 = (uint8_t *)calloc((size_t)N * (size_t)K / 2, 1);
      for (int n = 0; n < N; ++n) {
        double amax = 0.0;
        for (int k = 0; k < K; ++k) {
          double av = fabs(B_fp64[n * K + k]);
          if (av > amax) amax = av;
        }
        float sw = (amax > 1e-12) ? (float)(amax / 7.0) : 1.0f;
        w_scale[n] = sw;
        for (int k = 0; k < K; ++k) {
          long q = lround(B_fp64[n * K + k] / (double)sw);
          if (q < -8) q = -8;
          if (q > 7) q = 7;
          W_q[n * K + k] = (int8_t)q;
          uint8_t u4 = (uint8_t)(q + 8) & 0x0F;
          if (k & 1) {
            raw4[n * (K / 2) + (k >> 1)] |= (uint8_t)(u4 << 4);
          } else {
            raw4[n * (K / 2) + (k >> 1)] |= u4;
          }
        }
      }
      struct kai_rhs_pack_qs4cxs1s0_param kp = {1, 8};
      kai_run_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon(
          1, (size_t)N, (size_t)K, (size_t)N, 4, 1, raw4, w_bias, w_scale,
          rhs_packed, 0, &kp);
      free(raw4);
    } else if (bits == 8) {
      for (int n = 0; n < N; ++n) {
        double amax = 0.0;
        for (int k = 0; k < K; ++k) {
          double av = fabs(B_fp64[n * K + k]);
          if (av > amax) amax = av;
        }
        float sw = (amax > 1e-12) ? (float)(amax / 127.0) : 1.0f;
        w_scale[n] = sw;
        for (int k = 0; k < K; ++k) {
          long q = lround(B_fp64[n * K + k] / (double)sw);
          if (q < -127) q = -127;
          if (q > 127) q = 127;
          W_q[n * K + k] = (int8_t)q;
        }
      }
      struct kai_rhs_pack_qsi8cx_params kp = {1, 1.0f};
      kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
          1, (size_t)N, (size_t)K, (size_t)N, 4, 1, W_q, w_bias, w_scale,
          rhs_packed, 0, &kp);
    } else {
      // 2-bit asymmetric-range {-2, -1, 0, 1} via XNNPACK LUT {0, 1, -2, -1}
      // Inverse map from signed q in {-2,-1,0,1} to 2-bit code:
      //   0 -> 0, +1 -> 1, -2 -> 2, -1 -> 3
      uint8_t *raw2 = (uint8_t *)calloc((size_t)N * (size_t)K / 4, 1);
      for (int n = 0; n < N; ++n) {
        double amax = 0.0;
        for (int k = 0; k < K; ++k) {
          double av = fabs(B_fp64[n * K + k]);
          if (av > amax) amax = av;
        }
        // Choose MSE-optimal scale across symmetric/asymmetric 2-bit {-2,-1,0,+1} grid
        float cands[3] = {(amax > 1e-12) ? (float)(amax / 1.0) : 1.0f,
                          (amax > 1e-12) ? (float)(amax / 1.5) : 1.0f,
                          (amax > 1e-12) ? (float)(amax / 2.0) : 1.0f};
        float best_sw = cands[0];
        double best_mse = DBL_MAX;
        for (int ci = 0; ci < 3; ++ci) {
          double mse = 0.0;
          for (int k = 0; k < K; ++k) {
            long q = lround(B_fp64[n * K + k] / (double)cands[ci]);
            if (q < -2) q = -2;
            if (q > 1) q = 1;
            double diff = B_fp64[n * K + k] - (double)q * (double)cands[ci];
            mse += diff * diff;
          }
          if (mse < best_mse) {
            best_mse = mse;
            best_sw = cands[ci];
          }
        }
        float sw = best_sw;
        w_scale[n] = sw;
        for (int k = 0; k < K; ++k) {
          long q = lround(B_fp64[n * K + k] / (double)sw);
          if (q < -2) q = -2;
          if (q > 1) q = 1;
          W_q[n * K + k] = (int8_t)q;
          uint8_t code = (uint8_t)(q & 0x3);  // 0->0, 1->1, -2->2, -1->3
          raw2[n * (K / 4) + (k >> 2)] |= (uint8_t)(code << ((k & 3) * 2));
        }
      }
      struct kai_rhs_pack_qs4cxs1s0_param kp = {1, 2};
      kai_run_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(
          1, (size_t)N, (size_t)K, (size_t)N, 4, 1, raw2, w_bias, w_scale,
          rhs_packed, 0, &kp, lut2);
      free(raw2);
    }

    memset(C_out, 0, (size_t)M * (size_t)N * sizeof(float));
    fn((size_t)M, (size_t)N, (size_t)K, lhs_packed, rhs_packed, C_out,
       (size_t)N * sizeof(float), sizeof(float), &params);

    // Extract exact quantized LHS (q_a, neg_zp, recip_scale) from lhs_packed
    const int8_t *lhs_q_bytes = (const int8_t *)lhs_packed;
    const int32_t *neg_zps = (const int32_t *)(lhs_packed + (size_t)M * (size_t)K);
    const float *recip_scales =
        (const float *)(lhs_packed + (size_t)M * (size_t)K +
                        (size_t)M * sizeof(int32_t));

    for (int m = 0; m < M; ++m) {
      int32_t neg_zp = neg_zps[m];
      double sa = (double)recip_scales[m];
      for (int n = 0; n < N; ++n) {
        int64_t acc_i64 = 0;
        double acc_fp64 = bias_fp64[n];
        for (int k = 0; k < K; ++k) {
          int8_t qa = lhs_q_bytes[(k / 4) * M * 4 + m * 4 + (k % 4)];
          acc_i64 += ((int64_t)qa + (int64_t)neg_zp) * (int64_t)W_q[n * K + k];
          acc_fp64 += A_fp64[m * K + k] * B_fp64[n * K + k];
        }
        C_ref_a[m * N + n] =
            (double)acc_i64 * sa * (double)w_scale[n] + (double)w_bias[n];
        C_ref_b[m * N + n] = acc_fp64;
      }
    }

    double sqnr_b_min = (bits == 8) ? 35.0 : ((bits == 4) ? 14.0 : 5.5);
    double abs_b_tol = (bits == 8) ? 2.0 : ((bits == 4) ? 15.0 : 65.0);
    precision_metrics_t ma =
        compute_fp_metrics(C_out, C_ref_a, M * N, 2, 110.0, 1e-4);
    precision_metrics_t mb =
        compute_fp_metrics(C_out, C_ref_b, M * N, 0, sqnr_b_min, abs_b_tol);
    res.level_b_by_dist[d] = mb;
    merge_metrics(&res.level_a, &ma, d);
    merge_metrics(&res.level_b, &mb, d);
  }

  free(A_fp64);
  free(B_fp64);
  free(bias_fp64);
  free(C_ref_a);
  free(C_ref_b);
  free(A_f32);
  free(W_q);
  free(w_scale);
  free(w_bias);
  free(C_out);
  free(lhs_packed);
  free(rhs_packed);
  return res;
}

// -----------------------------------------------------------------------------
// 2. YNNPACK ARM64 Quantized Verifiers (Pair 1 DotProd, Pairs 2, 5, 6 SME/SME2)
// -----------------------------------------------------------------------------

// Self-calibrating layout prober for YNNPACK int8*int4 NEON DotProd (1x32x8)
static void calibrate_ynn_int4_neondot(int nibble_map[64][32]) {
  const int K = 64, N = 32;
  const int total_nibbles = K * N;
  int8_t A[64] __attribute__((aligned(64)));
  uint8_t B[1024] __attribute__((aligned(64)));
  int32_t C[32] __attribute__((aligned(64)));
  memset(nibble_map, 0, sizeof(int) * 64 * 32);

  for (int b = 0; b < 11; ++b) {
    memset(B, 0, sizeof(B));
    for (int nid = 0; nid < total_nibbles; ++nid) {
      if ((nid >> b) & 1) {
        int byte_idx = nid >> 1;
        if (nid & 1) {
          B[byte_idx] |= (uint8_t)(1 << 4);
        } else {
          B[byte_idx] |= 1;
        }
      }
    }
    for (int k = 0; k < K; ++k) {
      memset(A, 0, sizeof(A));
      A[k] = 1;
      memset(C, 0, sizeof(C));
      aeb_ynn_dot_int8_int4_1x32_neondot(1, 32, 1, 1, 64, 64, 0, 0, A, 0, 0, 16,
                                         B, 0, NULL, 32 * sizeof(int32_t), C,
                                         NULL);
      for (int n = 0; n < N; ++n) {
        if (C[n] == 1) nibble_map[k][n] |= (1 << b);
      }
    }
  }
}

static kernel_verification_result_t verify_ynn_int4_neondot(void) {
  kernel_verification_result_t res;
  memset(&res, 0, sizeof(res));
  const int M = 1, N = 32, K = 64;
  int nibble_map[64][32];
  calibrate_ynn_int4_neondot(nibble_map);

  double *A_fp64 = (double *)calloc(M * K, sizeof(double));
  double *B_fp64 = (double *)calloc(N * K, sizeof(double));
  double *bias_fp64 = (double *)calloc(N, sizeof(double));
  double *C_ref_a = (double *)calloc(M * N, sizeof(double));
  double *C_ref_b = (double *)calloc(M * N, sizeof(double));

  int8_t A_i8[64] __attribute__((aligned(64)));
  int8_t W_i4[32 * 64];
  uint8_t B_packed[1024] __attribute__((aligned(64)));
  int32_t C_i32[32] __attribute__((aligned(64)));
  float C_f32[32];

  for (int d = 0; d < NUM_DISTS; ++d) {
    fill_fp64_tensors((dist_kind_t)d, M, N, K, A_fp64, B_fp64, bias_fp64);

    // Dynamic per-row asymmetric int8 quantization of A (signed int8)
    double rmin = 0.0, rmax = 0.0;
    for (int k = 0; k < K; ++k) {
      if (A_fp64[k] < rmin) rmin = A_fp64[k];
      if (A_fp64[k] > rmax) rmax = A_fp64[k];
    }
    double sa = (rmax - rmin) > 1e-12 ? (rmax - rmin) / 255.0 : 1.0;
    int32_t zpa = (int32_t)lround(-128.0 - rmin / sa);
    if (zpa < -128) zpa = -128;
    if (zpa > 127) zpa = 127;
    for (int k = 0; k < K; ++k) {
      long q = lround(A_fp64[k] / sa) + zpa;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      A_i8[k] = (int8_t)q;
    }

    // Per-channel signed int4 [-8, 7] weights packed via calibrated nibble_map
    float w_scale[32];
    int32_t ksum[32];
    memset(B_packed, 0, sizeof(B_packed));
    for (int n = 0; n < N; ++n) {
      double amax = 0.0;
      for (int k = 0; k < K; ++k) {
        double av = fabs(B_fp64[n * K + k]);
        if (av > amax) amax = av;
      }
      w_scale[n] = (amax > 1e-12) ? (float)(amax / 7.0) : 1.0f;
      int32_t ws = 0;
      for (int k = 0; k < K; ++k) {
        long q = lround(B_fp64[n * K + k] / (double)w_scale[n]);
        if (q < -8) q = -8;
        if (q > 7) q = 7;
        W_i4[n * K + k] = (int8_t)q;
        ws += (int32_t)q;
        uint8_t nib = (uint8_t)(q & 0x0F);
        int nid = nibble_map[k][n];
        if (nid & 1) {
          B_packed[nid >> 1] |= (uint8_t)(nib << 4);
        } else {
          B_packed[nid >> 1] |= nib;
        }
      }
      ksum[n] = ws;
    }

    memset(C_i32, 0, sizeof(C_i32));
    aeb_ynn_dot_int8_int4_1x32_neondot(1, N, 1, 1, K, K, 0, 0, A_i8, 0, 0, 16,
                                       B_packed, 0, NULL, N * sizeof(int32_t),
                                       C_i32, NULL);

    for (int n = 0; n < N; ++n) {
      int64_t exact_dot = 0;
      double acc_fp64 = bias_fp64[n];
      for (int k = 0; k < K; ++k) {
        exact_dot += (int64_t)A_i8[k] * (int64_t)W_i4[n * K + k];
        acc_fp64 += A_fp64[k] * B_fp64[n * K + k];
      }
      // Level A: exact raw int32 accumulator comparison
      C_f32[n] = (float)C_i32[n];
      C_ref_a[n] = (double)exact_dot;
      // Level B: dequantized FP32 output vs unquantized FP64 golden
      C_ref_b[n] = acc_fp64;
    }
    precision_metrics_t ma =
        compute_fp_metrics(C_f32, C_ref_a, M * N, 0, 150.0, 1e-5);
    for (int n = 0; n < N; ++n) {
      int32_t corr = C_i32[n] - zpa * ksum[n];
      C_f32[n] = (float)((double)corr * sa * (double)w_scale[n] + bias_fp64[n]);
    }
    precision_metrics_t mb =
        compute_fp_metrics(C_f32, C_ref_b, M * N, 0, 14.0, 15.0);
    res.level_b_by_dist[d] = mb;
    merge_metrics(&res.level_a, &ma, d);
    merge_metrics(&res.level_b, &mb, d);
  }

  free(A_fp64);
  free(B_fp64);
  free(bias_fp64);
  free(C_ref_a);
  free(C_ref_b);
  return res;
}

// Unified verifier for YNNPACK SME / SME2 integer dot kernels (int4, int8, int2)
static kernel_verification_result_t verify_ynn_sme_int(ynn_dot_fn fn, int bits) {
  kernel_verification_result_t res;
  memset(&res, 0, sizeof(res));
  const int M = 16, N = 64, K = 64;
  const int dot_factor = (bits == 8) ? 4 : ((bits == 4) ? 8 : 16);
  const size_t b_stride_k1 = (size_t)N * (size_t)bits / 8;

  double *A_fp64 = (double *)calloc(M * K, sizeof(double));
  double *B_fp64 = (double *)calloc(N * K, sizeof(double));
  double *bias_fp64 = (double *)calloc(N, sizeof(double));
  double *C_ref_a = (double *)calloc(M * N, sizeof(double));
  double *C_ref_b = (double *)calloc(M * N, sizeof(double));

  int8_t A_i8[16 * 64];
  int8_t *A_trans = (int8_t *)alloc_aligned(16 * 64 + 128);
  int8_t W_q[64 * 64];
  uint8_t *B_packed = (uint8_t *)alloc_aligned(64 * 64 + 128);
  int32_t *C_i32 = (int32_t *)alloc_aligned(16 * 64 * sizeof(int32_t));
  float C_f32[16 * 64];

  for (int d = 0; d < NUM_DISTS; ++d) {
    fill_fp64_tensors((dist_kind_t)d, M, N, K, A_fp64, B_fp64, bias_fp64);

    double sa[16];
    int32_t zpa[16];
    for (int m = 0; m < M; ++m) {
      double rmin = 0.0, rmax = 0.0;
      for (int k = 0; k < K; ++k) {
        double v = A_fp64[m * K + k];
        if (v < rmin) rmin = v;
        if (v > rmax) rmax = v;
      }
      sa[m] = (rmax - rmin) > 1e-12 ? (rmax - rmin) / 255.0 : 1.0;
      zpa[m] = (int32_t)lround(-128.0 - rmin / sa[m]);
      if (zpa[m] < -128) zpa[m] = -128;
      if (zpa[m] > 127) zpa[m] = 127;
      for (int k = 0; k < K; ++k) {
        long q = lround(A_fp64[m * K + k] / sa[m]) + zpa[m];
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        A_i8[m * K + k] = (int8_t)q;
      }
    }

    // Pack transposed A for YNNPACK SME/SME2 (dot_flag::transpose_a)
    memset(A_trans, 0, 16 * 64);
    for (int kb = 0; kb < K; kb += dot_factor) {
      int8_t *blk = A_trans + (size_t)kb * (size_t)M;
      for (int m = 0; m < M; ++m) {
        for (int j = 0; j < dot_factor; ++j) {
          blk[m * dot_factor + j] = A_i8[m * K + kb + j];
        }
      }
    }

    // Quantize & pack B into YNNPACK SME/SME2 layout
    float w_scale[64];
    int32_t ksum[64];
    memset(B_packed, 0, 64 * 64);
    for (int n = 0; n < N; ++n) {
      double amax = 0.0;
      for (int k = 0; k < K; ++k) {
        double av = fabs(B_fp64[n * K + k]);
        if (av > amax) amax = av;
      }
      double qmax_div = (bits == 8) ? 127.0 : ((bits == 4) ? 7.0 : 1.5);
      long qlo = (bits == 8) ? -127 : ((bits == 4) ? -8 : -2);
      long qhi = (bits == 8) ? 127 : ((bits == 4) ? 7 : 1);
      if (bits == 2) {
        float cands[3] = {(amax > 1e-12) ? (float)(amax / 1.0) : 1.0f,
                          (amax > 1e-12) ? (float)(amax / 1.5) : 1.0f,
                          (amax > 1e-12) ? (float)(amax / 2.0) : 1.0f};
        float best_sw = cands[0];
        double best_mse = DBL_MAX;
        for (int ci = 0; ci < 3; ++ci) {
          double mse = 0.0;
          for (int k = 0; k < K; ++k) {
            long q = lround(B_fp64[n * K + k] / (double)cands[ci]);
            if (q < -2) q = -2;
            if (q > 1) q = 1;
            double diff = B_fp64[n * K + k] - (double)q * (double)cands[ci];
            mse += diff * diff;
          }
          if (mse < best_mse) {
            best_mse = mse;
            best_sw = cands[ci];
          }
        }
        w_scale[n] = best_sw;
      } else {
        w_scale[n] = (amax > 1e-12) ? (float)(amax / qmax_div) : 1.0f;
      }
      int32_t ws = 0;
      for (int k = 0; k < K; ++k) {
        long q = lround(B_fp64[n * K + k] / (double)w_scale[n]);
        if (q < qlo) q = qlo;
        if (q > qhi) q = qhi;
        W_q[n * K + k] = (int8_t)q;
        ws += (int32_t)q;
      }
      ksum[n] = ws;
    }

    for (int kb = 0; kb < K; kb += dot_factor) {
      uint8_t *blk = B_packed + (size_t)kb * b_stride_k1;
      for (int n = 0; n < N; ++n) {
        if (bits == 8) {
          for (int j = 0; j < 4; ++j) {
            blk[n * 4 + j] = (uint8_t)W_q[n * K + kb + j];
          }
        } else if (bits == 4) {
          for (int j = 0; j < 4; ++j) {
            uint8_t lo = (uint8_t)(W_q[n * K + kb + 2 * j] & 0x0F);
            uint8_t hi = (uint8_t)(W_q[n * K + kb + 2 * j + 1] & 0x0F);
            blk[n * 4 + j] = (uint8_t)((hi << 4) | lo);
          }
        } else {
          for (int j = 0; j < 4; ++j) {
            uint8_t c0 = (uint8_t)(W_q[n * K + kb + 4 * j + 0] & 0x03);
            uint8_t c1 = (uint8_t)(W_q[n * K + kb + 4 * j + 1] & 0x03);
            uint8_t c2 = (uint8_t)(W_q[n * K + kb + 4 * j + 2] & 0x03);
            uint8_t c3 = (uint8_t)(W_q[n * K + kb + 4 * j + 3] & 0x03);
            blk[n * 4 + j] = (uint8_t)((c3 << 6) | (c2 << 4) | (c1 << 2) | c0);
          }
        }
      }
    }

    memset(C_i32, 0, M * N * sizeof(int32_t));
    fn(M, N, 1, 1, K, (size_t)M, 0, 0, A_trans, 0, 0, b_stride_k1, B_packed, 0,
       NULL, (size_t)N * sizeof(int32_t), C_i32, NULL);

    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        int64_t exact_dot = 0;
        double acc_fp64 = bias_fp64[n];
        for (int k = 0; k < K; ++k) {
          exact_dot += (int64_t)A_i8[m * K + k] * (int64_t)W_q[n * K + k];
          acc_fp64 += A_fp64[m * K + k] * B_fp64[n * K + k];
        }
        C_f32[m * N + n] = (float)C_i32[m * N + n];
        C_ref_a[m * N + n] = (double)exact_dot;
        C_ref_b[m * N + n] = acc_fp64;
      }
    }

    precision_metrics_t ma =
        compute_fp_metrics(C_f32, C_ref_a, M * N, 0, 150.0, 1e-5);
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        int32_t corr = C_i32[m * N + n] - zpa[m] * ksum[n];
        C_f32[m * N + n] =
            (float)((double)corr * sa[m] * (double)w_scale[n] + bias_fp64[n]);
      }
    }
    double sqnr_b_min = (bits == 8) ? 35.0 : ((bits == 4) ? 14.0 : 5.5);
    double abs_b_tol = (bits == 8) ? 2.0 : ((bits == 4) ? 15.0 : 65.0);
    precision_metrics_t mb =
        compute_fp_metrics(C_f32, C_ref_b, M * N, 0, sqnr_b_min, abs_b_tol);
    res.level_b_by_dist[d] = mb;
    merge_metrics(&res.level_a, &ma, d);
    merge_metrics(&res.level_b, &mb, d);
  }

  free(A_fp64);
  free(B_fp64);
  free(bias_fp64);
  free(C_ref_a);
  free(C_ref_b);
  free(A_trans);
  free(B_packed);
  free(C_i32);
  return res;
}

// -----------------------------------------------------------------------------
// 3. Pair 3: FP32 Attention BatchMatMul Verifiers (XNNPACK SME2 vs YNNPACK SME2)
// -----------------------------------------------------------------------------

static kernel_verification_result_t verify_xnn_pf32_sme2(void) {
  kernel_verification_result_t res;
  memset(&res, 0, sizeof(res));
  const int M = 32, N = 32, K = 64;

  double *A_fp64 = (double *)calloc(M * K, sizeof(double));
  double *B_fp64 = (double *)calloc(N * K, sizeof(double));
  double *bias_fp64 = (double *)calloc(N, sizeof(double));
  double *C_ref = (double *)calloc(M * N, sizeof(double));

  float *A_f32 = (float *)alloc_aligned(M * K * sizeof(float));
  float *B_f32 = (float *)alloc_aligned(N * K * sizeof(float));
  float *bias_f32 = (float *)alloc_aligned(N * sizeof(float));
  uint8_t *A_packed = (uint8_t *)alloc_aligned(M * K * sizeof(float) + 256);
  uint8_t *B_packed = (uint8_t *)alloc_aligned(N * (K + 1) * sizeof(float) + 256);
  float *C_out = (float *)alloc_aligned(M * N * sizeof(float));
  aeb_xnn_f32_minmax_params_t params = {{-FLT_MAX, FLT_MAX}};

  for (int d = 0; d < NUM_DISTS; ++d) {
    fill_fp64_tensors((dist_kind_t)d, M, N, K, A_fp64, B_fp64, bias_fp64);
    for (int i = 0; i < M * K; ++i) A_f32[i] = (float)A_fp64[i];
    for (int i = 0; i < N * K; ++i) B_f32[i] = (float)B_fp64[i];
    for (int i = 0; i < N; ++i) bias_f32[i] = (float)bias_fp64[i];

    const void *ptrs[16];
    for (int b = 0; b < 2; ++b) {
      for (int r = 0; r < 16; ++r) ptrs[r] = A_f32 + (b * 16 + r) * K;
      kai_kernel_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme(
          16, K, ptrs, 0, A_packed + (size_t)b * 16 * K * sizeof(float));
      for (int r = 0; r < 16; ++r) ptrs[r] = B_f32 + (b * 16 + r) * K;
      kai_kernel_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme(
          16, K, ptrs, 0, B_packed + (size_t)b * 16 * (K + 1) * sizeof(float),
          bias_f32 + b * 16);
    }

    memset(C_out, 0, M * N * sizeof(float));
    xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2(
        M, N, K * sizeof(float), A_packed, B_packed, C_out, N * sizeof(float),
        sizeof(float), &params);

    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        double acc = bias_fp64[n];
        for (int k = 0; k < K; ++k) acc += A_fp64[m * K + k] * B_fp64[n * K + k];
        C_ref[m * N + n] = acc;
      }
    }
    precision_metrics_t ma =
        compute_fp_metrics(C_out, C_ref, M * N, 16, 120.0, 1e-3);
    precision_metrics_t mb =
        compute_fp_metrics(C_out, C_ref, M * N, 16, 120.0, 1e-3);
    res.level_b_by_dist[d] = mb;
    merge_metrics(&res.level_a, &ma, d);
    merge_metrics(&res.level_b, &mb, d);
  }

  free(A_fp64);
  free(B_fp64);
  free(bias_fp64);
  free(C_ref);
  free(A_f32);
  free(B_f32);
  free(bias_f32);
  free(A_packed);
  free(B_packed);
  free(C_out);
  return res;
}

static kernel_verification_result_t verify_ynn_fp32_sme2(void) {
  kernel_verification_result_t res;
  memset(&res, 0, sizeof(res));
  const int M = 16, N = 64, K = 64;

  double *A_fp64 = (double *)calloc(M * K, sizeof(double));
  double *B_fp64 = (double *)calloc(N * K, sizeof(double));
  double *bias_fp64 = (double *)calloc(N, sizeof(double));
  double *C_ref = (double *)calloc(M * N, sizeof(double));

  float *A_trans = (float *)alloc_aligned(K * M * sizeof(float));
  float *B_trans = (float *)alloc_aligned(K * N * sizeof(float));
  float *C_in = (float *)alloc_aligned(M * N * sizeof(float));
  float *C_out = (float *)alloc_aligned(M * N * sizeof(float));

  for (int d = 0; d < NUM_DISTS; ++d) {
    fill_fp64_tensors((dist_kind_t)d, M, N, K, A_fp64, B_fp64, bias_fp64);
    for (int k = 0; k < K; ++k) {
      for (int m = 0; m < M; ++m) A_trans[k * M + m] = (float)A_fp64[m * K + k];
      for (int n = 0; n < N; ++n) B_trans[k * N + n] = (float)B_fp64[n * K + k];
    }
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) C_in[m * N + n] = (float)bias_fp64[n];
    }

    memset(C_out, 0, M * N * sizeof(float));
    aeb_ynn_dot_fp32_sme2(M, N, 1, 1, K, M * sizeof(float), 0, 0, A_trans, 0, 0,
                          N * sizeof(float), B_trans, N * sizeof(float), C_in,
                          N * sizeof(float), C_out, NULL);

    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        double acc = bias_fp64[n];
        for (int k = 0; k < K; ++k) acc += A_fp64[m * K + k] * B_fp64[n * K + k];
        C_ref[m * N + n] = acc;
      }
    }
    precision_metrics_t ma =
        compute_fp_metrics(C_out, C_ref, M * N, 16, 120.0, 1e-3);
    precision_metrics_t mb =
        compute_fp_metrics(C_out, C_ref, M * N, 16, 120.0, 1e-3);
    res.level_b_by_dist[d] = mb;
    merge_metrics(&res.level_a, &ma, d);
    merge_metrics(&res.level_b, &mb, d);
  }

  free(A_fp64);
  free(B_fp64);
  free(bias_fp64);
  free(C_ref);
  free(A_trans);
  free(B_trans);
  free(C_in);
  free(C_out);
  return res;
}

// -----------------------------------------------------------------------------
// 4. Pair 4: FP32 Attention Softmax Verifier
// -----------------------------------------------------------------------------

static kernel_verification_result_t verify_softmax_pair(int is_ynn) {
  kernel_verification_result_t res;
  memset(&res, 0, sizeof(res));
  const int ROWS = 16, K = 1024;

  float *X_f32 = (float *)alloc_aligned(ROWS * K * sizeof(float));
  float *X_shift = (float *)alloc_aligned(ROWS * K * sizeof(float));
  float *Exp_f32 = (float *)alloc_aligned(ROWS * K * sizeof(float));
  float *P_f32 = (float *)alloc_aligned(ROWS * K * sizeof(float));
  float *P_shift = (float *)alloc_aligned(ROWS * K * sizeof(float));
  double *Exp_ref = (double *)calloc(ROWS * K, sizeof(double));
  double *P_ref = (double *)calloc(ROWS * K, sizeof(double));

  double max_simplex_err = 0.0, max_shift_err = 0.0;

  for (int d = 0; d < NUM_DISTS; ++d) {
    seed_rng(0x9876543210ABCDEFULL + (uint64_t)d * 7919ULL);
    for (int r = 0; r < ROWS; ++r) {
      for (int k = 0; k < K; ++k) {
        double v = 0.0;
        if (d == DIST_NORMAL) {
          v = rand_normal(0.0, 2.0);
        } else if (d == DIST_SATURATION) {
          v = (k % 2 == 0) ? 80.0 : -80.0;
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
        float max_val = -INFINITY, sum_val = 0.0f;
        xnn_f32_rmax_ukernel__neon_u16_acc4(K * sizeof(float), row_in, &max_val,
                                            NULL);
        xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc2(
            K * sizeof(float), row_in, &max_val, row_exp, &sum_val, NULL);
        for (int k = 0; k < K; ++k) row_p[k] = row_exp[k] / sum_val;

        float max_s = -INFINITY, sum_s = 0.0f;
        xnn_f32_rmax_ukernel__neon_u16_acc4(K * sizeof(float), X_shift + r * K,
                                            &max_s, NULL);
        xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc2(
            K * sizeof(float), X_shift + r * K, &max_s, P_shift + r * K, &sum_s,
            NULL);
        for (int k = 0; k < K; ++k) {
          P_shift[r * K + k] /= sum_s;
          double serr = fabs((double)P_shift[r * K + k] - (double)row_p[k]);
          if (serr > max_shift_err) max_shift_err = serr;
        }
      } else {
        float max_val = -INFINITY, sum_val = 0.0f;
        // YNNPACK k1 reductions take k in ELEMENTS (`K`) and stride in BYTES!
        aeb_ynn_max_k1_fp32_neon(1, (size_t)K, K * sizeof(float), row_in,
                                 &max_val, NULL);
        for (int k = 0; k < K; ++k) row_exp[k] = expf(row_in[k] - max_val);
        aeb_ynn_sum_k1_fp32_neon(1, (size_t)K, K * sizeof(float), row_exp,
                                 &sum_val, NULL);
        for (int k = 0; k < K; ++k) row_p[k] = row_exp[k] / sum_val;

        float max_s = -INFINITY, sum_s = 0.0f;
        aeb_ynn_max_k1_fp32_neon(1, (size_t)K, K * sizeof(float),
                                 X_shift + r * K, &max_s, NULL);
        for (int k = 0; k < K; ++k)
          P_shift[r * K + k] = expf(X_shift[r * K + k] - max_s);
        aeb_ynn_sum_k1_fp32_neon(1, (size_t)K, K * sizeof(float),
                                 P_shift + r * K, &sum_s, NULL);
        for (int k = 0; k < K; ++k) {
          P_shift[r * K + k] /= sum_s;
          double serr = fabs((double)P_shift[r * K + k] - (double)row_p[k]);
          if (serr > max_shift_err) max_shift_err = serr;
        }
      }

      double rmax = -DBL_MAX;
      for (int k = 0; k < K; ++k) {
        if ((double)row_in[k] > rmax) rmax = (double)row_in[k];
      }
      double rsum = 0.0, p_row_sum = 0.0;
      for (int k = 0; k < K; ++k) {
        double e = exp((double)row_in[k] - rmax);
        Exp_ref[r * K + k] = e;
        rsum += e;
        p_row_sum += (double)row_p[k];
      }
      for (int k = 0; k < K; ++k) P_ref[r * K + k] = Exp_ref[r * K + k] / rsum;
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
// 5. Pair 7: Matrix Packing / 32-Bit Transpose Verifier (Bit-Exact + Canary)
// -----------------------------------------------------------------------------

static kernel_verification_result_t verify_pack_transpose_pair(int is_ynn) {
  kernel_verification_result_t res;
  memset(&res, 0, sizeof(res));

  const int N_aligned = 64, K_aligned = 64;
  const int N_tail = 53, K_tail = 64;
  const uint32_t CANARY = 0xDEADBEEFU;
  const int GUARD = 64;

  uint32_t *in_buf =
      (uint32_t *)alloc_aligned((N_aligned * K_aligned + 256) * 4);
  uint32_t *bias_buf = (uint32_t *)alloc_aligned(N_aligned * 4);
  size_t out_words = (size_t)N_aligned * (size_t)(K_aligned + 1) + 2 * GUARD;
  uint32_t *out_buf = (uint32_t *)alloc_aligned(out_words * 4);
  uint32_t *ref_buf = (uint32_t *)alloc_aligned(out_words * 4);

  int canary_errors = 0;
  double tail_exact_pct = 100.0;

  for (int d = 0; d < NUM_DISTS; ++d) {
    int N = (d == DIST_TAILS) ? N_tail : N_aligned;
    int K = (d == DIST_TAILS) ? K_tail : K_aligned;
    int nr = 8;
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
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm(
          1, (size_t)N, (size_t)K, (size_t)nr, 1, 1, (size_t)K, in_buf,
          bias_buf, NULL, out_core, 0, NULL);

      for (int p = 0; p < panels; ++p) {
        int n_base = p * nr;
        uint32_t *panel_ref = ref_core + (size_t)p * (size_t)nr * (size_t)(K + 1);
        for (int lane = 0; lane < nr; ++lane) {
          int n = n_base + lane;
          panel_ref[lane] = (n < N) ? bias_buf[n] : 0U;
        }
        for (int k = 0; k < K; ++k) {
          for (int lane = 0; lane < nr; ++lane) {
            int n = n_base + lane;
            panel_ref[nr + k * nr + lane] = (n < N) ? in_buf[n * K + k] : 0U;
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
        aeb_ynn_transpose_x32_neon((size_t)K, (size_t)cur_m,
                                   (size_t)K * sizeof(uint32_t),
                                   (size_t)K * sizeof(uint32_t),
                                   in_buf + (size_t)m_idx * K,
                                   (size_t)N * sizeof(uint32_t),
                                   out_core + (size_t)m_idx);
        for (int n = 0; n < cur_m; ++n) {
          for (int k = 0; k < K; ++k) {
            ref_core[k * N + m_idx + n] = in_buf[(m_idx + n) * K + k];
          }
        }
      }
    }

    for (int i = 0; i < GUARD; ++i) {
      if (out_buf[i] != CANARY) canary_errors++;
      if (out_core[active_words + i] != CANARY) canary_errors++;
    }

    // Check bit-exactness of active words (for XNNPACK packw, ignore don't-care
    // tail padding lanes n >= N where the NEON kernel clamps row pointers)
    int exact = 0;
    size_t checked_words = 0;
    uint64_t max_diff = 0;
    if (!is_ynn) {
      for (int p = 0; p < panels; ++p) {
        int n_base = p * nr;
        const uint32_t *panel_act =
            out_core + (size_t)p * (size_t)nr * (size_t)(K + 1);
        const uint32_t *panel_ref =
            ref_core + (size_t)p * (size_t)nr * (size_t)(K + 1);
        for (int row = 0; row < K + 1; ++row) {
          for (int lane = 0; lane < nr; ++lane) {
            if (n_base + lane >= N) continue;
            checked_words++;
            uint32_t u_act = panel_act[(size_t)row * (size_t)nr + (size_t)lane];
            uint32_t u_ref = panel_ref[(size_t)row * (size_t)nr + (size_t)lane];
            if (u_act == u_ref) {
              exact++;
            } else {
              uint64_t diff = (u_act > u_ref) ? (uint64_t)(u_act - u_ref)
                                              : (uint64_t)(u_ref - u_act);
              if (diff > max_diff) max_diff = diff;
            }
          }
        }
      }
    } else {
      checked_words = active_words;
      for (size_t i = 0; i < active_words; ++i) {
        if (out_core[i] == ref_core[i]) {
          exact++;
        } else {
          uint64_t diff = (out_core[i] > ref_core[i])
                              ? (uint64_t)(out_core[i] - ref_core[i])
                              : (uint64_t)(ref_core[i] - out_core[i]);
          if (diff > max_diff) max_diff = diff;
        }
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
        100.0 * (double)exact / (double)(checked_words ? checked_words : 1);
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
// JSON Output Writer & Main
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

int main(int argc, char **argv) {
  const char *out_path = NULL;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      out_path = argv[++i];
    }
  }

  fprintf(stderr,
          "[verify] Running Level A (exact ukernel) & Level B (FP64 golden) "
          "verification across 7 ARM64 kernel pairs on Apple M5 Max...\n");

  kernel_verification_result_t xnn_res[7], ynn_res[7];

  // Pair 1: 4-Bit Decode GEMV (M=1)
  xnn_res[0] = verify_xnn_qp8_sme2(
      xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2, 1, 64, 64, 4);
  ynn_res[0] = verify_ynn_int4_neondot();

  // Pair 2: 4-Bit Prefill GEMM (M=16)
  xnn_res[1] = verify_xnn_qp8_sme2(
      xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x64c4__neonsme2, 16, 64, 64, 4);
  ynn_res[1] = verify_ynn_sme_int(aeb_ynn_dot_int8_int4_sme, 4);

  // Pair 3: FP32 Attention BatchMatMul (M=32/16)
  xnn_res[2] = verify_xnn_pf32_sme2();
  ynn_res[2] = verify_ynn_fp32_sme2();

  // Pair 4: FP32 Attention Softmax (ROWS=16, K=1024)
  xnn_res[3] = verify_softmax_pair(0);
  ynn_res[3] = verify_softmax_pair(1);

  // Pair 5: 8-Bit Quantized GEMM (M=16)
  xnn_res[4] = verify_xnn_qp8_sme2(
      xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x64c4__neonsme2, 16, 64, 64, 8);
  ynn_res[4] = verify_ynn_sme_int(aeb_ynn_dot_int8_int8_sme2, 8);

  // Pair 6: 2-Bit Weight-Only GEMM (M=16)
  xnn_res[5] = verify_xnn_qp8_sme2(
      xnn_qp8_f32_qc2w_gemm_minmax_ukernel_16x64c4__neonsme2, 16, 64, 64, 2);
  ynn_res[5] = verify_ynn_sme_int(aeb_ynn_dot_int8_int2_sme, 2);

  // Pair 7: Matrix Packing / 32-Bit Transpose
  xnn_res[6] = verify_pack_transpose_pair(0);
  ynn_res[6] = verify_pack_transpose_pair(1);

  const char *pair_ids[7] = {
      "pair_1_qc4w_gemv_decode",   "pair_2_qc4w_gemm_prefill",
      "pair_3_fp32_attn_bmm",      "pair_4_fp32_softmax",
      "pair_5_qc8w_gemm_multimodal", "pair_6_qc2w_gemm_embed",
      "pair_7_pack_transpose_x32"};

  const char *pair_titles[7] = {
      "4-Bit Quantized GEMV (Single-Token Decode: M=1)",
      "4-Bit Quantized GEMM (Batched Prefill: M=16)",
      "FP32 Attention BatchMatMul (SDPA Q*K^T & Scores*V)",
      "FP32 Attention Softmax (Max + Exp-Minus-Max + Sum)",
      "8-Bit Quantized GEMM (Vision ViT & Audio Conformer)",
      "2-Bit Weight-Only Quantized GEMM (Audio Conformer)",
      "Matrix Packing / 32-Bit Transpose (Weight/KV Layout)"};

  const char *xnn_symbols[7] = {
      "xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2",
      "xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x64c4__neonsme2",
      "xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2",
      "xnn_f32_rmax + xnn_f32_raddstoreexpminusmax__neonfma_rr1_lut64_p2",
      "xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x64c4__neonsme2",
      "xnn_qp8_f32_qc2w_gemm_minmax_ukernel_16x64c4__neonsme2",
      "xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm"};

  const char *ynn_symbols[7] = {
      "ynn::dot_int8_int4_int32_1x32x8_1x4x8_neondot",
      "ynn::dot_int8_int4_int32_sme",
      "ynn::dot_fp32_sme2",
      "ynn::max_k1_fp32_neon + ynn::sum_k1_fp32_neon",
      "ynn::dot_int8_int8_int32_sme2",
      "ynn::dot_int8_int2_int32_sme",
      "ynn::transpose_x32_neon"};

  FILE *fp = stdout;
  if (out_path && out_path[0]) {
    fp = fopen(out_path, "w");
    if (!fp) {
      perror("fopen output");
      return 1;
    }
  }

  int total_pass = 0;
  for (int i = 0; i < 7; ++i) {
    if (xnn_res[i].level_a.pass && xnn_res[i].level_b.pass) total_pass++;
    if (ynn_res[i].level_a.pass && ynn_res[i].level_b.pass) total_pass++;
  }

  fprintf(fp, "{\n");
  fprintf(fp, "  \"suite\": \"litert_microkernel_precision_verification\",\n");
  fprintf(fp, "  \"arch\": \"arm64_apple_m5_max\",\n");
  fprintf(fp, "  \"execution_mode\": \"direct_link_hardware_sme2\",\n");
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
  return (total_pass == 14) ? 0 : 1;
}
