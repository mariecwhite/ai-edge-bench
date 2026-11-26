// Shared Gemma 4 31B Level-2 (Layer Tier) Benchmark, Roofline & FP64 Golden
// Reference Utilities.
//
// Used identically by:
//   - layer/subgraphs/litert/bench_litert_layers.cc (LiteRT XNNPACK, LiteRT
//     YNNPACK decomposed/SDPA/hybrid/pure-blockwise, Direct YNNPACK C API)
//   - layer/subgraphs/ggml/bench_ggml_layers.cpp    (llama.cpp / ggml CPU
//     Flash Attention with KleidiAI NEON/i8mm and KleidiAI SME2)
//
// Provides:
//   1. Exact Gemma 4 31B (`google/gemma-4-31B-it`) layer hyperparameters.
//   2. Fast deterministic weight & activation initialization (identical seed
//      across every runtime and the FP64 reference).
//   3. Symmetric INT4 quantization (per-channel QC4W, blockwise-32 QB4W, and
//      FP16-scale Q4_0) with dequantization helpers for two-level verification.
//   4. Independent FP64 golden reference for all 5 canonical subgraphs
//      (local_swa_attn, global_full_attn, geglu_mlp, full_local_block,
//      full_global_block) supporting both unquantized (Level B) and
//      dequantized-weight (Level A) evaluation.
//   5. Whole-subgraph roofline op/byte accounting (boundary tensors only).
//   6. High-precision macOS timing (`CLOCK_MONOTONIC_RAW`), QoS setup, SLC
//      eviction flush for cold M=1 decode, 7-trial statistics, and memory
//      accounting (`phys_footprint` + `ru_maxrss`).

#ifndef AEB_LAYER_GEMMA4_COMMON_H_
#define AEB_LAYER_GEMMA4_COMMON_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <time.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task_info.h>
#include <pthread/qos.h>
#endif

namespace aeb_layer {

// -----------------------------------------------------------------------------
// Gemma 4 31B (`google/gemma-4-31B-it`) Architectural Constants
// -----------------------------------------------------------------------------

static constexpr int kHiddenSize = 5376;
static constexpr int kIntermediateSize = 21504;
static constexpr float kRmsNormEps = 1e-6f;
static constexpr float kAttentionScale = 1.0f;  // Gemma 4 uses QK/V-norm, scale=1.0
static constexpr int kBlockSize32 = 32;

// Local Sliding-Window Attention (50 of 60 layers)
static constexpr int kLocalNumHeads = 32;
static constexpr int kLocalNumKvHeads = 16;
static constexpr int kLocalHeadDim = 256;
static constexpr int kLocalSlidingWindow = 1024;
static constexpr float kLocalRopeTheta = 10000.0f;
static constexpr int kLocalRotaryDim = 256;  // partial_rotary_factor = 1.0

// Global Full Attention with K=V sharing (10 of 60 layers)
static constexpr int kGlobalNumHeads = 32;
static constexpr int kGlobalNumKvHeads = 4;
static constexpr int kGlobalHeadDim = 512;
static constexpr float kGlobalRopeTheta = 1000000.0f;
static constexpr int kGlobalRotaryDim = 128;  // partial_rotary_factor = 0.25

static constexpr double kPi = 3.14159265358979323846;
static constexpr double kSqrt2OverPi = 0.79788456080286535588;

// -----------------------------------------------------------------------------
// Timing, QoS, Memory Accounting & Cold-Cache SLC Flush
// -----------------------------------------------------------------------------

inline double NowSec() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

inline void SetHighQoS() {
#ifdef __APPLE__
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

struct MemoryMetrics {
  double phys_footprint_mb = 0.0;
  double peak_rss_mb = 0.0;
};

inline MemoryMetrics ReadMemoryMetrics() {
  MemoryMetrics m;
  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) == 0) {
#ifdef __APPLE__
    m.peak_rss_mb = static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);
#else
    m.peak_rss_mb = static_cast<double>(ru.ru_maxrss) / 1024.0;
#endif
  }
#ifdef __APPLE__
  task_vm_info_data_t vm_info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO,
                reinterpret_cast<task_info_t>(&vm_info), &count) ==
      KERN_SUCCESS) {
    m.phys_footprint_mb =
        static_cast<double>(vm_info.phys_footprint) / (1024.0 * 1024.0);
  }
#endif
  return m;
}

// 96 MB cache eviction buffer used outside timing windows before cold M=1
// decode invocations so single-layer weights stream from LPDDR5X Unified Memory
// rather than hitting Apple M5 Max's 32 MiB L2 / shared SLC.
class SlcFlusher {
 public:
  explicit SlcFlusher(size_t bytes = 96ULL * 1024 * 1024)
      : buf_(bytes / sizeof(uint64_t), 0x9e3779b97f4a7c15ULL) {}

  void Flush() {
    uint64_t acc = sink_;
    // Stride 128 bytes (Apple Silicon cache line = 128 B = 16 uint64_t)
    const size_t n = buf_.size();
    for (size_t i = 0; i < n; i += 16) {
      acc += buf_[i];
      buf_[i] = acc;
    }
    sink_ = acc;
  }

  volatile uint64_t sink_ = 0;

 private:
  std::vector<uint64_t> buf_;
};

// -----------------------------------------------------------------------------
// Deterministic Fast PRNG & Tensor Initializers
// -----------------------------------------------------------------------------

inline uint64_t SplitMix64(uint64_t& state) {
  uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

// Fast pseudo-Gaussian in [-3*sigma, +3*sigma] using sum of 3 uniform draws
// (exact variance = sigma^2, deterministic across platforms).
inline void FillNormal(std::vector<float>& dst, size_t n, uint64_t seed,
                       float sigma, float mean = 0.0f) {
  dst.resize(n);
  uint64_t st = seed ^ 0x243f6a8885a308d3ULL;
  // Sum of 3 independent U(-1, 1) has variance 3 * (1/3) = 1.0!
  constexpr double kInv2_21 = 1.0 / 2097152.0;
  for (size_t i = 0; i < n; ++i) {
    uint64_t r = SplitMix64(st);
    double u0 = (static_cast<double>(r & 0x1FFFFFULL) * kInv2_21) * 2.0 - 1.0;
    double u1 =
        (static_cast<double>((r >> 21) & 0x1FFFFFULL) * kInv2_21) * 2.0 - 1.0;
    double u2 =
        (static_cast<double>((r >> 42) & 0x1FFFFFULL) * kInv2_21) * 2.0 - 1.0;
    dst[i] = static_cast<float>(mean + sigma * (u0 + u1 + u2));
  }
}

// -----------------------------------------------------------------------------
// IEEE-754 FP16 Conversion Helpers (for Blockwise FP16 Scales & Q4_0 Parity)
// -----------------------------------------------------------------------------

inline uint16_t FloatToFp16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant = (mant | 0x800000u) >> (1 - exp);
    return static_cast<uint16_t>(sign | ((mant + 0x1000u) >> 13));
  }
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00u);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                               ((mant + 0x1000u) >> 13));
}

inline float Fp16ToFloat(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      exp = 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FFu;
      f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7F800000u | (mant << 13);
  } else {
    f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

// -----------------------------------------------------------------------------
// Quantized INT4 Linear Weight Representation (Per-Channel QC4W & Blockwise-32)
// -----------------------------------------------------------------------------

struct QuantLinearInt4 {
  int N = 0;  // out_features
  int K = 0;  // in_features (must be divisible by 32)
  int block_size = 0;  // 0 = per-channel QC4W, 32 = blockwise-32 QB4W

  // Packed signed 4-bit nibbles in [N, K/2] bytes:
  //   byte = (q[2i] & 0x0F) | ((q[2i+1] & 0x0F) << 4)
  // Matches TFLite TensorType_INT4 and YNNPACK ynn_type_int4 row-major layout.
  std::vector<uint8_t> packed_i4;

  // Per-channel FP32 scales [N] (when block_size == 0)
  std::vector<float> pc_scales;

  // Blockwise FP16 scales [N, K/32] (when block_size == 32, TFLite QB4W format)
  std::vector<uint16_t> bw_scales_f16;
  // Blockwise FP32 scales [N, K/32] (for Direct YNNPACK & FP64 dequant ref)
  std::vector<float> bw_scales_f32;
};

inline QuantLinearInt4 QuantizeLinearQC4W(const std::vector<float>& w_fp32,
                                          int N, int K) {
  QuantLinearInt4 q;
  q.N = N;
  q.K = K;
  q.block_size = 0;
  q.packed_i4.assign(static_cast<size_t>(N) * (K / 2), 0);
  q.pc_scales.resize(N);
  for (int n = 0; n < N; ++n) {
    const float* row = w_fp32.data() + static_cast<size_t>(n) * K;
    float amax = 0.0f;
    for (int k = 0; k < K; ++k) {
      float a = std::fabs(row[k]);
      if (a > amax) amax = a;
    }
    float scale = (amax > 0.0f) ? (amax / 7.0f) : 1.0f;
    q.pc_scales[n] = scale;
    float inv_s = 1.0f / scale;
    uint8_t* dst = q.packed_i4.data() + static_cast<size_t>(n) * (K / 2);
    for (int k = 0; k < K; k += 2) {
      int q0 = static_cast<int>(std::lrint(row[k] * inv_s));
      int q1 = static_cast<int>(std::lrint(row[k + 1] * inv_s));
      q0 = std::max(-7, std::min(7, q0));
      q1 = std::max(-7, std::min(7, q1));
      dst[k / 2] = static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
    }
  }
  return q;
}

inline QuantLinearInt4 QuantizeLinearQB4W32(const std::vector<float>& w_fp32,
                                            int N, int K,
                                            bool round_scales_f16 = true) {
  QuantLinearInt4 q;
  q.N = N;
  q.K = K;
  q.block_size = 32;
  const int nb = K / 32;
  q.packed_i4.assign(static_cast<size_t>(N) * (K / 2), 0);
  q.bw_scales_f16.resize(static_cast<size_t>(N) * nb);
  q.bw_scales_f32.resize(static_cast<size_t>(N) * nb);
  for (int n = 0; n < N; ++n) {
    const float* row = w_fp32.data() + static_cast<size_t>(n) * K;
    uint8_t* dst = q.packed_i4.data() + static_cast<size_t>(n) * (K / 2);
    for (int b = 0; b < nb; ++b) {
      float amax = 0.0f;
      for (int j = 0; j < 32; ++j) {
        float a = std::fabs(row[b * 32 + j]);
        if (a > amax) amax = a;
      }
      float s_raw = (amax > 0.0f) ? (amax / 7.0f) : 1.0f;
      uint16_t s_h = FloatToFp16(s_raw);
      float scale = round_scales_f16 ? Fp16ToFloat(s_h) : s_raw;
      if (scale == 0.0f) scale = s_raw;
      q.bw_scales_f16[static_cast<size_t>(n) * nb + b] = s_h;
      q.bw_scales_f32[static_cast<size_t>(n) * nb + b] = scale;
      float inv_s = 1.0f / scale;
      for (int j = 0; j < 32; j += 2) {
        int q0 = static_cast<int>(std::lrint(row[b * 32 + j] * inv_s));
        int q1 = static_cast<int>(std::lrint(row[b * 32 + j + 1] * inv_s));
        q0 = std::max(-7, std::min(7, q0));
        q1 = std::max(-7, std::min(7, q1));
        dst[(b * 32 + j) / 2] =
            static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
      }
    }
  }
  return q;
}

inline std::vector<float> DequantizeLinearInt4(const QuantLinearInt4& q) {
  std::vector<float> out(static_cast<size_t>(q.N) * q.K);
  const int nb = q.K / 32;
  for (int n = 0; n < q.N; ++n) {
    const uint8_t* src = q.packed_i4.data() + static_cast<size_t>(n) * (q.K / 2);
    float* dst = out.data() + static_cast<size_t>(n) * q.K;
    for (int k = 0; k < q.K; k += 2) {
      uint8_t byte = src[k / 2];
      int8_t q0 = static_cast<int8_t>(byte << 4) >> 4;
      int8_t q1 = static_cast<int8_t>(byte) >> 4;
      float s = (q.block_size == 32)
                    ? q.bw_scales_f32[static_cast<size_t>(n) * nb + (k / 32)]
                    : q.pc_scales[n];
      dst[k] = static_cast<float>(q0) * s;
      dst[k + 1] = static_cast<float>(q1) * s;
    }
  }
  return out;
}

// -----------------------------------------------------------------------------
// RoPE Tables (Half-Split / NeoX Style Matching Gemma 4)
// -----------------------------------------------------------------------------

// Builds cos/sin tables of shape [T, D_h] for token positions `pos_start..pos_start+T-1`.
// For proportional RoPE (rot_dim < D_h), entries in [rot_dim, D_h) have cos=1, sin=0
// so the identity pass-through is exact.
// Frequencies match Gemma 4 proportional RoPE:
//   freq_i = theta^(-2*i / D_h) for i in [0, rot_dim/2).
inline void BuildRopeCosSin(int T, int pos_start, int head_dim, int rot_dim,
                            float theta, std::vector<float>& cos_out,
                            std::vector<float>& sin_out) {
  cos_out.assign(static_cast<size_t>(T) * head_dim, 1.0f);
  sin_out.assign(static_cast<size_t>(T) * head_dim, 0.0f);
  const int half_rot = rot_dim / 2;
  for (int t = 0; t < T; ++t) {
    double pos = static_cast<double>(pos_start + t);
    float* c_row = cos_out.data() + static_cast<size_t>(t) * head_dim;
    float* s_row = sin_out.data() + static_cast<size_t>(t) * head_dim;
    for (int i = 0; i < half_rot; ++i) {
      double inv_freq =
          std::pow(static_cast<double>(theta),
                   -2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
      double angle = pos * inv_freq;
      float cv = static_cast<float>(std::cos(angle));
      float sv = static_cast<float>(std::sin(angle));
      c_row[i] = cv;
      c_row[i + half_rot] = cv;
      s_row[i] = sv;
      s_row[i + half_rot] = sv;
    }
  }
}

// -----------------------------------------------------------------------------
// Deterministic Layer Weights & Input Fixtures
// -----------------------------------------------------------------------------

struct AttnWeightsFp32 {
  int hidden = kHiddenSize;
  int n_heads = 32;
  int n_kv_heads = 16;
  int head_dim = 256;
  int rot_dim = 256;
  int window = 1024;  // 0 = full global attention
  float rope_theta = kLocalRopeTheta;
  bool k_eq_v = false;

  std::vector<float> pre_norm;   // [D]
  std::vector<float> wq;         // [n_heads * head_dim, D]
  std::vector<float> wk;         // [n_kv_heads * head_dim, D]
  std::vector<float> wv;         // [n_kv_heads * head_dim, D] (empty if k_eq_v)
  std::vector<float> q_norm;     // [head_dim]
  std::vector<float> k_norm;     // [head_dim]
  std::vector<float> wo;         // [D, n_heads * head_dim]
  std::vector<float> post_norm;  // [D]
};

struct MlpWeightsFp32 {
  int hidden = kHiddenSize;
  int inter = kIntermediateSize;
  std::vector<float> pre_norm;   // [D]
  std::vector<float> w_gate;     // [F, D]
  std::vector<float> w_up;       // [F, D]
  std::vector<float> w_down;     // [D, F]
  std::vector<float> post_norm;  // [D]
};

inline AttnWeightsFp32 MakeAttnWeights(bool is_global, uint64_t seed = 42) {
  AttnWeightsFp32 w;
  w.hidden = kHiddenSize;
  if (!is_global) {
    w.n_heads = kLocalNumHeads;
    w.n_kv_heads = kLocalNumKvHeads;
    w.head_dim = kLocalHeadDim;
    w.rot_dim = kLocalRotaryDim;
    w.window = kLocalSlidingWindow;
    w.rope_theta = kLocalRopeTheta;
    w.k_eq_v = false;
  } else {
    w.n_heads = kGlobalNumHeads;
    w.n_kv_heads = kGlobalNumKvHeads;
    w.head_dim = kGlobalHeadDim;
    w.rot_dim = kGlobalRotaryDim;
    w.window = 0;
    w.rope_theta = kGlobalRopeTheta;
    w.k_eq_v = true;
  }
  const int q_dim = w.n_heads * w.head_dim;
  const int kv_dim = w.n_kv_heads * w.head_dim;
  const float sig_in = 1.0f / std::sqrt(static_cast<float>(w.hidden));
  const float sig_o = 1.0f / std::sqrt(static_cast<float>(q_dim));

  FillNormal(w.pre_norm, w.hidden, seed + 1, 0.05f, 1.0f);
  FillNormal(w.wq, static_cast<size_t>(q_dim) * w.hidden, seed + 2, sig_in);
  FillNormal(w.wk, static_cast<size_t>(kv_dim) * w.hidden, seed + 3, sig_in);
  if (!w.k_eq_v) {
    FillNormal(w.wv, static_cast<size_t>(kv_dim) * w.hidden, seed + 4, sig_in);
  }
  FillNormal(w.q_norm, w.head_dim, seed + 5, 0.05f, 1.0f);
  FillNormal(w.k_norm, w.head_dim, seed + 6, 0.05f, 1.0f);
  FillNormal(w.wo, static_cast<size_t>(w.hidden) * q_dim, seed + 7, sig_o);
  FillNormal(w.post_norm, w.hidden, seed + 8, 0.05f, 1.0f);
  return w;
}

inline MlpWeightsFp32 MakeMlpWeights(uint64_t seed = 142) {
  MlpWeightsFp32 w;
  w.hidden = kHiddenSize;
  w.inter = kIntermediateSize;
  const float sig_up = 1.0f / std::sqrt(static_cast<float>(w.hidden));
  const float sig_dn = 1.0f / std::sqrt(static_cast<float>(w.inter));
  FillNormal(w.pre_norm, w.hidden, seed + 1, 0.05f, 1.0f);
  FillNormal(w.w_gate, static_cast<size_t>(w.inter) * w.hidden, seed + 2, sig_up);
  FillNormal(w.w_up, static_cast<size_t>(w.inter) * w.hidden, seed + 3, sig_up);
  FillNormal(w.w_down, static_cast<size_t>(w.hidden) * w.inter, seed + 4, sig_dn);
  FillNormal(w.post_norm, w.hidden, seed + 5, 0.05f, 1.0f);
  return w;
}

// -----------------------------------------------------------------------------
// Independent FP64 Golden Reference Implementations (All 5 Subgraphs)
// -----------------------------------------------------------------------------

namespace ref_fp64 {

inline void RmsNorm(const double* x, const float* weight, int rows, int dim,
                    double* out) {
  for (int r = 0; r < rows; ++r) {
    const double* xr = x + static_cast<size_t>(r) * dim;
    double* yr = out + static_cast<size_t>(r) * dim;
    double sum_sq = 0.0;
    for (int d = 0; d < dim; ++d) sum_sq += xr[d] * xr[d];
    double inv = 1.0 / std::sqrt(sum_sq / static_cast<double>(dim) +
                                 static_cast<double>(kRmsNormEps));
    if (weight) {
      for (int d = 0; d < dim; ++d) yr[d] = xr[d] * inv * weight[d];
    } else {
      for (int d = 0; d < dim; ++d) yr[d] = xr[d] * inv;
    }
  }
}

inline void Linear(const double* x, const float* w, int M, int N, int K,
                   double* out) {
  for (int m = 0; m < M; ++m) {
    const double* xm = x + static_cast<size_t>(m) * K;
    double* ym = out + static_cast<size_t>(m) * N;
    for (int n = 0; n < N; ++n) {
      const float* wn = w + static_cast<size_t>(n) * K;
      double acc = 0.0;
      for (int k = 0; k < K; ++k) acc += xm[k] * static_cast<double>(wn[k]);
      ym[n] = acc;
    }
  }
}

inline void ApplyRopeHalfSplit(double* x, int T, int n_heads, int head_dim,
                               int rot_dim, const std::vector<float>& cos_tab,
                               const std::vector<float>& sin_tab) {
  const int half = rot_dim / 2;
  for (int t = 0; t < T; ++t) {
    const float* c = cos_tab.data() + static_cast<size_t>(t) * head_dim;
    const float* s = sin_tab.data() + static_cast<size_t>(t) * head_dim;
    for (int h = 0; h < n_heads; ++h) {
      double* v = x + (static_cast<size_t>(t) * n_heads + h) * head_dim;
      for (int i = 0; i < half; ++i) {
        double x0 = v[i];
        double x1 = v[i + half];
        double cv = static_cast<double>(c[i]);
        double sv = static_cast<double>(s[i]);
        v[i] = x0 * cv - x1 * sv;
        v[i + half] = x0 * sv + x1 * cv;
      }
    }
  }
}

// Reference Gemma 4 Attention (Local SWA or Global K=V) with pre-populated KV
// cache history of length `S_past` (total active keys = S_kv = S_past + T).
// `k_past`: [n_kv_heads, S_past, head_dim]
// `v_past`: [n_kv_heads, S_past, head_dim]
// `x_in`:   [T, hidden]
// Returns residual output [T, hidden].
inline std::vector<double> RunAttnFp64(
    const AttnWeightsFp32& w, const std::vector<double>& x_in, int T,
    int S_past, const std::vector<float>& k_past,
    const std::vector<float>& v_past) {
  const int D = w.hidden;
  const int Hq = w.n_heads;
  const int Hkv = w.n_kv_heads;
  const int Dh = w.head_dim;
  const int g = Hq / Hkv;
  const int S_kv = S_past + T;

  std::vector<double> norm_x(static_cast<size_t>(T) * D);
  RmsNorm(x_in.data(), w.pre_norm.data(), T, D, norm_x.data());

  std::vector<double> q(static_cast<size_t>(T) * Hq * Dh);
  std::vector<double> k_new(static_cast<size_t>(T) * Hkv * Dh);
  std::vector<double> v_new(static_cast<size_t>(T) * Hkv * Dh);

  Linear(norm_x.data(), w.wq.data(), T, Hq * Dh, D, q.data());
  Linear(norm_x.data(), w.wk.data(), T, Hkv * Dh, D, k_new.data());
  if (w.k_eq_v) {
    v_new = k_new;
  } else {
    Linear(norm_x.data(), w.wv.data(), T, Hkv * Dh, D, v_new.data());
  }

  RmsNorm(q.data(), w.q_norm.data(), T * Hq, Dh, q.data());
  RmsNorm(k_new.data(), w.k_norm.data(), T * Hkv, Dh, k_new.data());
  RmsNorm(v_new.data(), nullptr, T * Hkv, Dh, v_new.data());

  std::vector<float> cos_tab, sin_tab;
  BuildRopeCosSin(T, S_past, Dh, w.rot_dim, w.rope_theta, cos_tab, sin_tab);
  ApplyRopeHalfSplit(q.data(), T, Hq, Dh, w.rot_dim, cos_tab, sin_tab);
  ApplyRopeHalfSplit(k_new.data(), T, Hkv, Dh, w.rot_dim, cos_tab, sin_tab);

  // Assemble full K, V of shape [Hkv, S_kv, Dh]
  std::vector<double> k_full(static_cast<size_t>(Hkv) * S_kv * Dh);
  std::vector<double> v_full(static_cast<size_t>(Hkv) * S_kv * Dh);
  for (int h = 0; h < Hkv; ++h) {
    for (int s = 0; s < S_past; ++s) {
      for (int d = 0; d < Dh; ++d) {
        size_t dst = (static_cast<size_t>(h) * S_kv + s) * Dh + d;
        size_t src = (static_cast<size_t>(h) * S_past + s) * Dh + d;
        k_full[dst] = static_cast<double>(k_past[src]);
        v_full[dst] = static_cast<double>(v_past[src]);
      }
    }
    for (int t = 0; t < T; ++t) {
      for (int d = 0; d < Dh; ++d) {
        size_t dst = (static_cast<size_t>(h) * S_kv + (S_past + t)) * Dh + d;
        size_t src = (static_cast<size_t>(t) * Hkv + h) * Dh + d;
        k_full[dst] = k_new[src];
        v_full[dst] = v_new[src];
      }
    }
  }

  // Attention: O[t, hq, d]
  std::vector<double> attn_ctx(static_cast<size_t>(T) * Hq * Dh, 0.0);
  std::vector<double> scores(S_kv);
  for (int hq = 0; hq < Hq; ++hq) {
    int hkv = hq / g;
    for (int t = 0; t < T; ++t) {
      int q_pos = S_past + t;
      int s_min = (w.window > 0) ? std::max(0, q_pos - w.window + 1) : 0;
      int s_max = q_pos;  // causal inclusive
      const double* qv = q.data() + (static_cast<size_t>(t) * Hq + hq) * Dh;
      double max_s = -1e300;
      for (int s = s_min; s <= s_max; ++s) {
        const double* kv =
            k_full.data() + (static_cast<size_t>(hkv) * S_kv + s) * Dh;
        double dot = 0.0;
        for (int d = 0; d < Dh; ++d) dot += qv[d] * kv[d];
        dot *= static_cast<double>(kAttentionScale);
        scores[s] = dot;
        if (dot > max_s) max_s = dot;
      }
      double sum_exp = 0.0;
      for (int s = s_min; s <= s_max; ++s) {
        scores[s] = std::exp(scores[s] - max_s);
        sum_exp += scores[s];
      }
      double inv_sum = 1.0 / sum_exp;
      double* ov = attn_ctx.data() + (static_cast<size_t>(t) * Hq + hq) * Dh;
      for (int s = s_min; s <= s_max; ++s) {
        double p = scores[s] * inv_sum;
        const double* vv =
            v_full.data() + (static_cast<size_t>(hkv) * S_kv + s) * Dh;
        for (int d = 0; d < Dh; ++d) ov[d] += p * vv[d];
      }
    }
  }

  std::vector<double> o_proj(static_cast<size_t>(T) * D);
  Linear(attn_ctx.data(), w.wo.data(), T, D, Hq * Dh, o_proj.data());

  std::vector<double> post(static_cast<size_t>(T) * D);
  RmsNorm(o_proj.data(), w.post_norm.data(), T, D, post.data());

  std::vector<double> out(static_cast<size_t>(T) * D);
  for (size_t i = 0; i < out.size(); ++i) out[i] = post[i] + x_in[i];
  return out;
}

inline std::vector<double> RunMlpFp64(const MlpWeightsFp32& w,
                                      const std::vector<double>& x_in, int T) {
  const int D = w.hidden;
  const int F = w.inter;

  std::vector<double> norm_x(static_cast<size_t>(T) * D);
  RmsNorm(x_in.data(), w.pre_norm.data(), T, D, norm_x.data());

  std::vector<double> gate(static_cast<size_t>(T) * F);
  std::vector<double> up(static_cast<size_t>(T) * F);
  Linear(norm_x.data(), w.w_gate.data(), T, F, D, gate.data());
  Linear(norm_x.data(), w.w_up.data(), T, F, D, up.data());

  std::vector<double> act(static_cast<size_t>(T) * F);
  for (size_t i = 0; i < act.size(); ++i) {
    double g = gate[i];
    double gelu =
        0.5 * g * (1.0 + std::tanh(kSqrt2OverPi * (g + 0.044715 * g * g * g)));
    act[i] = gelu * up[i];
  }

  std::vector<double> down(static_cast<size_t>(T) * D);
  Linear(act.data(), w.w_down.data(), T, D, F, down.data());

  std::vector<double> post(static_cast<size_t>(T) * D);
  RmsNorm(down.data(), w.post_norm.data(), T, D, post.data());

  std::vector<double> out(static_cast<size_t>(T) * D);
  for (size_t i = 0; i < out.size(); ++i) out[i] = post[i] + x_in[i];
  return out;
}

inline std::vector<double> RunFullBlockFp64(
    const AttnWeightsFp32& aw, const MlpWeightsFp32& mw, float layer_scalar,
    const std::vector<double>& x_in, int T, int S_past,
    const std::vector<float>& k_past, const std::vector<float>& v_past) {
  auto h = RunAttnFp64(aw, x_in, T, S_past, k_past, v_past);
  auto out = RunMlpFp64(mw, h, T);
  if (layer_scalar != 1.0f) {
    for (double& v : out) v *= static_cast<double>(layer_scalar);
  }
  return out;
}

}  // namespace ref_fp64

// -----------------------------------------------------------------------------
// Numerical Parity Metrics (Cosine Sim, Rel L2, SQNR dB, KL-Div, NaN/Inf)
// -----------------------------------------------------------------------------

struct ParityMetrics {
  double cosine_sim = 0.0;
  double rel_l2_err = 0.0;
  double sqnr_db = 0.0;
  double max_abs_err = 0.0;
  double kl_div = 0.0;
  int nan_inf_count = 0;
  bool passed = false;
};

inline ParityMetrics CompareAgainstFp64(const float* impl, const double* ref,
                                        int rows, int cols,
                                        double min_cos, double max_rel_l2,
                                        double min_sqnr, double max_kl) {
  ParityMetrics m;
  const size_t n = static_cast<size_t>(rows) * cols;
  double dot = 0.0, norm_a = 0.0, norm_b = 0.0, err_sq = 0.0;
  for (size_t i = 0; i < n; ++i) {
    float v = impl[i];
    if (!std::isfinite(v)) {
      m.nan_inf_count++;
      continue;
    }
    double a = static_cast<double>(v);
    double b = ref[i];
    dot += a * b;
    norm_a += a * a;
    norm_b += b * b;
    double d = a - b;
    err_sq += d * d;
    if (std::fabs(d) > m.max_abs_err) m.max_abs_err = std::fabs(d);
  }
  m.cosine_sim = (norm_a > 0.0 && norm_b > 0.0)
                     ? (dot / (std::sqrt(norm_a) * std::sqrt(norm_b)))
                     : 0.0;
  m.rel_l2_err = (norm_b > 0.0) ? std::sqrt(err_sq / norm_b) : 0.0;
  m.sqnr_db = (err_sq > 0.0 && norm_b > 0.0)
                  ? (10.0 * std::log10(norm_b / err_sq))
                  : 200.0;

  // Row-wise softmax KL divergence: KL(softmax(ref) || softmax(impl))
  double kl_sum = 0.0;
  for (int r = 0; r < rows; ++r) {
    const double* rr = ref + static_cast<size_t>(r) * cols;
    const float* ir = impl + static_cast<size_t>(r) * cols;
    double max_r = -1e300, max_i = -1e300;
    for (int c = 0; c < cols; ++c) {
      if (rr[c] > max_r) max_r = rr[c];
      if (std::isfinite(ir[c]) && ir[c] > max_i) max_i = ir[c];
    }
    double sum_r = 0.0, sum_i = 0.0;
    for (int c = 0; c < cols; ++c) {
      sum_r += std::exp(rr[c] - max_r);
      if (std::isfinite(ir[c])) sum_i += std::exp(static_cast<double>(ir[c]) - max_i);
    }
    double log_zr = std::log(sum_r);
    double log_zi = std::log(std::max(sum_i, 1e-300));
    double row_kl = 0.0;
    for (int c = 0; c < cols; ++c) {
      double log_p = (rr[c] - max_r) - log_zr;
      double p = std::exp(log_p);
      double log_q =
          (std::isfinite(ir[c]) ? (static_cast<double>(ir[c]) - max_i) : -1e6) -
          log_zi;
      if (p > 1e-30) row_kl += p * (log_p - log_q);
    }
    kl_sum += std::max(0.0, row_kl);
  }
  m.kl_div = (rows > 0) ? (kl_sum / rows) : 0.0;
  m.passed = (m.nan_inf_count == 0) && (m.cosine_sim >= min_cos) &&
             (m.rel_l2_err <= max_rel_l2) && (m.sqnr_db >= min_sqnr) &&
             (m.kl_div <= max_kl);
  return m;
}

// -----------------------------------------------------------------------------
// Whole-Subgraph Roofline Accounting (FLOPs & Boundary-Only Bytes)
// -----------------------------------------------------------------------------

struct RooflineSpec {
  double int4_ops = 0.0;
  double fp32_ops = 0.0;
  double total_ops = 0.0;
  double weight_params = 0.0;
  double weight_bytes_format = 0.0;  // Actual backend weight bytes (incl scales)
  double weight_bytes_ideal = 0.0;   // Canonical 0.5 B/param + norm FP32 weights
  double kv_bytes_format = 0.0;      // Actual backend KV read+write bytes
  double kv_bytes_ideal = 0.0;       // Canonical FP16 KV read+write bytes
  double io_bytes = 0.0;             // Boundary input + output FP32 tensors
  double min_bytes_format = 0.0;     // Total B_min (actual backend format)
  double min_bytes_ideal = 0.0;      // Total B_min (algorithmic ideal)
};

// `weight_quant`: "qc4w" | "qb4w32" | "q4_0"
// `kv_dtype`:     "fp32" | "fp16"
inline double WeightBytesForFormat(double num_params, double num_rows,
                                   const std::string& weight_quant) {
  double nibbles = num_params * 0.5;
  if (weight_quant == "qc4w") return nibbles + num_rows * 4.0;
  if (weight_quant == "qb4w32") return nibbles + (num_params / 32.0) * 2.0;  // FP16 block scales in TFLite
  if (weight_quant == "qb4w32_f32scale") return nibbles + (num_params / 32.0) * 4.0;
  if (weight_quant == "q4_0") return nibbles + (num_params / 32.0) * 2.0;  // 18 B / 32 weights
  return nibbles;
}

inline RooflineSpec ComputeSubgraphRoofline(const std::string& subgraph_id,
                                            int T, int S_kv,
                                            const std::string& weight_quant,
                                            const std::string& kv_dtype) {
  RooflineSpec r;
  const double kv_elem_b = (kv_dtype == "fp16") ? 2.0 : 4.0;
  const double D = kHiddenSize;
  const double F = kIntermediateSize;

  auto add_attn = [&](bool is_global) {
    const double Hq = is_global ? kGlobalNumHeads : kLocalNumHeads;
    const double Hkv = is_global ? kGlobalNumKvHeads : kLocalNumKvHeads;
    const double Dh = is_global ? kGlobalHeadDim : kLocalHeadDim;
    const double W = is_global ? 0.0 : kLocalSlidingWindow;
    const double q_dim = Hq * Dh;
    const double kv_dim = Hkv * Dh;

    // INT4 linear projections: Q, K, [V if !k_eq_v], O
    double params = q_dim * D + kv_dim * D + (is_global ? 0.0 : kv_dim * D) + D * q_dim;
    double rows = q_dim + kv_dim + (is_global ? 0.0 : kv_dim) + D;
    r.weight_params += params;
    r.int4_ops += 2.0 * T * params;
    r.weight_bytes_format += WeightBytesForFormat(params, rows, weight_quant) +
                             (2.0 * D + 2.0 * Dh) * 4.0;  // pre/post/q/k RMSNorm
    r.weight_bytes_ideal += params * 0.5 + (2.0 * D + 2.0 * Dh) * 4.0;

    // Effective causal / sliding-window attention pairs
    double S_past = std::max(0, S_kv - T);
    double active_pairs = 0.0;
    for (int t = 0; t < T; ++t) {
      double q_pos = S_past + t;
      double s_min = (W > 0.0) ? std::max(0.0, q_pos - W + 1.0) : 0.0;
      active_pairs += (q_pos - s_min + 1.0);
    }
    double S_eff_read = (W > 0.0) ? std::min(static_cast<double>(S_kv), W) : static_cast<double>(S_kv);
    // QK^T + PV MACs (2*Dh per pair each -> 4*Dh per pair) + softmax/norms/RoPE
    double attn_bmm_ops = 4.0 * Hq * Dh * active_pairs;
    double elem_ops = 5.0 * Hq * active_pairs + 8.0 * T * D + 12.0 * T * (q_dim + 2.0 * kv_dim);
    r.fp32_ops += attn_bmm_ops + elem_ops;

    // Boundary KV cache traffic: read S_eff_read past/present entries + write T new entries
    // (Both K and V are cached even when K=V because K has K_norm+RoPE while V has V_norm)
    double kv_elems_rw = 2.0 * Hkv * Dh * (S_eff_read + static_cast<double>(T));
    r.kv_bytes_format += kv_elems_rw * kv_elem_b;
    r.kv_bytes_ideal += kv_elems_rw * 2.0;  // FP16 KV ideal
  };

  auto add_mlp = [&]() {
    // Gate (F x D), Up (F x D), Down (D x F)
    double params = 3.0 * F * D;
    double rows = 2.0 * F + D;
    r.weight_params += params;
    r.int4_ops += 2.0 * T * params;
    r.weight_bytes_format += WeightBytesForFormat(params, rows, weight_quant) +
                             (2.0 * D) * 4.0;  // pre/post RMSNorm
    r.weight_bytes_ideal += params * 0.5 + (2.0 * D) * 4.0;
    // RMSNorm + GeGLU + residual
    r.fp32_ops += 10.0 * T * F + 8.0 * T * D;
  };

  if (subgraph_id == "local_swa_attn") {
    add_attn(false);
  } else if (subgraph_id == "global_full_attn") {
    add_attn(true);
  } else if (subgraph_id == "geglu_mlp") {
    add_mlp();
  } else if (subgraph_id == "full_local_block") {
    add_attn(false);
    add_mlp();
  } else if (subgraph_id == "full_global_block") {
    add_attn(true);
    add_mlp();
  }

  r.total_ops = r.int4_ops + r.fp32_ops;
  r.io_bytes = 2.0 * T * D * 4.0;  // FP32 boundary input + FP32 boundary output
  r.min_bytes_format = r.io_bytes + r.weight_bytes_format + r.kv_bytes_format;
  r.min_bytes_ideal = r.io_bytes + r.weight_bytes_ideal + r.kv_bytes_ideal;
  return r;
}

// -----------------------------------------------------------------------------
// 7-Trial Benchmark Statistics
// -----------------------------------------------------------------------------

struct TrialStats {
  int num_trials = 0;
  int iters_per_trial = 0;
  double median_ms = 0.0;
  double mean_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double p90_ms = 0.0;
  double p99_ms = 0.0;
  double stdev_ms = 0.0;
  double cv_pct = 0.0;
  std::vector<double> samples_ms;
};

inline TrialStats ComputeTrialStats(const std::vector<double>& samples_ms,
                                    int iters_per_trial) {
  TrialStats s;
  s.num_trials = static_cast<int>(samples_ms.size());
  s.iters_per_trial = iters_per_trial;
  s.samples_ms = samples_ms;
  if (samples_ms.empty()) return s;

  std::vector<double> sorted = samples_ms;
  std::sort(sorted.begin(), sorted.end());
  const int n = static_cast<int>(sorted.size());
  s.min_ms = sorted.front();
  s.max_ms = sorted.back();
  s.median_ms = (n % 2 == 1) ? sorted[n / 2]
                             : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
  auto pct = [&](double p) {
    double idx = p * (n - 1);
    int lo = static_cast<int>(std::floor(idx));
    int hi = std::min(n - 1, lo + 1);
    double f = idx - lo;
    return sorted[lo] * (1.0 - f) + sorted[hi] * f;
  };
  s.p90_ms = pct(0.90);
  s.p99_ms = pct(0.99);

  double sum = 0.0;
  for (double v : sorted) sum += v;
  s.mean_ms = sum / n;
  double var = 0.0;
  for (double v : sorted) var += (v - s.mean_ms) * (v - s.mean_ms);
  s.stdev_ms = (n > 1) ? std::sqrt(var / (n - 1)) : 0.0;
  s.cv_pct = (s.mean_ms > 0.0) ? (100.0 * s.stdev_ms / s.mean_ms) : 0.0;
  return s;
}

inline std::string JsonEscape(const std::string& s) {
  std::ostringstream o;
  for (char c : s) {
    if (c == '"') o << "\\\"";
    else if (c == '\\') o << "\\\\";
    else if (c == '\n') o << "\\n";
    else o << c;
  }
  return o.str();
}

}  // namespace aeb_layer

#endif  // AEB_LAYER_GEMMA4_COMMON_H_
