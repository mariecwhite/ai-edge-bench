// Level 2 (Layer Tier) Gemma 4 31B Subgraph Benchmark & Numerical Verifier for
// `llama.cpp` / `ggml` CPU (KleidiAI NEON/i8mm + Accelerate + Flash Attention,
// and KleidiAI SME2 + Flash Attention).
//
// Evaluates the exact same 5 Gemma 4 31B text decoder subgraphs and shares the
// exact same deterministic weights, inputs, and independent FP64 golden
// reference (`layer/subgraphs/common/gemma4_layer_common.h`) as the LiteRT and
// Direct YNNPACK C API harness:
//   - local_swa_attn     (H=32, KV=16, D=256, W=1024, full RoPE)
//   - global_full_attn   (H=32, KV=4,  D=512, K=V,    0.25x proportional RoPE)
//   - geglu_mlp          (5376 -> 21504 -> 5376, GeGLU)
//   - full_local_block   (local_swa_attn + geglu_mlp)
//   - full_global_block  (global_full_attn + geglu_mlp)
//
// Note: `GGML_KLEIDIAI_SME` is read by `ggml-cpu` at backend initialization, so
// invoke this binary twice:
//   GGML_KLEIDIAI_SME=0 ./bench_ggml_layers --backend-id ggml_cpu_neon_fa ...
//   GGML_KLEIDIAI_SME=1 ./bench_ggml_layers --backend-id ggml_cpu_sme_fa  ...

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "layer/subgraphs/common/gemma4_layer_common.h"

namespace {

using namespace aeb_layer;

struct RegimeCfg {
  std::string id;
  std::string mode;
  int T = 0;
  int S_kv = 0;
  int S_cap = 0;
};

std::vector<RegimeCfg> RegimesForSubgraph(const std::string& sg) {
  if (sg == "geglu_mlp") {
    return {
        {"prefill_128", "prefill", 128, 0, 0},
        {"prefill_1024", "prefill", 1024, 0, 0},
        {"decode_1", "decode", 1, 0, 0},
    };
  }
  if (sg == "local_swa_attn") {
    return {
        {"prefill_128", "prefill", 128, 128, 4096},
        {"prefill_1024", "prefill", 1024, 1024, 4096},
        {"decode_kv64", "decode", 1, 64, 4096},
        {"decode_kv256", "decode", 1, 256, 4096},
        {"decode_kv1024", "decode", 1, 1024, 4096},
        {"decode_kv1024_cap8192", "decode", 1, 1024, 8192},
    };
  }
  if (sg == "full_local_block") {
    return {
        {"prefill_128", "prefill", 128, 128, 4096},
        {"prefill_1024", "prefill", 1024, 1024, 4096},
        {"decode_kv64", "decode", 1, 64, 4096},
        {"decode_kv1024", "decode", 1, 1024, 4096},
        {"decode_kv1024_cap8192", "decode", 1, 1024, 8192},
    };
  }
  return {
      {"prefill_128", "prefill", 128, 128, 4096},
      {"prefill_1024", "prefill", 1024, 1024, 4096},
      {"prefill_1024_ctx4096", "prefill", 1024, 4096, 8192},
      {"decode_kv64", "decode", 1, 64, 4096},
      {"decode_kv1024", "decode", 1, 1024, 4096},
      {"decode_kv4096", "decode", 1, 4096, 8192},
      {"decode_kv8192", "decode", 1, 8192, 16384},
      {"decode_kv16384", "decode", 1, 16384, 16384},
  };
}

std::vector<std::string> SplitCsv(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

// Quantize a [N, K] row-major FP32 matrix into GGML_TYPE_Q4_0 and also return
// the exact dequantized FP32 weights for Level A verification.
std::vector<uint8_t> QuantizeQ4_0(const std::vector<float>& src, int N, int K,
                                  std::vector<float>* dequant_out = nullptr) {
  size_t nbytes = ggml_row_size(GGML_TYPE_Q4_0, K) * static_cast<size_t>(N);
  std::vector<uint8_t> dst(nbytes);
  ggml_quantize_chunk(GGML_TYPE_Q4_0, src.data(), dst.data(), 0, N, K, nullptr);
  if (dequant_out) {
    dequant_out->resize(static_cast<size_t>(N) * K);
    const ggml_type_traits* traits = ggml_get_type_traits(GGML_TYPE_Q4_0);
    traits->to_float(dst.data(), dequant_out->data(),
                     static_cast<int64_t>(N) * K);
  }
  return dst;
}

// Holds all quantized Q4_0 weights & FP32 norm weights in a ggml CPU backend
// buffer (which automatically uses KleidiAI repacked buffer type when enabled!).
struct GgmlLayerWeights {
  ggml_context* ctx = nullptr;
  ggml_backend_buffer_t buf_q = nullptr;   // Q4_0 weights (repacked buffer type!)
  ggml_backend_buffer_t buf_f = nullptr;   // FP32 norms & rope_factors

  ggml_tensor* attn_pre_norm = nullptr;
  ggml_tensor* wq = nullptr;
  ggml_tensor* wk = nullptr;
  ggml_tensor* wv = nullptr;
  ggml_tensor* q_norm = nullptr;
  ggml_tensor* k_norm = nullptr;
  ggml_tensor* wo = nullptr;
  ggml_tensor* attn_post_norm = nullptr;
  ggml_tensor* rope_factors = nullptr;

  ggml_tensor* mlp_pre_norm = nullptr;
  ggml_tensor* w_gate = nullptr;
  ggml_tensor* w_up = nullptr;
  ggml_tensor* w_down = nullptr;
  ggml_tensor* mlp_post_norm = nullptr;

  // Dequantized weights for Level A FP64 reference
  AttnWeightsFp32 dq_aw;
  MlpWeightsFp32 dq_mw;

  ~GgmlLayerWeights() {
    if (buf_q) ggml_backend_buffer_free(buf_q);
    if (buf_f) ggml_backend_buffer_free(buf_f);
    if (ctx) ggml_free(ctx);
  }

  bool Init(ggml_backend_t backend, const AttnWeightsFp32* aw,
            const MlpWeightsFp32* mw, bool need_dequant) {
    ggml_init_params p{ggml_tensor_overhead() * 64, nullptr, true};
    ctx = ggml_init(p);

    // Separate contexts or split allocations so Q4_0 weights go to CPU_REPACK
    // buffer type (enabling KleidiAI NEON/i8mm or SME2 repacking identically to
    // llama.cpp --no-mmap!) while FP32 norms go to regular CPU host buffer.
    ggml_backend_buffer_type_t buft_host =
        ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_type_t buft_repack = buft_host;
    ggml_backend_dev_t cpu_dev = ggml_backend_get_device(backend);
    if (cpu_dev) {
      ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
      auto get_extra_bufts = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
          ggml_backend_reg_get_proc_address(cpu_reg,
                                            "ggml_backend_dev_get_extra_bufts"));
      if (get_extra_bufts) {
        ggml_backend_buffer_type_t* extra = get_extra_bufts(cpu_dev);
        if (extra && extra[0]) buft_repack = extra[0];
      }
    }

    ggml_init_params pq{ggml_tensor_overhead() * 32, nullptr, true};
    ggml_context* ctx_q = ggml_init(pq);
    ggml_init_params pf{ggml_tensor_overhead() * 32, nullptr, true};
    ggml_context* ctx_f = ggml_init(pf);

    struct QItem {
      ggml_tensor** ptr;
      const std::vector<float>* src;
      std::vector<float>* dq_dst;
      int N;
      int K;
    };
    struct FItem {
      ggml_tensor** ptr;
      std::vector<float> vals;
    };
    std::vector<QItem> q_items;
    std::vector<FItem> f_items;

    if (aw) {
      if (need_dequant) dq_aw = *aw;
      int q_dim = aw->n_heads * aw->head_dim;
      int kv_dim = aw->n_kv_heads * aw->head_dim;

      wq = ggml_new_tensor_2d(ctx_q, GGML_TYPE_Q4_0, aw->hidden, q_dim);
      q_items.push_back({&wq, &aw->wq, need_dequant ? &dq_aw.wq : nullptr, q_dim, aw->hidden});
      wk = ggml_new_tensor_2d(ctx_q, GGML_TYPE_Q4_0, aw->hidden, kv_dim);
      q_items.push_back({&wk, &aw->wk, need_dequant ? &dq_aw.wk : nullptr, kv_dim, aw->hidden});
      if (!aw->k_eq_v) {
        wv = ggml_new_tensor_2d(ctx_q, GGML_TYPE_Q4_0, aw->hidden, kv_dim);
        q_items.push_back({&wv, &aw->wv, need_dequant ? &dq_aw.wv : nullptr, kv_dim, aw->hidden});
      }
      wo = ggml_new_tensor_2d(ctx_q, GGML_TYPE_Q4_0, q_dim, aw->hidden);
      q_items.push_back({&wo, &aw->wo, need_dequant ? &dq_aw.wo : nullptr, aw->hidden, q_dim});

      attn_pre_norm = ggml_new_tensor_1d(ctx_f, GGML_TYPE_F32, aw->hidden);
      f_items.push_back({&attn_pre_norm, aw->pre_norm});
      q_norm = ggml_new_tensor_1d(ctx_f, GGML_TYPE_F32, aw->head_dim);
      f_items.push_back({&q_norm, aw->q_norm});
      k_norm = ggml_new_tensor_1d(ctx_f, GGML_TYPE_F32, aw->head_dim);
      f_items.push_back({&k_norm, aw->k_norm});
      attn_post_norm = ggml_new_tensor_1d(ctx_f, GGML_TYPE_F32, aw->hidden);
      f_items.push_back({&attn_post_norm, aw->post_norm});

      if (aw->rot_dim < aw->head_dim) {
        // Proportional RoPE frequency factors so ggml_rope_ext(n_dims=rot_dim,
        // freq_base=theta) uses theta^(-2i/head_dim) = theta^(-2i/rot_dim) / factor[i].
        int half = aw->rot_dim / 2;
        std::vector<float> ff(half);
        for (int i = 0; i < half; ++i) {
          double exp_diff =
              2.0 * i * (1.0 / aw->head_dim - 1.0 / aw->rot_dim);
          ff[i] = static_cast<float>(std::pow(static_cast<double>(aw->rope_theta), exp_diff));
        }
        rope_factors = ggml_new_tensor_1d(ctx_f, GGML_TYPE_F32, half);
        f_items.push_back({&rope_factors, std::move(ff)});
      }
    }

    if (mw) {
      if (need_dequant) dq_mw = *mw;
      w_gate = ggml_new_tensor_2d(ctx_q, GGML_TYPE_Q4_0, mw->hidden, mw->inter);
      q_items.push_back({&w_gate, &mw->w_gate, need_dequant ? &dq_mw.w_gate : nullptr, mw->inter, mw->hidden});
      w_up = ggml_new_tensor_2d(ctx_q, GGML_TYPE_Q4_0, mw->hidden, mw->inter);
      q_items.push_back({&w_up, &mw->w_up, need_dequant ? &dq_mw.w_up : nullptr, mw->inter, mw->hidden});
      w_down = ggml_new_tensor_2d(ctx_q, GGML_TYPE_Q4_0, mw->inter, mw->hidden);
      q_items.push_back({&w_down, &mw->w_down, need_dequant ? &dq_mw.w_down : nullptr, mw->hidden, mw->inter});

      mlp_pre_norm = ggml_new_tensor_1d(ctx_f, GGML_TYPE_F32, mw->hidden);
      f_items.push_back({&mlp_pre_norm, mw->pre_norm});
      mlp_post_norm = ggml_new_tensor_1d(ctx_f, GGML_TYPE_F32, mw->hidden);
      f_items.push_back({&mlp_post_norm, mw->post_norm});
    }

    buf_q = ggml_backend_alloc_ctx_tensors_from_buft(ctx_q, buft_repack);
    buf_f = ggml_backend_alloc_ctx_tensors_from_buft(ctx_f, buft_host);
    if (!buf_q || !buf_f) {
      ggml_free(ctx_q);
      ggml_free(ctx_f);
      return false;
    }

    for (auto& qi : q_items) {
      auto qbytes = QuantizeQ4_0(*qi.src, qi.N, qi.K, qi.dq_dst);
      ggml_backend_tensor_set(*qi.ptr, qbytes.data(), 0, qbytes.size());
    }
    for (auto& fi : f_items) {
      ggml_backend_tensor_set(*fi.ptr, fi.vals.data(), 0,
                              fi.vals.size() * sizeof(float));
    }

    // Keep tensor metadata alive via separate contexts owned in a helper?
    // Wait: we must keep ctx_q and ctx_f alive until destructor! Store them.
    ctx_q_ = ctx_q;
    ctx_f_ = ctx_f;
    return true;
  }

  ggml_context* ctx_q_ = nullptr;
  ggml_context* ctx_f_ = nullptr;
  void CleanupExtra() {
    if (ctx_q_) { ggml_free(ctx_q_); ctx_q_ = nullptr; }
    if (ctx_f_) { ggml_free(ctx_f_); ctx_f_ = nullptr; }
  }
};

struct RunResult {
  std::string backend_id;
  std::string subgraph_id;
  std::string regime_id;
  std::string mode;
  int T = 0;
  int S_kv = 0;
  int S_cap = 0;
  int threads = 0;
  TrialStats stats;
  RooflineSpec roofline;
  MemoryMetrics mem;
  bool ok = false;
  std::string error;
  bool has_verify = false;
  ParityMetrics level_a;
  ParityMetrics level_b;
  bool deterministic_18t = true;
  double max_repro_diff = 0.0;
};

RunResult RunOneGgml(ggml_backend_t backend, const std::string& backend_id,
                     const std::string& sg_id, const RegimeCfg& reg,
                     int threads, int trials, bool verify_mode,
                     bool use_flash_attn, const AttnWeightsFp32* aw,
                     const MlpWeightsFp32* mw, GgmlLayerWeights& lw,
                     SlcFlusher& flusher, const std::vector<double>* ref_unq,
                     const std::vector<double>* ref_dq) {
  RunResult res;
  res.backend_id = backend_id;
  res.subgraph_id = sg_id;
  res.regime_id = reg.id;
  res.mode = reg.mode;
  res.T = reg.T;
  res.S_kv = reg.S_kv;
  res.S_cap = reg.S_cap;
  res.threads = threads;
  res.roofline = ComputeSubgraphRoofline(sg_id, reg.T, reg.S_kv, "q4_0", "fp16");

  ggml_backend_cpu_set_n_threads(backend, threads);

  const int T = reg.T;
  const int S_kv = reg.S_kv;
  const int S_past = (aw != nullptr) ? (S_kv - T) : 0;

  // Prepare deterministic shared inputs
  std::vector<float> x_in, k_past, v_past;
  FillNormal(x_in, static_cast<size_t>(T) * kHiddenSize, 9001, 0.5f);
  if (aw && S_past > 0) {
    size_t kv_n = static_cast<size_t>(aw->n_kv_heads) * S_past * aw->head_dim;
    FillNormal(k_past, kv_n, 9101, 0.25f);
    FillNormal(v_past, kv_n, 9201, 0.25f);
  }

  // Allocate graph inputs in a dedicated persistent CPU buffer (separate from
  // ggml_gallocr transient activation memory) so ggml_gallocr never aliases
  // downstream outputs (e.g., MLP residual in full_local_block) onto inp_x!
  ggml_init_params inp_p{ggml_tensor_overhead() * 16, nullptr, true};
  ggml_context* ctx_inp = ggml_init(inp_p);

  ggml_tensor* inp_x = ggml_new_tensor_2d(ctx_inp, GGML_TYPE_F32, kHiddenSize, T);
  ggml_set_input(inp_x);

  ggml_tensor* inp_pos = nullptr;
  ggml_tensor* inp_k_cache = nullptr;
  ggml_tensor* inp_v_cache = nullptr;
  ggml_tensor* inp_mask = nullptr;
  int kv_pad = 0;

  if (aw != nullptr) {
    const int Hkv = aw->n_kv_heads;
    const int Dh = aw->head_dim;
    kv_pad = (use_flash_attn) ? ((S_kv + 31) & ~31) : S_kv;
    inp_pos = ggml_new_tensor_1d(ctx_inp, GGML_TYPE_I32, T);
    ggml_set_input(inp_pos);
    inp_k_cache = ggml_new_tensor_3d(ctx_inp, GGML_TYPE_F16, Dh, kv_pad, Hkv);
    ggml_set_input(inp_k_cache);
    inp_v_cache = ggml_new_tensor_3d(ctx_inp, GGML_TYPE_F16, Dh, kv_pad, Hkv);
    ggml_set_input(inp_v_cache);
    int t_mask = use_flash_attn ? ((T + 15) & ~15) : T;
    inp_mask = ggml_new_tensor_2d(ctx_inp,
                                  use_flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32,
                                  kv_pad, t_mask);
    ggml_set_input(inp_mask);
  }
  ggml_backend_buffer_t buf_inp = ggml_backend_alloc_ctx_tensors(ctx_inp, backend);

  // Build graph in no-alloc context
  size_t meta_size = ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(4096, false);
  std::vector<uint8_t> meta_buf(meta_size);
  ggml_init_params gp{meta_size, meta_buf.data(), true};
  ggml_context* ctx0 = ggml_init(gp);
  ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

  ggml_tensor* cur = inp_x;

  if (aw != nullptr) {
    const int Hq = aw->n_heads;
    const int Hkv = aw->n_kv_heads;
    const int Dh = aw->head_dim;

    ggml_tensor* h = ggml_rms_norm(ctx0, cur, kRmsNormEps);
    h = ggml_mul(ctx0, h, lw.attn_pre_norm);

    ggml_tensor* q = ggml_mul_mat(ctx0, lw.wq, h);
    q = ggml_reshape_3d(ctx0, q, Dh, Hq, T);
    q = ggml_rms_norm(ctx0, q, kRmsNormEps);
    q = ggml_mul(ctx0, q, lw.q_norm);
    q = ggml_rope_ext(ctx0, q, inp_pos, lw.rope_factors, aw->rot_dim,
                      GGML_ROPE_TYPE_NEOX, 0, aw->rope_theta, 1.0f, 0.0f, 1.0f,
                      0.0f, 0.0f);

    ggml_tensor* k_new = ggml_mul_mat(ctx0, lw.wk, h);
    ggml_tensor* v_new = aw->k_eq_v ? k_new : ggml_mul_mat(ctx0, lw.wv, h);

    k_new = ggml_reshape_3d(ctx0, k_new, Dh, Hkv, T);
    v_new = ggml_reshape_3d(ctx0, v_new, Dh, Hkv, T);

    k_new = ggml_rms_norm(ctx0, k_new, kRmsNormEps);
    k_new = ggml_mul(ctx0, k_new, lw.k_norm);
    k_new = ggml_rope_ext(ctx0, k_new, inp_pos, lw.rope_factors, aw->rot_dim,
                          GGML_ROPE_TYPE_NEOX, 0, aw->rope_theta, 1.0f, 0.0f,
                          1.0f, 0.0f, 0.0f);

    v_new = ggml_rms_norm(ctx0, v_new, kRmsNormEps);

    // Append new K, V into FP16 KV cache at position [S_past .. S_past + T)
    // k_new is [Dh, Hkv, T] -> permute to [Dh, T, Hkv] and copy into slice of inp_k_cache
    ggml_tensor* k_new_p = ggml_permute(ctx0, k_new, 0, 2, 1, 3);  // [Dh, T, Hkv]
    ggml_tensor* v_new_p = ggml_permute(ctx0, v_new, 0, 2, 1, 3);  // [Dh, T, Hkv]

    size_t k_off = static_cast<size_t>(S_past) * inp_k_cache->nb[1];
    ggml_tensor* k_dst = ggml_view_3d(ctx0, inp_k_cache, Dh, T, Hkv,
                                      inp_k_cache->nb[1], inp_k_cache->nb[2], k_off);
    ggml_tensor* v_dst = ggml_view_3d(ctx0, inp_v_cache, Dh, T, Hkv,
                                      inp_v_cache->nb[1], inp_v_cache->nb[2], k_off);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, k_new_p, k_dst));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, v_new_p, v_dst));

    // Q for flash_attn_ext: [Dh, T, Hq, 1] (permuted from [Dh, Hq, T])
    ggml_tensor* q_fa = ggml_permute(ctx0, q, 0, 2, 1, 3);

    ggml_tensor* attn_out = nullptr;
    if (use_flash_attn) {
      attn_out = ggml_flash_attn_ext(ctx0, q_fa, inp_k_cache, inp_v_cache,
                                     inp_mask, kAttentionScale, 0.0f, 0.0f);
      ggml_prec_set_acc(attn_out, GGML_PREC_F32);
      // Output of ggml_flash_attn_ext is [Dh, Hq, T, 1] -> reshape to [Hq*Dh, T]
      attn_out = ggml_reshape_2d(ctx0, attn_out, Hq * Dh, T);
    } else {
      // Unfused attention baseline
      ggml_tensor* kq = ggml_mul_mat(ctx0, inp_k_cache, q_fa);  // [kv_pad, T, Hq]
      ggml_prec_set_acc(kq, GGML_PREC_F32);
      kq = ggml_soft_max_ext(ctx0, kq, inp_mask, kAttentionScale, 0.0f);
      ggml_tensor* v_t = ggml_cont(ctx0, ggml_transpose(ctx0, inp_v_cache));  // [kv_pad, Dh, Hkv]
      ggml_tensor* kqv = ggml_mul_mat(ctx0, v_t, kq);  // [Dh, T, Hq]
      attn_out = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));  // [Dh, Hq, T]
      attn_out = ggml_reshape_2d(ctx0, attn_out, Hq * Dh, T);
    }

    ggml_tensor* o = ggml_mul_mat(ctx0, lw.wo, attn_out);
    o = ggml_rms_norm(ctx0, o, kRmsNormEps);
    o = ggml_mul(ctx0, o, lw.attn_post_norm);
    cur = ggml_add(ctx0, o, cur);
  }

  if (mw != nullptr) {
    ggml_tensor* h = ggml_rms_norm(ctx0, cur, kRmsNormEps);
    h = ggml_mul(ctx0, h, lw.mlp_pre_norm);

    ggml_tensor* gate = ggml_mul_mat(ctx0, lw.w_gate, h);
    ggml_tensor* up = ggml_mul_mat(ctx0, lw.w_up, h);
    gate = ggml_gelu(ctx0, gate);
    ggml_tensor* gated = ggml_mul(ctx0, gate, up);
    ggml_tensor* down = ggml_mul_mat(ctx0, lw.w_down, gated);

    down = ggml_rms_norm(ctx0, down, kRmsNormEps);
    down = ggml_mul(ctx0, down, lw.mlp_post_norm);
    cur = ggml_add(ctx0, down, cur);
  }

  ggml_set_output(cur);
  ggml_build_forward_expand(gf, cur);

  ggml_gallocr_t allocr =
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
  if (!ggml_gallocr_alloc_graph(allocr, gf)) {
    res.error = "ggml_gallocr_alloc_graph failed";
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    ggml_backend_buffer_free(buf_inp);
    ggml_free(ctx_inp);
    return res;
  }

  // Copy input tensors
  ggml_backend_tensor_set(inp_x, x_in.data(), 0, x_in.size() * sizeof(float));

  if (aw != nullptr) {
    const int Hkv = aw->n_kv_heads;
    const int Dh = aw->head_dim;
    std::vector<int32_t> pos(T);
    for (int t = 0; t < T; ++t) pos[t] = S_past + t;
    ggml_backend_tensor_set(inp_pos, pos.data(), 0, pos.size() * sizeof(int32_t));

    // Populate FP16 KV cache [Dh, kv_pad, Hkv]
    std::vector<uint16_t> kc_h(static_cast<size_t>(Hkv) * kv_pad * Dh, 0);
    std::vector<uint16_t> vc_h(static_cast<size_t>(Hkv) * kv_pad * Dh, 0);
    for (int h = 0; h < Hkv; ++h) {
      for (int s = 0; s < S_past; ++s) {
        const float* ks =
            k_past.data() + (static_cast<size_t>(h) * S_past + s) * Dh;
        const float* vs =
            v_past.data() + (static_cast<size_t>(h) * S_past + s) * Dh;
        uint16_t* kd = kc_h.data() + (static_cast<size_t>(h) * kv_pad + s) * Dh;
        uint16_t* vd = vc_h.data() + (static_cast<size_t>(h) * kv_pad + s) * Dh;
        for (int d = 0; d < Dh; ++d) {
          kd[d] = FloatToFp16(ks[d]);
          vd[d] = FloatToFp16(vs[d]);
        }
      }
    }
    ggml_backend_tensor_set(inp_k_cache, kc_h.data(), 0,
                            kc_h.size() * sizeof(uint16_t));
    ggml_backend_tensor_set(inp_v_cache, vc_h.data(), 0,
                            vc_h.size() * sizeof(uint16_t));

    int t_mask = use_flash_attn ? ((T + 15) & ~15) : T;
    if (use_flash_attn) {
      std::vector<uint16_t> mh(static_cast<size_t>(t_mask) * kv_pad,
                               FloatToFp16(-INFINITY));
      for (int t = 0; t < T; ++t) {
        int q_pos = S_past + t;
        int s_min = (aw->window > 0) ? std::max(0, q_pos - aw->window + 1) : 0;
        for (int s = s_min; s <= q_pos; ++s) {
          mh[static_cast<size_t>(t) * kv_pad + s] = FloatToFp16(0.0f);
        }
      }
      ggml_backend_tensor_set(inp_mask, mh.data(), 0,
                              mh.size() * sizeof(uint16_t));
    } else {
      std::vector<float> mf(static_cast<size_t>(t_mask) * kv_pad, -INFINITY);
      for (int t = 0; t < T; ++t) {
        int q_pos = S_past + t;
        int s_min = (aw->window > 0) ? std::max(0, q_pos - aw->window + 1) : 0;
        for (int s = s_min; s <= q_pos; ++s) {
          mf[static_cast<size_t>(t) * kv_pad + s] = 0.0f;
        }
      }
      ggml_backend_tensor_set(inp_mask, mf.data(), 0, mf.size() * sizeof(float));
    }
  }

  // Warmup
  for (int w = 0; w < 2; ++w) {
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
      res.error = "ggml_backend_graph_compute warmup failed";
      ggml_gallocr_free(allocr);
      ggml_free(ctx0);
      ggml_backend_buffer_free(buf_inp);
      ggml_free(ctx_inp);
      return res;
    }
  }

  std::vector<float> y_out(static_cast<size_t>(T) * kHiddenSize);
  ggml_backend_tensor_get(cur, y_out.data(), 0, y_out.size() * sizeof(float));

  if (verify_mode) {
    res.has_verify = true;
    res.level_a = CompareAgainstFp64(y_out.data(), ref_dq->data(), T,
                                     kHiddenSize, 0.9990, 0.045, 27.0, 0.005);
    res.level_b = CompareAgainstFp64(y_out.data(), ref_unq->data(), T,
                                     kHiddenSize, 0.930, 0.38, 8.5, 0.09);
    std::vector<float> y_chk(y_out.size());
    res.max_repro_diff = 0.0;
    for (int r = 0; r < 3; ++r) {
      ggml_backend_graph_compute(backend, gf);
      ggml_backend_tensor_get(cur, y_chk.data(), 0, y_chk.size() * sizeof(float));
      for (size_t i = 0; i < y_out.size(); ++i) {
        double d = std::fabs(static_cast<double>(y_chk[i]) - y_out[i]);
        if (d > res.max_repro_diff) res.max_repro_diff = d;
      }
    }
    res.deterministic_18t = (res.max_repro_diff == 0.0);
    res.mem = ReadMemoryMetrics();
    res.ok = true;
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    ggml_backend_buffer_free(buf_inp);
    ggml_free(ctx_inp);
    return res;
  }

  double t0 = NowSec();
  ggml_backend_graph_compute(backend, gf);
  double one_ms = (NowSec() - t0) * 1e3;
  int iters = (reg.mode == "decode")
                  ? std::max(1, std::min(12, static_cast<int>(std::ceil(60.0 / std::max(one_ms, 0.2)))))
                  : std::max(1, std::min(6, static_cast<int>(std::ceil(80.0 / std::max(one_ms, 1.0)))));

  std::vector<double> samples_ms;
  samples_ms.reserve(trials);
  for (int tr = 0; tr < trials; ++tr) {
    double acc_s = 0.0;
    for (int it = 0; it < iters; ++it) {
      if (reg.mode == "decode") flusher.Flush();
      double s0 = NowSec();
      ggml_backend_graph_compute(backend, gf);
      acc_s += (NowSec() - s0);
    }
    samples_ms.push_back((acc_s / iters) * 1e3);
  }
  res.stats = ComputeTrialStats(samples_ms, iters);
  res.mem = ReadMemoryMetrics();
  res.ok = true;
  ggml_gallocr_free(allocr);
  ggml_free(ctx0);
  ggml_backend_buffer_free(buf_inp);
  ggml_free(ctx_inp);
  return res;
}

void WriteResultsJson(const std::string& path, const std::string& mode,
                      const std::string& llama_commit,
                      const std::vector<RunResult>& results) {
  std::ofstream os(path);
  os << std::fixed << std::setprecision(6);
  os << "{\n";
  os << "  \"harness\": \"bench_ggml_layers\",\n";
  os << "  \"mode\": \"" << JsonEscape(mode) << "\",\n";
  os << "  \"llama_cpp_commit\": \"" << JsonEscape(llama_commit) << "\",\n";
  os << "  \"results\": [\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const auto& r = results[i];
    os << "    {\n";
    os << "      \"backend_id\": \"" << JsonEscape(r.backend_id) << "\",\n";
    os << "      \"subgraph_id\": \"" << JsonEscape(r.subgraph_id) << "\",\n";
    os << "      \"regime_id\": \"" << JsonEscape(r.regime_id) << "\",\n";
    os << "      \"mode\": \"" << JsonEscape(r.mode) << "\",\n";
    os << "      \"weight_quant\": \"q4_0\",\n";
    os << "      \"kv_dtype\": \"fp16\",\n";
    os << "      \"T\": " << r.T << ", \"S_kv\": " << r.S_kv
       << ", \"S_cap\": " << r.S_cap << ", \"threads\": " << r.threads << ",\n";
    os << "      \"ok\": " << (r.ok ? "true" : "false") << ",\n";
    if (!r.error.empty()) {
      os << "      \"error\": \"" << JsonEscape(r.error) << "\",\n";
    }
    os << "      \"roofline_inputs\": {\n";
    os << "        \"int4_ops\": " << r.roofline.int4_ops << ",\n";
    os << "        \"fp32_ops\": " << r.roofline.fp32_ops << ",\n";
    os << "        \"total_ops\": " << r.roofline.total_ops << ",\n";
    os << "        \"weight_params\": " << r.roofline.weight_params << ",\n";
    os << "        \"weight_bytes_format\": " << r.roofline.weight_bytes_format << ",\n";
    os << "        \"weight_bytes_ideal\": " << r.roofline.weight_bytes_ideal << ",\n";
    os << "        \"kv_bytes_format\": " << r.roofline.kv_bytes_format << ",\n";
    os << "        \"kv_bytes_ideal\": " << r.roofline.kv_bytes_ideal << ",\n";
    os << "        \"io_bytes\": " << r.roofline.io_bytes << ",\n";
    os << "        \"min_bytes_format\": " << r.roofline.min_bytes_format << ",\n";
    os << "        \"min_bytes_ideal\": " << r.roofline.min_bytes_ideal << "\n";
    os << "      },\n";
    os << "      \"memory\": {\"phys_footprint_mb\": " << r.mem.phys_footprint_mb
       << ", \"peak_rss_mb\": " << r.mem.peak_rss_mb << "}";
    if (r.has_verify) {
      auto emit_pm = [&](const ParityMetrics& p) {
        os << "{\"cosine_sim\": " << std::setprecision(8) << p.cosine_sim
           << ", \"rel_l2_err\": " << p.rel_l2_err
           << ", \"sqnr_db\": " << std::setprecision(4) << p.sqnr_db
           << ", \"max_abs_err\": " << std::setprecision(6) << p.max_abs_err
           << ", \"kl_div\": " << std::setprecision(8) << p.kl_div
           << ", \"nan_inf_count\": " << p.nan_inf_count
           << ", \"passed\": " << (p.passed ? "true" : "false") << "}";
      };
      os << ",\n      \"verify\": {\n";
      os << "        \"level_a_dequant_fp64\": ";
      emit_pm(r.level_a);
      os << ",\n        \"level_b_unquant_fp64\": ";
      emit_pm(r.level_b);
      os << ",\n        \"deterministic\": " << (r.deterministic_18t ? "true" : "false")
         << ", \"max_repro_diff\": " << std::scientific << r.max_repro_diff
         << std::fixed << "\n      }\n";
    } else {
      os << ",\n      \"timing\": {\n";
      os << "        \"num_trials\": " << r.stats.num_trials
         << ", \"iters_per_trial\": " << r.stats.iters_per_trial << ",\n";
      os << "        \"median_ms\": " << r.stats.median_ms
         << ", \"mean_ms\": " << r.stats.mean_ms
         << ", \"min_ms\": " << r.stats.min_ms
         << ", \"max_ms\": " << r.stats.max_ms
         << ", \"p90_ms\": " << r.stats.p90_ms
         << ", \"p99_ms\": " << r.stats.p99_ms
         << ", \"stdev_ms\": " << r.stats.stdev_ms
         << ", \"cv_pct\": " << r.stats.cv_pct << ",\n";
      os << "        \"samples_ms\": [";
      for (size_t j = 0; j < r.stats.samples_ms.size(); ++j) {
        if (j > 0) os << ", ";
        os << r.stats.samples_ms[j];
      }
      os << "]\n      }\n";
    }
    os << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
  }
  os << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
  SetHighQoS();
  std::string mode = "bench";
  std::string backend_id = "ggml_cpu_neon_fa";
  std::string subgraphs_arg =
      "local_swa_attn,global_full_attn,geglu_mlp,full_local_block,full_global_block";
  std::string regimes_arg = "";
  std::string threads_arg = "6,18";
  std::string output_path = "ggml_layer_results.json";
  std::string llama_commit = "e0a9118fe9ed9229b715f4bd83fd9767963659c6";
  bool use_flash_attn = true;
  int trials = 7;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      return (i + 1 < argc) ? std::string(argv[++i]) : "";
    };
    if (a == "--mode") mode = next();
    else if (a == "--backend-id") backend_id = next();
    else if (a == "--subgraphs") subgraphs_arg = next();
    else if (a == "--regimes") regimes_arg = next();
    else if (a == "--threads") threads_arg = next();
    else if (a == "--trials") trials = std::atoi(next().c_str());
    else if (a == "--no-flash-attn") use_flash_attn = false;
    else if (a == "--output") output_path = next();
    else if (a == "--llama-commit") llama_commit = next();
  }

  auto want_subgraphs = SplitCsv(subgraphs_arg);
  auto want_regimes = SplitCsv(regimes_arg);
  std::vector<int> thread_list;
  for (const auto& ts : SplitCsv(threads_arg)) {
    thread_list.push_back(std::atoi(ts.c_str()));
  }

  ggml_backend_t backend = ggml_backend_cpu_init();
  if (!backend) {
    fprintf(stderr, "ERROR: ggml_backend_cpu_init failed\n");
    return 1;
  }

  const char* sme_env = std::getenv("GGML_KLEIDIAI_SME");
  fprintf(stderr,
          "[bench_ggml_layers] backend_id=%s mode=%s flash_attn=%d "
          "GGML_KLEIDIAI_SME=%s trials=%d\n",
          backend_id.c_str(), mode.c_str(), use_flash_attn ? 1 : 0,
          sme_env ? sme_env : "(unset)", trials);

  AttnWeightsFp32 local_aw = MakeAttnWeights(/*is_global=*/false, 42);
  AttnWeightsFp32 global_aw = MakeAttnWeights(/*is_global=*/true, 72);
  MlpWeightsFp32 mlp_w = MakeMlpWeights(142);

  SlcFlusher flusher(96ULL * 1024 * 1024);
  std::vector<RunResult> results;

  for (const auto& sg_id : want_subgraphs) {
    const AttnWeightsFp32* aw = nullptr;
    const MlpWeightsFp32* mw = nullptr;
    if (sg_id == "local_swa_attn") {
      aw = &local_aw;
    } else if (sg_id == "global_full_attn") {
      aw = &global_aw;
    } else if (sg_id == "geglu_mlp") {
      mw = &mlp_w;
    } else if (sg_id == "full_local_block") {
      aw = &local_aw;
      mw = &mlp_w;
    } else if (sg_id == "full_global_block") {
      aw = &global_aw;
      mw = &mlp_w;
    } else {
      continue;
    }

    GgmlLayerWeights lw;
    if (!lw.Init(backend, aw, mw, /*need_dequant=*/mode == "verify")) {
      fprintf(stderr, "ERROR: failed to init ggml weights for %s\n",
              sg_id.c_str());
      continue;
    }

    std::vector<RegimeCfg> regs;
    if (mode == "verify") {
      if (sg_id == "geglu_mlp") {
        regs = {{"verify_prefill_8", "prefill", 8, 0, 0},
                {"verify_decode_1", "decode", 1, 0, 0}};
      } else {
        regs = {{"verify_prefill_8_kv32", "prefill", 8, 32, 128},
                {"verify_decode_1_kv64", "decode", 1, 64, 128}};
      }
    } else {
      for (const auto& r : RegimesForSubgraph(sg_id)) {
        if (want_regimes.empty() ||
            std::find(want_regimes.begin(), want_regimes.end(), r.id) !=
                want_regimes.end()) {
          regs.push_back(r);
        }
      }
    }

    for (const auto& reg : regs) {
      std::vector<double> ref_unq, ref_dq;
      if (mode == "verify") {
        std::vector<float> x_in, kp, vp;
        FillNormal(x_in, static_cast<size_t>(reg.T) * kHiddenSize, 9001, 0.5f);
        int S_past = (aw != nullptr) ? (reg.S_kv - reg.T) : 0;
        if (aw && S_past > 0) {
          size_t kv_n =
              static_cast<size_t>(aw->n_kv_heads) * S_past * aw->head_dim;
          FillNormal(kp, kv_n, 9101, 0.25f);
          FillNormal(vp, kv_n, 9201, 0.25f);
        }
        std::vector<double> x_d(x_in.begin(), x_in.end());
        auto run_ref = [&](const AttnWeightsFp32* a, const MlpWeightsFp32* m) {
          if (a && m)
            return ref_fp64::RunFullBlockFp64(*a, *m, 1.0f, x_d, reg.T, S_past,
                                              kp, vp);
          if (a) return ref_fp64::RunAttnFp64(*a, x_d, reg.T, S_past, kp, vp);
          return ref_fp64::RunMlpFp64(*m, x_d, reg.T);
        };
        ref_unq = run_ref(aw, mw);
        ref_dq = run_ref(aw ? &lw.dq_aw : nullptr, mw ? &lw.dq_mw : nullptr);
      }

      for (int thr : thread_list) {
        RunResult rr = RunOneGgml(backend, backend_id, sg_id, reg, thr, trials,
                                  mode == "verify", use_flash_attn, aw, mw, lw,
                                  flusher, &ref_unq, &ref_dq);
        if (mode == "verify") {
          fprintf(stderr,
                  "  [VERIFY] %-24s %-18s %-22s %2dT ok=%d | "
                  "LvA(cos=%.6f l2=%.4f sqnr=%.1fdB kl=%.2e pass=%d) | "
                  "LvB(cos=%.6f l2=%.4f sqnr=%.1fdB kl=%.2e pass=%d) | "
                  "det=%d(%.2e)\n",
                  rr.backend_id.c_str(), sg_id.c_str(), reg.id.c_str(), thr,
                  rr.ok ? 1 : 0, rr.level_a.cosine_sim, rr.level_a.rel_l2_err,
                  rr.level_a.sqnr_db, rr.level_a.kl_div,
                  rr.level_a.passed ? 1 : 0, rr.level_b.cosine_sim,
                  rr.level_b.rel_l2_err, rr.level_b.sqnr_db, rr.level_b.kl_div,
                  rr.level_b.passed ? 1 : 0, rr.deterministic_18t ? 1 : 0,
                  rr.max_repro_diff);
        } else {
          double eff_gops =
              (rr.stats.median_ms > 0.0)
                  ? (rr.roofline.total_ops / (rr.stats.median_ms * 1e-3) / 1e9)
                  : 0.0;
          double eff_gbps =
              (rr.stats.median_ms > 0.0)
                  ? (rr.roofline.min_bytes_format / (rr.stats.median_ms * 1e-3) / 1e9)
                  : 0.0;
          fprintf(stderr,
                  "  [BENCH]  %-24s %-18s %-22s %2dT ok=%d | "
                  "med=%8.3fms cv=%4.2f%% | %7.1f GOPS | %6.1f GB/s\n",
                  rr.backend_id.c_str(), sg_id.c_str(), reg.id.c_str(), thr,
                  rr.ok ? 1 : 0, rr.stats.median_ms, rr.stats.cv_pct, eff_gops,
                  eff_gbps);
        }
        results.push_back(std::move(rr));
        WriteResultsJson(output_path, mode, llama_commit, results);
      }
    }
    lw.CleanupExtra();
  }

  ggml_backend_free(backend);
  fprintf(stderr, "[bench_ggml_layers] Wrote %zu records to %s\n",
          results.size(), output_path.c_str());
  return 0;
}
