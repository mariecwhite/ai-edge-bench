// Level 2 (Layer Tier) Gemma 4 31B Subgraph Benchmark & Numerical Verifier for:
//   1. LiteRT XNNPACK delegate (QC4W per-channel & QB4W-32 blockwise INT4)
//   2. LiteRT YNNPACK delegate (QC4W decomposed, QC4W fused odml.sdpa_transposed,
//      QB4W-32 hybrid YNNPACK+XNNPACK fallback matching LiteRT-LM, and
//      QB4W-32 pure YNNPACK with enable_blockwise_quantization=true)
//   3. Direct YNNPACK C API (QC4W & QB4W-32 with Slinky threadpool, zero TFLite
//      interpreter overhead)
//
// Evaluates all 5 Gemma 4 31B text decoder compound subgraphs:
//   - local_swa_attn     (H=32, KV=16, D=256, W=1024, full RoPE)
//   - global_full_attn   (H=32, KV=4,  D=512, K=V,    0.25x proportional RoPE)
//   - geglu_mlp          (5376 -> 21504 -> 5376, GeGLU)
//   - full_local_block   (local_swa_attn + geglu_mlp)
//   - full_global_block  (global_full_attn + geglu_mlp)
//
// Usage:
//   ./bench_litert_layers \
//       --mode bench|verify \
//       --backends litert_xnn_qc4w,litert_xnn_qb4w32,... \
//       --subgraphs local_swa_attn,global_full_attn,geglu_mlp,full_local_block,full_global_block \
//       --threads 6,18 \
//       --trials 7 \
//       --output results.json

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/flexbuffers.h"
#include "layer/subgraphs/common/gemma4_layer_common.h"
#include "slinky/base/thread_pool_impl.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#include "tflite/delegates/ynnpack/ynnpack_delegate.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"
#include "tflite/schema/schema_generated.h"
#include "ynnpack/composites/composites.h"
#include "ynnpack/include/ynnpack.h"

namespace {

using namespace aeb_layer;

// Dummy registration for custom `odml.sdpa_transposed` so TFLite interpreter
// constructs the graph cleanly prior to YNNPACK delegation.
TfLiteStatus DummySdpaPrepare(TfLiteContext* ctx, TfLiteNode* node) {
  const TfLiteTensor* q = &ctx->tensors[node->inputs->data[0]];
  TfLiteTensor* out = &ctx->tensors[node->outputs->data[0]];
  TfLiteIntArray* dims = TfLiteIntArrayCopy(q->dims);
  return ctx->ResizeTensor(ctx, out, dims);
}
TfLiteStatus DummySdpaEval(TfLiteContext*, TfLiteNode*) {
  return kTfLiteError;  // Must be delegated to YNNPACK.
}
TfLiteRegistration* RegisterDummySdpa() {
  static TfLiteRegistration r = {nullptr, nullptr, DummySdpaPrepare,
                                 DummySdpaEval};
  return &r;
}

// -----------------------------------------------------------------------------
// TFLite FlatBuffer Graph Builder for Gemma 4 31B Subgraphs
// -----------------------------------------------------------------------------

class TfLiteSubgraphBuilder {
 public:
  TfLiteSubgraphBuilder() {
    // Buffer 0 is always empty (non-constant tensors).
    buffers_.push_back(tflite::CreateBuffer(fbb_));
  }

  int32_t AddBuffer(const void* data, size_t bytes) {
    auto vec = fbb_.CreateVector(reinterpret_cast<const uint8_t*>(data), bytes);
    buffers_.push_back(tflite::CreateBuffer(fbb_, vec));
    return static_cast<int32_t>(buffers_.size() - 1);
  }

  int32_t AddTensor(const std::vector<int32_t>& shape, tflite::TensorType type,
                    const std::string& name, int32_t buf_idx = 0,
                    flatbuffers::Offset<tflite::QuantizationParameters> qp = 0) {
    auto t = tflite::CreateTensor(
        fbb_, fbb_.CreateVector(shape), type, static_cast<uint32_t>(buf_idx),
        fbb_.CreateString(name), qp);
    tensors_.push_back(t);
    return static_cast<int32_t>(tensors_.size() - 1);
  }

  int32_t AddInput(const std::vector<int32_t>& shape, tflite::TensorType type,
                   const std::string& name) {
    int32_t id = AddTensor(shape, type, name, 0);
    inputs_.push_back(id);
    return id;
  }

  template <typename T>
  int32_t AddConst(const std::vector<int32_t>& shape, tflite::TensorType type,
                   const std::vector<T>& vals, const std::string& name) {
    int32_t buf = AddBuffer(vals.data(), vals.size() * sizeof(T));
    return AddTensor(shape, type, name, buf);
  }

  int32_t AddConstScalarF32(float v, const std::string& name) {
    int32_t buf = AddBuffer(&v, sizeof(float));
    return AddTensor({1}, tflite::TensorType_FLOAT32, name, buf);
  }

  uint32_t GetOrAddBuiltinOpCode(tflite::BuiltinOperator op, int32_t version = 1) {
    for (size_t i = 0; i < op_keys_.size(); ++i) {
      if (op_keys_[i].first == static_cast<int32_t>(op) &&
          op_keys_[i].second == version && custom_names_[i].empty()) {
        return static_cast<uint32_t>(i);
      }
    }
    int8_t dep_code = (static_cast<int32_t>(op) < 127)
                          ? static_cast<int8_t>(op)
                          : static_cast<int8_t>(tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES);
    op_codes_.push_back(tflite::CreateOperatorCode(fbb_, dep_code, 0, version, op));
    op_keys_.push_back({static_cast<int32_t>(op), version});
    custom_names_.push_back("");
    return static_cast<uint32_t>(op_codes_.size() - 1);
  }

  uint32_t GetOrAddCustomOpCode(const std::string& name) {
    for (size_t i = 0; i < custom_names_.size(); ++i) {
      if (custom_names_[i] == name) return static_cast<uint32_t>(i);
    }
    op_codes_.push_back(tflite::CreateOperatorCode(
        fbb_, static_cast<int8_t>(tflite::BuiltinOperator_CUSTOM),
        fbb_.CreateString(name), 1, tflite::BuiltinOperator_CUSTOM));
    op_keys_.push_back({static_cast<int32_t>(tflite::BuiltinOperator_CUSTOM), 1});
    custom_names_.push_back(name);
    return static_cast<uint32_t>(op_codes_.size() - 1);
  }

  void AddBuiltinOp(tflite::BuiltinOperator op,
                    tflite::BuiltinOptions opts_type,
                    flatbuffers::Offset<void> opts,
                    const std::vector<int32_t>& ins,
                    const std::vector<int32_t>& outs, int32_t version = 1) {
    uint32_t code = GetOrAddBuiltinOpCode(op, version);
    ops_.push_back(tflite::CreateOperator(
        fbb_, code, fbb_.CreateVector(ins), fbb_.CreateVector(outs), opts_type,
        opts));
  }

  void AddCustomOp(const std::string& name, const std::vector<uint8_t>& custom_opts,
                   const std::vector<int32_t>& ins,
                   const std::vector<int32_t>& outs) {
    uint32_t code = GetOrAddCustomOpCode(name);
    ops_.push_back(tflite::CreateOperator(
        fbb_, code, fbb_.CreateVector(ins), fbb_.CreateVector(outs),
        tflite::BuiltinOptions_NONE, 0,
        custom_opts.empty() ? 0 : fbb_.CreateVector(custom_opts),
        tflite::CustomOptionsFormat_FLEXBUFFERS));
  }

  // Gemma 4 RMSNorm: x * rsqrt(mean(x^2, axis=rank-1, keep_dims=true) + eps) * w
  int32_t AddRmsNorm(int32_t x_id, const std::vector<int32_t>& x_shape,
                     const std::vector<float>* weight, const std::string& pfx) {
    int rank = static_cast<int>(x_shape.size());
    std::vector<int32_t> red_shape = x_shape;
    red_shape[rank - 1] = 1;

    int32_t sq = AddTensor(x_shape, tflite::TensorType_FLOAT32, pfx + "_sq");
    AddBuiltinOp(tflite::BuiltinOperator_MUL, tflite::BuiltinOptions_MulOptions,
                 tflite::CreateMulOptions(fbb_).Union(), {x_id, x_id}, {sq});

    int32_t axis_t = AddConst<int32_t>({1}, tflite::TensorType_INT32,
                                       {rank - 1}, pfx + "_axis");
    int32_t ms = AddTensor(red_shape, tflite::TensorType_FLOAT32, pfx + "_ms");
    AddBuiltinOp(tflite::BuiltinOperator_MEAN,
                 tflite::BuiltinOptions_ReducerOptions,
                 tflite::CreateReducerOptions(fbb_, /*keep_dims=*/true).Union(),
                 {sq, axis_t}, {ms});

    int32_t eps_t = AddConstScalarF32(kRmsNormEps, pfx + "_eps");
    int32_t ms_eps =
        AddTensor(red_shape, tflite::TensorType_FLOAT32, pfx + "_ms_eps");
    AddBuiltinOp(tflite::BuiltinOperator_ADD, tflite::BuiltinOptions_AddOptions,
                 tflite::CreateAddOptions(fbb_).Union(), {ms, eps_t}, {ms_eps});

    int32_t rsq =
        AddTensor(red_shape, tflite::TensorType_FLOAT32, pfx + "_rsqrt");
    AddBuiltinOp(tflite::BuiltinOperator_RSQRT, tflite::BuiltinOptions_NONE, 0,
                 {ms_eps}, {rsq});

    int32_t normed =
        AddTensor(x_shape, tflite::TensorType_FLOAT32, pfx + "_norm");
    AddBuiltinOp(tflite::BuiltinOperator_MUL, tflite::BuiltinOptions_MulOptions,
                 tflite::CreateMulOptions(fbb_).Union(), {x_id, rsq}, {normed});

    if (weight != nullptr) {
      int32_t w_t = AddConst<float>({x_shape.back()}, tflite::TensorType_FLOAT32,
                                    *weight, pfx + "_w");
      int32_t out =
          AddTensor(x_shape, tflite::TensorType_FLOAT32, pfx + "_out");
      AddBuiltinOp(tflite::BuiltinOperator_MUL,
                   tflite::BuiltinOptions_MulOptions,
                   tflite::CreateMulOptions(fbb_).Union(), {normed, w_t}, {out});
      return out;
    }
    return normed;
  }

  // INT4 Linear projection (QC4W per-channel OR QB4W-32 blockwise with FP16 scales)
  int32_t AddLinearInt4(int32_t x_id, const std::vector<int32_t>& x_shape,
                        const QuantLinearInt4& qw, const std::string& pfx) {
    int32_t w_id = -1;
    if (qw.block_size == 32) {
      // Create FP16 scale tensor FIRST so we know its tensor index for BlockwiseQuantization
      int32_t nb = qw.K / 32;
      int32_t scale_buf = AddBuffer(qw.bw_scales_f16.data(),
                                    qw.bw_scales_f16.size() * sizeof(uint16_t));
      int32_t scale_idx = AddTensor({qw.N, nb}, tflite::TensorType_FLOAT16,
                                    pfx + "_scales", scale_buf);
      auto bw_details = tflite::CreateBlockwiseQuantization(
          fbb_, scale_idx, /*zero_point=*/-1, /*block_size=*/32);
      auto qp = tflite::CreateQuantizationParameters(
          fbb_, 0, 0, 0, 0, tflite::QuantizationDetails_BlockwiseQuantization,
          bw_details.Union(), /*quantized_dimension=*/0);
      int32_t w_buf = AddBuffer(qw.packed_i4.data(), qw.packed_i4.size());
      w_id = AddTensor({qw.N, qw.K}, tflite::TensorType_INT4, pfx + "_w", w_buf,
                       qp);
    } else {
      std::vector<int64_t> zps(qw.N, 0);
      auto qp = tflite::CreateQuantizationParameters(
          fbb_, 0, 0, fbb_.CreateVector(qw.pc_scales), fbb_.CreateVector(zps),
          tflite::QuantizationDetails_NONE, 0, /*quantized_dimension=*/0);
      int32_t w_buf = AddBuffer(qw.packed_i4.data(), qw.packed_i4.size());
      w_id = AddTensor({qw.N, qw.K}, tflite::TensorType_INT4, pfx + "_w", w_buf,
                       qp);
    }

    std::vector<int32_t> out_shape = x_shape;
    out_shape.back() = qw.N;
    int32_t out_id = AddTensor(out_shape, tflite::TensorType_FLOAT32, pfx + "_y");
    auto fc_opts = tflite::CreateFullyConnectedOptions(
        fbb_, tflite::ActivationFunctionType_NONE,
        tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
        /*keep_num_dims=*/true, /*asymmetric_quantize_inputs=*/false);
    // Version 12+ supports INT4 weights and blockwise quantization.
    AddBuiltinOp(tflite::BuiltinOperator_FULLY_CONNECTED,
                 tflite::BuiltinOptions_FullyConnectedOptions, fc_opts.Union(),
                 {x_id, w_id, -1}, {out_id}, /*version=*/12);
    return out_id;
  }

  int32_t AddReshape(int32_t x_id, const std::vector<int32_t>& new_shape,
                     const std::string& pfx) {
    int32_t shape_t =
        AddConst<int32_t>({static_cast<int32_t>(new_shape.size())},
                          tflite::TensorType_INT32, new_shape, pfx + "_shp");
    int32_t out = AddTensor(new_shape, tflite::TensorType_FLOAT32, pfx);
    AddBuiltinOp(tflite::BuiltinOperator_RESHAPE,
                 tflite::BuiltinOptions_ReshapeOptions,
                 tflite::CreateReshapeOptions(fbb_, fbb_.CreateVector(new_shape))
                     .Union(),
                 {x_id, shape_t}, {out});
    return out;
  }

  int32_t AddTranspose(int32_t x_id, const std::vector<int32_t>& x_shape,
                       const std::vector<int32_t>& perm,
                       const std::string& pfx) {
    std::vector<int32_t> out_shape(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) out_shape[i] = x_shape[perm[i]];
    int32_t perm_t =
        AddConst<int32_t>({static_cast<int32_t>(perm.size())},
                          tflite::TensorType_INT32, perm, pfx + "_perm");
    int32_t out = AddTensor(out_shape, tflite::TensorType_FLOAT32, pfx);
    AddBuiltinOp(tflite::BuiltinOperator_TRANSPOSE,
                 tflite::BuiltinOptions_TransposeOptions,
                 tflite::CreateTransposeOptions(fbb_).Union(), {x_id, perm_t},
                 {out});
    return out;
  }

  int32_t AddSlice(int32_t x_id, const std::vector<int32_t>& begin,
                   const std::vector<int32_t>& size, const std::string& pfx) {
    int32_t b_t = AddConst<int32_t>({static_cast<int32_t>(begin.size())},
                                    tflite::TensorType_INT32, begin, pfx + "_b");
    int32_t s_t = AddConst<int32_t>({static_cast<int32_t>(size.size())},
                                    tflite::TensorType_INT32, size, pfx + "_s");
    int32_t out = AddTensor(size, tflite::TensorType_FLOAT32, pfx);
    AddBuiltinOp(tflite::BuiltinOperator_SLICE,
                 tflite::BuiltinOptions_SliceOptions,
                 tflite::CreateSliceOptions(fbb_).Union(), {x_id, b_t, s_t},
                 {out});
    return out;
  }

  int32_t AddConcat(const std::vector<int32_t>& ins,
                    const std::vector<int32_t>& out_shape, int32_t axis,
                    const std::string& pfx) {
    int32_t out = AddTensor(out_shape, tflite::TensorType_FLOAT32, pfx);
    AddBuiltinOp(tflite::BuiltinOperator_CONCATENATION,
                 tflite::BuiltinOptions_ConcatenationOptions,
                 tflite::CreateConcatenationOptions(fbb_, axis).Union(), ins,
                 {out});
    return out;
  }

  // Half-split (NeoX) RoPE with optional proportional rotary_dim <= head_dim.
  // x: [1, T, H, Dh], cos_rot/sin_rot: [1, T, 1, rot_dim]
  int32_t AddRope(int32_t x_id, int T, int H, int Dh, int rot_dim,
                  int32_t cos_id, int32_t sin_id, const std::string& pfx) {
    int half = rot_dim / 2;
    int32_t x_rot = x_id;
    int32_t x_pass = -1;
    if (rot_dim < Dh) {
      x_rot = AddSlice(x_id, {0, 0, 0, 0}, {1, T, H, rot_dim}, pfx + "_rot");
      x_pass = AddSlice(x_id, {0, 0, 0, rot_dim}, {1, T, H, Dh - rot_dim},
                        pfx + "_pass");
    }
    int32_t x0 = AddSlice(x_rot, {0, 0, 0, 0}, {1, T, H, half}, pfx + "_x0");
    int32_t x1 = AddSlice(x_rot, {0, 0, 0, half}, {1, T, H, half}, pfx + "_x1");

    int32_t neg_x1 =
        AddTensor({1, T, H, half}, tflite::TensorType_FLOAT32, pfx + "_negx1");
    AddBuiltinOp(tflite::BuiltinOperator_NEG, tflite::BuiltinOptions_NegOptions,
                 tflite::CreateNegOptions(fbb_).Union(), {x1}, {neg_x1});

    int32_t rot_half =
        AddConcat({neg_x1, x0}, {1, T, H, rot_dim}, /*axis=*/3, pfx + "_rh");

    int32_t t_cos =
        AddTensor({1, T, H, rot_dim}, tflite::TensorType_FLOAT32, pfx + "_mc");
    AddBuiltinOp(tflite::BuiltinOperator_MUL, tflite::BuiltinOptions_MulOptions,
                 tflite::CreateMulOptions(fbb_).Union(), {x_rot, cos_id},
                 {t_cos});

    int32_t t_sin =
        AddTensor({1, T, H, rot_dim}, tflite::TensorType_FLOAT32, pfx + "_ms");
    AddBuiltinOp(tflite::BuiltinOperator_MUL, tflite::BuiltinOptions_MulOptions,
                 tflite::CreateMulOptions(fbb_).Union(), {rot_half, sin_id},
                 {t_sin});

    int32_t out_rot =
        AddTensor({1, T, H, rot_dim}, tflite::TensorType_FLOAT32, pfx + "_add");
    AddBuiltinOp(tflite::BuiltinOperator_ADD, tflite::BuiltinOptions_AddOptions,
                 tflite::CreateAddOptions(fbb_).Union(), {t_cos, t_sin},
                 {out_rot});

    if (rot_dim < Dh) {
      return AddConcat({out_rot, x_pass}, {1, T, H, Dh}, /*axis=*/3,
                       pfx + "_out");
    }
    return out_rot;
  }

  // Adds Attention block (Local SWA or Global K=V) with KV cache inputs and
  // either decomposed BMM+ADD+SOFTMAX+BMM or fused `odml.sdpa_transposed`.
  int32_t AddAttentionBlock(
      int32_t x_id, int T, int S_cap, const AttnWeightsFp32& aw,
      int block_size, bool use_fused_sdpa, const std::string& pfx,
      int32_t& out_cos_id, int32_t& out_sin_id, int32_t& out_k_cache_id,
      int32_t& out_v_cache_id, int32_t& out_mask_id, int32_t& out_s_active_id,
      int32_t& out_k_start_id, int32_t& out_v_start_id,
      int32_t& out_k_upd_id, int32_t& out_v_upd_id) {
    const int D = aw.hidden;
    const int Hq = aw.n_heads;
    const int Hkv = aw.n_kv_heads;
    const int Dh = aw.head_dim;
    const int g = Hq / Hkv;

    auto quant = [&](const std::vector<float>& w, int N, int K) {
      return (block_size == 32) ? QuantizeLinearQB4W32(w, N, K, true)
                                : QuantizeLinearQC4W(w, N, K);
    };

    int32_t h = AddRmsNorm(x_id, {1, T, D}, &aw.pre_norm, pfx + "_prenorm");
    int32_t q_lin =
        AddLinearInt4(h, {1, T, D}, quant(aw.wq, Hq * Dh, D), pfx + "_wq");
    int32_t k_lin =
        AddLinearInt4(h, {1, T, D}, quant(aw.wk, Hkv * Dh, D), pfx + "_wk");
    int32_t v_lin = aw.k_eq_v
                        ? k_lin
                        : AddLinearInt4(h, {1, T, D}, quant(aw.wv, Hkv * Dh, D),
                                        pfx + "_wv");

    int32_t q_4d = AddReshape(q_lin, {1, T, Hq, Dh}, pfx + "_q4d");
    int32_t k_4d = AddReshape(k_lin, {1, T, Hkv, Dh}, pfx + "_k4d");
    int32_t v_4d = AddReshape(v_lin, {1, T, Hkv, Dh}, pfx + "_v4d");

    int32_t q_n =
        AddRmsNorm(q_4d, {1, T, Hq, Dh}, &aw.q_norm, pfx + "_qnorm");
    int32_t k_n =
        AddRmsNorm(k_4d, {1, T, Hkv, Dh}, &aw.k_norm, pfx + "_knorm");
    int32_t v_n = AddRmsNorm(v_4d, {1, T, Hkv, Dh}, nullptr, pfx + "_vnorm");

    out_cos_id = AddInput({1, T, 1, aw.rot_dim}, tflite::TensorType_FLOAT32,
                          pfx + "_cos");
    out_sin_id = AddInput({1, T, 1, aw.rot_dim}, tflite::TensorType_FLOAT32,
                          pfx + "_sin");

    int32_t q_rope =
        AddRope(q_n, T, Hq, Dh, aw.rot_dim, out_cos_id, out_sin_id, pfx + "_qr");
    int32_t k_rope = AddRope(k_n, T, Hkv, Dh, aw.rot_dim, out_cos_id,
                             out_sin_id, pfx + "_kr");

    // Transpose Q -> [1, Hq, T, Dh], K -> [1, Hkv, T, Dh], V -> [1, Hkv, Dh, T]
    int32_t q_t = AddTranspose(q_rope, {1, T, Hq, Dh}, {0, 2, 1, 3}, pfx + "_qt");
    int32_t k_t =
        AddTranspose(k_rope, {1, T, Hkv, Dh}, {0, 2, 1, 3}, pfx + "_kt");
    int32_t v_t =
        AddTranspose(v_n, {1, T, Hkv, Dh}, {0, 2, 3, 1}, pfx + "_vt");

    // KV caches (transposed V matching LiteRT-LM & odml.sdpa_transposed)
    out_k_cache_id = AddInput({1, Hkv, S_cap, Dh}, tflite::TensorType_FLOAT32,
                              pfx + "_kcache");
    out_v_cache_id = AddInput({1, Hkv, Dh, S_cap}, tflite::TensorType_FLOAT32,
                              pfx + "_vcache");
    out_k_start_id = AddInput({4}, tflite::TensorType_INT32, pfx + "_kstart");
    out_v_start_id = AddInput({4}, tflite::TensorType_INT32, pfx + "_vstart");

    out_k_upd_id = AddTensor({1, Hkv, S_cap, Dh}, tflite::TensorType_FLOAT32,
                             pfx + "_kupd");
    AddBuiltinOp(tflite::BuiltinOperator_DYNAMIC_UPDATE_SLICE,
                 tflite::BuiltinOptions_DynamicUpdateSliceOptions,
                 tflite::CreateDynamicUpdateSliceOptions(fbb_).Union(),
                 {out_k_cache_id, k_t, out_k_start_id}, {out_k_upd_id});

    out_v_upd_id = AddTensor({1, Hkv, Dh, S_cap}, tflite::TensorType_FLOAT32,
                             pfx + "_vupd");
    AddBuiltinOp(tflite::BuiltinOperator_DYNAMIC_UPDATE_SLICE,
                 tflite::BuiltinOptions_DynamicUpdateSliceOptions,
                 tflite::CreateDynamicUpdateSliceOptions(fbb_).Union(),
                 {out_v_cache_id, v_t, out_v_start_id}, {out_v_upd_id});

    int32_t ctx_t = -1;
    if (use_fused_sdpa) {
      out_mask_id = AddInput({1, 1, T, S_cap}, tflite::TensorType_BOOL,
                             pfx + "_bmask");
      out_s_active_id = AddInput({1}, tflite::TensorType_INT32, pfx + "_sact");

      flexbuffers::Builder fb;
      fb.Map([&]() { fb.Float("scale", kAttentionScale); });
      fb.Finish();

      ctx_t = AddTensor({1, Hq, T, Dh}, tflite::TensorType_FLOAT32,
                        pfx + "_sdpa_out");
      AddCustomOp("odml.sdpa_transposed", fb.GetBuffer(),
                  {q_t, out_k_upd_id, out_v_upd_id, out_mask_id, out_s_active_id},
                  {ctx_t});
    } else {
      out_s_active_id = -1;
      // Decomposed GQA matching LiteRT-LM:
      // q_g: [1, Hkv, g*T, Dh], k_upd: [1, Hkv, S_cap, Dh]
      int32_t q_g = AddReshape(q_t, {1, Hkv, g * T, Dh}, pfx + "_qg");
      int32_t scores = AddTensor({1, Hkv, g * T, S_cap},
                                 tflite::TensorType_FLOAT32, pfx + "_scores");
      AddBuiltinOp(tflite::BuiltinOperator_BATCH_MATMUL,
                   tflite::BuiltinOptions_BatchMatMulOptions,
                   tflite::CreateBatchMatMulOptions(fbb_, false, true).Union(),
                   {q_g, out_k_upd_id}, {scores});

      // Float mask [1, 1, g*T, S_cap] (identical mask repeated across g heads)
      out_mask_id = AddInput({1, 1, g * T, S_cap}, tflite::TensorType_FLOAT32,
                             pfx + "_fmask");
      int32_t masked = AddTensor({1, Hkv, g * T, S_cap},
                                 tflite::TensorType_FLOAT32, pfx + "_masked");
      AddBuiltinOp(tflite::BuiltinOperator_ADD,
                   tflite::BuiltinOptions_AddOptions,
                   tflite::CreateAddOptions(fbb_).Union(),
                   {scores, out_mask_id}, {masked});

      int32_t probs = AddTensor({1, Hkv, g * T, S_cap},
                                tflite::TensorType_FLOAT32, pfx + "_probs");
      AddBuiltinOp(tflite::BuiltinOperator_SOFTMAX,
                   tflite::BuiltinOptions_SoftmaxOptions,
                   tflite::CreateSoftmaxOptions(fbb_, kAttentionScale).Union(),
                   {masked}, {probs});

      int32_t ctx_g = AddTensor({1, Hkv, g * T, Dh}, tflite::TensorType_FLOAT32,
                                pfx + "_ctxg");
      AddBuiltinOp(tflite::BuiltinOperator_BATCH_MATMUL,
                   tflite::BuiltinOptions_BatchMatMulOptions,
                   tflite::CreateBatchMatMulOptions(fbb_, false, true).Union(),
                   {probs, out_v_upd_id}, {ctx_g});
      ctx_t = AddReshape(ctx_g, {1, Hq, T, Dh}, pfx + "_ctxt");
    }

    int32_t ctx_seq =
        AddTranspose(ctx_t, {1, Hq, T, Dh}, {0, 2, 1, 3}, pfx + "_ctxseq");
    int32_t ctx_flat = AddReshape(ctx_seq, {1, T, Hq * Dh}, pfx + "_ctxflat");

    int32_t o_lin = AddLinearInt4(ctx_flat, {1, T, Hq * Dh},
                                  quant(aw.wo, D, Hq * Dh), pfx + "_wo");
    int32_t post =
        AddRmsNorm(o_lin, {1, T, D}, &aw.post_norm, pfx + "_postnorm");
    int32_t res = AddTensor({1, T, D}, tflite::TensorType_FLOAT32, pfx + "_res");
    AddBuiltinOp(tflite::BuiltinOperator_ADD, tflite::BuiltinOptions_AddOptions,
                 tflite::CreateAddOptions(fbb_).Union(), {post, x_id}, {res});
    return res;
  }

  int32_t AddMlpBlock(int32_t x_id, int T, const MlpWeightsFp32& mw,
                      int block_size, const std::string& pfx) {
    const int D = mw.hidden;
    const int F = mw.inter;
    auto quant = [&](const std::vector<float>& w, int N, int K) {
      return (block_size == 32) ? QuantizeLinearQB4W32(w, N, K, true)
                                : QuantizeLinearQC4W(w, N, K);
    };

    int32_t h = AddRmsNorm(x_id, {1, T, D}, &mw.pre_norm, pfx + "_prenorm");
    int32_t gate =
        AddLinearInt4(h, {1, T, D}, quant(mw.w_gate, F, D), pfx + "_gate");
    int32_t up = AddLinearInt4(h, {1, T, D}, quant(mw.w_up, F, D), pfx + "_up");

    int32_t gelu =
        AddTensor({1, T, F}, tflite::TensorType_FLOAT32, pfx + "_gelu");
    AddBuiltinOp(tflite::BuiltinOperator_GELU,
                 tflite::BuiltinOptions_GeluOptions,
                 tflite::CreateGeluOptions(fbb_, /*approximate=*/true).Union(),
                 {gate}, {gelu});

    int32_t gated =
        AddTensor({1, T, F}, tflite::TensorType_FLOAT32, pfx + "_gated");
    AddBuiltinOp(tflite::BuiltinOperator_MUL, tflite::BuiltinOptions_MulOptions,
                 tflite::CreateMulOptions(fbb_).Union(), {gelu, up}, {gated});

    int32_t down =
        AddLinearInt4(gated, {1, T, F}, quant(mw.w_down, D, F), pfx + "_down");
    int32_t post =
        AddRmsNorm(down, {1, T, D}, &mw.post_norm, pfx + "_postnorm");
    int32_t res = AddTensor({1, T, D}, tflite::TensorType_FLOAT32, pfx + "_res");
    AddBuiltinOp(tflite::BuiltinOperator_ADD, tflite::BuiltinOptions_AddOptions,
                 tflite::CreateAddOptions(fbb_).Union(), {post, x_id}, {res});
    return res;
  }

  void AddOutput(int32_t id) { outputs_.push_back(id); }

  std::vector<uint8_t> Finish() {
    auto sg = tflite::CreateSubGraph(
        fbb_, fbb_.CreateVector(tensors_), fbb_.CreateVector(inputs_),
        fbb_.CreateVector(outputs_), fbb_.CreateVector(ops_),
        fbb_.CreateString("main"));
    std::vector<flatbuffers::Offset<tflite::SubGraph>> sgs = {sg};
    auto model = tflite::CreateModel(
        fbb_, 3, fbb_.CreateVector(op_codes_), fbb_.CreateVector(sgs),
        fbb_.CreateString("aeb_gemma4_layer"), fbb_.CreateVector(buffers_));
    tflite::FinishModelBuffer(fbb_, model);
    return std::vector<uint8_t>(fbb_.GetBufferPointer(),
                                fbb_.GetBufferPointer() + fbb_.GetSize());
  }

 private:
  flatbuffers::FlatBufferBuilder fbb_;
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers_;
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors_;
  std::vector<flatbuffers::Offset<tflite::OperatorCode>> op_codes_;
  std::vector<std::pair<int32_t, int32_t>> op_keys_;
  std::vector<std::string> custom_names_;
  std::vector<flatbuffers::Offset<tflite::Operator>> ops_;
  std::vector<int32_t> inputs_;
  std::vector<int32_t> outputs_;
};

// -----------------------------------------------------------------------------
// Direct YNNPACK C API Subgraph Builder (Pure YNNPACK + Slinky, No TFLite)
// -----------------------------------------------------------------------------

#define CHECK_YNN(expr)                                                  \
  do {                                                                   \
    ynn_status _s = (expr);                                              \
    if (_s != ynn_status_success) {                                      \
      fprintf(stderr, "YNNPACK error %d at %s:%d: %s\n", (int)_s,        \
              __FILE__, __LINE__, #expr);                                \
      return false;                                                      \
    }                                                                    \
  } while (0)

class DirectYnnSubgraph {
 public:
  ~DirectYnnSubgraph() {
    if (runtime_) ynn_delete_runtime(runtime_);
    if (subgraph_) ynn_delete_subgraph(subgraph_);
  }

  bool Build(const std::string& subgraph_id, int T, int S_kv, int S_cap,
             const AttnWeightsFp32* aw, const MlpWeightsFp32* mw,
             int block_size, int num_threads) {
    T_ = T;
    S_kv_ = S_kv;
    S_cap_ = S_cap;
    aw_ = aw;
    mw_ = mw;
    block_size_ = block_size;
    has_attn_ = (aw != nullptr);
    has_mlp_ = (mw != nullptr);

    // External IDs:
    // 0: x_in [1, T, D]
    // 1: y_out [1, T, D]
    // If has_attn:
    // 2: cos [1, T, 1, rot_dim]
    // 3: sin [1, T, 1, rot_dim]
    // 4: k_past [1, Hkv, S_past, Dh] (if S_past > 0)
    // 5: v_past [1, Hkv, S_past, Dh] (if S_past > 0)
    // 6: fmask  [1, 1, T, S_kv]
    uint32_t num_ext = has_attn_ ? 7 : 2;
    CHECK_YNN(ynn_create_subgraph(num_ext, /*flags=*/0, &subgraph_));

    size_t x_dims[3] = {1, static_cast<size_t>(T), static_cast<size_t>(kHiddenSize)};
    uint32_t x_id = 0;
    CHECK_YNN(ynn_define_tensor(subgraph_, ynn_type_fp32, 3, x_dims, nullptr,
                                YNN_VALUE_FLAG_EXTERNAL_INPUT, &x_id));

    uint32_t cur_id = x_id;
    if (has_attn_) {
      if (!DefineAttn(cur_id, cur_id)) return false;
    }
    if (has_mlp_) {
      if (!DefineMlp(cur_id, cur_id)) return false;
    }

    uint32_t y_id = 1;
    CHECK_YNN(ynn_define_tensor(subgraph_, ynn_type_fp32, 3, x_dims, nullptr,
                                YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &y_id));
    CHECK_YNN(ynn_define_copy(subgraph_, cur_id, &y_id, 0));

    if (num_threads > 1) {
      tp_ = std::make_unique<slinky::thread_pool_impl>(num_threads - 1);
    }
    ynn_threadpool_t ynn_tp = reinterpret_cast<ynn_threadpool_t>(tp_.get());
    CHECK_YNN(ynn_optimize_subgraph(subgraph_, ynn_tp, 0));
    CHECK_YNN(ynn_create_runtime(subgraph_, ynn_tp, 0, &runtime_));
    ynn_delete_subgraph(subgraph_);
    subgraph_ = nullptr;

    // Set external input shapes and reshape runtime before any invocation.
    CHECK_YNN(ynn_set_external_value_shape(runtime_, 0, 3, x_dims));
    if (has_attn_) {
      size_t cs_dims[4] = {1, static_cast<size_t>(T), 1,
                           static_cast<size_t>(aw_->rot_dim)};
      CHECK_YNN(ynn_set_external_value_shape(runtime_, 2, 4, cs_dims));
      CHECK_YNN(ynn_set_external_value_shape(runtime_, 3, 4, cs_dims));
      if (S_kv_ > T_) {
        size_t past_dims[4] = {1, static_cast<size_t>(aw_->n_kv_heads),
                               static_cast<size_t>(S_kv_ - T_),
                               static_cast<size_t>(aw_->head_dim)};
        CHECK_YNN(ynn_set_external_value_shape(runtime_, 4, 4, past_dims));
        CHECK_YNN(ynn_set_external_value_shape(runtime_, 5, 4, past_dims));
      }
      size_t mask_dims[4] = {1, 1, static_cast<size_t>(T),
                             static_cast<size_t>(S_kv_)};
      CHECK_YNN(ynn_set_external_value_shape(runtime_, 6, 4, mask_dims));
    }
    CHECK_YNN(ynn_reshape_runtime(runtime_));
    return true;
  }

  bool Invoke(const std::vector<float>& x_in, std::vector<float>& y_out,
              const std::vector<float>* cos_rot,
              const std::vector<float>* sin_rot,
              const std::vector<float>* k_past,
              const std::vector<float>* v_past,
              const std::vector<float>* fmask) {
    CHECK_YNN(ynn_set_external_value_data(runtime_, 0,
                                          const_cast<float*>(x_in.data())));
    CHECK_YNN(ynn_set_external_value_data(runtime_, 1, y_out.data()));
    if (has_attn_) {
      CHECK_YNN(ynn_set_external_value_data(
          runtime_, 2, const_cast<float*>(cos_rot->data())));
      CHECK_YNN(ynn_set_external_value_data(
          runtime_, 3, const_cast<float*>(sin_rot->data())));
      if (S_kv_ > T_) {
        CHECK_YNN(ynn_set_external_value_data(
            runtime_, 4, const_cast<float*>(k_past->data())));
        CHECK_YNN(ynn_set_external_value_data(
            runtime_, 5, const_cast<float*>(v_past->data())));
      }
      CHECK_YNN(ynn_set_external_value_data(
          runtime_, 6, const_cast<float*>(fmask->data())));
    }
    CHECK_YNN(ynn_invoke_runtime(runtime_));
    return true;
  }

 private:
  bool DefineConstF32(const std::vector<size_t>& dims,
                      const std::vector<float>& vals, uint32_t& out_id) {
    out_id = YNN_INVALID_VALUE_ID;
    const_f32_.push_back(vals);
    CHECK_YNN(ynn_define_tensor(subgraph_, ynn_type_fp32, dims.size(),
                                dims.data(), const_f32_.back().data(), 0,
                                &out_id));
    return true;
  }

  bool DefineRmsNorm(uint32_t x_id, int rank, const std::vector<float>* w,
                     uint32_t& out_id) {
    uint32_t sq = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn_define_binary(subgraph_, ynn_binary_multiply, x_id, x_id, &sq, 0));

    int32_t axis = rank - 1;
    uint32_t ms = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn::define_reduce_sum(
        subgraph_, 1, &axis, sq, YNN_INVALID_VALUE_ID, YNN_INVALID_VALUE_ID,
        /*keep_dims=*/true, /*mean=*/true, /*squared=*/false, ynn_type_fp32,
        YNN_INVALID_VALUE_ID, YNN_INVALID_VALUE_ID, ms));

    float eps_v = kRmsNormEps;
    uint32_t eps_id = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn_define_tensor(subgraph_, ynn_type_fp32, 0, nullptr, &eps_v,
                                YNN_VALUE_FLAG_COPY_DATA_FP32, &eps_id));

    uint32_t ms_eps = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn_define_binary(subgraph_, ynn_binary_add, ms, eps_id, &ms_eps, 0));

    uint32_t inv = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn_define_unary(subgraph_, ynn_unary_reciprocal_square_root,
                               ms_eps, &inv, 0));

    uint32_t normed = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn_define_binary(subgraph_, ynn_binary_multiply, x_id, inv,
                                &normed, 0));
    if (w != nullptr) {
      uint32_t w_id = YNN_INVALID_VALUE_ID;
      if (!DefineConstF32({w->size()}, *w, w_id)) return false;
      out_id = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn_define_binary(subgraph_, ynn_binary_multiply, normed, w_id,
                                  &out_id, 0));
    } else {
      out_id = normed;
    }
    return true;
  }

  // Dynamic INT8 activation quantization + INT4 weight dot (QC4W or QB4W-32)
  bool DefineLinearInt4(uint32_t x_id, const std::vector<float>& w_fp32,
                        int N, int K, uint32_t& out_id) {
    // 1. Dynamically quantize x along last axis (K) to int8
    int32_t red_axis = -1;
    uint32_t mm_id = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn_define_reduce(subgraph_, ynn_reduce_min_max, 1, &red_axis,
                                x_id, YNN_INVALID_VALUE_ID, &mm_id,
                                YNN_NODE_FLAG_KEEP_DIMS));
    uint32_t a_zp = YNN_INVALID_VALUE_ID, a_scale = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn_define_dynamic_quantization(subgraph_, mm_id, ynn_type_int8,
                                              &a_zp, &a_scale, 0));
    uint32_t a_q = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn_define_quantize(subgraph_, x_id, ynn_type_int8, a_zp, a_scale,
                                  &a_q, 0));

    // 2. Pack INT4 weights in [N, K] then transpose statically to [K, N]
    QuantLinearInt4 qw = (block_size_ == 32)
                             ? QuantizeLinearQB4W32(w_fp32, N, K, true)
                             : QuantizeLinearQC4W(w_fp32, N, K);
    const_u8_.push_back(std::move(qw.packed_i4));
    size_t w_dims[2] = {static_cast<size_t>(N), static_cast<size_t>(K)};
    uint32_t b_nk = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn_define_tensor(subgraph_, ynn_type_int4, 2, w_dims,
                                const_u8_.back().data(), 0, &b_nk));
    const int32_t perm_t[2] = {1, 0};
    uint32_t b_kn = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn_define_static_transpose(subgraph_, 2, perm_t, b_nk, &b_kn, 0));

    if (block_size_ == 32) {
      // b_scale has shape [N, num_blocks] per composites.h
      size_t nb = static_cast<size_t>(K / 32);
      const_u16_.push_back(std::move(qw.bw_scales_f16));
      size_t s_dims[2] = {static_cast<size_t>(N), nb};
      uint32_t b_scale = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn_define_tensor(subgraph_, ynn_type_fp16, 2, s_dims,
                                  const_u16_.back().data(), 0, &b_scale));
      out_id = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn::define_blockwise_dot(
          subgraph_, a_q, a_zp, a_scale, b_kn, YNN_INVALID_VALUE_ID, b_scale,
          /*block_size=*/32, YNN_INVALID_VALUE_ID, ynn_type_fp32, out_id,
          YNN_NODE_FLAG_SYMMETRIC_B));
    } else {
      // Per-channel QC4W: b_scale [1, N]
      uint32_t b_scale = YNN_INVALID_VALUE_ID;
      if (!DefineConstF32({1, static_cast<size_t>(N)}, qw.pc_scales, b_scale))
        return false;

      uint32_t zp_term = YNN_INVALID_VALUE_ID, dot_scale = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn::define_dot_quantization(
          subgraph_, /*num_k_dims=*/1, a_q, a_zp, a_scale, b_kn,
          YNN_INVALID_VALUE_ID, b_scale, zp_term, dot_scale));

      uint32_t dot_i32 = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn_define_dot(subgraph_, 1, a_q, b_kn, YNN_INVALID_VALUE_ID,
                               &dot_i32, 0));

      uint32_t sub_i32 = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn_define_binary(subgraph_, ynn_binary_subtract, dot_i32,
                                  zp_term, &sub_i32, 0));

      out_id = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn_define_dequantize(subgraph_, sub_i32, YNN_INVALID_VALUE_ID,
                                      dot_scale, ynn_type_fp32, &out_id, 0));
    }
    return true;
  }

  bool DefineRope(uint32_t x_id, int T, int H, int Dh, int rot_dim,
                  uint32_t cos_id, uint32_t sin_id, uint32_t& out_id) {
    int half = rot_dim / 2;
    uint32_t x_rot = x_id;
    uint32_t x_pass = YNN_INVALID_VALUE_ID;
    int32_t ax3 = 3;
    if (rot_dim < Dh) {
      x_rot = YNN_INVALID_VALUE_ID;
      int64_t b0 = 0, e0 = rot_dim, st = 1;
      CHECK_YNN(ynn_define_static_slice(subgraph_, 1, &ax3, &b0, &e0, &st, x_id,
                                        &x_rot, 0));
      int64_t b1 = rot_dim, e1 = Dh;
      CHECK_YNN(ynn_define_static_slice(subgraph_, 1, &ax3, &b1, &e1, &st, x_id,
                                        &x_pass, 0));
    }
    int64_t b0 = 0, e0 = half, b1 = half, e1 = rot_dim, st = 1;
    uint32_t x0 = YNN_INVALID_VALUE_ID, x1 = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn_define_static_slice(subgraph_, 1, &ax3, &b0, &e0, &st, x_rot,
                                      &x0, 0));
    CHECK_YNN(ynn_define_static_slice(subgraph_, 1, &ax3, &b1, &e1, &st, x_rot,
                                      &x1, 0));
    uint32_t neg_x1 = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn_define_unary(subgraph_, ynn_unary_negate, x1, &neg_x1, 0));

    uint32_t pair_in[2] = {neg_x1, x0};
    uint32_t rot_half = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn_define_concatenate(subgraph_, 3, 2, pair_in, &rot_half, 0));

    uint32_t mc = YNN_INVALID_VALUE_ID, ms = YNN_INVALID_VALUE_ID,
             out_rot = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn_define_binary(subgraph_, ynn_binary_multiply, x_rot, cos_id,
                                &mc, 0));
    CHECK_YNN(ynn_define_binary(subgraph_, ynn_binary_multiply, rot_half,
                                sin_id, &ms, 0));
    CHECK_YNN(ynn_define_binary(subgraph_, ynn_binary_add, mc, ms, &out_rot, 0));

    if (rot_dim < Dh) {
      uint32_t cat_in[2] = {out_rot, x_pass};
      out_id = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn_define_concatenate(subgraph_, 3, 2, cat_in, &out_id, 0));
    } else {
      out_id = out_rot;
    }
    return true;
  }

  bool DefineAttn(uint32_t x_id, uint32_t& out_id) {
    const AttnWeightsFp32& aw = *aw_;
    const int D = aw.hidden;
    const int Hq = aw.n_heads;
    const int Hkv = aw.n_kv_heads;
    const int Dh = aw.head_dim;
    const int g = Hq / Hkv;
    const int S_past = S_kv_ - T_;

    uint32_t h = YNN_INVALID_VALUE_ID;
    if (!DefineRmsNorm(x_id, 3, &aw.pre_norm, h)) return false;

    uint32_t q_lin = YNN_INVALID_VALUE_ID, k_lin = YNN_INVALID_VALUE_ID,
             v_lin = YNN_INVALID_VALUE_ID;
    if (!DefineLinearInt4(h, aw.wq, Hq * Dh, D, q_lin)) return false;
    if (!DefineLinearInt4(h, aw.wk, Hkv * Dh, D, k_lin)) return false;
    if (aw.k_eq_v) {
      v_lin = k_lin;
    } else {
      if (!DefineLinearInt4(h, aw.wv, Hkv * Dh, D, v_lin)) return false;
    }

    size_t q4_dims[4] = {1, static_cast<size_t>(T_), static_cast<size_t>(Hq),
                         static_cast<size_t>(Dh)};
    size_t kv4_dims[4] = {1, static_cast<size_t>(T_), static_cast<size_t>(Hkv),
                          static_cast<size_t>(Dh)};
    uint32_t q4 = YNN_INVALID_VALUE_ID, k4 = YNN_INVALID_VALUE_ID,
             v4 = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn_define_static_reshape(subgraph_, 4, q4_dims, q_lin, &q4, 0));
    CHECK_YNN(ynn_define_static_reshape(subgraph_, 4, kv4_dims, k_lin, &k4, 0));
    CHECK_YNN(ynn_define_static_reshape(subgraph_, 4, kv4_dims, v_lin, &v4, 0));

    uint32_t qn = YNN_INVALID_VALUE_ID, kn = YNN_INVALID_VALUE_ID,
             vn = YNN_INVALID_VALUE_ID;
    if (!DefineRmsNorm(q4, 4, &aw.q_norm, qn)) return false;
    if (!DefineRmsNorm(k4, 4, &aw.k_norm, kn)) return false;
    if (!DefineRmsNorm(v4, 4, nullptr, vn)) return false;

    size_t cs_dims[4] = {1, static_cast<size_t>(T_), 1,
                         static_cast<size_t>(aw.rot_dim)};
    uint32_t cos_id = 2, sin_id = 3;
    CHECK_YNN(ynn_define_tensor(subgraph_, ynn_type_fp32, 4, cs_dims, nullptr,
                                YNN_VALUE_FLAG_EXTERNAL_INPUT, &cos_id));
    CHECK_YNN(ynn_define_tensor(subgraph_, ynn_type_fp32, 4, cs_dims, nullptr,
                                YNN_VALUE_FLAG_EXTERNAL_INPUT, &sin_id));

    uint32_t qr = YNN_INVALID_VALUE_ID, kr = YNN_INVALID_VALUE_ID;
    if (!DefineRope(qn, T_, Hq, Dh, aw.rot_dim, cos_id, sin_id, qr)) return false;
    if (!DefineRope(kn, T_, Hkv, Dh, aw.rot_dim, cos_id, sin_id, kr)) return false;

    // Transpose to [1, H, T, Dh]
    const int32_t perm0213[4] = {0, 2, 1, 3};
    uint32_t qt = YNN_INVALID_VALUE_ID, kt = YNN_INVALID_VALUE_ID,
             vt = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn_define_static_transpose(subgraph_, 4, perm0213, qr, &qt, 0));
    CHECK_YNN(ynn_define_static_transpose(subgraph_, 4, perm0213, kr, &kt, 0));
    CHECK_YNN(ynn_define_static_transpose(subgraph_, 4, perm0213, vn, &vt, 0));

    uint32_t k_full = kt;
    uint32_t v_full = vt;
    if (S_past > 0) {
      size_t past_dims[4] = {1, static_cast<size_t>(Hkv),
                             static_cast<size_t>(S_past),
                             static_cast<size_t>(Dh)};
      uint32_t kp_id = 4, vp_id = 5;
      CHECK_YNN(ynn_define_tensor(subgraph_, ynn_type_fp32, 4, past_dims,
                                  nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT,
                                  &kp_id));
      CHECK_YNN(ynn_define_tensor(subgraph_, ynn_type_fp32, 4, past_dims,
                                  nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT,
                                  &vp_id));
      uint32_t kc_in[2] = {kp_id, kt};
      uint32_t vc_in[2] = {vp_id, vt};
      k_full = YNN_INVALID_VALUE_ID;
      v_full = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn_define_concatenate(subgraph_, 2, 2, kc_in, &k_full, 0));
      CHECK_YNN(ynn_define_concatenate(subgraph_, 2, 2, vc_in, &v_full, 0));
    }

    // Direct YNNPACK GQA attention over exact active S_kv:
    // Use 5D [1, Hkv, g, T, Dh] for prefill and folded [1, Hkv, g*T, Dh] for small Q
    // matching YNNPACK's `is_small_q` / `gqa_batch` strategy in attention.cc!
    size_t mask_dims[4] = {1, 1, static_cast<size_t>(T_),
                           static_cast<size_t>(S_kv_)};
    uint32_t mask_id = 6;
    CHECK_YNN(ynn_define_tensor(subgraph_, ynn_type_fp32, 4, mask_dims, nullptr,
                                YNN_VALUE_FLAG_EXTERNAL_INPUT, &mask_id));

    const bool is_small_q = (T_ * g <= 8);
    uint32_t ctx_t = YNN_INVALID_VALUE_ID;

    if (is_small_q) {
      // Fold [1, Hq, T, Dh] -> [1, Hkv, g*T, Dh], compute (K @ Q^T)^T to avoid
      // repacking K (exact `is_small_q` path in attention.cc).
      size_t qg_dims[4] = {1, static_cast<size_t>(Hkv),
                           static_cast<size_t>(g * T_),
                           static_cast<size_t>(Dh)};
      uint32_t qg = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn_define_static_reshape(subgraph_, 4, qg_dims, qt, &qg, 0));

      const int32_t swap2[2] = {-1, -2};
      uint32_t qg_t = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn_define_static_transpose(subgraph_, 2, swap2, qg, &qg_t,
                                            YNN_NODE_FLAG_KEEP_DIMS));
      uint32_t scores_t = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn_define_dot(subgraph_, 1, k_full, qg_t, YNN_INVALID_VALUE_ID,
                               &scores_t, 0));
      uint32_t scores = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn_define_static_transpose(subgraph_, 2, swap2, scores_t,
                                            &scores, YNN_NODE_FLAG_KEEP_DIMS));
      uint32_t masked = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn_define_binary(subgraph_, ynn_binary_add, scores, mask_id,
                                  &masked, 0));
      uint32_t probs = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn::define_softmax(subgraph_, masked, kAttentionScale, probs));

      uint32_t ctx_g = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn_define_dot(subgraph_, 1, probs, v_full,
                               YNN_INVALID_VALUE_ID, &ctx_g, 0));
      size_t ctxt_dims[4] = {1, static_cast<size_t>(Hq), static_cast<size_t>(T_),
                             static_cast<size_t>(Dh)};
      CHECK_YNN(
          ynn_define_static_reshape(subgraph_, 4, ctxt_dims, ctx_g, &ctx_t, 0));
    } else {
      // 5D batched GQA: Q [1, Hkv, g, T, Dh], K [1, Hkv, 1, Dh, S_kv],
      // V [1, Hkv, 1, S_kv, Dh]
      size_t q5_dims[5] = {1, static_cast<size_t>(Hkv), static_cast<size_t>(g),
                           static_cast<size_t>(T_), static_cast<size_t>(Dh)};
      uint32_t q5 = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn_define_static_reshape(subgraph_, 5, q5_dims, qt, &q5, 0));

      const int32_t exp_ax = 2;
      uint32_t k5 = YNN_INVALID_VALUE_ID, v5 = YNN_INVALID_VALUE_ID,
               m5 = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn_define_static_expand_dims(subgraph_, 1, &exp_ax, k_full,
                                              &k5, 0));
      CHECK_YNN(ynn_define_static_expand_dims(subgraph_, 1, &exp_ax, v_full,
                                              &v5, 0));
      CHECK_YNN(ynn_define_static_expand_dims(subgraph_, 1, &exp_ax, mask_id,
                                              &m5, 0));

      const int32_t swap2[2] = {-1, -2};
      uint32_t k5_t = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn_define_static_transpose(subgraph_, 2, swap2, k5, &k5_t,
                                            YNN_NODE_FLAG_KEEP_DIMS));
      uint32_t scores = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn_define_dot(subgraph_, 1, q5, k5_t, YNN_INVALID_VALUE_ID,
                               &scores, 0));
      uint32_t masked = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn_define_binary(subgraph_, ynn_binary_add, scores, m5,
                                  &masked, 0));
      uint32_t probs = YNN_INVALID_VALUE_ID;
      CHECK_YNN(ynn::define_softmax(subgraph_, masked, kAttentionScale, probs));

      uint32_t ctx5 = YNN_INVALID_VALUE_ID;
      CHECK_YNN(
          ynn_define_dot(subgraph_, 1, probs, v5, YNN_INVALID_VALUE_ID, &ctx5, 0));
      size_t ctxt_dims[4] = {1, static_cast<size_t>(Hq), static_cast<size_t>(T_),
                             static_cast<size_t>(Dh)};
      CHECK_YNN(
          ynn_define_static_reshape(subgraph_, 4, ctxt_dims, ctx5, &ctx_t, 0));
    }

    uint32_t ctx_seq = YNN_INVALID_VALUE_ID;
    CHECK_YNN(
        ynn_define_static_transpose(subgraph_, 4, perm0213, ctx_t, &ctx_seq, 0));
    size_t flat_dims[3] = {1, static_cast<size_t>(T_),
                           static_cast<size_t>(Hq * Dh)};
    uint32_t ctx_flat = YNN_INVALID_VALUE_ID;
    CHECK_YNN(
        ynn_define_static_reshape(subgraph_, 3, flat_dims, ctx_seq, &ctx_flat, 0));

    uint32_t o_lin = YNN_INVALID_VALUE_ID;
    if (!DefineLinearInt4(ctx_flat, aw.wo, D, Hq * Dh, o_lin)) return false;

    uint32_t post = YNN_INVALID_VALUE_ID;
    if (!DefineRmsNorm(o_lin, 3, &aw.post_norm, post)) return false;

    out_id = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn_define_binary(subgraph_, ynn_binary_add, post, x_id, &out_id, 0));
    return true;
  }

  bool DefineMlp(uint32_t x_id, uint32_t& out_id) {
    const MlpWeightsFp32& mw = *mw_;
    uint32_t h = YNN_INVALID_VALUE_ID;
    if (!DefineRmsNorm(x_id, 3, &mw.pre_norm, h)) return false;

    uint32_t gate = YNN_INVALID_VALUE_ID, up = YNN_INVALID_VALUE_ID;
    if (!DefineLinearInt4(h, mw.w_gate, mw.inter, mw.hidden, gate)) return false;
    if (!DefineLinearInt4(h, mw.w_up, mw.inter, mw.hidden, up)) return false;

    uint32_t gelu = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn::define_approx_gelu(subgraph_, gate, gelu));

    uint32_t gated = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn_define_binary(subgraph_, ynn_binary_multiply, gelu, up,
                                &gated, 0));

    uint32_t down = YNN_INVALID_VALUE_ID;
    if (!DefineLinearInt4(gated, mw.w_down, mw.hidden, mw.inter, down))
      return false;

    uint32_t post = YNN_INVALID_VALUE_ID;
    if (!DefineRmsNorm(down, 3, &mw.post_norm, post)) return false;

    out_id = YNN_INVALID_VALUE_ID;
    CHECK_YNN(ynn_define_binary(subgraph_, ynn_binary_add, post, x_id, &out_id, 0));
    return true;
  }

  int T_ = 0, S_kv_ = 0, S_cap_ = 0, block_size_ = 0;
  bool has_attn_ = false, has_mlp_ = false;
  const AttnWeightsFp32* aw_ = nullptr;
  const MlpWeightsFp32* mw_ = nullptr;
  ynn_subgraph_t subgraph_ = nullptr;
  ynn_runtime_t runtime_ = nullptr;
  std::unique_ptr<slinky::thread_pool_impl> tp_;
  std::vector<std::vector<float>> const_f32_;
  std::vector<std::vector<uint8_t>> const_u8_;
  std::vector<std::vector<uint16_t>> const_u16_;
};

// -----------------------------------------------------------------------------
// Backend Spec & Regime Table
// -----------------------------------------------------------------------------

struct BackendCfg {
  std::string id;
  std::string kind;  // "litert" | "direct_ynn"
  std::string delegate;  // "xnn" | "ynn" | "ynn_hybrid"
  int block_size = 0;  // 0 = qc4w, 32 = qb4w32
  bool use_fused_sdpa = false;
  bool ynn_enable_blockwise = false;
  std::string weight_quant;  // "qc4w" | "qb4w32"
};

struct RegimeCfg {
  std::string id;
  std::string mode;  // "prefill" | "decode"
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
  // global_full_attn and full_global_block
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

std::vector<BackendCfg> AllBackends() {
  return {
      {"litert_xnn_qc4w", "litert", "xnn", 0, false, false, "qc4w"},
      {"litert_xnn_qb4w32", "litert", "xnn", 32, false, false, "qb4w32"},
      {"litert_ynn_qc4w_decomposed", "litert", "ynn", 0, false, false, "qc4w"},
      {"litert_ynn_qc4w_sdpa", "litert", "ynn", 0, true, false, "qc4w"},
      {"litert_ynn_qb4w32_hybrid_sdpa", "litert", "ynn_hybrid", 32, true, false, "qb4w32"},
      {"litert_ynn_qb4w32_pure_sdpa", "litert", "ynn", 32, true, true, "qb4w32"},
      {"direct_ynn_capi_qc4w", "direct_ynn", "direct", 0, true, false, "qc4w"},
      {"direct_ynn_capi_qb4w32", "direct_ynn", "direct", 32, true, true, "qb4w32"},
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

// -----------------------------------------------------------------------------
// Build, Populate Inputs, Run & Measure One (Backend, Subgraph, Regime, Threads)
// -----------------------------------------------------------------------------

struct RunResult {
  std::string backend_id;
  std::string subgraph_id;
  std::string regime_id;
  std::string mode;
  std::string weight_quant;
  int T = 0;
  int S_kv = 0;
  int S_cap = 0;
  int threads = 0;
  int num_delegated_nodes = 0;
  int num_total_nodes = 0;
  TrialStats stats;
  RooflineSpec roofline;
  MemoryMetrics mem;
  bool ok = false;
  std::string error;

  // Populated in verify mode:
  bool has_verify = false;
  ParityMetrics level_a;
  ParityMetrics level_b;
  bool deterministic_18t = true;
  double max_repro_diff = 0.0;
};

struct RuntimeSetup {
  std::vector<uint8_t> fb_buf;
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interp;
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> del1{nullptr, [](TfLiteDelegate*) {}};
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> del2{nullptr, [](TfLiteDelegate*) {}};

  int32_t x_id = -1, y_id = -1;
  int32_t cos_id = -1, sin_id = -1;
  int32_t k_cache_id = -1, v_cache_id = -1;
  int32_t mask_id = -1, s_active_id = -1;
  int32_t k_start_id = -1, v_start_id = -1;
  int32_t k_upd_id = -1, v_upd_id = -1;
  int num_delegated = 0;
  int num_total = 0;
};

bool BuildLiteRtRuntime(const BackendCfg& bcfg, const std::string& sg_id,
                        int T, int S_cap, const AttnWeightsFp32* aw,
                        const MlpWeightsFp32* mw, int num_threads,
                        RuntimeSetup& rs, std::string& err) {
  TfLiteSubgraphBuilder b;
  rs.x_id = b.AddInput({1, T, kHiddenSize}, tflite::TensorType_FLOAT32, "x_in");
  int32_t cur = rs.x_id;
  if (aw != nullptr) {
    cur = b.AddAttentionBlock(
        cur, T, S_cap, *aw, bcfg.block_size, bcfg.use_fused_sdpa, "attn",
        rs.cos_id, rs.sin_id, rs.k_cache_id, rs.v_cache_id, rs.mask_id,
        rs.s_active_id, rs.k_start_id, rs.v_start_id, rs.k_upd_id, rs.v_upd_id);
  }
  if (mw != nullptr) {
    cur = b.AddMlpBlock(cur, T, *mw, bcfg.block_size, "mlp");
  }
  rs.y_id = cur;
  b.AddOutput(rs.y_id);
  rs.fb_buf = b.Finish();

  rs.model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(rs.fb_buf.data()), rs.fb_buf.size());
  if (!rs.model) {
    err = "FlatBufferModel::BuildFromBuffer failed";
    return false;
  }

  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  if (bcfg.use_fused_sdpa) {
    resolver.AddCustom("odml.sdpa_transposed", RegisterDummySdpa());
  }
  tflite::InterpreterBuilder builder(*rs.model, resolver);
  builder.SetNumThreads(num_threads);
  if (builder(&rs.interp) != kTfLiteOk || !rs.interp) {
    err = "InterpreterBuilder failed";
    return false;
  }

  // Record node count before delegation
  rs.num_total = static_cast<int>(rs.interp->execution_plan().size());

  auto apply_ynn = [&](bool enable_bw) -> bool {
    TfLiteYNNPackDelegateOptions yopts = TfLiteYNNPackDelegateOptionsDefault();
    yopts.num_threads = num_threads;
    yopts.static_shape = true;
    yopts.enable_blockwise_quantization = enable_bw;
    TfLiteDelegate* yd = TfLiteYNNPackDelegateCreate(&yopts);
    if (!yd) {
      err = "TfLiteYNNPackDelegateCreate failed";
      return false;
    }
    rs.del1 = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
        yd, TfLiteYNNPackDelegateDelete);
    if (rs.interp->ModifyGraphWithDelegate(rs.del1.get()) != kTfLiteOk) {
      err = "ModifyGraphWithDelegate(YNNPACK) failed";
      return false;
    }
    return true;
  };

  auto apply_xnn = [&](bool as_second) -> bool {
    TfLiteXNNPackDelegateOptions xopts = TfLiteXNNPackDelegateOptionsDefault();
    xopts.num_threads = num_threads;
    TfLiteDelegate* xd = TfLiteXNNPackDelegateCreate(&xopts);
    if (!xd) {
      err = "TfLiteXNNPackDelegateCreate failed";
      return false;
    }
    auto ptr = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
        xd, TfLiteXNNPackDelegateDelete);
    if (as_second) {
      rs.del2 = std::move(ptr);
      if (rs.interp->ModifyGraphWithDelegate(rs.del2.get()) != kTfLiteOk) {
        err = "ModifyGraphWithDelegate(XNNPACK fallback) failed";
        return false;
      }
    } else {
      rs.del1 = std::move(ptr);
      if (rs.interp->ModifyGraphWithDelegate(rs.del1.get()) != kTfLiteOk) {
        err = "ModifyGraphWithDelegate(XNNPACK) failed";
        return false;
      }
    }
    return true;
  };

  if (bcfg.delegate == "xnn") {
    if (!apply_xnn(false)) return false;
  } else if (bcfg.delegate == "ynn") {
    if (!apply_ynn(bcfg.ynn_enable_blockwise)) return false;
  } else if (bcfg.delegate == "ynn_hybrid") {
    if (!apply_ynn(false)) return false;
    if (!apply_xnn(true)) return false;
  }

  if (rs.interp->AllocateTensors() != kTfLiteOk) {
    err = "AllocateTensors failed";
    return false;
  }

  // Count how many execution-plan nodes are delegate nodes
  rs.num_delegated = 0;
  for (int node_idx : rs.interp->execution_plan()) {
    const auto* nr = rs.interp->node_and_registration(node_idx);
    if (nr && nr->first.delegate != nullptr) rs.num_delegated++;
  }
  return true;
}

// Fill deterministic input buffers shared across ALL runtimes and FP64 ref.
void PrepareSharedInputs(const AttnWeightsFp32* aw, int T, int S_kv,
                         std::vector<float>& x_in,
                         std::vector<float>& cos_rot,
                         std::vector<float>& sin_rot,
                         std::vector<float>& k_past,
                         std::vector<float>& v_past) {
  FillNormal(x_in, static_cast<size_t>(T) * kHiddenSize, 9001, 0.5f);
  if (aw != nullptr) {
    const int S_past = S_kv - T;
    std::vector<float> cos_full, sin_full;
    BuildRopeCosSin(T, S_past, aw->head_dim, aw->rot_dim, aw->rope_theta,
                    cos_full, sin_full);
    cos_rot.resize(static_cast<size_t>(T) * aw->rot_dim);
    sin_rot.resize(static_cast<size_t>(T) * aw->rot_dim);
    for (int t = 0; t < T; ++t) {
      std::memcpy(cos_rot.data() + static_cast<size_t>(t) * aw->rot_dim,
                  cos_full.data() + static_cast<size_t>(t) * aw->head_dim,
                  static_cast<size_t>(aw->rot_dim) * sizeof(float));
      std::memcpy(sin_rot.data() + static_cast<size_t>(t) * aw->rot_dim,
                  sin_full.data() + static_cast<size_t>(t) * aw->head_dim,
                  static_cast<size_t>(aw->rot_dim) * sizeof(float));
    }
    if (S_past > 0) {
      size_t kv_n = static_cast<size_t>(aw->n_kv_heads) * S_past * aw->head_dim;
      FillNormal(k_past, kv_n, 9101, 0.25f);
      FillNormal(v_past, kv_n, 9201, 0.25f);
    } else {
      k_past.clear();
      v_past.clear();
    }
  }
}

void PopulateLiteRtTensors(RuntimeSetup& rs, const BackendCfg& bcfg,
                           const AttnWeightsFp32* aw, int T, int S_kv,
                           int S_cap, const std::vector<float>& x_in,
                           const std::vector<float>& cos_rot,
                           const std::vector<float>& sin_rot,
                           const std::vector<float>& k_past,
                           const std::vector<float>& v_past) {
  auto copy_f32 = [&](int32_t tid, const float* src, size_t n) {
    float* dst = rs.interp->typed_tensor<float>(tid);
    std::memcpy(dst, src, n * sizeof(float));
  };
  copy_f32(rs.x_id, x_in.data(), x_in.size());

  if (aw != nullptr) {
    const int Hkv = aw->n_kv_heads;
    const int Dh = aw->head_dim;
    const int g = aw->n_heads / Hkv;
    const int S_past = S_kv - T;

    copy_f32(rs.cos_id, cos_rot.data(), cos_rot.size());
    copy_f32(rs.sin_id, sin_rot.data(), sin_rot.size());

    // Populate K cache [1, Hkv, S_cap, Dh] and V cache [1, Hkv, Dh, S_cap]
    float* kc = rs.interp->typed_tensor<float>(rs.k_cache_id);
    float* vc = rs.interp->typed_tensor<float>(rs.v_cache_id);
    std::memset(kc, 0, static_cast<size_t>(Hkv) * S_cap * Dh * sizeof(float));
    std::memset(vc, 0, static_cast<size_t>(Hkv) * Dh * S_cap * sizeof(float));
    for (int h = 0; h < Hkv; ++h) {
      for (int s = 0; s < S_past; ++s) {
        const float* ks =
            k_past.data() + (static_cast<size_t>(h) * S_past + s) * Dh;
        const float* vs =
            v_past.data() + (static_cast<size_t>(h) * S_past + s) * Dh;
        std::memcpy(kc + (static_cast<size_t>(h) * S_cap + s) * Dh, ks,
                    static_cast<size_t>(Dh) * sizeof(float));
        for (int d = 0; d < Dh; ++d) {
          vc[(static_cast<size_t>(h) * Dh + d) * S_cap + s] = vs[d];
        }
      }
    }

    int32_t* ks = rs.interp->typed_tensor<int32_t>(rs.k_start_id);
    ks[0] = 0; ks[1] = 0; ks[2] = S_past; ks[3] = 0;
    int32_t* vs = rs.interp->typed_tensor<int32_t>(rs.v_start_id);
    vs[0] = 0; vs[1] = 0; vs[2] = 0; vs[3] = S_past;

    if (bcfg.use_fused_sdpa) {
      // Bool mask [1, 1, T, S_cap]: true = attend, false = masked
      bool* bm = rs.interp->typed_tensor<bool>(rs.mask_id);
      std::memset(bm, 0, static_cast<size_t>(T) * S_cap * sizeof(bool));
      for (int t = 0; t < T; ++t) {
        int q_pos = S_past + t;
        int s_min = (aw->window > 0) ? std::max(0, q_pos - aw->window + 1) : 0;
        for (int s = s_min; s <= q_pos; ++s) {
          bm[static_cast<size_t>(t) * S_cap + s] = true;
        }
      }
      int32_t* sa = rs.interp->typed_tensor<int32_t>(rs.s_active_id);
      sa[0] = S_kv;
    } else {
      // Float mask [1, 1, g*T, S_cap]: 0.0f = attend, -1e9f = masked
      float* fm = rs.interp->typed_tensor<float>(rs.mask_id);
      std::fill(fm, fm + static_cast<size_t>(g * T) * S_cap, -1e9f);
      for (int gi = 0; gi < g; ++gi) {
        for (int t = 0; t < T; ++t) {
          int q_pos = S_past + t;
          int s_min = (aw->window > 0) ? std::max(0, q_pos - aw->window + 1) : 0;
          float* row = fm + static_cast<size_t>(gi * T + t) * S_cap;
          for (int s = s_min; s <= q_pos; ++s) row[s] = 0.0f;
        }
      }
    }
  }
}

void BuildDirectFloatMask(const AttnWeightsFp32& aw, int T, int S_kv,
                          std::vector<float>& fmask) {
  const int S_past = S_kv - T;
  fmask.assign(static_cast<size_t>(T) * S_kv, -1e9f);
  for (int t = 0; t < T; ++t) {
    int q_pos = S_past + t;
    int s_min = (aw.window > 0) ? std::max(0, q_pos - aw.window + 1) : 0;
    float* row = fmask.data() + static_cast<size_t>(t) * S_kv;
    for (int s = s_min; s <= q_pos; ++s) row[s] = 0.0f;
  }
}

RunResult ExecuteCase(const BackendCfg& bcfg, const std::string& sg_id,
                      const RegimeCfg& reg, int threads, int trials,
                      bool verify_mode, const AttnWeightsFp32* aw,
                      const MlpWeightsFp32* mw, SlcFlusher& flusher,
                      const std::vector<double>* ref_unquant,
                      const std::vector<double>* ref_dequant) {
  RunResult res;
  res.backend_id = bcfg.id;
  res.subgraph_id = sg_id;
  res.regime_id = reg.id;
  res.mode = reg.mode;
  res.weight_quant = bcfg.weight_quant;
  res.T = reg.T;
  res.S_kv = reg.S_kv;
  res.S_cap = reg.S_cap;
  res.threads = threads;
  res.roofline = ComputeSubgraphRoofline(sg_id, reg.T, reg.S_kv,
                                         bcfg.weight_quant, "fp32");

  std::vector<float> x_in, cos_rot, sin_rot, k_past, v_past, direct_mask;
  PrepareSharedInputs(aw, reg.T, reg.S_kv, x_in, cos_rot, sin_rot, k_past,
                      v_past);

  std::vector<float> y_out(static_cast<size_t>(reg.T) * kHiddenSize, 0.0f);

  if (bcfg.kind == "litert") {
    RuntimeSetup rs;
    if (!BuildLiteRtRuntime(bcfg, sg_id, reg.T, reg.S_cap, aw, mw, threads, rs,
                            res.error)) {
      res.ok = false;
      return res;
    }
    res.num_delegated_nodes = rs.num_delegated;
    res.num_total_nodes = rs.num_total;

    PopulateLiteRtTensors(rs, bcfg, aw, reg.T, reg.S_kv, reg.S_cap, x_in,
                          cos_rot, sin_rot, k_past, v_past);

    // Warmup
    for (int w = 0; w < 2; ++w) {
      if (rs.interp->Invoke() != kTfLiteOk) {
        res.error = "LiteRT Invoke failed during warmup";
        return res;
      }
    }

    const float* y_ptr = rs.interp->typed_tensor<float>(rs.y_id);
    std::memcpy(y_out.data(), y_ptr, y_out.size() * sizeof(float));

    if (verify_mode) {
      res.has_verify = true;
      res.level_a = CompareAgainstFp64(y_out.data(), ref_dequant->data(), reg.T,
                                       kHiddenSize, 0.9990, 0.045, 27.0, 0.005);
      res.level_b = CompareAgainstFp64(y_out.data(), ref_unquant->data(), reg.T,
                                       kHiddenSize, 0.930, 0.38, 8.5, 0.09);
      // Determinism check: invoke 3 more times and measure max bitwise diff
      res.max_repro_diff = 0.0;
      for (int r = 0; r < 3; ++r) {
        rs.interp->Invoke();
        const float* yp = rs.interp->typed_tensor<float>(rs.y_id);
        for (size_t i = 0; i < y_out.size(); ++i) {
          double d = std::fabs(static_cast<double>(yp[i]) - y_out[i]);
          if (d > res.max_repro_diff) res.max_repro_diff = d;
        }
      }
      res.deterministic_18t = (res.max_repro_diff == 0.0);
      res.mem = ReadMemoryMetrics();
      res.ok = true;
      return res;
    }

    // Calibrate iterations per trial
    double t0 = NowSec();
    rs.interp->Invoke();
    double one_ms = (NowSec() - t0) * 1e3;
    int iters = 1;
    if (reg.mode == "decode") {
      iters = std::max(1, std::min(12, static_cast<int>(std::ceil(60.0 / std::max(one_ms, 0.2)))));
    } else {
      iters = std::max(1, std::min(6, static_cast<int>(std::ceil(80.0 / std::max(one_ms, 1.0)))));
    }

    std::vector<double> samples_ms;
    samples_ms.reserve(trials);
    for (int tr = 0; tr < trials; ++tr) {
      double acc_s = 0.0;
      for (int it = 0; it < iters; ++it) {
        if (reg.mode == "decode") flusher.Flush();
        double s0 = NowSec();
        rs.interp->Invoke();
        acc_s += (NowSec() - s0);
      }
      samples_ms.push_back((acc_s / iters) * 1e3);
    }
    res.stats = ComputeTrialStats(samples_ms, iters);
    res.mem = ReadMemoryMetrics();
    res.ok = true;
    return res;
  }

  // Direct YNNPACK C API path
  DirectYnnSubgraph dy;
  if (!dy.Build(sg_id, reg.T, reg.S_kv, reg.S_cap, aw, mw, bcfg.block_size,
                threads)) {
    res.error = "DirectYnnSubgraph::Build failed";
    return res;
  }
  res.num_delegated_nodes = 1;
  res.num_total_nodes = 1;
  if (aw != nullptr) BuildDirectFloatMask(*aw, reg.T, reg.S_kv, direct_mask);

  for (int w = 0; w < 2; ++w) {
    if (!dy.Invoke(x_in, y_out, &cos_rot, &sin_rot, &k_past, &v_past,
                   &direct_mask)) {
      res.error = "DirectYnnSubgraph::Invoke warmup failed";
      return res;
    }
  }

  if (verify_mode) {
    res.has_verify = true;
    res.level_a = CompareAgainstFp64(y_out.data(), ref_dequant->data(), reg.T,
                                     kHiddenSize, 0.9990, 0.045, 27.0, 0.005);
    res.level_b = CompareAgainstFp64(y_out.data(), ref_unquant->data(), reg.T,
                                     kHiddenSize, 0.930, 0.38, 8.5, 0.09);
    std::vector<float> y_chk(y_out.size());
    res.max_repro_diff = 0.0;
    for (int r = 0; r < 3; ++r) {
      dy.Invoke(x_in, y_chk, &cos_rot, &sin_rot, &k_past, &v_past, &direct_mask);
      for (size_t i = 0; i < y_out.size(); ++i) {
        double d = std::fabs(static_cast<double>(y_chk[i]) - y_out[i]);
        if (d > res.max_repro_diff) res.max_repro_diff = d;
      }
    }
    res.deterministic_18t = (res.max_repro_diff == 0.0);
    res.mem = ReadMemoryMetrics();
    res.ok = true;
    return res;
  }

  double t0 = NowSec();
  dy.Invoke(x_in, y_out, &cos_rot, &sin_rot, &k_past, &v_past, &direct_mask);
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
      dy.Invoke(x_in, y_out, &cos_rot, &sin_rot, &k_past, &v_past, &direct_mask);
      acc_s += (NowSec() - s0);
    }
    samples_ms.push_back((acc_s / iters) * 1e3);
  }
  res.stats = ComputeTrialStats(samples_ms, iters);
  res.mem = ReadMemoryMetrics();
  res.ok = true;
  return res;
}

// Helper to create dequantized FP32 weight copy for Level A FP64 reference
AttnWeightsFp32 DequantAttnWeights(const AttnWeightsFp32& aw, int block_size) {
  AttnWeightsFp32 d = aw;
  auto dq = [&](const std::vector<float>& w, int N, int K) {
    auto q = (block_size == 32) ? QuantizeLinearQB4W32(w, N, K, true)
                                : QuantizeLinearQC4W(w, N, K);
    return DequantizeLinearInt4(q);
  };
  int q_dim = aw.n_heads * aw.head_dim;
  int kv_dim = aw.n_kv_heads * aw.head_dim;
  d.wq = dq(aw.wq, q_dim, aw.hidden);
  d.wk = dq(aw.wk, kv_dim, aw.hidden);
  if (!aw.k_eq_v) d.wv = dq(aw.wv, kv_dim, aw.hidden);
  d.wo = dq(aw.wo, aw.hidden, q_dim);
  return d;
}

MlpWeightsFp32 DequantMlpWeights(const MlpWeightsFp32& mw, int block_size) {
  MlpWeightsFp32 d = mw;
  auto dq = [&](const std::vector<float>& w, int N, int K) {
    auto q = (block_size == 32) ? QuantizeLinearQB4W32(w, N, K, true)
                                : QuantizeLinearQC4W(w, N, K);
    return DequantizeLinearInt4(q);
  };
  d.w_gate = dq(mw.w_gate, mw.inter, mw.hidden);
  d.w_up = dq(mw.w_up, mw.inter, mw.hidden);
  d.w_down = dq(mw.w_down, mw.hidden, mw.inter);
  return d;
}

void WriteResultsJson(const std::string& path, const std::string& mode,
                      const std::string& litert_commit,
                      const std::string& xnnpack_commit,
                      const std::vector<RunResult>& results) {
  std::ofstream os(path);
  os << std::fixed << std::setprecision(6);
  os << "{\n";
  os << "  \"harness\": \"bench_litert_layers\",\n";
  os << "  \"mode\": \"" << JsonEscape(mode) << "\",\n";
  os << "  \"litert_commit\": \"" << JsonEscape(litert_commit) << "\",\n";
  os << "  \"xnnpack_commit\": \"" << JsonEscape(xnnpack_commit) << "\",\n";
  os << "  \"results\": [\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const auto& r = results[i];
    os << "    {\n";
    os << "      \"backend_id\": \"" << JsonEscape(r.backend_id) << "\",\n";
    os << "      \"subgraph_id\": \"" << JsonEscape(r.subgraph_id) << "\",\n";
    os << "      \"regime_id\": \"" << JsonEscape(r.regime_id) << "\",\n";
    os << "      \"mode\": \"" << JsonEscape(r.mode) << "\",\n";
    os << "      \"weight_quant\": \"" << JsonEscape(r.weight_quant) << "\",\n";
    os << "      \"kv_dtype\": \"fp32\",\n";
    os << "      \"T\": " << r.T << ", \"S_kv\": " << r.S_kv
       << ", \"S_cap\": " << r.S_cap << ", \"threads\": " << r.threads << ",\n";
    os << "      \"ok\": " << (r.ok ? "true" : "false") << ",\n";
    if (!r.error.empty()) {
      os << "      \"error\": \"" << JsonEscape(r.error) << "\",\n";
    }
    os << "      \"delegated_nodes\": " << r.num_delegated_nodes
       << ", \"total_nodes\": " << r.num_total_nodes << ",\n";
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
  std::string backends_arg =
      "litert_xnn_qc4w,litert_xnn_qb4w32,litert_ynn_qc4w_decomposed,"
      "litert_ynn_qc4w_sdpa,litert_ynn_qb4w32_hybrid_sdpa,"
      "litert_ynn_qb4w32_pure_sdpa,direct_ynn_capi_qc4w,direct_ynn_capi_qb4w32";
  std::string subgraphs_arg =
      "local_swa_attn,global_full_attn,geglu_mlp,full_local_block,full_global_block";
  std::string regimes_arg = "";  // empty = all regimes for each subgraph
  std::string threads_arg = "6,18";
  std::string output_path = "litert_layer_results.json";
  std::string litert_commit = "unknown";
  std::string xnnpack_commit = "unknown";
  int trials = 7;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      return (i + 1 < argc) ? std::string(argv[++i]) : "";
    };
    if (a == "--mode") mode = next();
    else if (a == "--backends") backends_arg = next();
    else if (a == "--subgraphs") subgraphs_arg = next();
    else if (a == "--regimes") regimes_arg = next();
    else if (a == "--threads") threads_arg = next();
    else if (a == "--trials") trials = std::atoi(next().c_str());
    else if (a == "--output") output_path = next();
    else if (a == "--litert-commit") litert_commit = next();
    else if (a == "--xnnpack-commit") xnnpack_commit = next();
  }

  auto want_backends = SplitCsv(backends_arg);
  auto want_subgraphs = SplitCsv(subgraphs_arg);
  auto want_regimes = SplitCsv(regimes_arg);
  std::vector<int> thread_list;
  for (const auto& ts : SplitCsv(threads_arg)) {
    thread_list.push_back(std::atoi(ts.c_str()));
  }

  auto all_b = AllBackends();
  std::vector<BackendCfg> active_backends;
  for (const auto& wid : want_backends) {
    for (const auto& b : all_b) {
      if (b.id == wid) active_backends.push_back(b);
    }
  }

  fprintf(stderr,
          "[bench_litert_layers] mode=%s backends=%zu subgraphs=%zu trials=%d\n",
          mode.c_str(), active_backends.size(), want_subgraphs.size(), trials);

  // Initialize deterministic Gemma 4 31B weights once
  AttnWeightsFp32 local_aw = MakeAttnWeights(/*is_global=*/false, 42);
  AttnWeightsFp32 global_aw = MakeAttnWeights(/*is_global=*/true, 72);
  MlpWeightsFp32 mlp_w = MakeMlpWeights(142);

  // Dequantized weights for Level A reference in verify mode
  AttnWeightsFp32 local_aw_qc4, local_aw_qb4, global_aw_qc4, global_aw_qb4;
  MlpWeightsFp32 mlp_w_qc4, mlp_w_qb4;
  if (mode == "verify") {
    local_aw_qc4 = DequantAttnWeights(local_aw, 0);
    local_aw_qb4 = DequantAttnWeights(local_aw, 32);
    global_aw_qc4 = DequantAttnWeights(global_aw, 0);
    global_aw_qb4 = DequantAttnWeights(global_aw, 32);
    mlp_w_qc4 = DequantMlpWeights(mlp_w, 0);
    mlp_w_qb4 = DequantMlpWeights(mlp_w, 32);
  }

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
      fprintf(stderr, "Unknown subgraph: %s\n", sg_id.c_str());
      continue;
    }

    std::vector<RegimeCfg> regs;
    if (mode == "verify") {
      // Verify BOTH a multi-token prefill (T=8, S_kv=32) AND a decode step (T=1, S_kv=64)
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
      std::vector<double> ref_unq, ref_dq_qc4, ref_dq_qb4;
      if (mode == "verify") {
        std::vector<float> x_in, cos_r, sin_r, kp, vp;
        PrepareSharedInputs(aw, reg.T, reg.S_kv, x_in, cos_r, sin_r, kp, vp);
        std::vector<double> x_d(x_in.begin(), x_in.end());
        int S_past = reg.S_kv - reg.T;
        auto run_ref = [&](const AttnWeightsFp32* a, const MlpWeightsFp32* m) {
          if (a && m)
            return ref_fp64::RunFullBlockFp64(*a, *m, 1.0f, x_d, reg.T, S_past,
                                              kp, vp);
          if (a) return ref_fp64::RunAttnFp64(*a, x_d, reg.T, S_past, kp, vp);
          return ref_fp64::RunMlpFp64(*m, x_d, reg.T);
        };
        ref_unq = run_ref(aw, mw);
        const AttnWeightsFp32* aw_qc4 =
            (aw == &local_aw) ? &local_aw_qc4
                              : (aw == &global_aw ? &global_aw_qc4 : nullptr);
        const AttnWeightsFp32* aw_qb4 =
            (aw == &local_aw) ? &local_aw_qb4
                              : (aw == &global_aw ? &global_aw_qb4 : nullptr);
        ref_dq_qc4 = run_ref(aw_qc4, mw ? &mlp_w_qc4 : nullptr);
        ref_dq_qb4 = run_ref(aw_qb4, mw ? &mlp_w_qb4 : nullptr);
      }

      for (const auto& bcfg : active_backends) {
        // For geglu_mlp (no attention), skip duplicate sdpa-only variants
        if (sg_id == "geglu_mlp" && bcfg.id == "litert_ynn_qc4w_sdpa") continue;

        for (int thr : thread_list) {
          const std::vector<double>* dq_ref =
              (bcfg.block_size == 32) ? &ref_dq_qb4 : &ref_dq_qc4;
          RunResult rr = ExecuteCase(bcfg, sg_id, reg, thr, trials,
                                     mode == "verify", aw, mw, flusher,
                                     &ref_unq, dq_ref);
          if (mode == "verify") {
            fprintf(stderr,
                    "  [VERIFY] %-30s %-18s %-22s %2dT ok=%d | "
                    "LvA(cos=%.6f l2=%.4f sqnr=%.1fdB kl=%.2e pass=%d) | "
                    "LvB(cos=%.6f l2=%.4f sqnr=%.1fdB kl=%.2e pass=%d) | "
                    "det=%d(%.2e)\n",
                    rr.backend_id.c_str(), sg_id.c_str(), reg.id.c_str(), thr,
                    rr.ok ? 1 : 0, rr.level_a.cosine_sim, rr.level_a.rel_l2_err,
                    rr.level_a.sqnr_db, rr.level_a.kl_div,
                    rr.level_a.passed ? 1 : 0, rr.level_b.cosine_sim,
                    rr.level_b.rel_l2_err, rr.level_b.sqnr_db,
                    rr.level_b.kl_div, rr.level_b.passed ? 1 : 0,
                    rr.deterministic_18t ? 1 : 0, rr.max_repro_diff);
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
                    "  [BENCH]  %-30s %-18s %-22s %2dT ok=%d del=%d/%d | "
                    "med=%8.3fms cv=%4.2f%% | %7.1f GOPS | %6.1f GB/s\n",
                    rr.backend_id.c_str(), sg_id.c_str(), reg.id.c_str(), thr,
                    rr.ok ? 1 : 0, rr.num_delegated_nodes, rr.num_total_nodes,
                    rr.stats.median_ms, rr.stats.cv_pct, eff_gops, eff_gbps);
          }
          results.push_back(std::move(rr));
          // Flush intermediate progress to JSON after every case
          WriteResultsJson(output_path, mode, litert_commit, xnnpack_commit,
                           results);
        }
      }
    }
  }

  fprintf(stderr, "[bench_litert_layers] Wrote %zu records to %s\n",
          results.size(), output_path.c_str());
  return 0;
}
