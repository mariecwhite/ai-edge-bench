# Compound Subgraph Implementations (`layer/subgraphs/`)

Self-contained, deterministic C++17 subgraph harnesses implementing the exact **Gemma 4 31B (`google/gemma-4-31B-it`)** text decoder architecture (`hidden_size = 5376`, `intermediate_size = 21504`, `rms_norm_eps = 1e-6`, `attention_scale = 1.0`, QK-norm, parameter-free V-norm, NeoX/half-split RoPE, `K = V` global sharing with `0.25×` proportional RoPE, GeGLU `gelu_pytorch_tanh`, and pre/post-norm residuals):

1. **[common/gemma4_layer_common.h](common/gemma4_layer_common.h)** — Shared deterministic SplitMix64 weight/input generators, symmetric INT4 quantizers (`QC4W` per-channel & `QB4W-32` blockwise-32 with FP16 scales), independent **FP64 golden reference** for all 5 subgraphs (supporting Level A dequantized-weight and Level B unquantized verification), numerical parity evaluator (`cosine_sim`, `rel_l2_err`, `sqnr_db`, `max_abs_err`, softmax `kl_div`, `nan_inf_count`), boundary-only roofline op/byte calculator, `CLOCK_MONOTONIC_RAW` + `QOS_CLASS_USER_INTERACTIVE` timing, 96 MB SLC cache flusher for cold `M = 1` decode, and macOS `phys_footprint` / `ru_maxrss` accounting.
2. **[litert/bench_litert_layers.cc](litert/bench_litert_layers.cc)** — Builds genuine TFLite FlatBuffer graphs and Direct YNNPACK C API subgraphs for all 5 subgraphs (`local_swa_attn`, `global_full_attn`, `geglu_mlp`, `full_local_block`, `full_global_block`) across **8 backend configurations**:
   - `litert_xnn_qc4w` — LiteRT `TfLiteXNNPackDelegate` (per-channel INT4, decomposed attention)
   - `litert_xnn_qb4w32` — LiteRT `TfLiteXNNPackDelegate` (blockwise-32 INT4, decomposed attention)
   - `litert_ynn_qc4w_decomposed` — LiteRT `TfLiteYNNPackDelegate` (per-channel INT4, decomposed attention)
   - `litert_ynn_qc4w_sdpa` — LiteRT `TfLiteYNNPackDelegate` (per-channel INT4, fused `odml.sdpa_transposed` + bool mask + dynamic `s_active`)
   - `litert_ynn_qb4w32_hybrid_sdpa` — LiteRT `TfLiteYNNPackDelegate` (`enable_blockwise_quantization=false`) + `TfLiteXNNPackDelegate` fallback + fused `odml.sdpa_transposed` (reproducing LiteRT-LM's hybrid execution)
   - `litert_ynn_qb4w32_pure_sdpa` — LiteRT `TfLiteYNNPackDelegate` (`enable_blockwise_quantization=true`, zero XNNPACK fallback) + fused `odml.sdpa_transposed`
   - `direct_ynn_capi_qc4w` — Direct YNNPACK C API + Slinky threadpool (per-channel INT4, dynamically sliced KV)
   - `direct_ynn_capi_qb4w32` — Direct YNNPACK C API + `ynn::define_blockwise_dot` + Slinky threadpool (blockwise-32 INT4)
3. **[ggml/bench_ggml_layers.cpp](ggml/bench_ggml_layers.cpp)** — Builds the exact same 5 Gemma 4 31B subgraphs directly with `ggml` (`llama.cpp` `b12469` CPU backend with `Q4_0` KleidiAI-repacked weights, `FP16` KV cache, and `ggml_flash_attn_ext` plus optional `--no-flash-attn` unfused ablation):
   - `ggml_cpu_neon_fa` (`GGML_KLEIDIAI_SME=0`, KleidiAI NEON/`i8mm` + Accelerate + Flash Attention)
   - `ggml_cpu_sme_fa` (`GGML_KLEIDIAI_SME=1`, KleidiAI SME2 + Flash Attention)
   - `ggml_cpu_neon_unfused` (`GGML_KLEIDIAI_SME=0 --no-flash-attn`, unfused attention ablation)

## Build & Run (macOS ARM64 / Apple Silicon)

```bash
# 1. Build bench_ggml_layers against llama.cpp (CPU-only with KleidiAI):
cmake -G Ninja -S llama.cpp -B build/llama.cpp \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DGGML_NATIVE=ON \
  -DGGML_CPU_KLEIDIAI=ON -DBUILD_SHARED_LIBS=ON
cmake --build build/llama.cpp -j 18 --target ggml ggml-cpu ggml-base

clang++ -O3 -std=c++17 -mcpu=native \
  -I. -Illama.cpp/ggml/include \
  layer/subgraphs/ggml/bench_ggml_layers.cpp \
  -Lbuild/llama.cpp/bin -lggml -lggml-cpu -lggml-base \
  -Wl,-rpath,build/llama.cpp/bin -o bench_ggml_layers

# 2. Build bench_litert_layers against upstream google-ai-edge/LiteRT (Bazel):
#    Link //tflite:framework, //tflite/delegates/xnnpack:xnnpack_delegate,
#    //tflite/delegates/ynnpack:ynnpack_delegate, //tflite/kernels:builtin_ops,
#    @XNNPACK//ynnpack:ynnpack, @XNNPACK//ynnpack/composites:composites,
#    @slinky//slinky/base:thread_pool_impl, @flatbuffers//:runtime_cc

# 3. Verify FP64 numerical parity & 18T determinism, then run 7-trial benchmarks:
./bench_litert_layers --mode verify --threads 18 --output litert_verify.json
GGML_KLEIDIAI_SME=0 ./bench_ggml_layers --mode verify --threads 18 --output ggml_neon_verify.json
./bench_litert_layers --mode bench --threads 6,18 --trials 7 --output bench_litert_all.json
GGML_KLEIDIAI_SME=0 ./bench_ggml_layers --mode bench --threads 6,18 --trials 7 --output bench_ggml_neon.json
GGML_KLEIDIAI_SME=1 ./bench_ggml_layers --mode bench --backend-id ggml_cpu_sme_fa --threads 6,18 --trials 7 --output bench_ggml_sme.json
```
