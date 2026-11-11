## Key architecture & methodology notes (Gemma 4 31B on x86 Sapphire Rapids — BF16, INT8, and INT4)

1. **Model Architecture (`google/gemma-4-31B-it`).**
   - Dense **60-layer** transformer (`hidden_size = 5376`, `intermediate_size = 21504`, `num_attention_heads = 32`, `vocab_size = 262144`, `hidden_size_per_layer_input = 0`, no Per-Layer Embeddings).
   - **Hybrid attention schedule (50 sliding-window + 10 full global layers):** Every 6th layer (5, 11, 17, 23, 29, 35, 41, 47, 53, 59) uses full global causal attention (`head_dim = 512`, `num_key_value_heads = 4`, `k_eq_v = True` where $K$ and $V$ share a single projection), while the remaining 50 layers use local sliding-window attention (`sliding_window = 1024`, `head_dim = 256`, `num_key_value_heads = 16`, separate $K$ and $V$ projections).

2. **Precision Matrix & Framework Versions (`BF16`, `INT8`, and `INT4`).**
   - **LiteRT-LM (`v0.19.1`, commit `7a03724cece4`, built from source with `--define=litert_enable_ynnpack=true`)**:
     - `litert-lm/matched` (XNNPACK delegate, 32 threads), `litert-lm/fastest` (XNNPACK delegate, 56 physical cores), `litert-lm/ynnpack` (experimental **YNNPACK** delegate via `--ynnpack` / `litert_lm_engine_settings_set_enable_ynnpack`, 32 threads), and `litert-lm/ynnpack-fastest` (YNNPACK delegate, 56 physical cores).
     - Executes `litert-community/gemma-4-31B-it-litert-lm/gemma-4-31B-it.litertlm` (`16.7 GB`, exported with `dynamic_wi4_afp32` from `google/gemma-4-31B-it-qat-q4_0-unquantized`).
   - **`llama.cpp` (`b12247`, commit `c1a4febe9d10`, compiled with AVX-512 + VNNI + BF16 + AMX)**:
     - `llama.cpp/matched` (`google/gemma-4-31B-it-qat-q4_0-gguf/gemma-4-31B_q4_0-it.gguf`, 32 threads), `llama.cpp/fastest` (`ggml-org/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_0.gguf`, 32 threads, `--batch 2048 --ubatch 512 --flash-attn on`), `llama.cpp/q8_0` (`gemma-4-31B-it-Q8_0.gguf`, 32 threads), and `llama.cpp/bf16` (`gemma-4-31B-it-BF16.gguf`, 32 threads).
   - **OpenVINO GenAI (`2026.4.1.0` / `openvino==2026.4.1`)**:
     - `openvino/matched` (`OpenVINO/gemma-4-31B-it-int4-ov`, 32 threads), `openvino/fastest` (`OpenVINO/gemma-4-31B-it-int4-ov`, 56 physical cores), and `openvino/int8` (`OpenVINO/gemma-4-31B-it-int8-ov`, 56 physical cores) using oneDNN AMX/AVX-512 kernels.
     - Stateless runs disable internal prefix caching (`SchedulerConfig(enable_prefix_caching=False)`); `--cache-prefix` runs pre-populate the 896-token shared prefix and evaluate incremental 128-token suffixes.
   - **OpenXLA (`jax==0.12.0.1` / `jaxlib==0.12.0.1`, `flax==0.12.12`, `qwix==0.1.8`, `google-tunix==0.1.7`)**:
     - `xla/bf16-ynnpack` (YNNPACK `__ynn_fusion` backend via `--xla_cpu_use_ynn=true`, 56 threads) on `google/gemma-4-31B-it` (`60.4 GB` BF16 safetensors), achieving ~116.6 tok/s prefill (8,781 ms TTFT on 1,024 tokens, 2.8× faster prefill than `llama.cpp/bf16`).
     - Note on `xla/bf16-ynnpack` single-token decode: In `jaxlib==0.12.0.1`, XLA:CPU's `IsDotSupportedByYnn` rejects dot contractions with `m == 1` (`T=1` decode), causing `FloatNormalization` to upcast the 60 GB BF16 weights to 120 GB FP32 buffers for Eigen GEMV (~181 GB peak RSS, ~0.33 tok/s decode).
     - **Intel oneDNN backend (`--onednn`) disabled:** `xla/bf16-onednn` (`--xla_cpu_use_onednn=true`) segfaults (`SIGSEGV`, exit code `-11`) inside `ThunkExecutor::Execute` when executing 1,024-token batched 4D attention contractions (`BTKGH,BSKH->BTKGS`) across 56 IntraOp threads on Intel AMX.

3. **Accuracy & Validity Gate Findings.**
   - **BF16 Reference Equivalence (`llama.cpp/bf16` vs. `xla/bf16-ynnpack`):** `llama.cpp/bf16` achieves **86.5% (`173/200`)** on MMLU and `xla/bf16-ynnpack` achieves **86.0% (`172/200`)**, confirming numerical consistency between `llama.cpp` and Tunix/XLA:CPU YNNPACK on the 60-layer Gemma 4 31B BF16 weights.
   - **No stock `Q4_0` degradation on Gemma 4 31B (unlike E2B):** Because Gemma 4 31B has no Per-Layer Embeddings (`hidden_size_per_layer_input = 0`), `ggml-org`'s stock `Q4_0` GGUF (`llama.cpp/fastest`) achieves identical accuracy to Google's QAT `Q4_0` GGUF (`llama.cpp/matched`): **100.0% (`50/50`) on GSM8K** and **85.5% (`171/200`) on MMLU** (within 1.0 percentage point of `llama.cpp/bf16`'s **86.5%**, `p = 0.77`).
   - **OpenVINO GenAI (`2026.4.1.0`):** `openvino/fastest` (INT4) achieves **100.0% (`50/50`) on GSM8K** and **86.0% (`172/200`) on MMLU**, matching `openvino/int8` (**86.0%** MMLU), `xla/bf16-ynnpack` (**86.0%** MMLU), and `llama.cpp/bf16` (**86.5%** MMLU).
   - **LiteRT-LM XNNPACK vs. Experimental YNNPACK (`v0.19.1`):**
     - `litert-lm/matched` (XNNPACK delegate) achieves **94.0% (`47/50`) on GSM8K** and **71.0% (`142/200`) on MMLU**, clearing both validity thresholds (`>= 40%` GSM8K, `>= 45%` MMLU).
     - Conversely, `litert-lm/ynnpack` (`--ynnpack` / `ForceEnableYnnPack`) fails the accuracy validity gate (**0.0% MMLU, 0.0% GSM8K**, 100% parse failures): on `gemma-4-31B-it.litertlm`, the experimental YNNPACK TFLite delegate produces corrupted logits (repeating `" thẻ\n"`) on the 31B 4-bit block32 weights, and is automatically excluded from the input-length sequence sweep by the accuracy gate.

4. **Hardware & NUMA Pinning (`Sapphire Rapids`).**
   - Dual-socket **4th Gen Intel Xeon Scalable (Sapphire Rapids)** server (2× 56 physical cores, ~1,007.6 GiB DDR5 RAM across 2 NUMA sockets).
   - ISA extensions present: `avx512f`, `avx512bw`, `avx512dq`, `avx512vl`, `avx512_vnni`, `avx512_bf16`, `avx512_fp16`, `amx_tile`, `amx_int8`, `amx_bf16`.
   - All single-stream performance (`perf`, `prefix-cache`, `sequence`) processes are strictly pinned to **Socket 0 physical cores** via `numactl --physcpubind=... --localalloc`, avoiding cross-socket UPI traffic and SMT sibling contention.
   - During parallel `repro` and 2-shard `accuracy` runs (`--parallel 2`), Queue 0 is pinned to Socket 0 physical cores and Queue 1 is pinned to Socket 1 physical cores.

## Caveats
- **`litert-lm/ynnpack` and `litert-lm/ynnpack-fastest` fail the accuracy validity gate (`0.0%` MMLU/GSM8K):** In LiteRT-LM `v0.19.1`, enabling the experimental YNNPACK TFLite delegate (`--ynnpack`) on `gemma-4-31B-it.litertlm` produces corrupted token logits (`" thẻ\n"` repetition). Their throughput numbers are included in the headline performance tables for diagnostic comparison against XNNPACK, but they do not produce valid text on 31B and are excluded from the sequence sweep by the accuracy gate.
- **`xla/bf16-ynnpack` single-token decode fallback:** In `jaxlib==0.12.0.1`, XLA:CPU's `IsDotSupportedByYnn` rejects dot contractions with `m == 1`, so `xla/bf16-ynnpack` uses YNNPACK `__ynn_fusion` for multi-token prefill (`116.6 tok/s`, 2.8× faster than `llama.cpp/bf16`) but falls back to FP32 upcast + Eigen GEMV for single-token decode (`0.33 tok/s`, `181.5 GB` peak RSS).
- **`xla/bf16-onednn` disabled:** XLA:CPU's experimental oneDNN contraction rewriter (`--xla_cpu_use_onednn=true`) segfaults (`SIGSEGV`, exit code `-11`) on 1,024-token batched 4D attention contractions on Intel AMX.
