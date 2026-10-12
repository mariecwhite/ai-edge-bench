## Key architecture & methodology notes (x86 Sapphire Rapids — FP32, BF16, INT8, and INT4/INT2)

1. **Precision Matrix (`FP32`, `BF16`, `INT8`, plus `INT4`/`INT2` where pre-built artifacts exist).**
   - Every framework is evaluated across `FP32`, `BF16`, `INT8`, and `INT4`/`INT2`
     **only where pre-existing model artifacts and runtime support are available**:
     - **OpenXLA (`jax==0.11.2` / `jaxlib==0.11.2`, official Google `tunix.models.gemma4` + `qwix` on YNNPACK)**:
       - `xla/fp32-ynnpack` (`FP32`), `xla/bf16-ynnpack` (`BF16`), `xla/int8-ynnpack`
         (`qwix` `int8` RTN weight-only PTQ on `google/gemma-4-E2B-it`), and
         `xla/matched-ynnpack` (`qwix` `int4` RTN weight-only PTQ on `google/gemma-4-E2B-it`).
     - **OpenXLA with Intel oneDNN** (`xla/*-onednn`, currently disabled):
       - `xla/fp32-onednn`, `xla/bf16-onednn`, `xla/int8-onednn`, and `xla/matched-onednn`
         are marked `"disabled": true` in the suite because XLA's experimental oneDNN
         custom-call rewriter (`--xla_cpu_experimental_onednn_custom_call=true`) crashes
         during compilation on Tunix Gemma 4's multi-axis Einsum contractions (`dot_general`
         with rank > 3) and does not support `int4` weight tensors.
     - **LiteRT-LM (`v0.17.1`)**:
       - `litert-lm/matched` (XNNPACK) and `litert-lm/fastest` (**YNNPACK**, `--ynnpack`)
         executing `litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm`
         (mixed `INT2`/`INT4`/`INT8` mobile QAT weights). Omitted from `FP32`, `BF16`, and
         pure `INT8` because `.litertlm` only ships as the pre-compiled mobile QAT bundle.
     - **`llama.cpp` (`b11259`, Sapphire Rapids AVX-512 + AMX backend)**:
       - `llama.cpp/bf16` (`BF16`), `llama.cpp/int8` (`Q8_0`),
         `llama.cpp/stock-q4_0` (official `ggml-org` `Q4_0` INT4 QAT checkpoint),
         `llama.cpp/matched` (bit-identical mobile QAT mixed `INT2`/`INT4`/`INT8` in
         `Q2_K`/`Q4_0`/`Q8_0`), and `llama.cpp/fastest` (mobile QAT weights with `INT2`
         tensors stored in `Q4_0` blocks).
       - **Omitted from `FP32`:** `ggml-org/gemma-4-E2B-it-GGUF` publishes pre-built
         `BF16`, `Q8_0`, and `Q4_0` GGUFs, but **no `F32` GGUF**. Because the benchmark
         protocol only includes `llama.cpp` data types with pre-existing GGUF artifacts
         (and avoids synthesizing an un-published ~17.4 GB `F32` GGUF from `BF16` weights),
         `llama.cpp` is not included in the `FP32` row.
     - **OpenVINO GenAI (`2026.4.1.0`)**:
       - `openvino/fp32` (`INFERENCE_PRECISION_HINT=f32`), `openvino/bf16`
         (`INFERENCE_PRECISION_HINT=bf16` on Intel AMX), `openvino/int8`
         (`OpenVINO/gemma-4-E2B-it-int8-ov`), and `openvino/int4`
         (`OpenVINO/gemma-4-E2B-it-int4-ov`).
     - **ONNX Runtime GenAI (`0.17.1`)**:
       - `onnxruntime/q4_k_m` (`Q4_K_M` 4-bit block quantization from
         `onnx-community/gemma-4-E2B-it-ONNX`).

2. **Caveats: Missing Data-Type Cells & Differences Between INT8, INT4, and INT2 Variants.**
   - **Why certain (framework, data-type) pairs are omitted:**
     - If a pre-built GGUF or framework artifact is not already available for a given
       data type, that framework is excluded from that data type in the benchmark:
       - `llama.cpp` has no published `F32` GGUF in `ggml-org/gemma-4-E2B-it-GGUF` (the
         base checkpoint `google/gemma-4-E2B-it` is natively `bfloat16`, so `BF16.gguf`
         is already lossless).
       - `LiteRT-LM` only publishes the mobile QAT `.litertlm` artifact (`INT2`/`INT4`/`INT8`).
       - `ONNX Runtime GenAI` is evaluated on its staged `Q4_K_M` package.
   - **Weight quantization schemes are not identical across checkpoints:**
     - **Mobile QAT (`litert-lm/*`, `llama.cpp/matched`, `llama.cpp/fastest`)**:
       Share the **exact same 278 quantized text tensors** (proven bit-identical
       in `models/aeb/litert-parity.json`) from `google/gemma-4-E2B-it-qat-mobile-transformers`:
       per-channel symmetric **INT2** on token embeddings, `lm_head`, and the double-wide
       MLPs of layers 15–34; **INT4** on attention projections, MLPs of layers 0–14, and
       per-layer embeddings (PLE); and **INT8** on per-layer input gates/projections.
     - **Stock `ggml-org` `Q4_0` and `Q8_0` (`llama.cpp/stock-q4_0`, `llama.cpp/int8`)**:
       Use 32-element block quantization (1 `fp16` scale per 32 weights) exported from
       Google's separate `Q4_0` QAT and `BF16` checkpoints (`ggml-org/gemma-4-E2B-it-GGUF`).
       Unlike the mobile QAT checkpoint, layers 15–34 MLPs are 4-bit (`Q4_0`) or 8-bit
       (`Q8_0`), not 2-bit.
     - **OpenVINO `int4` and `int8` (`openvino/int4`, `openvino/int8`)**:
       Use OpenVINO NNCF group-wise asymmetric/symmetric INT4 and INT8 weight compression
       (`OpenVINO/gemma-4-E2B-it-{int4,int8}-ov`) derived from the BF16 base checkpoint.
     - **OpenXLA `int8` and `int4` (`xla/int8-ynnpack`, `xla/matched-ynnpack`)**:
       Use official `qwix` (`qwix.quantize_model` with `QwixQuantizationProvider`)
       round-to-nearest (`int8` or `int4`) weight-only quantization applied to the official
       Tunix `tunix.models.gemma4` Flax NNX model loaded from `google/gemma-4-E2B-it`.
     - **ONNX Runtime `q4_k_m` (`onnxruntime/q4_k_m`)**:
       Uses `MatMulNBits` 4-bit block quantization (`onnx-community/gemma-4-E2B-it-ONNX`).
   - **Activation & KV-cache quantization differences (impact on accuracy and speed):**
     - **Static INT8 Activation Rounding (SRQ):** The mobile QAT checkpoint was trained
       with static per-tensor INT8 activation rounding (SRQ) and fixed-scale INT8 KV cache.
       `LiteRT-LM` executes SRQ natively; `llama.cpp` does not apply SRQ.
       Because the mobile QAT weights were co-adapted to SRQ rounding, running those
       same weights without SRQ (`llama.cpp/matched`, `llama.cpp/fastest`) lowers MMLU/GSM8K
       accuracy compared to `LiteRT-LM` or non-SRQ checkpoints (`llama.cpp/stock-q4_0`,
       `openvino/int4`, `xla/matched-ynnpack`).
     - **`Q2_K` vs. `Q4_0` kernel coverage in `llama.cpp`:** `llama.cpp` lacks an
       AVX-512 VNNI / AMX repacked kernel path for `Q2_K` super-blocks. Storing the exact
       same mobile QAT INT2 integer weights in `Q4_0` blocks (`llama.cpp/fastest`) enables
       repacked AVX-512/AMX kernels at the cost of +0.4 GB of storage (`2.7 GB` vs `2.3 GB`).

3. **Hardware & NUMA pinning (`Sapphire Rapids`).**
   - Dual-socket **Intel Xeon 6985P-C (Sapphire Rapids)** server (`cpuid` family 6, model 173, 40 physical cores / 80 threads per socket, 160 logical CPUs total, ~772 GB DDR5 across 2 NUMA nodes).
   - ISA extensions present: `avx512f`, `avx512bw`, `avx512dq`, `avx512vl`, `avx512_vnni`, `avx512_bf16`, `avx512_fp16`, `amx_tile`, `amx_int8`, `amx_bf16`.
   - All single-stream performance and reproducibility runs are strictly pinned to **NUMA node 0 physical cores (`0–39`, memory node `0`)** via `numactl --physcpubind=0-39 --localalloc` inside `scripts/dock_shim.py`, avoiding cross-socket UPI traffic and SMT sibling contention.
   - During the 2-shard parallel accuracy runs (`--parallel 2 --shards 2`), shard 0 is pinned to Socket 0 physical cores and shard 1 is pinned to Socket 1 physical cores.

4. **OpenXLA CPU: Official Tunix + Qwix on YNNPACK (`__ynn_fusion`) vs. Intel oneDNN.**
   - `xla_driver.py` uses the official Google Tunix Gemma 4 implementation (`tunix.models.gemma4`) and official `qwix` quantization (`qwix.quantize_model`) on Flax NNX/JAX (`jaxlib==0.11.2`), lowering matrix multiplications and elementwise fusions into `__ynn_fusion` custom calls backed by **YNNPACK**.
   - Enabling XLA's experimental Intel oneDNN custom-call rewriter (`--xla_cpu_experimental_onednn_custom_call=true --xla_cpu_experimental_ynn_fusion_type=invalid`) currently fails during XLA compilation on Tunix Gemma 4's multi-axis Einsum contractions (`dot_general` with rank > 3) and does not support `int4` weight quantization; the `xla/*-onednn` configurations are therefore disabled until upstream XLA oneDNN custom-call support handles higher-rank contractions and sub-byte weights.

5. **Data-Type-Aware Phase Selection & Apples-to-Apples Accuracy Protocol.**
   - **Phase 3 (`perf`):** Evaluates all 15 active configurations (2 interleaved rounds × 3 measured requests after 1 warm-up at 1,024 prompt → 256 generated tokens).
   - **Phase 4 (`accuracy`):** With `accuracy_select_fastest_by_dtype: true`, the runner groups active configurations by `(framework, dtype)` pair and selects the empirically fastest configuration (lowest median `e2e_ms` in Phase 3) for each of the 12 `(framework, dtype)` cells. All 12 `(framework, dtype)` winners are evaluated on the **identical 2,000-item stratified MMLU subset (`mmlu_limit: 2000`)** so accuracy and paired McNemar agreement tests are strictly apples-to-apples across every framework and data type, plus a 250-item GSM8K subset (`gsm8k_limit: 250`) on the fast `int4` configurations.
   - **Phase 5 (`sequence`):** With `sequence_sweep.select_fastest_by_dtype: true` and `min_accuracy: {"mmlu": 0.40, "gsm8k": 0.40}`, the input-length sweep (`64, 256, 1024, 4096` prompt tokens → `128` generated tokens) runs on the fastest configuration per `(framework, dtype)` cell that passes the accuracy gate.
