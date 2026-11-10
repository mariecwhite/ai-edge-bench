# Kernel Benchmark Suites (`kernel/suites/`)

Suite definitions in this directory specify the operator families, shape grids, data layouts, and numerical precision gates evaluated at the kernel tier.

## Workload Selection Criteria

Kernel workloads are selected based on their frequency of occurrence and execution-time weight in target models, as well as the data types and target hardware backends:

1. **GEMM / GEMV / MatMul (`(M, N, K)`):**
   - Low-batch / memory-bound shapes (`M = 1..16`, large `N, K`) and batched / compute-bound shapes (`M = 64..4096`, large `N, K`).
   - Non-power-of-2 dimensions and prime remainders to detect tail-handling cliffs, register spills, and cache-set associativity conflicts.
   - Quantized weight-only and weight-activation schemes (`INT2`, `INT4`, `INT8`, `FP16`, `BF16`, `FP32`).
2. **Weight & Activation Packing / Swizzling:**
   - Panel packing, blockwise scale/zero-point interleaving, and tile transposition overhead.
3. **Reduction & Normalization Microkernels:**
   - Softmax, RMSNorm, LayerNorm, and vector dot products.
4. **Elementwise, Positional & Activation Microkernels:**
   - Gated activations, rotary positional embeddings, and dequantize-on-load kernels.

## Gemma 4 31B (`google/gemma-4-31B-it`) Kernel Shapes & User-Journey Mapping

In Gemma 4 31B (`60` transformer layers: `50` local sliding-window attention layers with $W = 1024$ and `10` full global attention layers with unified $K = V$, `hidden_size = 5376`, `intermediate_size = 21504`, `vocab_size = 262144`), the active kernel shapes `(M, N, K)` in LiteRT-LM (`XNNPACK` and `YNNPACK`) are determined by the sequence signature ($M = 1$ for `decode`, $M = 128$ for `prefill_128`, $M = 1024$ for `prefill_1024`) and the KV cache length (`max_num_tokens`):

### Linear Projection `(M, N, K)` GEMM / GEMV Shapes per Layer

| Projection | Layer Scope | `M` (`decode` / `prefill_128` / `prefill_1024`) | `N` (Output Channels) | `K` (Input Channels) | Weight Quantization (`dynamic_wi4_afp32`) |
| :--- | :--- | :--- | ---: | ---: | :--- |
| `q_proj` (Local) | 50 sliding layers | `1` / `128` / `1024` | `8,192` ($32 \times 256$) | `5,376` | `QC4W` (channelwise INT4) |
| `k_proj`, `v_proj` (Local) | 50 sliding layers | `1` / `128` / `1024` | `4,096` ($16 \times 256$) | `5,376` | `QC4W` (channelwise INT4) |
| `o_proj` (Local) | 50 sliding layers | `1` / `128` / `1024` | `5,376` | `8,192` | `QC4W` (channelwise INT4) |
| `q_proj` (Global) | 10 global layers | `1` / `128` / `1024` | `16,384` ($32 \times 512$) | `5,376` | `QC4W` (channelwise INT4) |
| `k_proj` ($K=V$ Global) | 10 global layers | `1` / `128` / `1024` | `2,048` ($4 \times 512$) | `5,376` | `QC4W` (channelwise INT4) |
| `o_proj` (Global) | 10 global layers | `1` / `128` / `1024` | `5,376` | `16,384` | `QC4W` (channelwise INT4) |
| `gate_proj`, `up_proj` | All 60 layers | `1` / `128` / `1024` | `21,504` | `5,376` | `QC4W` (channelwise INT4) |
| `down_proj` | All 60 layers | `1` / `128` / `1024` | `5,376` | `21,504` | `QC4W` (channelwise INT4) |
| `lm_head` (Logits) | Final output | `1` (decode only) | `262,144` | `5,376` | `QC8W` (channelwise INT8 tied embedding) |
| `vision_encoder` (`q/k/v/o_proj`) | 27 ViT layers | `630` / `1260` / `2520` / `5040` / `10080` | `1,152` ($16 \times 72$) | `1,152` | `QC8W` (`INT8 × INT8 -> INT32`) |
| `vision_encoder` (`gate/up_proj`) | 27 ViT layers | `630` / `1260` / `2520` / `5040` / `10080` | `4,304` | `1,152` | `QC8W` (`INT8 × INT8 -> INT32`) |
| `vision_encoder` (`down_proj`) | 27 ViT layers | `630` / `1260` / `2520` / `5040` / `10080` | `1,152` | `4,304` | `QC8W` (`INT8 × INT8 -> INT32`) |
| `vision_adapter` (`mm_projection`) | 1 adapter | `70` / `140` / `280` / `560` / `1120` | `5,376` | `1,152` | `QC8W` (`INT8 × INT8 -> INT32`) |
| `audio_encoder` (`conformer`) | 12 Conformer layers (`E2B`/`E4B` ref) | `204` (`T=816` subsampled $4\times$) | `1,024` / `4,096` | `1,024` | `QC2W` (2-bit INT2) & `QC4W` (4-bit INT4) |

### End-to-End Multimodal Input Shapes (Minimum, User Journeys, Maximum)

See [benchmark_methodology.md](../../model/docs/benchmark_methodology.md) and [extract_litert_lm_kernels.py](../scripts/extract_litert_lm_kernels.py) for the complete rationale and kernel time-proportion extraction across:
- **Text (`gemma-4-31B-it`, 60 layers)**:
  - **Minimum supported text shape (`p1_d1`):** `prompt_tokens = 1, decode_tokens = 1, max_num_tokens = 1024` (single-row $M=1$ GEMV only; `max_num_tokens < 1024` is rejected by `prefill_1024`'s `DYNAMIC_UPDATE_SLICE` shape check).
  - **Autocomplete (`p16_d128`):** `16` prompt, `128` decode (`max_num_tokens = 1024`).
  - **Interactive chat / voice turn (`p64_d64`):** `64` prompt, `64` decode (`max_num_tokens = 1024`).
  - **`prefill_128` bucket saturation / IDE completion (`p128_d128`):** `128` prompt, `128` decode (`max_num_tokens = 1024`).
  - **RAG Q&A / multi-turn chat (`p256_d128`):** `256` prompt, `128` decode (`max_num_tokens = 1024`).
  - **Document summarization / headline workload (`p1024_d128`):** `1024` prompt, `128` decode (`max_num_tokens = 4096`).
  - **Long-context document & repo analysis (`p4096_d128`):** `4096` prompt, `128` decode (`max_num_tokens = 8192`).
  - **Maximum supported KV cache capacity (`p16384_d128`):** `16384` prompt, `128` decode at `max_num_tokens = 32768` (`cache_length = 32771`; at `65536`, global attention logits `[1, 32, 1024, 65536]` reach $2^{31}$ elements and overflow signed 32-bit TFLite tensor indexing).
- **Vision (`gemma-4-31B-it`, 27-layer ViT + Adapter)**:
  - **`vis_70` (`70` soft tokens / `630` patches)**: Minimum supported vision token budget (thumbnail / icon classification).
  - **`vis_140` (`140` soft tokens / `1,260` patches)**: Low-resolution UI widget / mobile screenshot.
  - **`vis_280` (`280` soft tokens / `2,520` patches)**: Default Gemma 4 vision token budget (standard photo VQA).
  - **`vis_560` (`560` soft tokens / `5,040` patches)**: High-resolution chart / slide / document page VQA.
  - **`vis_1120` (`1,120` soft tokens / `10,080` patches)**: Maximum supported vision token budget (dense OCR / multi-column PDF / 4K diagram).
- **Audio (`audio_config: null` in 31B; `gemma-4-E2B-it` Conformer Reference)**:
  - **`audio_204` (`204` soft tokens / `816` mel frames)**: 8.2s voice command / speech clip (`[1, 816, 128] -> [1, 204, 1536]`).
