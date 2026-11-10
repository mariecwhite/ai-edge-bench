# Layer Benchmark Suites (`layer/suites/`)

Suite configurations in this directory define the subgraph shapes, concurrency settings (thread counts, GPU streams/command buffers), and precision tracks evaluated at the Layer tier.

## Suite Structure

Each suite JSON specifies:
- `subgraph`: Target subgraph family (`gqa`, `mlp`, `moe_block`, `transformer_block`) and architectural hyperparameters (`hidden_size`, `intermediate_size`, `num_heads`, `num_kv_heads`, `head_dim`, `num_experts`, `top_k_experts`).
- `regimes`:
  - **Prefill (`T > 1`):** Batch token lengths (`64`, `256`, `1024`, `4096`) testing compute-bound tile fusion.
  - **Decode (`T = 1`):** Single-token step across KV context depths (`S_kv = 64..128K`) testing memory-bound weight + KV streaming and multi-thread scaling.
- `quality_gate`: Maximum allowable relative $L_2$ error, minimum cosine similarity, and maximum KL-divergence against the `FP32` reference subgraph.

## Gemma 4 31B (`google/gemma-4-31B-it`) Subgraph Dimensions & Input Regimes

Gemma 4 31B is a dense 60-layer hybrid-attention transformer (`hidden_size = 5376`, `intermediate_size = 21504`, `vocab_size = 262144`, `hidden_size_per_layer_input = 0`) composed of two distinct transformer block types repeated in a 5:1 pattern (`10` blocks of `5` sliding-window layers + `1` global layer):

1. **Local Sliding-Window Attention Block (`50` of `60` layers):**
   - `num_attention_heads = 32`, `num_key_value_heads = 16` (GQA 2:1), `head_dim = 256`, `sliding_window = 1024`, RoPE $\theta = 10{,}000$.
   - Effective KV horizon is bounded by $\min(S_{\text{kv}}, 1024)$ regardless of total sequence length.
2. **Global Full-Attention Block (`10` of `60` layers, indices `5, 11, 17, ..., 59`):**
   - `num_attention_heads = 32`, `num_global_key_value_heads = 4` (GQA 8:1), `global_head_dim = 512`, `attention_k_eq_v = True` (unified $K=V$ projection), Proportional RoPE (`partial_rotary_factor = 0.25`, $\theta = 1{,}000{,}000$).
   - Scales across the full KV cache capacity (`max_num_tokens` from `1,024` up to `32,768` in LiteRT-LM CPU/GPU flatbuffers).
3. **GeGLU Feed-Forward Network (`MLP`, all `60` layers):**
   - `gate_proj`: `5376 -> 21504`, `up_proj`: `5376 -> 21504`, `gelu_pytorch_tanh` gated activation, `down_proj`: `21504 -> 5376`.
4. **Multimodal Vision Encoder (`vision_encoder`, `27` ViT layers) & Vision Adapter (`vision_adapter`):**
   - `hidden_size = 1152`, `intermediate_size = 4304`, `num_attention_heads = 16`, `head_dim = 72`, `patch_size = 16`, `pooling_kernel_size = 3`, `adapter_output_dim = 5376` (`INT8 × INT8 -> INT32` `QC8W` weights + unmasked 2D bidirectional FP32 self-attention).
   - Evaluates across the 5 official Gemma 4 soft-token budgets: `70`, `140`, `280` (default), `560`, and `1,120` soft tokens (`630` to `10,080` `16×16` patches).
5. **Multimodal Audio Conformer Reference (`audio_encoder` + `audio_adapter`, `gemma-4-E2B-it` reference):**
   - The official `google/gemma-4-31B-it` checkpoint has `"audio_config": null` (only `Gemma 4 E2B` and `Gemma 4 E4B` include `model.audio_tower`). We include the Gemma 4 Conformer Audio Encoder + Adapter (`[1, 816, 128] -> [1, 204, 1536]`, `204` soft tokens) as the Gemma 4 audio subgraph reference.

### Input Shape Regimes Tested Across Subgraphs and End-to-End Model

| Regime | Shape ID | Modality | Prompt / Soft Tokens ($P$) | Decode ($D$) | Patches / Context (`max_num_tokens`) | Subgraph & Kernel Regime |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| **Minimum text boundary** | `p1_d1` | `text` | `1` | `1` | `1,024` | Pure single-token `decode` ($T=1$ GEMV); no `prefill_*` invocation. |
| **Decode-dominated** | `p16_d128` | `text` | `16` | `128` | `1,024` | 1x `prefill_128` + 128x `decode` ($T=1$ memory-bandwidth-bound). |
| **Interactive chat** | `p64_d64` | `text` | `64` | `64` | `1,024` | Balanced conversational turn within `prefill_128` and local window $W=1024$. |
| **Bucket boundary** | `p128_d128` | `text` | `128` | `128` | `1,024` | Saturates `prefill_128` ($T=128$) before switching to `prefill_1024`. |
| **RAG / multi-turn** | `p256_d128` | `text` | `256` | `128` | `1,024` | First `prefill_1024` ($T=1024$ AMX/AVX-512 GEMM) regime. |
| **Headline workload** | `p1024_d128` | `text` | `1,024` | `128` | `4,096` | Full `prefill_1024` tile utilization at the exact $W=1024$ sliding-window boundary. |
| **Long context** | `p4096_d128` | `text` | `4,096` | `128` | `8,192` | $4\times$ `prefill_1024` chunks; global attention layers scale to $S_{\text{kv}}=4096$ while local layers remain capped at $W=1024$. |
| **Maximum text boundary** | `p16384_d128` | `text` | `16,384` | `128` | `32,768` | $16\times$ `prefill_1024` chunks at the maximum 32K KV cache capacity (`cache_length=32771`). |
| **Minimum vision boundary** | `vis_70` | `vision` | `70` | `0` | `630` | `630` patches (`~400×400 px`) -> `70` soft tokens; 8-bit AMX `QC8W` GEMM (`33.0%`) outpaces FP32 attention (`21.4%`). |
| **Low-res UI vision** | `vis_140` | `vision` | `140` | `0` | `1,260` | `1,260` patches (`~560×576 px`) -> `140` soft tokens; transition regime where FP32 attention (`39.0%`) overtakes `QC8W` GEMM (`29.3%`). |
| **Default photo VQA** | `vis_280` | `vision` | `280` | `0` | `2,520` | `2,520` patches (`~800×800 px`) -> `280` soft tokens; default Gemma 4 vision budget (`60.3%` FP32 attention vs. `21.4%` `QC8W` GEMM). |
| **High-res chart VQA** | `vis_560` | `vision` | `560` | `0` | `5,040` | `5,040` patches (`~1120×1152 px`) -> `560` soft tokens (`74.9%` FP32 attention vs. `12.6%` `QC8W` GEMM). |
| **Maximum vision boundary** | `vis_1120` | `vision` | `1,120` | `0` | `10,080` | `10,080` patches (`~1600×1600 px`) -> `1,120` soft tokens; quadratic FP32 `BatchMatMul + Softmax` reaches `84.9%` of TFLite time (`88.9%` of kernel cycles). |
| **Audio Conformer ref** | `audio_204` | `audio` | `204` | `0` | `816` | `816` mel frames -> `204` soft tokens (`gemma-4-E2B-it` ref); 2-bit `QC2W`/`INT2` & 4-bit `QC4W`/`INT4` Conformer linear + depthwise conv. |
