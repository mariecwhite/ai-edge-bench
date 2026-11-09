# Level 2: Layer-Tier Benchmarks (`layer/`)

The `layer/` tier benchmarks logically distinct **compound subgraphs** (e.g., attention blocks, decoder/encoder blocks, normalization + residual fusions, and feed-forward / Mixture-of-Experts subgraphs), introducing **multi-thread, multi-stream, and memory-hierarchy concurrency** between the [Kernel](../kernel/README.md) and [Model](../model/README.md) tiers.

---

## Subgraph Construction

Logical subgraphs are extracted from target model architectures and constructed via:

1. **LiteRT Tensor API:** Subgraphs are authored in the LiteRT Tensor API and lowered directly to target backends (XNNPACK, YNNPACK, ML Drift, WebGPU/Metal/Vulkan/CUDA).
2. **Competitor Library Subgraphs:** Equivalent subgraphs built in third-party runtimes for apples-to-apples subgraph comparison.

---

## Roofline Measurement

Rather than evaluating each operator in isolation, the Layer tier measures **Practical Roofline Utilization (%) across the entire subgraph as a single unit**.

We calculate the subgraph's theoretical minimum latency ($T_{\text{min}}$) assuming all intermediate activations remain fused in registers or on-chip cache throughout the computation, and only the unavoidable boundary tensors cross main memory:

$$T_{\text{min}}(\text{subgraph}) = \max\left(\frac{F_{\text{subgraph}}}{C_{\text{prac}}},\; \frac{B_{\text{min\_subgraph}}}{BW_{\text{prac}}}\right)$$

$$\text{Whole-Subgraph Roofline Utilization (\%)} = \frac{T_{\text{min}}(\text{subgraph})}{T_{\text{measured}}(\text{subgraph})} \times 100\%$$

Where:
- $F_{\text{subgraph}} = \sum_{op \in \text{subgraph}} F_{op}$: Total algorithmic math operations across all steps in the subgraph.
- $B_{\text{min\_subgraph}} = B_{\text{inputs}} + B_{\text{weights}} + B_{\text{state\_read/update}} + B_{\text{outputs}}$: Minimum bytes that must cross main memory — counting **only** the subgraph's initial inputs, model weights, state/KV-cache updates, and final outputs.
- $C_{\text{prac}}$ and $BW_{\text{prac}}$: Practical hardware compute and bandwidth ceilings established by Level 1 [kernel probes](../kernel/probes/README.md).

### Why Whole-Subgraph Roofline Matters
Because intermediate activation buffers are excluded from $B_{\text{min\_subgraph}}$, this formula directly penalizes:
- Unnecessary intermediate DRAM/cache round-trips between unfused operators.
- Redundant data layout conversions, packing, and dequantization steps.
- Multi-thread barrier synchronization and GPU kernel launch / dispatch overhead.

---

## Optimization Search Space

To keep the search space tractable for agentic loops, individual microkernels and shaders are initially kept frozen at this tier (leaving joint kernel-layer co-optimization as a subsequent stage).

Optimizations at the Layer tier are classified into two categories:

### 1. Accuracy-Preserving Optimizations
- **Tile & Block Fusion:** Outer, middle, and inner-level block/tile fusion, operator ordering, and multi-worker thread scheduling (e.g., attention tiling, fused normalization + projection, fused gate/up MLP + activation).
- **Layout Swizzling & Packing:** Weight and activation packing/swizzling across tile sizes and cache hierarchy levels.

### 2. Non-Accuracy-Preserving Optimizations
- **Mixed-Precision & Subgraph Quantization:** Per-operator or per-tensor bit-width assignment (`INT2`, `INT4`, `INT8`, `FP8`, `FP16`, `BF16`) and activation quantization schemes.
- **Mandatory Quality Gate:** Every non-accuracy-preserving optimization must be qualified with a subgraph numerical fidelity evaluation against an `FP32` reference subgraph:
  - Output cosine similarity and relative $L_2$ error.
  - Output distribution KL-divergence.
  - Per-layer quantization sensitivity profiles to guide model-tier converter recipes.

---

## Directory Layout

| Directory | Purpose |
| :--- | :--- |
| [subgraphs/](subgraphs/README.md) | LiteRT Tensor API and competitor subgraph implementations (attention, normalization, MLP, MoE, and full blocks). |
| [suites/](suites/README.md) | Layer benchmark suite definitions (shapes, sequence/context depths, thread counts, and precision configurations). |
| [scripts/](scripts/README.md) | Layer benchmark runners, whole-subgraph roofline calculators, and numerical error propagation evaluators. |
