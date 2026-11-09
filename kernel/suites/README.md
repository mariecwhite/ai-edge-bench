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
