# Level 1: Kernel-Tier Benchmarks (`kernel/`)

The `kernel/` tier benchmarks individual CPU microkernels and GPU shaders executing on a **single thread, Streaming Multiprocessor (SM), or compute unit**. It forms the foundation of AI Edge Bench's three-tier (**Kernel → Layer → Model**) benchmarking and closed-loop agentic optimization architecture.

---

## Objectives

1. **Establish Empirical Hardware Ceilings:** Measure practical compute and memory-hierarchy bandwidth limits via lightweight software probes ([probes/](probes/)).
2. **Report Practical Roofline Utilization (%):** Evaluate every microkernel and GPU shader against its empirical roofline ceiling rather than raw latency alone.
3. **Provide a Fast, Low-Noise Reward Oracle for Agents:** Enable sub-second, deterministic evaluation of candidate kernels/shaders without whole-model build or initialization overhead.
4. **Map Performance Cliffs & Cost Models:** Systematically sweep shape tuples `(M, N, K)` (including non-powers of 2 and unaligned tails) to identify Pareto-optimal shapes for model co-design and calibrate runtime kernel selection cost models.

---

## Roofline Measurement

For every target CPU and GPU architecture, `kernel/probes/` establishes two sets of empirical ceilings through software probes:

1. **Compute Ceilings ($C_{\text{prac}}(\text{op}, \text{dtype})$):** The maximum sustained arithmetic throughput (FLOP/s or OPS/s, and MACs/cycle) for the kernel's instruction family (e.g., ARM SME2/SVE2/NEON, x86 AMX/AVX-512 VNNI/BF16, Apple Metal simdgroup matrix, Vulkan cooperative matrix, CUDA Tensor Core) and data type (`FP32`, `BF16`, `FP16`, `INT8`, `INT4`, `INT2`).
2. **Memory Hierarchy Ceilings ($BW_{\text{prac}}(\text{WorkingSet})$):** The maximum sustained read/write bandwidth (GB/s) across L1D, L2, L3 / System Level Cache (SLC), and main memory (DRAM) as a function of working set size.

### Practical Roofline Utilization Formula

For a matrix/dot kernel parameterized by shape $(M, N, K)$ (and generalized to shape tuple $s$ for elementwise, reduction, dequantization, or packing kernels), **Practical Roofline Utilization (%)** is defined as:

$$\text{Practical Roofline Utilization }(M, N, K) = \frac{\text{Achieved Throughput }(M, N, K)}{\min\left(C_{\text{prac}}(\text{op}, \text{dtype}),\; I(M, N, K) \times BW_{\text{prac}}(\text{WorkingSet}(M, N, K))\right)} \times 100\%$$

Where:
- $I(M, N, K) = \frac{\text{Algorithmic Ops }(M, N, K)}{\text{Minimum Memory Bytes }(M, N, K)}$: **Operational Intensity** — algorithmic operations per byte transferred from the active memory hierarchy level.
- $C_{\text{prac}}(\text{op}, \text{dtype})$: **Practical Compute Ceiling** — maximum sustained throughput for the kernel's instruction and data type.
- $BW_{\text{prac}}(\text{WorkingSet})$: **Practical Bandwidth Ceiling** when streaming a working set of size $\text{WorkingSet}(M, N, K)$ from the corresponding cache level or main memory.

The ridge point $I_{\text{ridge}} = \frac{C_{\text{prac}}(\text{op}, \text{dtype})}{BW_{\text{prac}}(\text{WorkingSet})}$ separates the memory-bound regime ($I < I_{\text{ridge}}$, typical of GEMV / single-token decode) from the compute-bound regime ($I > I_{\text{ridge}}$, typical of batched GEMM / prefill).

---

## Numerical Precision & Quality Gate

No kernel optimization can graduate if it introduces unverified numerical drift:

- **Reference Tolerance:** Bit-exact match or bounded relative/absolute error (`rtol` / `atol`) against a golden `FP32` / `FP64` reference implementation.
- **Quantization Fidelity:** Signal-to-Quantization-Noise Ratio (SQNR) and ULP drift tracking across `FP16`, `BF16`, `INT8`, `INT4`, and `INT2` block/channel formats.
- **Accumulator Integrity:** Explicit edge-case inputs testing subnormal handling, NaN/Inf propagation, and integer/FP16 accumulator saturation.

---

## Directory Layout

| Directory | Purpose |
| :--- | :--- |
| [probes/](probes/README.md) | Hardware ceiling software probes for compute ($C_{\text{prac}}$) and cache/DRAM bandwidth ($BW_{\text{prac}}$). |
| [suites/](suites/README.md) | Kernel benchmark suite specifications (operator families, `(M, N, K)` shape grids, data types, and numeric tolerances). |
| [scripts/](scripts/README.md) | Runners, roofline utilization calculators, and cost-model fitting scripts. |
