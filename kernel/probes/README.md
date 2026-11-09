# Hardware Ceiling Software Probes (`kernel/probes/`)

Software probes empirically measure the practical hardware ceilings used as denominators in the [Kernel](../README.md), [Layer](../../layer/README.md), and [Model](../../model/README.md) roofline formulas.

## Probe Categories

### 1. Compute Ceiling Probes (`C_prac(op, dtype)`)
Tight register-resident instruction loops that measure maximum sustained arithmetic throughput without memory stalls across target ISAs and accelerators:
- **CPU ISAs:**
  - `arm64`: NEON dotprod (`sdot`/`udot`), i8mm (`smmla`), FP16/BF16 FMLA, SVE2, SME2 outer-product tiles (`fmopa`/`smopa`).
  - `x86_64`: AVX2 FMA, AVX-512 (FMA, VNNI `vpdpbusd`, BF16 `vdpbf16ps`, FP16), Intel AMX (`tdpbf16ps`, `tdpbssd`).
- **GPU Shaders:**
  - Apple Metal (`simdgroup_multiply_accumulate`), Vulkan (`VK_KHR_cooperative_matrix`), CUDA Tensor Cores (`mma.sync` / `wgmma`).

### 2. Memory Hierarchy Ceiling Probes (`BW_prac(WorkingSet)`)
Streaming read, write, and read-modify-write triad probes swept across working set sizes from `4 KB` to `4 GB+` to map sustained bandwidth plateaus across:
- **L1 Data Cache / Shared Memory / Threadgroup Memory**
- **L2 Cache**
- **L3 / System Level Cache (SLC)**
- **Main Memory (DRAM / Unified LPDDR5X / HBM)**

## Output Contract

Each probe run emits a machine-keyed JSON profile (`results/<machine>/probes/ceilings.json`) containing:
- `compute_ceilings`: map of `(op, dtype, isa)` to sustained GFLOP/s or GOPS/s and MACs/cycle.
- `bandwidth_ceilings`: piecewise bandwidth curve `BW_prac(working_set_bytes)` in GB/s with identified L1/L2/L3/SLC/DRAM regimes.
