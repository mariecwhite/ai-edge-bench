# Hardware Ceiling Software Probes (`kernel/probes/`)

Software probes empirically measure the practical hardware ceilings used as denominators in the [Kernel](../README.md), [Layer](../../layer/README.md), and [Model](../../model/README.md) roofline formulas.

## Available Probes

- [cpu_ceilings.c](cpu_ceilings.c) — Repeatable x86_64 / Intel Sapphire Rapids hardware ceiling probe measuring both single-core (`1T`, physical core `0`) and multi-core (`28T`, NUMA Node 0 physical cores `0..27`) compute and memory-hierarchy ceilings:
  - **Compute Ceilings ($C_{\text{prac}}(\text{op}, \text{dtype})$)**:
    - `amx_int8_tdpbssd` (`tdpbssd` `16×16×64`, `32,768` INT8 OPs/inst)
    - `amx_bf16_tdpbf16ps` (`tdpbf16ps` `16×16×32`, `16,384` BF16 FLOPs/inst)
    - `avx512_vnni_int8` (`vpdpbusd %zmm`, `128` INT8 OPs/inst)
    - `avx_vnni_256_int8` (`vpdpbusd %ymm`, `64` INT8 OPs/inst)
    - `avx512_fp32_fma` (`vfmadd231ps %zmm`, `32` FP32 FLOPs/inst)
  - **Memory Hierarchy Bandwidth Ceilings ($BW_{\text{prac}}(\text{WorkingSet})$)**:
    - **L1D Cache** (`16 KB`/core): Streaming `vmovaps` Read (`6,691.22 GB/s` 28C) and Read+Write Copy (`5,560.20 GB/s` 28C)
    - **L2 Cache** (`512 KB`/core): Streaming `vmovaps` Read (`4,473.83 GB/s` 28C) and Read+Write Copy (`2,216.39 GB/s` 28C)
    - **Shared L3 LLC** (`32 MB` shared pool, `8 MB`/thread window): Streaming Read (`780.35 GB/s` 28C) and Read+Write Copy (`535.86 GB/s` 28C)
    - **DDR5 DRAM** (`896 MB` across 28 physical cores on Socket 0): Streaming Read (`246.60 GB/s` 28C), Read+Write Copy (`134.61 GB/s` 28C), and STREAM Triad (`151.58 GB/s` 28C)

## Repeatability Design

Every probe in [cpu_ceilings.c](cpu_ceilings.c) enforces strict measurement repeatability:
1. **Physical Core Pinning (`sched_setaffinity`)**: Threads `0..N-1` are pinned 1-to-1 to physical cores `0..N-1` on Socket 0, preventing OS thread migration, SMT sibling contention, and cross-socket UPI traffic.
2. **Monotonic Hardware Timing (`CLOCK_MONOTONIC_RAW`)**: Immune to NTP slew adjustments.
3. **Barrier-Synchronized Windows (`#pragma omp barrier`)**: All threads synchronize immediately before and after the timed loop, eliminating OpenMP thread-pool fork/join overhead from timed windows.
4. **Explicit Warmup + Multi-Trial Statistics**: Runs an untimed warmup pass to transition cores to sustained AVX-512/AMX turbo frequencies and warm TLBs/caches, followed by `N` independent timed trials (`--trials 7` by default), emitting `median`, `mean`, `min`, `max`, `stdev`, and Coefficient of Variation (`cv_pct`).

## Usage

```bash
gcc -O3 -march=sapphirerapids -fopenmp kernel/probes/cpu_ceilings.c -o /tmp/cpu_ceilings -lm
/tmp/cpu_ceilings --threads 28 --trials 7 \
    --output kernel/reports/gemma4-31b-sapphire-rapids/ceilings_sapphire_rapids.json
```

## Reference Output

- [ceilings_sapphire_rapids.json](../reports/gemma4-31b-sapphire-rapids/ceilings_sapphire_rapids.json) — Full 7-trial empirical ceiling profile for Intel Sapphire Rapids.
- [roofline_gemma4_31b_sapphire_rapids.md](../reports/gemma4-31b-sapphire-rapids/roofline_gemma4_31b_sapphire_rapids.md) — Roofline comparison against load-bearing XNNPACK and YNNPACK microkernels.
