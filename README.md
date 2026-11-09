# AI Edge Bench

AI Edge Bench is an open-source-first, **three-tier benchmarking architecture (Kernel → Layer → Model)** for measuring, comparing, and agentically optimizing on-device AI inference across CPUs, GPUs, and NPUs.

We take a bottoms-up, roofline-grounded approach across every operational tier so that:
- **Empirical roofline targets** are established from hardware probes up to full model execution.
- **Modularized components** at the kernel, layer, and model levels serve as fast, low-noise reward oracles for agentic optimization loops and hill climbing.
- **Multi-dimensional trade-offs** are explicitly quantified across **Throughput / Speed**, **Accuracy**, **Memory**, **Power**, **Code Size**, and **Numerical Consistency**.

---

## Three-Tier Benchmarking Architecture

| Tier & Directory | Operational Scope | Roofline & Performance Metrics | Quality & Numerical Precision Gate |
| :--- | :--- | :--- | :--- |
| **Level 1: Kernel**<br>[`kernel/`](kernel/README.md) | Single thread / SM / compute-unit microkernels & GPU shaders (GEMM, dot, packing, reductions across ARM SME2/SVE2/NEON, x86 AVX-512/AMX, Metal, Vulkan, WebGPU, CUDA). | **Practical Roofline Utilization (%)** via software probes ([`kernel/probes/`](kernel/probes/README.md)) for compute ceilings ($C_{\text{prac}}$) and cache/DRAM bandwidth ceilings ($BW_{\text{prac}}$). | Bit-exact or relative `FP32` reference tolerance (`rtol`/`atol`); SQNR & ULP drift across `FP16`/`BF16`/`INT8`/`INT4`/`INT2`; accumulator overflow checks. |
| **Level 2: Layer**<br>[`layer/`](layer/README.md) | Logically distinct compound subgraphs (attention, normalization, MLP, MoE, full blocks) built with LiteRT Tensor API or competitor libraries; introduces multi-thread & memory-hierarchy concurrency. | **Whole-Subgraph Practical Roofline Utilization (%)** calculated from total subgraph ops and boundary-only DRAM bytes—penalizing intermediate memory round-trips, layout conversions, and launch overhead. | Subgraph numerical error propagation (cosine similarity, relative $L_2$ error, KL-divergence vs. `FP32` reference); per-layer quantization sensitivity. |
| **Level 3: Model**<br>[`model/`](model/README.md) | End-to-end model inference pipelines across prefill, decode (with prefix caching), context-length sweeps, and stress tests. | **Dual-Regime Roofline Ceilings** (**Prefill Ceiling** & **Decode Ceiling**); TTFT, prefill & decode throughput, inter-token latency (p50/p99), peak RSS / GPU footprint, context scaling. | Standardized accuracy benchmarks with 95% confidence intervals; instruction following & structured schema adherence; within/across-process & thread-count reproducibility. |

---

## Shared Infrastructure & Reproducibility

To ensure external reproducibility, every software, data, and hardware variable is locked through **layered containerization**:

1. **Base Image ([`docker/base/`](docker/base/Dockerfile) & [`docker/bin/`](docker/bin/)):** Shared across `kernel/`, `layer/`, and `model/`. Locks OS packages, system utilities, the `/models` and `/results` volume contract, and hardware/OS telemetry capture (`aeb-sysinfo`).
2. **Inference Framework Images (e.g., [`model/docker/`](model/docker/)):** Per-framework images built from pinned upstream revisions, recording exact commit SHAs and build flags in `/opt/aeb/build-info.json`.
3. **Benchmark Suite & Machine Overlays (e.g., [`model/suites/`](model/suites/) & [`model/compose/machines/`](model/compose/machines/)):** Lock checksum-verified model artifacts, evaluation datasets, CPU core pinning / NUMA affinity, clock settings, and runtime configurations.

### Workloads & Hardware Selection

Benchmarking is strictly purpose-driven:
- **Workloads:** Kernels and layers are selected based on their frequency of occurrence and execution-time weight in target models. Models are analyzed to decompose logical subgraphs into [`layer/`](layer/README.md) benchmarks and underlying compute/memory primitives into [`kernel/`](kernel/README.md) benchmarks.
- **Reports & Dashboards:** Benchmark results, historical series, and interactive dashboards are stored directly in the repository (see [model/dashboards/index.html](model/dashboards/index.html), [model/reports/](model/reports), and [model/reports/history/](model/reports/history)).

---

## Quick Start

```bash
# Build the shared base container image (used across tiers)
make base

# Explore tier-specific targets and documentation
make help
make -C kernel help
make -C layer help
make -C model help

# Run Level 3 (Model) containerized benchmark workflow
make MACHINE=<machine> images
make MACHINE=<machine> models
make MACHINE=<machine> SUITE=suites/<suite>.json suite-perf
make MACHINE=<machine> SUITE=suites/<suite>.json suite-accuracy
make MACHINE=<machine> SUITE=suites/<suite>.json report
```

See:
- **Level 1 (Kernel):** [kernel/README.md](kernel/README.md)
- **Level 2 (Layer):** [layer/README.md](layer/README.md)
- **Level 3 (Model):** [model/README.md](model/README.md), [model/docs/benchmark_methodology.md](model/docs/benchmark_methodology.md), [model/docs/containers.md](model/docs/containers.md), and [model/docs/harness.md](model/docs/harness.md)

---

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a framework adapter, kernel/layer/model benchmark, or result. This project is licensed under the [Apache License 2.0](LICENSE).
