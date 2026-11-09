# Level 3: Model-Tier Benchmarks (`model/`)

The `model/` tier benchmarks end-to-end model inference pipelines to:
1. Understand the macro-level impact of optimizations made at the [kernel](../kernel/README.md) and [layer](../layer/README.md) tiers.
2. Guide global, whole-model optimization decisions across runtimes and hardware backends.

---

## Reports, History & Dashboards

- **Interactive Dashboard:** [dashboards/index.html](dashboards/index.html)
- **Generated Reports:** [reports/](reports)
- **Performance History:** [reports/history/](reports/history)
- **Benchmark Suites:** [suites/](suites)

---

## Methodology & Roofline Measurement

All model-level benchmarks are grounded in real user scenarios (see [docs/benchmark_methodology.md](docs/benchmark_methodology.md)):

- **Prefill and Decode Regimes:** Evaluated separately across cold stateless KV requests (`perf`) and warm cached-prefix KV requests (`prefix-cache`).
- **Context Length Performance Decay (`sequence`):** Sweeps prompt lengths across the advertised context window (`64` to `128K+` tokens) to measure attention and state/KV-cache scaling.
- **Accuracy & Determinism Gate (`accuracy`, `repro`):**
  - *Knowledge & Reasoning:* Stratified MMLU (reporting Wilson 95% CIs) and GSM8K (reporting normal-approximation 95% CIs).
  - *Instruction Following & Structured Output:* Verifiable instruction-following constraints (IFEval) and JSON/tool-call schema adherence.
  - *Reproducibility:* Within-process, across-process, and across-thread-count determinism checks under greedy (`temp=0`) and seeded sampling.
- **Stress Testing:** Pushes each runtime to advertised context and concurrency limits, recording explicit failure boundaries (e.g., OOM or accelerator command-buffer aborts).

### Dual-Regime Roofline Formulas

Because autoregressive inference shifts from being **memory-bandwidth-bound** during single-token decode to **compute-bound** during batch prefill, model-level roofline utilization is audited separately across the two regimes using the practical hardware ceilings ($\pi_{\text{peak}}$ and $B_{\text{DRAM/SLC}}$) established by [Level 1 kernel probes](../kernel/probes/README.md):

$$\text{Decode Ceiling (tokens/s)} = \frac{B_{\text{DRAM}}}{W_{\text{active\_bytes}} + S_{\text{KV/state\_bytes}}}$$

$$\text{Prefill Ceiling (tokens/s)} = \frac{T \times \pi_{\text{peak}}}{F_{\text{model}}(T)}$$

Where:
- $B_{\text{DRAM}}$: Practical sustained main-memory bandwidth in bytes/s.
- $W_{\text{active\_bytes}}$: Active weight bytes streamed per generated token. For dense models, this is the quantized footprint of all layers; for Mixture-of-Experts (MoE) models, it counts only the routed active parameters per token.
- $S_{\text{KV/state\_bytes}}$: Total KV-cache and recurrent/conv state bytes read and updated across all layers at context depth $S_{\text{kv}}$. As $S_{\text{kv}}$ scales from `64` to `128K+`, $S_{\text{KV/state\_bytes}}$ grows to dominate the denominator, capturing theoretical roofline decay across long contexts.
- $T$: Prompt token length.
- $\pi_{\text{peak}}$: Practical peak hardware compute (ops/s) measured via kernel-level probes for the target instruction set and data type.
- $F_{\text{model}}(T)$: Total algorithmic model prefill FLOPs required to process $T$ prompt tokens ($\approx 2 \cdot N_{\text{active\_params}} \cdot T + F_{\text{attention/SSM}}(T)$).

### Optimization Search Space

- **Accuracy-Preserving Optimizations:**
  - Whole-graph memory and pass lifecycle management (e.g., sharing packed weight caches across prefill and decode).
  - Cross-layer command-buffer coalescing and persistent zero-copy KV residency.
  - Core-affinity and NUMA-local thread scheduling.
  - Heterogeneous compute partitioning (e.g., MoE expert placement across GPU, CPU, and NPU).
- **Non-Accuracy-Preserving Optimizations** (must clear the accuracy gate):
  - Non-uniform, sensitivity-aware cross-layer quantization.
  - KV-cache quantization.

---

## Quick Start

All commands below can be run directly from `model/` (or from the repository root via `make <target>` / `make -C model <target>`).

### 1. Containerized Benchmark Suites

Composable Docker images ([docs/containers.md](docs/containers.md)) and per-machine overlays in [compose/machines/](compose/machines) provide reproducible execution environments:

```bash
cd model
make base                          # build shared aeb/base image (from ../docker/base)
make MACHINE=<machine> images      # build inference framework images
make MACHINE=<machine> models      # download model weights into ./models
make MACHINE=<machine> bench       # run native-tool baseline benchmarks
```

### 2. Cross-Framework Harness Evaluation

The framework-neutral harness ([docs/harness.md](docs/harness.md)) drives every engine in-process with identical tokenized prompts, measurement boundaries, resource monitoring, and accuracy gates:

```bash
cd model
make tools                                                 # offline tools image
make datasets                                              # stage pinned MMLU + GSM8K into ./datasets
make MACHINE=<machine> SUITE=suites/<suite>.json suite-perf     # prime caches + interleaved timing rounds
make MACHINE=<machine> SUITE=suites/<suite>.json suite-accuracy # MMLU / GSM8K accuracy gate
make MACHINE=<machine> SUITE=suites/<suite>.json suite-repro    # greedy + seeded output reproducibility
make MACHINE=<machine> SUITE=suites/<suite>.json report         # generate report, charts, history, and dashboard
```

### 3. Native Host GPU Execution (macOS Metal)

Suites configured with `"execution": "native"` run directly on the host when containers cannot access the target accelerator (e.g., Metal on macOS) through the same harness, accuracy gate, and reporting pipeline:

```bash
cd model
make native                                                # build pinned host stack in build/native/
make gpu-models                                            # stage pinned, checksum-verified artifacts
make MACHINE=<machine> SUITE=suites/<suite>.json suite-perf
make MACHINE=<machine> SUITE=suites/<suite>.json native-report
```

See [docs/harness.md](docs/harness.md) and [docs/benchmark_methodology.md](docs/benchmark_methodology.md) for full details.
