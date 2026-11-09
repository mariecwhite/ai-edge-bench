# Kernel Runner & Analysis Scripts (`kernel/scripts/`)

This directory holds host-side orchestration scripts for the kernel tier:

- **Probe & Kernel Runner:** Executes isolated single-thread / single-compute-unit microbenchmarks with locked clocks/affinity and records median, p90, p99 latency, hardware counters, and numerical drift (`rtol`/`atol`, SQNR, ULP).
- **Roofline Utilization Calculator:** Joins kernel timing results with empirical ceiling profiles from `kernel/probes/` to compute Practical Roofline Utilization (%).
- **Pareto & Cost-Model Exporter:** Identifies Pareto-optimal kernel variants across `(M, N, K)` regions and exports cost-model tables for runtime dispatchers.
