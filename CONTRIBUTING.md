# Contributing to AI Edge Bench

Contributions and suggestions are welcome. The harness has unit tests that need
no models or accelerators (`make test`); there are no CI checks yet.
Open an [issue](https://github.com/mariecwhite/ai-edge-bench/issues) to discuss
new frameworks, metrics, or changes to the comparison methodology before
implementing them.

## Proposing an adapter, benchmark, or optimization

- Place benchmarks and adapters in the appropriate tier directory:
  - [`kernel/`](kernel/README.md) for single-thread / compute-unit microkernels, GPU shaders, and hardware ceiling probes.
  - [`layer/`](layer/README.md) for compound subgraphs (LiteRT Tensor API or competitor libraries) and whole-subgraph roofline evaluations.
  - [`model/`](model/README.md) for end-to-end model pipeline benchmarks, accuracy gates, and reproducibility suites.
- Always use the official model weights and the official model implementation
  available in each inference framework or open-source library (e.g. Hugging
  Face libraries, Google Tunix/Flax, native framework runtimes). Do not write
  custom model implementations inside model-tier harness drivers.
- Describe the framework and version, supported platform and backend, model
  format, and how the model will be obtained. Do not commit model weights or
  downloaded datasets.
- Explain how prompt formatting, token counts, generation settings, and
  measurement boundaries map to the shared
  [model methodology](model/docs/benchmark_methodology.md) or tier-specific
  roofline formulas ([`kernel/`](kernel/README.md), [`layer/`](layer/README.md)). Document unsupported settings
  instead of presenting non-equivalent runs as directly comparable.
- Include instructions for setup and execution, plus a small, automated test
  that does not require model downloads or accelerator hardware when code is
  introduced (for model-tier harness changes, add it to `model/docker/harness/tests/`).
- Keep raw run directories (`results/`) out of version control. A published
  model-tier comparison is a generated report under `model/reports/` (README, charts and
  `data.json` with per-request measurements) plus its line in
  `model/reports/history/`; add caveats for the machine in `model/reports/notes/`.

## Graduating agentic optimizations

When graduating agent-generated or hand-tuned optimizations into production:

- **Prioritize across-the-board Pareto wins:** Favor fundamental approaches (e.g., FlashAttention, kernel fusion, weight packing) that improve performance across dimensions without numerical precision loss.
- **Verify generalization:** Prefer optimizations that generalize across CPU/GPU families, documenting any hardware-specific specialization.
- **Document macro-level trade-offs:** If an optimization improves one dimension (e.g., throughput) while regressing another (e.g., memory or code size), explicitly document why the trade-off is acceptable.
- **Prefer architectural clarity over brittle micro-patches:** Avoid overfitting to single shapes or masking broader design flaws; favor changes that reduce code complexity and binary size.

Keep pull requests focused, update the relevant documentation, and explain
whether your change affects existing measurements or their interpretation.

This file describes this repository's current contribution process. It does
not adopt LiteRT-LM's contribution restrictions or imply ownership by the
google-ai-edge organization.
