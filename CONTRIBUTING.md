# Contributing to AI Edge Bench

Contributions and suggestions are welcome. This repository currently contains
documentation only; it has no benchmark runner, test command, or CI checks yet.
Open an [issue](https://github.com/mariecwhite/ai-edge-bench/issues) to discuss
new frameworks, metrics, or changes to the comparison methodology before
implementing them.

## Proposing an adapter or benchmark

- Describe the framework and version, supported platform and backend, model
  format, and how the model will be obtained. Do not commit model weights or
  downloaded datasets.
- Explain how prompt formatting, token counts, generation settings, and
  measurement boundaries map to the shared
  [methodology](docs/benchmark_methodology.md). Document unsupported settings
  instead of presenting non-equivalent runs as directly comparable.
- Include instructions for setup and execution, plus a small, automated test
  that does not require model downloads or accelerator hardware when code is
  introduced. Until tooling exists, describe how the change was checked in the
  pull request.
- Keep generated results out of version control by default. For a proposed
  published comparison, include the configuration, environment, per-run raw
  measurements, summary calculation, and any limitations alongside the result.

Keep pull requests focused, update the relevant documentation, and explain
whether your change affects existing measurements or their interpretation.

This file describes this repository's current contribution process. It does
not adopt LiteRT-LM's contribution restrictions or imply ownership by the
google-ai-edge organization.
