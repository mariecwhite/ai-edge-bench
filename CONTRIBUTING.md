# Contributing to AI Edge Bench

Contributions and suggestions are welcome. The harness has unit tests that need
no models or accelerators (`make test`); there are no CI checks yet.
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
  introduced (add it to `docker/harness/tests/`).
- Keep raw run directories (`results/`) out of version control. A published
  comparison is a generated report under `reports/` (README, charts and
  `data.json` with per-request measurements) plus its line in
  `reports/history/`; add caveats for the machine in `reports/notes/`.

Keep pull requests focused, update the relevant documentation, and explain
whether your change affects existing measurements or their interpretation.

This file describes this repository's current contribution process. It does
not adopt LiteRT-LM's contribution restrictions or imply ownership by the
google-ai-edge organization.
