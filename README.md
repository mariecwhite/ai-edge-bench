# AI Edge Bench

AI Edge Bench is a starting point for reproducible, on-device LLM benchmarks
across inference frameworks.

**Status:** Reproducible benchmark containers for llama.cpp and LiteRT-LM
([container setup](docs/containers.md)), a framework-neutral harness that times
both frameworks the same way and checks their accuracy ([harness](docs/harness.md)),
and the first published comparison:

- [Gemma 4 E2B on CPU, Apple M5 Max (Linux VM), 2026-10-01](reports/2026-10-01-gemma4-e2b-cpu-apple-m5-max/README.md)
- [Three-framework ARM CPU comparison, 2026-10-02](reports/2026-10-02-gemma4-e2b-cpu-arm-runtimes-apple-m5-max/README.md):
  LiteRT-LM, llama.cpp and ONNX Runtime.
- Performance history: [reports/history/](reports/history)

The optional ONNX Runtime GenAI CPU adapter for the Apple ARM VM has a
separate [exploratory suite](suites/gemma4-e2b-cpu-arm-runtimes.json). Its
independently converted model files are **not** weight-matched to the first
report; see [model staging and validation](docs/harness.md#optional-arm-cpu-runtimes).

## Benchmark containers

Composable images and a per-machine compose overlay cover AMD EPYC 7443P
(Zen 3), AMD Strix Halo (Zen 5), NVIDIA DGX Spark (GB10/Cortex-X925) and
Apple M5 Max:

```bash
cp .env.example .env
make base                          # shared harness image
make MACHINE=epyc-7443p images     # build llama.cpp + LiteRT-LM
make MACHINE=epyc-7443p models     # download Gemma 4 E2B weights
make MACHINE=epyc-7443p bench      # llama.cpp, then LiteRT-LM ±YNNPACK
```

Run `make machines` for the available targets. Read
[docs/containers.md](docs/containers.md) for the reproducibility contract, the
accelerator caveats, and what the YNNPACK comparison does and does not measure.

## Cross-framework comparison

```bash
make tools                               # offline tools image
make MACHINE=apple-m5-max matched-models # GGUF with the .litertlm's exact weights
make datasets                            # pinned MMLU + GSM8K
make MACHINE=apple-m5-max suite-perf     # interleaved timing rounds
make MACHINE=apple-m5-max suite-accuracy # accuracy gate
make MACHINE=apple-m5-max report         # report, charts, history
```

The suite definition ([suites/gemma4-e2b-cpu.json](suites/gemma4-e2b-cpu.json))
holds three tracks: **matched** (bit-identical weights, same prompt tokens,
decoding, context and threads), **fastest** (each framework's best tuned,
accuracy-gated configuration) and **reference** (stock llama.cpp model and
defaults). See [docs/harness.md](docs/harness.md).

## Benchmark goals

- Compare the same model, quantization, input, generation settings, and hardware
  where each framework supports them.
- Report latency and throughput with consistent measurement boundaries.
- Publish enough configuration and environment detail to reproduce each result,
  and call out settings that cannot be made equivalent.
- Keep raw run data separate from summaries so comparisons can be checked.

See [benchmark methodology](docs/benchmark_methodology.md) for the
measurement, accuracy-gate, reporting and history-tracking rules.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a framework adapter,
benchmark, or result. This project is licensed under the
[Apache License 2.0](LICENSE).
