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
- [Gemma 4 26B A4B on the Apple M5 Max GPU (native macOS, Metal), 2026-10-22](reports/2026-10-22-gemma4-26b-a4b-gpu-apple-m5-max/README.md):
  LiteRT-LM (ML Drift GPU), llama.cpp, MLX and BaseRT.
  [HTML version](reports/2026-10-22-gemma4-26b-a4b-gpu-apple-m5-max/README.html).
- Performance history: [reports/history/](reports/history)

The optional ONNX Runtime GenAI CPU adapter for the Apple ARM VM has a
separate [exploratory suite](suites/gemma4-e2b-cpu-arm-runtimes.json). Its
independently converted model files are **not** weight-matched to the first
report; see [model staging and validation](docs/harness.md#optional-arm-cpu-runtimes).

## Native Apple GPU suite

[suites/gemma4-26b-a4b-gpu.json](suites/gemma4-26b-a4b-gpu.json) benchmarks
Gemma 4 26B A4B on the Apple M5 Max GPU with LiteRT-LM (ML Drift GPU
backend), llama.cpp (Metal), MLX and BaseRT. Docker on macOS cannot reach
Metal, so this suite runs natively on the macOS host through the same harness,
runner, accuracy gate and report:

```bash
make native                                   # pinned host stack in build/native/
make gpu-models                               # pinned, checksum-verified artifacts
make MACHINE=apple-m5-max SUITE=suites/gemma4-26b-a4b-gpu.json suite-perf
make MACHINE=apple-m5-max SUITE=suites/gemma4-26b-a4b-gpu.json native-report
```

See [harness.md](docs/harness.md#native-apple-gpu-suite) for the full
sequence (sweep, accuracy, reproducibility, native-tool cross-check), the
artifacts and what the GPU-side metrics mean.

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

- Always use the official model weights and official model implementations (e.g.
  Hugging Face libraries, Google Tunix/Flax, native framework runtimes)
  available in each inference framework.
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
