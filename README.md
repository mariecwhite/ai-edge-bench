# AI Edge Bench

AI Edge Bench is a starting point for reproducible, on-device LLM benchmarks
across inference frameworks.

**Status:** This repository provides reproducible benchmark *containers* — see
[container setup](docs/containers.md) — for llama.cpp and LiteRT-LM on Gemma 4.
It does not yet contain published results or a cross-framework results parser.
The documentation establishes how to build and compare these frameworks; it is
not a claim that any framework has been benchmarked.

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

## Benchmark goals

- Compare the same model, quantization, input, generation settings, and hardware
  where each framework supports them.
- Report latency and throughput with consistent measurement boundaries.
- Publish enough configuration and environment detail to reproduce each result,
  and call out settings that cannot be made equivalent.
- Keep raw run data separate from summaries so comparisons can be checked.

See [benchmark methodology](docs/benchmark_methodology.md) for proposed
measurement and reporting rules.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a framework adapter,
benchmark, or result. This project is licensed under the
[Apache License 2.0](LICENSE).
