# AI Edge Bench

AI Edge Bench is a starting point for reproducible, on-device LLM benchmarks
across inference frameworks.

**Status:** This repository does not yet contain a benchmark runner, framework
adapters, published results, or installation instructions. The initial
documentation establishes how to build and compare them; it is not a claim
that any framework has been benchmarked.

## Benchmark goals

- Compare the same model, quantization, input, generation settings, and hardware
  where each framework supports them.
- Report latency and throughput with consistent measurement boundaries.
- Publish enough configuration and environment detail to reproduce each result,
  and call out settings that cannot be made equivalent.
- Keep raw run data separate from summaries so comparisons can be checked.

See [benchmark methodology](docs/benchmark_methodology.md) for proposed
measurement and reporting rules. Framework support and runnable commands will
be documented here as implementations are added.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a framework adapter,
benchmark, or result. This project is licensed under the
[Apache License 2.0](LICENSE).
