# Benchmark containers

This directory tree provides reproducible containers for benchmarking Gemma 4
with [llama.cpp](https://github.com/ggml-org/llama.cpp) and
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) on four target
machines. The native-tool targets (`make bench`) produce the run directories
described below; comparable cross-framework numbers come from the
framework-neutral harness that runs inside the same images (see
[harness.md](harness.md)), which implements
[benchmark_methodology.md](benchmark_methodology.md).

## Layout

| Path | Purpose |
| --- | --- |
| [docker/base/Dockerfile](../docker/base/Dockerfile) | Harness-only image: run scripts, `/models` + `/results` contract, metadata capture. Carries no framework. |
| [docker/llama-cpp/Dockerfile](../docker/llama-cpp/Dockerfile) | Builds `llama-bench`. CPU, CUDA and Vulkan variants come from build arguments. |
| [docker/litert-lm/Dockerfile](../docker/litert-lm/Dockerfile) | Builds `litert_lm_main` (and `litert_lm_advanced_main` when available) with Bazel. |
| [docker/bin/](../docker/bin) | Shell entry points: `aeb-fetch-model`, `aeb-sysinfo`, `aeb-bench-llama-cpp`, `aeb-bench-litert-lm` (native tools), `aeb-prep-matched-gguf`. |
| [docker/harness/](../docker/harness) | Framework-neutral Python harness (drivers, runners, monitor, report); see [harness.md](harness.md). |
| [docker/tools/Dockerfile](../docker/tools/Dockerfile) | Offline tools image: model re-encoding, dataset staging, reports. Never timed. |
| [compose.yaml](../compose.yaml) | *What* is benchmarked: frameworks, variants, workload. |
| [compose/machines/](../compose/machines) | *Where* it runs: base images, ISA/backend flags, thread counts, device access. |

## How the pieces compose

Three independent axes are kept separate so that one change never forces a fork
of another file:

1. **Harness layer.** `aeb/base` holds every script. The framework images do
   `COPY --from=aeb/base /opt/aeb /opt/aeb`, so they can keep a framework-
   specific runtime base (plain Ubuntu for CPU, `nvidia/cuda` for CUDA) and
   still behave identically.
2. **Framework layer.** One Dockerfile per framework. Backends and target ISA
   are build arguments (`BUILDER_BASE`, `RUNTIME_BASE`, `CMAKE_EXTRA_FLAGS`,
   `BAZEL_CONFIG`, …), never `if` branches.
3. **Machine layer.** A compose override file per machine supplies those build
   arguments plus device access and thread counts. Adding a fifth machine means
   adding one file.

```
docker compose -f compose.yaml -f compose/machines/<machine>.yaml <command>
```

The `Makefile` wraps that pairing: `make MACHINE=<machine> bench`.

## Quick start

```bash
cp .env.example .env

make base                          # shared harness image, once per host
make MACHINE=epyc-7443p images     # build llama.cpp + LiteRT-LM
make MACHINE=epyc-7443p models     # download Gemma 4 E2B weights into ./models
make MACHINE=epyc-7443p bench      # llama.cpp, then LiteRT-LM ±YNNPACK
```

Machine names: `epyc-7443p`, `strix-halo`, `dgx-spark`, `apple-m5-max`
(`make machines`).

## Target machines

| Machine | Arch | llama.cpp | LiteRT-LM | Accelerator |
| --- | --- | --- | --- | --- |
| AMD EPYC 7443P (Zen 3) | linux/amd64 | CPU | CPU | none |
| AMD Strix Halo (Zen 5) | linux/amd64 | CPU, Vulkan | CPU | Radeon 8060S iGPU via Vulkan (`make bench-gpu`) |
| NVIDIA DGX Spark GB10 (Cortex-X925) | linux/arm64 | CPU, CUDA | CPU | Blackwell GPU, `sm_121`, CUDA 13 (`make bench-gpu`) |
| Apple M5 Max | linux/arm64 | CPU | CPU | **none reachable** — see below |

The CPU column is the comparison that is valid across all four machines. The
accelerator column is not: three different backends on three different vendors
are not a like-for-like comparison and must be reported separately.

## Reproducibility contract

* **Upstream revisions are pinned** — llama.cpp `b11259`, LiteRT-LM `v0.17.1`,
  set in `.env.example` and the compose defaults. The resolved commit SHA is
  baked into `/opt/aeb/build-info.json` inside each image and copied into every
  run's `metadata.json`. Never compare runs built from different refs.
* **`GGML_NATIVE=OFF`** for llama.cpp, with `GGML_CPU_ALL_VARIANTS=ON`. The
  image is portable across Zen 3, Zen 5 and Arm, and selects the best ISA
  variant at runtime instead of being tuned to whichever host built it.
* **Model bytes are checksummed.** `aeb-fetch-model` writes a `.sha256` and a
  `.manifest.json` next to every download, and the checksum is recorded in each
  run. Downloads land on a `.part` file first, so an interrupted transfer is
  never mistaken for a complete model.
* **Threads are explicit.** The harness derives a default from the cgroup CPU
  quota rather than trusting a framework's auto-detection, and every machine
  overlay pins a value.
* **One run per directory**:
  `results/<machine>/<framework>/<variant>/<timestamp>-<id>/` containing
  `metadata.json`, `sysinfo.json`, `command.txt`, `raw.log`, and for llama.cpp
  `llama-bench.json`.
* **Never `docker compose up`.** Use `run --rm` (as the Makefile does) so that
  only one benchmark container competes for the cores at a time.

Containers run as root, so on Linux hosts the files under `./results` and
`./models` are root-owned. Add `--user "$(id -u):$(id -g)"` to `docker compose
run` if that matters on your host; the harness writes nothing outside those two
directories.

## Framework notes

### llama.cpp

`llama-bench` is the native measurement tool. It performs its own warm-up pass
before the measured repetitions and excludes tokenization and sampling from
its timings, which is narrower than the end-to-end boundary described in the
methodology — report it as prompt-processing (`pp`) and token-generation (`tg`)
throughput, not as end-to-end latency or TTFT. Its prompt is random token ids
and `tg` decodes from an empty context by default; set `AEB_DEPTH` (`-d`) to
the prompt length to measure decode after a prompt (on the M5 Max VM, Gemma 4
E2B decodes ~19% slower at depth 1024 than at depth 0).

The image also builds `aeb-llama-driver`, the harness's in-process driver,
against the same `libllama`.

### LiteRT-LM and YNNPACK

`--enable_ynnpack` is the with/without switch, exposed as the two services
`litert-lm-xnnpack` and `litert-lm-ynnpack`. Two things about it are easy to
get wrong:

* **XNNPACK cannot be disabled.** LiteRT-LM's CPU backend always configures
  XNNPACK; there is no upstream build flag or runtime flag that turns it off.
  `--enable_ynnpack` gives YNNPACK *first pick* of the ops and leaves the
  remainder to XNNPACK. So the comparison is "XNNPACK only" versus
  "YNNPACK where possible, XNNPACK elsewhere" — not "with and without a
  delegate". Label results accordingly.
* **The runtime flag needs a matching build.** It only has an effect when the
  binary was compiled with `--define=litert_enable_ynnpack=true`. That is
  already the default for `--config=linux_arm64`; the images pass it
  explicitly so the x86-64 builds get it too. If the x86-64 Bazel build fails
  on that define, clear it with
  `AEB_LITERTLM_BAZEL_FLAGS=` and expect `--enable_ynnpack` to be a no-op on
  that machine — record that fact rather than reporting a null result as
  "no difference". YNNPACK is flagged experimental upstream.

LiteRT-LM prints its benchmark table as text and has no machine-readable
export, so `raw.log` is the authoritative record for those runs; `metadata.json`
says so explicitly.

`aeb-bench-litert-lm` uses `litert_lm_advanced_main`. At `v0.17.1`
`litert_lm_main` accepts but **ignores** `--benchmark_prefill_tokens` and
`--benchmark_decode_tokens`: it runs a built-in ~18-token prompt and decodes
until EOS, so its numbers do not describe the configured workload. The
advanced binary honours the flags but needs `--max_num_tokens` (set from
`AEB_CTX`, default 4096) for a 1024-token prefill; without it the KV cache is
sized too small and `DYNAMIC_UPDATE_SLICE` fails. Its benchmark prefill is
the tokenized default prompt zero-padded to the requested length.

The image also builds the C API library (`//c:litert-lm` →
`liblitert-lm.so`) from the same source and flags, plus the upstream ctypes
bindings, for the harness's in-process driver.

The GPU backend is not enabled in these images. On Linux, LiteRT-LM's GPU path
is WebGPU/Dawn via prebuilt shared objects and requires
`--define=litert_runtime_link_mode=dynamic` plus co-locating those libraries;
OpenCL is compiled out upstream. NPU support is Android-only.

## Machine-specific caveats

**EPYC 7443P / Strix Halo (x86-64).** LiteRT-LM's `linux_x86_64` Bazel config
compiles with `-mavx2`. Both Zen 3 and Zen 5 satisfy that; an older host would
produce an illegal-instruction crash at runtime. Threads default to one per
physical core — SMT siblings add variance to the memory-bound decode phase.

**DGX Spark (arm64).** The overlay pins `cpuset: 0-9` and 10 threads so runs
land on the Cortex-X925 performance cores rather than being spread across the
A725 cores by the scheduler. CUDA 13 is required for GB10; the compute
capability is `sm_121`. The NVIDIA Container Toolkit must be installed on the
host. Note that upstream LiteRT-LM has no arm64 CI for `litert_lm_main` and
ships no arm64 release artifact, so budget for build breakage there.

**Apple M5 Max.** Containers run inside a `linux/arm64` VM under Apple
Virtualization. There is **no GPU passthrough**: Metal and the Neural Engine
are unreachable, so only CPU numbers are available. Those numbers include VM
overhead and are *not* comparable to a native macOS llama.cpp build using
Metal. The vCPU count and memory come from Docker Desktop settings, not from
compose, and the VM scheduler decides P-core versus E-core placement. If you
need a Metal baseline for this machine, build llama.cpp natively on macOS and
publish it as a clearly separate, non-containerised result.

## Extending

* **New machine**: add `compose/machines/<name>.yaml` overriding the build args
  and thread/device settings. No Dockerfile change.
* **New llama.cpp backend**: add a service with different `CMAKE_EXTRA_FLAGS`
  and base images, as `strix-halo.yaml` does for Vulkan.
* **New framework**: add `docker/<framework>/Dockerfile` that copies
  `/opt/aeb` from `aeb/base`, plus an `aeb-bench-<framework>` script that
  writes the same run-directory layout.
