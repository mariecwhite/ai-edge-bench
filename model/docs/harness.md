# Framework-neutral harness

The container images ship native benchmark tools (`llama-bench`,
`litert_lm_main`), but those tools time different things: `llama-bench` feeds
random token ids and decodes from an empty context; LiteRT-LM's benchmark mode
zero-pads the prompt and reports its own phase timers. Their numbers cannot be
put side by side. The harness in [docker/harness/aeb](../docker/harness/aeb)
drives both frameworks the same way and measures them from the outside. It
implements [benchmark_methodology.md](benchmark_methodology.md).

## Pieces

| Module | Runs in | Purpose |
| --- | --- | --- |
| [aeb-llama-driver.cpp](../docker/llama-cpp/aeb-llama-driver.cpp) | llama.cpp image | In-process llama.cpp driver (links the image's `libllama`). |
| [drivers/litert_lm_driver.py](../docker/harness/aeb/drivers/litert_lm_driver.py) | LiteRT-LM image | In-process LiteRT-LM driver over the C API (`liblitert-lm.so`, built from the pinned source with the same Bazel flags as the CLI). |
| [drivers/xla_driver.py](../docker/harness/aeb/drivers/xla_driver.py) | OpenXLA image | OpenXLA (JAX / XLA:CPU) driver using the official Google Tunix (`tunix.models.gemma4`) and `qwix` model implementation on Flax NNX. |
| [drivers/openvino_driver.py](../docker/harness/aeb/drivers/openvino_driver.py) | OpenVINO image | OpenVINO GenAI CPU driver (`openvino_genai`) using models exported via Hugging Face `optimum-intel`. |
| [drivers/onnxruntime_driver.py](../docker/harness/aeb/drivers/onnxruntime_driver.py) | ONNX Runtime image | ONNX Runtime GenAI CPU driver (`onnxruntime_genai`); separate E2B GenAI package. |
| [drivers/mlx_driver.py](../docker/harness/aeb/drivers/mlx_driver.py) | macOS host | MLX (`mlx-lm`) Metal driver; native Apple GPU suite only. |
| [drivers/basert_driver.py](../docker/harness/aeb/drivers/basert_driver.py) | macOS host | BaseRT Metal driver over the engine's C API; native Apple GPU suite only. |
| [macos.py](../docker/harness/aeb/macos.py) | macOS host | Native process, CPU and GPU probes used by the monitor when there is no `/proc`. |
| [gpucheck.py](../docker/harness/aeb/gpucheck.py) | macOS host | Finds Metal GPU errors (hangs, aborted command buffers) per run in the unified log; such runs are excluded from reports. |
| [perf.py](../docker/harness/aeb/perf.py) | framework images | One timed process: load, warm-up, measured requests, resource sampling. |
| [monitor.py](../docker/harness/aeb/monitor.py) | framework images | 50 ms `/proc` + cgroup sampler for the driver process. |
| [accuracy.py](../docker/harness/aeb/accuracy.py) | framework images | MMLU / GSM8K validity gate through the same drivers. |
| [repro.py](../docker/harness/aeb/repro.py) | framework images | Output reproducibility (greedy and seeded) across repeats, processes and thread counts. |
| [workload.py](../docker/harness/aeb/workload.py) | framework images | Token-exact chat prompt from a public-domain passage. |
| [prep/](../docker/harness/aeb/prep) | tools image | Matched-weights GGUF conversion and `.litertlm` parity proof; `hf_reference.py` is an optional diagnostic that reruns the PyTorch reference of the QAT checkpoint with/without its activation quantization (`make diagnose-activations`, image `docker/tools/hf.Dockerfile`). |
| [datasets.py](../docker/harness/aeb/datasets.py) | tools image | Pinned MMLU / GSM8K staging into `./datasets`. |
| [report.py](../docker/harness/aeb/report.py) | tools image | Report, charts and performance history. |
| [scripts/suite.py](../scripts/suite.py) | host | Runs a [suite](../suites) one container (or, for `"execution": "native"`, one host process) at a time, interleaved; supports `--skip-existing` (idempotent resumption and automatic accuracy `--resume`) and data-type-aware selection (`accuracy_select_fastest_by_dtype` and `sequence_sweep.select_fastest_by_dtype`). |
| [scripts/native-apple-gpu.sh](../scripts/native-apple-gpu.sh) | macOS host | Builds the pinned native stack for the Apple GPU suite into `build/native/`. |
| [scripts/native_tools_crosscheck.py](../scripts/native_tools_crosscheck.py) | macOS host | Runs each framework's own benchmark tool on the suite's shapes for the cross-check. |

Runtime modules use only the Python standard library, so the framework images
use their own inference libraries; the shared orchestration modules use only
the Python standard library.

## Driver protocol

Drivers read one JSON object per line on stdin and answer one per line on
stdout (logs go to stderr). On start a driver loads the model and prints
`{"event": "ready", "load_ns": …, "info": {…}}`.

| Request | Response |
| --- | --- |
| `{"op": "tokenize", "text": …, "add_bos": true}` | `{"ok": true, "ids": [...]}` |
| `{"op": "generate", "prompt": …, "max_tokens": N, "ignore_eos": bool, "stop_ids": [...], "sampling": {"temperature", "top_k", "top_p", "seed"}}` | `{"ok": true, "n_prompt", "n_gen", "t_req_ns", "t_first_ns", "t_end_ns", "token_ns": [...], "text", "native": {…}}` |
| `{"op": "quit"}` | `{"ok": true}` |

Rules every driver follows:

- Always use the official model weights and the official model implementation
  available in the target inference framework or open-source library (for
  example, `libllama`, `liblitert-lm`, `tunix.models.gemma4` + `qwix`,
  `openvino_genai`, `onnxruntime_genai`, or Hugging Face `transformers`); never
  implement custom model architectures inside a driver.
- The prompt is sent without `<bos>`; the framework adds it. No chat template
  is applied by the framework (the harness renders it), so both frameworks
  receive the same token ids.
- Each request starts from an empty KV cache and decodes greedily unless
  `sampling` with `temperature > 0` is given (llama.cpp: top-k → top-p →
  temperature → seeded `dist` sampler, created per request).
- Timestamps are `CLOCK_MONOTONIC` nanoseconds. `t_req_ns` is taken before the
  prompt is tokenized; `token_ns[i]` when token *i* is available to the caller.
- `ignore_eos` decodes exactly `max_tokens` tokens. LiteRT-LM can only do this
  through its engine-level benchmark parameter, so its driver must be started
  with `--fixed-decode-tokens N` for timing runs.
- `native` carries the framework's own counters (llama.cpp perf context,
  LiteRT-LM BenchmarkInfo) for cross-checking.

Framework-specific notes:

- **LiteRT-LM sampling.** At v0.17.1 the CPU sampler implements only TOP_P;
  GREEDY and TOP_K return `UNIMPLEMENTED`. The driver uses TOP_P with
  `top_k = 1`, which is arg-max. The executor sampler is created once, from
  the first session's parameters, and reused (RNG state included) for the
  engine's lifetime: later sessions' sampler settings are ignored. Never mix
  sampling configurations in one LiteRT-LM driver process.
- **LiteRT-LM weight cache.** The engine silently skips the XNNPACK weight
  cache when `--cache-dir` does not exist; the driver creates it, and
  `aeb.perf` records whether the cache was warm. YNNPACK packs its own weights
  at load and writes no cache.
- **llama.cpp SWA cache.** The driver uses `swa_full = false` (the
  `llama-bench` / `llama-cli` default; the library default is `true`).
- **OpenXLA (`xla_driver.py`).** Uses the official Google Tunix Gemma 4 model
  (`tunix.models.gemma4`) on Flax NNX/JAX (`jaxlib`) with either the default
  **YNNPACK** backend (`__ynn_fusion`) or **Intel oneDNN** (`--onednn`, which
  sets `--xla_cpu_experimental_onednn_custom_call=true --xla_cpu_experimental_ynn_fusion_type=invalid --xla_disable_hlo_passes=onednn-ops-rewriter`).
  Quantization is only executed when the checkpoint itself provides pre-quantized
  weights and scales in `config.json` (`quantization_config`, e.g.
  `google/gemma-4-E2B-it-qat-mobile-transformers`); `xla_driver.py` refuses to
  run dynamic activation quantization or on-the-fly channel-wise PTQ on
  unquantized checkpoints.
- **ONNX Runtime.** The GenAI E2B model is a directory, not an `.onnx` file.
  The driver loads a temporary config overlay to set CPU thread counts; it
  never changes the staged package. Model hashing covers each package file.

## Run directory

`results/<machine>/perf/<framework>/<label>/<run_id>/`:

| File | Content |
| --- | --- |
| `meta.json` | Driver command, ready info, workload identity (prompt token-id SHA-256), model SHA-256, build info (framework commit, harness digest, repo revision), cache state, cgroup peak. |
| `requests.jsonl` | Every warm-up and measured request with all timestamps and generated text. |
| `timeseries.csv` | `t_ns, cpu_ticks, rss_kb, rss_anon_kb, rss_file_kb, rss_shmem_kb, hwm_kb, threads, pss_kb, cg_mem_bytes, sys_busy_ticks, sys_total_ticks`. |
| `summary.json` | Per-request metrics and median/min/max/CV over measured requests. |
| `driver.log`, `sysinfo.json` | Driver stderr; hardware snapshot. |

Accuracy runs go to `results/<machine>/accuracy-<task>/…` with
`predictions.jsonl` (every item, prediction and generated text) and
`summary.json`. Reproducibility runs go to `results/<machine>/repro/…` with
`outputs.jsonl` (text and token ids of every generation) and `summary.json`.

## Typical session

Both `make report` and `make native-report` use the same report layout and
HTML styling, following the
[Sapphire Rapids report](../reports/2026-10-15-gemma4-e2b-cpu-sapphire-rapids-sapphire-rapids/README.md):

1. **Summary**: accuracy/latency Pareto chart when accuracy exists, followed
   by a data-type-grouped comparison table ordered by end-to-end latency.
   The lowest-latency row is highlighted; only the best individual values
   are bolded.
2. **Results**: throughput and input-length sweep, latency, optional prefix
   caching, CPU, memory, combined resource timelines, optional thread scaling,
   detailed metrics/ratios, and generated-text checks. Native GPU suites add
   GPU utilization without substituting Linux memory metrics for Metal memory.
3. **Validation and context**: accuracy, reproducibility, suite-specific notes,
   environment, configurations, optional quantization equivalence, data-quality
   checks, caveats, and reproduction commands.

Outputs are `README.md`, identical `README.html` / `index.html`, `data.json`,
and PNG charts. The HTML embeds local chart images and CSS; no network access
or JavaScript is required. Keep the directory structure for relative data
and methodology links. Optional sections and charts appear only when their
measurements exist; missing data is never replaced with estimates.
Thread-scaling runs must match a headline model's framework, SHA-256 and
execution platform; suite-tagged runs must also match the suite. This keeps
legacy untagged CPU sweeps out of unrelated GPU/model reports. Regeneration
removes stale optional chart files when the corresponding data is absent.

Native suites declare `dtype` and optional `report` metadata per configuration:
`checkpoint`, `weights`, `activations`, `kv_cache`, and `thread_usage`.
Unsupported precision details display as `not exposed` rather than inheriting
CPU-framework assumptions. Their summary uses system GPU in-use memory above
the pre-load baseline, since process RSS/footprint do not account for Metal
weights consistently. CPU summaries retain peak RSS.

```bash
make MACHINE=apple-m5-max images tools     # framework + tools images
make MACHINE=apple-m5-max models           # stock GGUF + .litertlm
make MACHINE=apple-m5-max matched-models   # QAT checkpoint -> matched GGUFs (+ proofs)
make MACHINE=apple-m5-max datasets         # MMLU + GSM8K, pinned revisions
make MACHINE=apple-m5-max suite-perf       # prime caches, 3 interleaved rounds (stateless cold KV)
make MACHINE=apple-m5-max suite-prefix-cache # warm KV prefix-caching benchmark (if defined in suite)
make MACHINE=apple-m5-max suite-accuracy   # validity gate
make MACHINE=apple-m5-max suite-repro      # output reproducibility
make MACHINE=apple-m5-max report           # reports/<date>-<suite>-<machine>/ (README.md + README.html)
```

## Optional ARM CPU runtimes

### Input-length throughput sweep

Suites may define `sequence_sweep` (for example, N = 64, 128, 256, 512, 1,024,
2,048, 4,096, 8,192 and 16,384 prompt tokens with 512 generated tokens in the
exploratory ARM suite, or N = 64, 256, 1,024, 4,096 prompt tokens with 128
generated tokens in the x86 Sapphire Rapids suite). Run it separately from the
headline measurements:

```bash
python3 scripts/suite.py sequence --machine apple-m5-max \
  --suite suites/gemma4-e2b-cpu-arm-runtimes.json --dev --skip-existing
make tools
make MACHINE=apple-m5-max SUITE=suites/gemma4-e2b-cpu-arm-runtimes.json report
```

`make suite-sequence` is the equivalent runner target for images containing
the updated harness. Each length uses the suite's or `sequence_sweep`'s
warm-up/repetition counts and interleaved rounds, with one container at a time.
When `sequence_sweep.select_fastest_by_dtype` is `true`, `scripts/suite.py`
selects the empirically fastest configuration per `(framework, dtype)` pair
from the `perf` phase that also clears `sequence_sweep.min_accuracy`.
Context capacity stays at 16,896 for every point; model artifacts and thread
counts stay unchanged. The prompt includes BOS and chat-template tokens. Beyond
the passage's length, the harness repeats it deterministically; shorter
existing prompts are unchanged.

Reports retain the original 1,024 -> 256 headline and history and add separate
prefill/decode charts, tables and per-request data for the sweep. Prefill
throughput is N divided by TTFT (including tokenization and first-token
generation); decode throughput is `(N_gen - 1)` divided by first-to-last-token
time. Token-ID identity is checked separately at each N. Failed or wrong-length
processes are excluded with explicit diagnostics; unavailable points are
gaps, not extrapolated values.

### Model staging

The [exploratory ARM suite](../suites/gemma4-e2b-cpu-arm-runtimes.json) adds
ONNX Runtime GenAI to the two original CPU engines. It deliberately
does not claim a matched-weight comparison: the published GGUF, LiteRT-LM and ONNX
Q4_K_M conversions come from separate distributions and may have different
quantization, activation numerics, memory use and accuracy. Do not rank a
configuration that fails the accuracy gate.

Stage the complete model packages in the shared `models/` directory with
`make MACHINE=apple-m5-max arm-models` (builds the tools image and downloads
the pinned snapshots). Alternatively, with the Hugging Face `hf` CLI on the
host, use the equivalent commands below. Model assets stay outside git.
Pinned revisions prevent subsequent upstream updates from silently altering
results:

```bash
hf download justinchuby/gemma-4-e2b-it-onnx \
  --revision 9bcf2cb1c2878b1c68a5f94db037272dfb278384 \
  --include 'Q4_K_M/default/**' \
  --local-dir models/aeb/gemma-4-e2b-onnx
make MACHINE=apple-m5-max arm-images
python3 scripts/suite.py plan --machine apple-m5-max \
  --suite suites/gemma4-e2b-cpu-arm-runtimes.json
```

Stage the existing GGUF and `.litertlm` with `make MACHINE=apple-m5-max models`
and build their images with `make MACHINE=apple-m5-max images` if not already
done. Run the suite with `--suite suites/gemma4-e2b-cpu-arm-runtimes.json` for
the `perf`, `accuracy`, and `repro` actions; perform accuracy before publishing
or interpreting speed. Run `make MACHINE=apple-m5-max SUITE=suites/gemma4-e2b-cpu-arm-runtimes.json report`
only after the runs and their quality checks complete. Use `--only` to smoke
test one new configuration without rerunning the original engines.

## x86 Sapphire Rapids (Intel AMX) multi-precision suite

The [Sapphire Rapids suite](../suites/gemma4-e2b-cpu-sapphire-rapids.json)
evaluates five CPU runtimes (`litert-lm`, `llama.cpp`, `xla`, `openvino`, and
`onnxruntime`) across `fp32`, `bf16`, `int8`, and `int4` precision tiers on
Intel Xeon Sapphire Rapids (`numactl --physcpubind=0-55 --localalloc`):

- [scripts/setup_amx_machine.sh](../scripts/setup_amx_machine.sh): Stages images,
  datasets, and all pre-built model artifacts on the remote AMX host.
- [scripts/run_amx_benchmarks.sh](../scripts/run_amx_benchmarks.sh): Executes the
  8-phase pipeline (`smoke`, `sweep`, `perf`, `prefix-cache`, `accuracy`,
  `sequence`, `repro`, and `report`) with `--skip-existing` so interrupted runs
  resume cleanly without repeating finished phases.
- [scripts/fetch_results.sh](../scripts/fetch_results.sh): Pulls the generated
  reports and raw `results/sapphire-rapids` directory back to the local
  workstation.

`suites/<suite>.json` defines the configurations. The `machines.<machine>`
section holds the per-machine thread count and the tuned "fastest" settings
together with how they were selected; add one before running a new machine.

## Native Apple GPU suite

[suites/gemma4-26b-a4b-gpu.json](../suites/gemma4-26b-a4b-gpu.json) runs
Gemma 4 26B A4B on the Apple M5 Max GPU with LiteRT-LM (ML Drift GPU backend),
llama.cpp (Metal), MLX (`mlx-lm`) and BaseRT. Docker on macOS runs a Linux VM
without Metal, so this suite sets `"execution": "native"`: `scripts/suite.py`
starts every process directly on the macOS host, still one at a time and
interleaved, with the pinned stack from `scripts/native-apple-gpu.sh`. Suite
files keep the container paths (`/models`, `/results`, `/datasets`); the
runner maps them to the repository directories.

```bash
brew install cmake uv                       # plus Xcode Command Line Tools
make native                                 # build/native: venv, llama.cpp Metal, BaseRT
make gpu-models                             # pinned artifacts, SHA-256 verified (~62 GB)
S=suites/gemma4-26b-a4b-gpu.json
make MACHINE=apple-m5-max SUITE=$S suite-perf       # prime + 3 interleaved rounds
make MACHINE=apple-m5-max SUITE=$S suite-sequence   # input-length sweep
make MACHINE=apple-m5-max SUITE=$S suite-accuracy   # validity gate (always sequential here)
make MACHINE=apple-m5-max SUITE=$S suite-repro      # output reproducibility
make MACHINE=apple-m5-max native-crosscheck         # native benchmark tools, same shapes
PYTHONPATH=docker/harness build/native/venv/bin/python -m aeb.gpucheck \
  --results results --machine apple-m5-max          # GPU errors in accuracy/repro runs too
make MACHINE=apple-m5-max SUITE=$S native-report    # report with the native venv
```

Close other GPU applications first. The runner and the cross-check script keep
the machine and display awake (`caffeinate -dis`) for as long as they run, so
the machine cannot sleep mid-run and the window server's GPU load does not change.
For long sessions with gaps between steps (downloads, several `make` targets),
also hold `caffeinate -dims` in another terminal. After the run, confirm
there was no sleep in the run window:
`pmset -g log | grep -E "Entering Sleep|Wake from|DarkWake from"`. Accuracy runs
never run in parallel on the single GPU (`--parallel` is ignored) and keep the
configured thread count.

### Artifacts and frameworks

Every configuration is a **reference** configuration: each framework's own
published ~4-bit artifact with framework defaults. They are not weight-matched
(see the suite description and the report caveats).

| Config | Artifact (pinned revision) | Framework build |
| --- | --- | --- |
| `litert-lm/gpu` | `litert-community/gemma-4-26B-A4B-it-litert-lm` `gemma-4-26B-A4B-it-gpu.litertlm` (QAT, blockwise int4 weights, float activations) | `litert-lm-api==0.18.1` macOS wheel; ML Drift (Metal) is linked into `liblitert-lm.dylib`; `--backend gpu` |
| `llama.cpp/metal` | `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` `gemma-4-26B_q4_0-it.gguf` (Google's QAT Q4_0) | `b11259`, `GGML_METAL=ON`, `GGML_METAL_EMBED_LIBRARY=ON` (no Xcode Metal compiler needed); all layers offloaded (`--n-gpu-layers 999`) |
| `mlx/metal` | `lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit` (QAT checkpoint; experts 4-bit g64 affine, dense MLP and router 8-bit) | `mlx==0.32.4`, `mlx-lm==0.32.2` |
| `basert/metal` | `basecompute/gemma-4-26B-A4B-it` `gemma-4-26B-A4B-it-Q4.base` (BaseRT catalog `default-q4`; checksum from the v0.4.0 catalog) | BaseRT `v0.4.0` prebuilt engine (proprietary; tarball SHA-256 checked) with its Apache-2.0 Python binding |

The 26B A4B chat template closes a non-thinking model turn with an empty
thought channel (`<|channel>thought\n<channel|>`), unlike E2B. The suite sets
`workload.chat_template` to `gemma4-empty-thought`; the runner passes it to
every process as `AEB_CHAT_TEMPLATE` and each run records it.

Driver notes:

- **LiteRT-LM GPU** uses the same driver as the CPU suites with
  `--backend gpu`; greedy is still TOP_P with `top_k = 1`, and fixed-length
  decoding still uses the engine's benchmark `num_decode_tokens`. The engine's
  GPU program/weight cache is primed into `--cache-dir` before timing.
  `gpu_decode_steps_per_sync` and speculative decoding stay at the engine's
  defaults.
- **llama.cpp** is the container driver source compiled natively. On macOS it
  timestamps with `CLOCK_UPTIME_RAW`, the clock Python 3.12's
  `time.monotonic_ns` reads; libc++'s `steady_clock` there also counts sleep.
  `aeb.perf` flags any request whose driver timestamps fall outside its own
  request window.
- **MLX** generates with `mlx_lm.stream_generate` (fresh prompt cache per
  request, `make_sampler(temp=0)` for greedy). `ignore_eos` empties the
  tokenizer's EOS set for that request. The driver sets the Metal wired
  limit once at load, as `mlx_lm.server` does. mlx-lm grows its KV cache on
  demand, so `--ctx` is a request limit, not a preallocation.
- **BaseRT** calls `baseRT_reset` and then `baseRT_generate` per request.
  `baseRT_generate` has no ignore-EOS option, so fixed-length timing biases every registered
  stop id to −1e30 (the other drivers feed a sampled stop token back instead;
  per-step work is the same). n-gram speculation stays at its default (off).

### What the GPU-side metrics mean

The native monitor samples the driver process every 50 ms with
[macos.py](../docker/harness/aeb/macos.py):

| Column / metric | Source | Meaning and limits |
| --- | --- | --- |
| `cpu_ticks` → CPU cores, CPU s | `proc_pid_rusage` user + system time | Microsecond ticks (`clk_tck` = 1e6 in `meta.json`). |
| `footprint_kb` → peak phys_footprint | `ri_phys_footprint` | Private memory as macOS accounts it. It includes **some** Metal allocations (a 1 GiB MLX array adds ~1.1 GiB; MLX's 15 GB of weights appear) but not others: llama.cpp's no-copy mmap'd weights and LiteRT-LM's and BaseRT's GPU weights do not. Clean mmap'd file pages are not included. Lifetime maximum from `ri_lifetime_max_phys_footprint`. |
| `rss_kb` → resident size | `ri_resident_size` | Resident pages, including file-backed ones. There is no anonymous/file split or PSS on macOS. |
| `gpu_util_pct` → GPU utilization (prefill/decode) | IOAccelerator `Device Utilization %` | **System-wide**, not per process. It reads 99–100% at saturation and ~0% at idle, but it is coarse. With 1 s on/off load it tracks the true duty cycle (0.25 → 0.30, 0.50 → 0.48), but with 100 ms cycles it overstates idle gaps (0.25 → 0.51). Use it as an indicator. The report also gives the mean between requests as the background load. |
| `gpu_mem_kb` → GPU memory above baseline | IOAccelerator `In use system memory` | System-wide GPU-mapped memory. The report's comparable memory figure is the process-lifetime peak minus the value at process start (before the model loads; ~0.8 GB of window-server and other clients' allocations, stable across processes). It is system-wide, so other GPU clients must stay idle. |
| `energy_nj` → CPU energy | `ri_energy_nj` | The kernel's CPU energy estimate. It **excludes GPU work** (~0 W for a GPU-saturating process). GPU power needs `powermetrics` (root) and is not measured. |
| `gpu_ns` | AGXDeviceUserClient `AppUsage.accumulatedGPUTime` | Recorded raw only. Its ratio to wall time varied 0.32–0.83 across equally saturating kernels, so it is not a busy-time measure and is not reported. |

### GPU errors invalidate a run

After a GPU hang, macOS restarts the GPU and ignores further command buffers
from the offending process. An engine that does not check command-buffer
status keeps "running" without doing GPU work: it returns stale output with
impossible timings and reports success. This happened once with LiteRT-LM
0.18.1 in the 2026-10-22 report: a 16,384-token prefill "took 12 ms".
After every native process, `aeb.perf` therefore reads the
macOS unified log for aborted Metal command buffers during the run, writes
`gpu_check.json` and marks the run `error` if the driver process had any.
The report excludes such runs and lists them under data-quality checks;
it never averages or replaces them. To check accuracy and
reproducibility runs, or runs made before this check existed, run the scan
while the unified log still holds that period (a few days):

```bash
PYTHONPATH=docker/harness build/native/venv/bin/python -m aeb.gpucheck \
  --results results --machine apple-m5-max --since 2026-10-22T14:00:00Z
```

Native macOS numbers are not comparable with the Linux-VM CPU suites.
