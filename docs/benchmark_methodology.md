# Benchmark methodology

This document defines how AI Edge Bench measures and reports on-device LLM
inference. The framework-neutral harness in [docker/harness](../docker/harness)
implements it (see [harness.md](harness.md)). Record any departure from this
protocol with the result; never silently compare runs that use different
measurement boundaries.

## Comparable workloads

Use the same model family, weights and quantization, input content, and output
limit across frameworks where supported. Record the exact model artifact and
its source and checksum, tokenizer, prompt text or input identifier, chat
template, stop conditions, and generation settings (including temperature,
sampling parameters, and seed). If a framework needs a different model format,
template, tokenizer, or setting, disclose the difference and do not label the
runs an apples-to-apples comparison.

Record framework name, version or commit, runtime/backend, threading and
acceleration settings; device, CPU/GPU/NPU, memory, operating system, and
power/thermal conditions. Run competing frameworks on the same device where
possible, without concurrent heavy workloads.

### Official model weights and framework implementations

- Always use the official published model weights and the official model
  implementation available in each inference framework or open-source library
  (for example, Hugging Face `transformers` / `optimum-intel`, Google `tunix` +
  `qwix` on Flax NNX/JAX, `libllama` in `llama.cpp`, `liblitert-lm` in
  LiteRT-LM, `onnxruntime_genai`, and `openvino_genai`).
- Never implement custom, hand-written model architectures or bespoke layer
  graphs inside benchmark harness drivers. Harness drivers must remain thin
  protocol adapters that load official checkpoints into the target framework's
  official model/runtime API and record timing and token events.
- When a framework requires an offline format conversion (such as GGUF,
  OpenVINO IR via `optimum-cli`, or ONNX Runtime GenAI), generate the artifact
  from the official upstream checkpoint using the framework's official
  conversion tools and record the source revision and conversion steps.
- **No on-the-fly dynamic or channel-wise quantization on unquantized checkpoints.**
  Do not apply runtime/on-the-fly weight quantization (such as dynamic channel-wise
  PTQ) or dynamic activation quantization to an unquantized checkpoint (e.g., in
  OpenXLA) unless the checkpoint's weights and metadata (`quantization_config`)
  explicitly provide pre-quantized weights and scales. When evaluating a quantized
  precision tier (`INT8`, `INT4`, `QAT-mobile`), only include frameworks that have
  a pre-quantized checkpoint or pre-built quantized artifact for that precision.

### Quantization equivalence is proven on tensors, not on labels

"4-bit" on two model cards rarely means the same weights. Before calling a
comparison matched:

1. Identify the checkpoint each artifact was produced from and the per-tensor
   bit width, granularity (per-channel, block size), symmetry and scale dtype.
   Inspect the files themselves; model cards are not specific enough.
2. If the frameworks need different containers (e.g. `.litertlm` vs GGUF),
   build one artifact from the other's source integers with a **lossless**
   re-encoding, and verify it: every integer round-trips exactly, and a sample
   of tensors in both final files is compared element by element.
3. Document every remaining numerical difference: scale storage precision,
   activation quantization (static vs dynamic, granularity), KV-cache
   precision, and anything a framework applies at run time (repacking,
   fast-math kernels).

If equivalence cannot be established, label the comparison by quantization
scheme and treat it as an open (best-effort) comparison only.

### Identical, realistic inputs

- Use a real-text prompt rendered with the model's chat template, not random
  token ids (`llama-bench`) or zero padding (LiteRT-LM
  `--benchmark_prefill_tokens`). Kernel fast paths and activation ranges can
  depend on content.
- Trim the prompt with each framework's own tokenizer until it is exactly the
  target length (including `<bos>`), and record the SHA-256 of the token ids
  per run. A report must show the hashes are identical across frameworks.
- Decode greedily to a fixed number of tokens (stop tokens ignored) for
  timing, so every framework does the same amount of work. Use the
  framework's own stop handling for accuracy runs.
- Allocate the same context length (KV-cache size) everywhere; it changes
  memory and can change attention cost.

## Configuration tracks

Report results in separate tracks and never rank across them. This follows
MLPerf's closed/open divisions and SPEC's base/peak split.

| Track | Rule |
| --- | --- |
| **Matched** (closed / base) | Same weights (proven as above), prompt tokens, decoding, context, thread count and the closest available KV-cache precision. Framework-internal scheduling knobs (batch splitting, thread pools) stay at their defaults and are recorded. |
| **Fastest** (open / peak) | Each framework's best configuration for the same workload: any supported kernels, weight encoding, delegate, thread count or numerics-relaxing option (for example YNNPACK `fast_math`, f16 KV cache). Must pass the accuracy gate. Publish the sweep that selected it and the selection objective (here: lowest median end-to-end latency). |
| **Reference** | Out-of-the-box usage, e.g. the stock community model file with framework defaults, for context only. |

Thread count in the matched track equals the number of CPUs allocated to the
container (`nproc`). Because the best thread count differs per framework and
per phase, always publish a thread-scaling sweep alongside.

## Measurement boundaries

Benchmark inference separately from setup. Complete model download, load, and
initialization before starting a warm-run timer. If setup time is measured,
report it as a separate metric and specify whether it includes download,
compilation, cache population, or model loading. State whether a run starts
with an empty context/cache; use the same cache policy for comparisons.

- **Cache policy.** Warm framework caches (for example LiteRT-LM's XNNPACK
  weight cache) with an unmeasured priming run before any measured process,
  record whether each cache existed at process start, and make sure the cache
  directory exists — LiteRT-LM silently disables the weight cache otherwise,
  which changes both load time and resident memory.
- **Fresh KV per request (stateless headline benchmark).** Every request in the
  headline `perf` benchmark starts from an empty KV cache; no prefix or response
  caching is allowed between requests.
- **Cached-prefix KV benchmark (`prefix-cache` track).** Because multi-turn chat
  and system-prompt workloads reuse a shared prompt prefix across turns, suites
  may also define a separate `prefix_cache` benchmark (`--cache-prefix`) that
  keeps a shared prefix of `P` tokens (for example, `896` tokens of a `1,024`-token
  prompt) warm in the KV cache while varying the final `S = N - P` suffix tokens
  on every request. Report cached-prefix results in their own dedicated section
  alongside the stateless baseline, never mixed into cold-KV headline numbers.

For each measured request, start a monotonic timer immediately before handing
the prepared input to the framework and stop when generation finishes. Capture
the first generated token's availability at the API boundary if the framework
exposes it; do not infer it from the time the whole response is returned.
Exclude rendering, logging, and output file writing from the timed section.
Describe synchronization with asynchronous backends so reported times include
completed work rather than only dispatch.

Drive every framework **in-process** through the same thin protocol (the
harness drivers), and cross-check the result against each framework's native
benchmark tool. Differences must be explained before publishing. Example:
`llama-bench`'s `tg` test decodes from an empty context, which overstates
decode throughput after a long prompt; use `-d <prompt tokens>` when
comparing.

Report these metrics using the **actual** number of generated output tokens
(`N`), not the configured maximum:

| Metric | Definition |
| --- | --- |
| End-to-end latency | Time from request start to completion, in milliseconds. |
| Time to first token (TTFT) | Time from request start to first output token available at the API boundary, in milliseconds. Includes prompt processing. |
| Prefill throughput | Prompt tokens ÷ TTFT, in tokens/second. |
| Suffix prefill throughput (cached-prefix mode) | Uncached suffix tokens (`N_prompt - N_cached`) ÷ warm TTFT, in tokens/second. |
| Decode throughput | `(N - 1) / (last-token time - first-token time)`, in tokens/second, for `N >= 2`. The first token is excluded because TTFT includes prefill. |
| Inter-token latency | Distribution of gaps between consecutive streamed tokens (p50, p99). |

Use framework token IDs/counts when available, and document whether generated
special or stop tokens are included. Do not compare token-based throughput
across different tokenizers without identifying that limitation. If token
events are unavailable, report end-to-end latency and output length only; mark
TTFT and decode throughput unavailable rather than estimating them from total
time. If fewer than two output tokens are timed, decode throughput is
unavailable.

### CPU and memory

Run the framework in its own process and sample it, not the harness:

- **CPU use**: `(utime + stime)` deltas from `/proc/<pid>/stat` divided by wall
  time, reported per phase (prefill, decode) as average cores busy, plus CPU
  seconds per request. Busy-waiting worker threads count as used CPU — that is
  a real cost on a shared device.
- **Memory over time**: sample `/proc/<pid>/status` every 50 ms and keep the
  time series. Report anonymous and file-backed RSS separately (mmap'd weights
  are file-backed, shared and reclaimable; repacked or converted weights, KV
  cache and activations are anonymous), PSS from `smaps_rollup` at a lower
  rate, the kernel high-water mark `VmHWM`, and the container's cgroup
  `memory.peak` (which also counts page cache).
- Record whole-machine CPU busy time during each request to detect background
  load.

## Cached-prefix KV benchmark (`prefix_cache`)

To measure how effectively each runtime reuses KV state across requests that
share a common prefix (such as a system prompt or document context), suites may
define a `prefix_cache` section (`prompt_tokens`, `prefix_tokens`, `gen_tokens`,
`warmup`, `repetitions`, `rounds`):

- The harness constructs a sequence of `warmup + repetitions` distinct prompts
  of exact token length `prompt_tokens` that share the identical first
  `prefix_tokens` token IDs (`P`) and differ in the trailing
  `S = prompt_tokens - prefix_tokens` suffix tokens, so full-response caching
  cannot short-circuit suffix evaluation or decoding.
- Each framework driver is started with `--cache-prefix` and reuses or snapshots
  the KV state for the `P`-token prefix from the warm-up turn, evaluating only
  the `S` new suffix tokens before decoding `gen_tokens` tokens.
- Reports compare **Cold TTFT** (from the stateless `perf` run at the same total
  prompt length) against **Warm TTFT**, **TTFT Speedup (`Cold ÷ Warm`)**,
  **Effective Prefill Throughput (`prompt_tokens ÷ Warm TTFT`)**,
  **Suffix Prefill Throughput (`S ÷ Warm TTFT`)**, and **Decode Throughput**.

## Input-length throughput sweep

In addition to the headline workload (`1,024` prompt tokens → `256` generated
tokens), suites may define an input-length sequence sweep (`sequence_sweep`)
across multiple prompt lengths (for example `64` to `4,096` or `16,384` tokens)
at a fixed decode length:

- Every prompt length uses the same real-text passage, chat template, and
  token-exact trimming, with SHA-256 token-id verification at each length `N`.
- In multi-precision suites (`select_fastest_by_dtype: true`), the sequence
  sweep automatically selects the empirically fastest configuration per
  `(framework, dtype)` pair from the headline `perf` runs that also passes the
  minimum accuracy threshold (`min_accuracy`), ensuring scaling curves only
  include configurations with verified output quality.

## Accuracy gate

Timing a configuration that produces wrong tokens is meaningless. Every
configuration or `(framework, dtype)` cell in a report must be checked for
output validity with the same drivers, prompts and decoding used for timing:

- Use pinned revisions of industry-standard datasets (here the full MMLU test
  set, 14,042 questions, and GSM8K for long greedy generations) and record the
  dataset checksums.
- Matched single-dtype suites run the full set; fastest/reference or
  multi-precision matrix suites run a deterministic stratified subset
  (≥ 2,000 MMLU items).
- **Apples-to-apples item limits across frameworks and data types.** Every
  configuration compared in a report must use the **exact same stratified item
  limit** (for example, `mmlu_limit: 2000` across all `fp32`, `bf16`, `int8`,
  and `int4` configurations) so that accuracy intervals and paired McNemar
  agreement tests are computed over the identical set of questions.
- **Data-type-aware winner selection (`accuracy_select_fastest_by_dtype`).**
  When a suite includes multiple candidate configurations for the same
  `(framework, dtype)` pair (such as matched vs. fastest kernel/quantization
  variants), the runner can group configurations by `(framework, dtype)` and
  evaluate accuracy on the empirically fastest configuration (lowest median
  end-to-end latency in `perf`) for each `(framework, dtype)` cell.
- **Checkpointing and resumption.** Accuracy runs stream per-item predictions to
  `predictions.jsonl` and support `--resume` (automatically enabled by
  `scripts/suite.py --skip-existing`) so interrupted evaluations continue from
  already-scored item indices without repeating work.
- Accuracy runs do not need an idle machine and may run side by side, but
  only after showing that outputs do not depend on the thread count, and
  only if the total number of threads stays within the allocated CPUs.
  llama.cpp's worker threads spin while waiting, so oversubscription slows
  every process sharply; one oversubscribed run here fell to about 1 output
  per minute.
- Put the model's publicly reported scores next to the measured ones, with
  each source's setup (shots, thinking mode, token cap, precision). They
  are a sanity check, not a threshold, unless the setup matches exactly.
- Report accuracy with a 95% confidence interval and count parse failures as
  wrong, never dropped. Compare frameworks with each other on identical items;
  differences larger than the intervals indicate a numerical problem to
  investigate before publishing performance numbers.

## Output reproducibility

A framework is reproducible if the same model file, framework build, prompt
and generation settings produce the same output every time. Users depend on
this for debugging, caching, regression tests and evaluations. Test it as its
own benchmark, separately from timing, and report the result per
configuration.

**Workload.** A fixed set of 12 prompts covering short answers, multiple
choice, arithmetic, word problems, code, summarization, structured JSON,
translation and open-ended text (`data/repro_prompts.json`). Each request uses
a fresh session/KV cache, the chat template, stop tokens honoured and a fixed
token cap (128 new tokens by default, or configured via `repro.max_tokens` in
the suite JSON). Both modes are tested:

- **Greedy**: temperature 0.
- **Seeded sampling**: the model's recommended sampling (Gemma: temperature
  1.0, top-k 64, top-p 0.95) with a fixed seed.

**One sampling configuration per process.** Some engines fix the sampler
when the engine is created rather than per request. LiteRT-LM v0.17.1 builds
its executor sampler from the first session's parameters and reuses it, RNG
state included. Mixing modes in one process would therefore test the wrong
thing. Run each mode, and each seed, in its own process.

**Levels.** Report each separately; a lower level never implies a higher one.
Suites may configure `repro.repeats` (default `3`, minimum `2`) and
`repro.slots` in the suite JSON when constraining total runtime on large CPU
matrices.

| Level | Comparison | Expected for a deterministic engine |
| --- | --- | --- |
| Within process | Each prompt repeated `N`× (`N = 3` by default, or `repro.repeats`) in one process, interleaved with the other prompts (A B C … A B C …), so state carried between requests becomes visible. | identical |
| Across processes | A second fresh process with the same request sequence. | identical |
| Across thread counts | A process with half the threads. Thread-dependent reduction order is a common source of divergence. | identical (report if not) |
| Seed control | The seeded process rerun with seed + 1. | outputs **change** for most prompts; if not, the seed is ignored or sampling has collapsed to greedy |
| Across frameworks | Same weights, different framework, greedy. | informational only: different activation/KV quantization or kernels legitimately move near-tie logits |

**Verdicts.** *Reproducible per request* means identical within and across
processes. *Reproducible per process only* means identical across processes
but not within one: the output depends on earlier requests (for example a
sampler RNG that is not reseeded per request). *Not reproducible* means
neither. For mismatches, report the token index of the first divergence,
using the token ids the framework's tokenizer assigns to the output text.

**Over time.** Store a fingerprint (hash of all greedy outputs) with each
history record. If it changes while the model file and framework commit stay
the same, flag it as a reproducibility regression. If it changes after a
framework bump, record that numerics changed.

The timing runs provide a free extra check: their measured requests repeat
one long (1,024 + 256 token) greedy generation per process, and the report
states whether all of them were identical.

## Repetition and reporting

Unless a benchmark explicitly studies cold starts, perform at least one
unmeasured warm-up request followed by multiple measured requests (default five,
or configured via `workload.repetitions` in the suite JSON) with the same
inputs and configuration. Keep warm-up and measured runs distinct; state the
repetition count and order, and disclose thermal throttling or background
activity. For cold starts, use a separately specified cache and process-reset
procedure rather than mixing cold and warm timings.

- **Repeat at the process level.** Run each configuration across multiple
  separate processes (`workload.rounds`, default three; pyperf, SPEC), each with
  its own warm-up. Process-level effects (memory layout, thread placement) are
  often larger than request-to-request noise.
- **Interleave.** Alternate configurations and reverse the order every other
  round (A B C, C B A, …) with a fixed cool-down between processes, so drift
  and thermal effects spread across configurations instead of biasing the last
  one.
- **Pool and describe.** Report the median of all measured requests with the
  min–max range and the coefficient of variation, plus each process's median.
  Flag decode CV above 5% and between-process spread above 5%. Flag outliers
  with the modified z-score rule (|x − median| > 10 robust σ) but keep them.
- **Compare with intervals.** Express framework ratios as ratio of medians
  with a bootstrap 95% interval; do not call differences inside the interval.

Preserve per-run measurements and output-token counts. Report the median and
range (minimum to maximum) for each available metric, with units and number of
measured runs. Identify errors, truncated outputs, and excluded runs; never
silently discard them. Publish the exact command/configuration and software
and hardware metadata needed to reproduce results. Do not present results
from different conditions as a single framework ranking.

## Tracking performance over time

- Every report appends one record per configuration to a committed history
  file (`reports/history/<suite>/<machine>.jsonl`) holding medians, ranges,
  CV, accuracy, framework commit, harness digest, repository revision and
  model hash.
- Compare a configuration only with earlier records of the same suite,
  machine and configuration id. Flag a change when the median moves by more
  than max(3%, 3 × the larger CV) — a noise-aware threshold in the spirit of
  asv, Perfherder and LNT.
- When the harness digest or model hash changes, annotate the history entry;
  attribute a step change to the framework only when those stayed fixed.
- Re-run the full suite (all tracks, accuracy gate included) whenever a
  framework revision is bumped; never mix revisions inside one report.

## Native GPU runs (Apple silicon)

Containers on macOS cannot reach Metal, so GPU suites run natively on the host
(`"execution": "native"`, see [harness.md](harness.md#native-apple-gpu-suite)).
The rules above still apply: one framework process at a time (the GPU is never
shared between benchmark processes, accuracy runs included), interleaved
rounds, warm-up, process-level repetition, the accuracy gate and pinned
framework builds and model files.

- **Synchronization.** A token timestamp is taken when the token id is
  available to the caller on the CPU, which requires the GPU work that
  produced it to have completed. Dispatch alone is never timed. Engines that
  pipeline the next step (mlx-lm) or sync every few steps
  (`gpu_decode_steps_per_sync`) keep their defaults; that shows up in
  inter-token latency, not in decode throughput.
- **Memory.** Unified memory makes "GPU memory" and "CPU memory" one pool,
  and macOS does not attribute every Metal allocation to the allocating
  process's `phys_footprint`. Compare frameworks on the rise in system GPU
  in-use memory above the value at process start (other GPU clients idle), and
  report `phys_footprint` and resident size alongside it.
- **Utilization and energy.** Without root, macOS exposes only a system-wide,
  coarse GPU utilization counter and a CPU-only per-process energy estimate.
  Report them as indicators with their calibration, state the background GPU
  load between requests, and do not present either as GPU efficiency.
- **GPU errors.** A GPU hang or fault can leave an engine silently returning
  stale output with impossible timings. Check the system's GPU error log for
  every process (`aeb.gpucheck`). Exclude affected processes and report them;
  never average or replace them.
- **Idle time.** Prefill (TTFT) can depend on how long the GPU idled before
  the request. Keep the inter-request idle time fixed (the suite's
  `cooldown_s`), state it, and when it matters publish the sensitivity.
- **Conditions.** Record power source, low-power mode and thermal warnings.
  Keep the display awake for the whole run so the window server's GPU load
  stays the same, and close other GPU applications.

## Virtualized and shared hosts

When the container runs in a VM (e.g. Docker on macOS), the host scheduler,
not the container, decides core placement and frequency; governors, turbo and
isolation cannot be set. Rely on interleaving, process-level repetition and
medians, state the vCPU count and host model, and do not compare such numbers
with native runs.

Keep the host awake for the whole run. On macOS, idle sleep pauses the Docker
VM: the guest's monotonic clock stops, so a request that spans a sleep looks
only slightly slow rather than obviously broken. The suite runner holds a
`caffeinate` assertion; still check the host power log (`pmset -g log`) for
sleep/wake events inside the run window and disclose any overlap.
