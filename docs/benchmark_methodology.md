# Benchmark methodology

This document defines the intended protocol for future AI Edge Bench runners
and reports. No benchmark implementation or published measurements exist yet.
Record departures from this protocol with each result; do not silently compare
runs with different measurement boundaries.

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

## Measurement boundaries

Benchmark inference separately from setup. Complete model download, load, and
initialization before starting a warm-run timer. If setup time is measured,
report it as a separate metric and specify whether it includes download,
compilation, cache population, or model loading. State whether a run starts
with an empty context/cache; use the same cache policy for comparisons.

For each measured request, start a monotonic timer immediately before handing
the prepared input to the framework and stop when generation finishes. Capture
the first generated token's availability at the API boundary if the framework
exposes it; do not infer it from the time the whole response is returned.
Exclude rendering, logging, and output file writing from the timed section.
Describe synchronization with asynchronous backends so reported times include
completed work rather than only dispatch.

Report these metrics using the **actual** number of generated output tokens
(`N`), not the configured maximum:

| Metric | Definition |
| --- | --- |
| End-to-end latency | Time from request start to completion, in milliseconds. |
| Time to first token (TTFT) | Time from request start to first output token available at the API boundary, in milliseconds. Includes prompt processing. |
| Decode throughput | `(N - 1) / (last-token time - first-token time)`, in tokens/second, for `N >= 2`. The first token is excluded because TTFT includes prefill. |

Use framework token IDs/counts when available, and document whether generated
special or stop tokens are included. Do not compare token-based throughput
across different tokenizers without identifying that limitation. If token
events are unavailable, report end-to-end latency and output length only; mark
TTFT and decode throughput unavailable rather than estimating them from total
time. If fewer than two output tokens are timed, decode throughput is
unavailable.

## Repetition and reporting

Unless a benchmark explicitly studies cold starts, perform at least one
unmeasured warm-up request followed by at least five measured requests with
the same inputs and configuration. Keep warm-up and measured runs distinct;
state the repetition count and order, and disclose thermal throttling or
background activity. For cold starts, use a separately specified cache and
process-reset procedure rather than mixing cold and warm timings.

Preserve per-run measurements and output-token counts. Report the median and
range (minimum to maximum) for each available metric, with units and number of
measured runs. Identify errors, truncated outputs, and excluded runs; never
silently discard them. Publish the exact command/configuration and software
and hardware metadata needed to reproduce results. Do not present results
from different conditions as a single framework ranking.
