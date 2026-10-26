## Key findings

1. **MLX has the lowest end-to-end latency on the headline workload**
   (1,024 -> 256 tokens): 2.41 s, against 2.61 s for llama.cpp (Metal), 2.83 s
   for LiteRT-LM (ML Drift GPU) and 3.59 s for BaseRT. Relative to LiteRT-LM,
   decode throughput is 1.14x for MLX (118.5 vs 103.7 tok/s), 1.06x for
   llama.cpp (110.3) and 0.75x for BaseRT (77.6), all with tight 95% bootstrap
   intervals. These are reference configurations with different ~4-bit
   artifacts, not a matched-weight or tuned comparison.
2. **Prefill depends on input length, and the order of the frameworks changes
   with it.** At 1,024 tokens (2 s idle before each request), MLX prefills at
   4,002 tok/s, llama.cpp 3,413, BaseRT 3,369 and LiteRT-LM 2,752. LiteRT-LM is
   slowest on short prompts (551 tok/s at N = 128) but scales best. It passes
   llama.cpp from N = 4,096 and reaches 3,201 tok/s at N = 16,384, against
   2,333 for llama.cpp. MLX prefills fastest from N = 1,024 upwards (4,303 tok/s
   at 16,384); llama.cpp is fastest at N <= 512.
3. **LiteRT-LM and BaseRT hold decode speed best as context grows.** From
   N = 128 to 16,384 with 512-token decode, LiteRT-LM drops 5% (100.0 -> 94.8
   tok/s) and BaseRT 5% (82.1 -> 77.9), against 19% for llama.cpp
   (116.5 -> 93.9) and 34% for MLX (126.8 -> 83.8). At 16,384 tokens, LiteRT-LM
   and llama.cpp decode fastest.
4. **Accuracy is equivalent across all four frameworks**, so the speed
   comparison is valid. MMLU (2,000 items) is 78.4–79.5% and GSM8K (250 items)
   93.6–94.0%. No paired difference is significant (McNemar p >= 0.06 for
   all 12 pairs). All four are reproducible per request, greedy and seeded,
   within and across processes and thread counts, and all use identical
   prompt token ids.
5. **Resource use differs more than speed.** Host CPU time per request is
   0.15 s for BaseRT, 0.59 s for LiteRT-LM, 0.59 s for llama.cpp and 1.87 s for
   MLX, which keeps 0.83 cores busy during decode. GPU memory above the
   pre-load baseline is 14.9 GB for llama.cpp and MLX, 15.2 GB for LiteRT-LM
   and 17.3 GB for BaseRT. Warm-cache load time is 0.8 s for BaseRT, 2.1 s for
   MLX, 2.6 s for llama.cpp and 8.3 s for LiteRT-LM.
6. **One LiteRT-LM process hit a GPU hang and silently returned invalid
   output.** At N = 16,384 (sweep round 3), a LiteRT-LM command buffer caused a
   GPU hang (`kIOGPUCommandBufferCallbackErrorHang`). macOS restarted the GPU
   and ignored the process's further command buffers, but LiteRT-LM raised no
   error. It kept returning stale, slightly corrupted text with impossible
   timings (a 16,384-token prefill "in 12 ms" by its own counter). The harness
   now checks the macOS unified log after every native run. The process is
   excluded (that point has n = 10 instead of 15), not replaced. This was 1 of
   30 LiteRT-LM GPU timing processes (its 9 accuracy and reproducibility
   processes were clean) and did not recur in 3 immediate reruns of the same
   point. No other process in the session, of any framework, logged a GPU
   error.
7. **TTFT depends on how long the GPU has been idle** (see Caveats). After
   0.5 s idle, LiteRT-LM prefills ~22% faster than in the headline. After
   5 s, MLX prefill halves. Decode is unaffected.

## Caveats

**Scope and model equivalence**
- This is an open/reference comparison: each framework runs its own published
  ~4-bit artifact of Gemma 4 26B A4B with framework defaults. Quantization
  differs (LiteRT-LM GPU package: QAT, blockwise int4 weights, float
  activations; Google QAT Q4_0 GGUF; LM Studio QAT MLX: 4-bit g64 experts,
  8-bit dense MLP and router; BaseRT catalog `default-q4`). No weight
  equivalence was established, so speed must be read together with the
  accuracy gate. The four configurations are not tuned "fastest" settings.
- Prompt token ids are identical in every configuration (the hash in the
  Environment table). The prompt uses the 26B A4B chat template's non-thinking
  generation prompt, which ends with an empty thought channel
  (`<|channel>thought\n<channel|>`). The E2B suites don't use it, because the
  E2B template has no such block.

**Environment and measurement**
- Native macOS 27.0.1 on the Apple M5 Max (40-core GPU, 64 GB unified memory,
  AC power, low-power mode off), not a VM. These results are not comparable
  with the Linux-VM CPU reports.
- The machine and display were kept awake for the whole session
  (`caffeinate -dims`, plus the runner's own assertion), and `pmset -g log`
  shows no sleep or wake events in the run window. VS Code (which hosted the
  session) and the window server stayed open; mean system GPU utilization
  between requests was 6–10%, and GPU in-use memory before each model load was
  a stable ~0.81 GB.
- **GPU utilization is the system-wide IOAccelerator counter**, used as an
  indicator only. It reads 99–100% at saturation and ~0% at idle, but
  overstates idle gaps shorter than ~1 s (a 25% duty cycle at 100 ms periods
  read as 51%). It is not per process, and there is no GPU power or energy
  figure: `powermetrics` needs root. *CPU energy* is the kernel's per-process
  CPU estimate and excludes GPU work.
- **Memory:** compare frameworks on *GPU in-use memory above the pre-load
  baseline* (system-wide). `phys_footprint` includes MLX's Metal buffers but
  not llama.cpp's no-copy mmap'd weights, nor LiteRT-LM's or BaseRT's GPU
  weights, so footprint alone understates their memory use.
- llama.cpp's driver timestamps tokens with `CLOCK_UPTIME_RAW` on macOS, the
  clock the monitor (Python 3.12 `time.monotonic_ns`) uses; libc++'s
  `steady_clock` there also counts sleep. Every request passed the
  clock-domain check.

**Workload and framework behaviour**
- Headline: 1,024-token real-text prompt -> 256 greedy tokens (stop tokens
  ignored), context 4,096, 2 warm-up + 5 measured requests per process, 3
  interleaved processes per configuration, 2 s idle between requests and 10 s
  between processes.
- **TTFT depends on how long the GPU idled before the request.** In a separate
  idle-gap experiment (4 measured requests per point; runs labelled `idlegap-*`,
  not part of the headline), median prefill throughput in tok/s was:

  | Idle before request | `litert-lm/gpu` | `llama.cpp/metal` | `mlx/metal` | `basert/metal` |
  | --- | --- | --- | --- | --- |
  | 0.5 s | 3,348 | 3,445 | 4,202 | 3,549 |
  | 2 s (headline) | 2,662 | 3,420 | 3,971 | 3,405 |
  | 5 s | 2,644 | 3,410 | 1,965 | 3,374 |

  Decode throughput did not change. The headline uses 2 s, so it is neither
  the back-to-back best case nor the cold-GPU case.
- **Fixed-length decoding differs in mechanism, not work.** llama.cpp and MLX
  feed a sampled stop token back, and LiteRT-LM uses its engine's
  `num_decode_tokens`. BaseRT stops on stop tokens internally and has no
  ignore-EOS option, so its driver counts the stop token and resumes with
  `baseRT_generate_continue` (`eos_resumed` in the raw data). A logit bias on
  the stop ids was rejected because it moved BaseRT off its fast decode path
  (77.6 -> 42–58 tok/s).
- **BaseRT streams in bursts.** Tokens reach the caller in groups of about
  four (inter-token p50 ≈ 0.006 ms, p99 ≈ 52.5 ms). Decode throughput is
  unaffected, but TTFT is measured when the first token is available to the
  caller, which can be part of a burst. With the release's standard layout, the
  v0.4.0 engine logs `get_pipeline: no function named
  'moe_simd_gelu_gate_up_q4_sbf16'`. Neither the standalone nor the embedded
  kernel library contains that kernel, and loading only the embedded copy
  gave the same speed and output.
- **Load times depend on the page cache.** The four artifacts total ~62 GB on
  a 64 GB machine, so a model may have been partly evicted when its process
  started (e.g. one LiteRT-LM load took 48 s against ~8.2 s otherwise).
  LiteRT-LM GPU wrote no program cache to its `--cache-dir`.
- MLX grows its KV cache on demand; the other engines allocate for the
  configured context (4,096 headline; 16,896 sweep).

**Native-tool cross-check (same 1,024 / 256 shape)**

Each framework's own benchmark tool, on the same model files (raw output
under `results/apple-m5-max/native-tools/`). The harness rows are the headline
medians.

| Config | Native tool | Tool prefill / decode (tok/s) | Harness prefill / decode (tok/s) | Explanation |
| --- | --- | --- | --- | --- |
| `litert-lm/gpu` | `litert-lm benchmark --backend gpu -p 1024 -d 256` | 1,219 / 107.4 | 2,752 / 103.7 | The tool creates a fresh engine per iteration, so each timed prefill is an engine's first prefill (~0.85 s; the harness's unmeasured first warm-up is ~0.88–0.94 s). Decode within 4%. |
| `llama.cpp/metal` | `llama-bench -ngl 999 -p 1024`; `-n 256 -d 1024` | 3,592 / 121.5 | 3,413 / 110.3 | Random token ids; excludes tokenization, and llama-bench does not copy the full 262k-entry logits to the host for an arg-max every token, which the driver does. |
| `mlx/metal` | `mlx_lm.benchmark -p 1024 -g 256` | 4,585 / 119.5 | 4,002 / 118.5 | Back-to-back trials with no idle gap (cf. 4,202 tok/s at 0.5 s idle); random prompt tokens. |
| `basert/metal` | `basert-bench -p 1024 -n 256 --tg-at 1024 --fill corpus:<harness passage>` | 3,879 / 80.4 (92.1 from an empty context) | 3,369 / 77.6 | Back-to-back repetitions (cf. 3,549 tok/s at 0.5 s idle); decode is best-of-5, against the harness median; excludes tokenization. |

**Accuracy and output reproducibility**
- Accuracy uses the same deterministic 2,000-item MMLU subset and 250-item
  GSM8K subset as the earlier reports. Prompts are zero-shot, greedy and
  non-thinking, with MMLU answer-prefilled. Neither score is comparable with
  Google's published (thinking-mode, bf16) scores, which are for other
  benchmarks.
- On ~3–5% of MMLU items, 26B A4B hedges ("not explicitly provided in the
  options …") within the 8-token answer budget. These are scored as wrong
  parse failures, not dropped, on identical items for every framework.
- Accuracy and reproducibility runs overlapped with model downloads (network
  and disk only), as the methodology allows; timing runs did not. The
  recorded harness digest varies across runs because of report-rendering
  changes and driver fixes made during the session. Each framework's driver
  was unchanged across all of that framework's accuracy, reproducibility and
  timing processes: the MLX BOS fix and the BaseRT EOS-resume and config fixes
  came before any of those frameworks' recorded runs.
