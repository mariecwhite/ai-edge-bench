## Key findings

1. **Same weights, proven.** The `.litertlm` and Google's mobile QAT checkpoint
   hold identical integers and per-channel scales in all 278 quantized text
   tensors. The matched GGUF re-encodes those integers losslessly (INT4→Q4_0,
   INT8→Q8_0, INT2→Q2_K). The only stored difference is fp16 instead of fp32
   per-row scales: ≤0.05% relative error, ≤0.26% for 4 tensors with subnormal
   scales.
2. **Matched track, 12 threads.** The frameworks split the work very
   differently, and end-to-end latency for 1,024 → 256 tokens comes out
   equal (7.86 s vs 7.89 s):
   - LiteRT-LM prefills 4.2× faster: 1,014 vs 241 tok/s, TTFT 1.0 s vs 4.3 s.
   - llama.cpp decodes 1.9× faster: 70.2 vs 37.2 tok/s.
   - llama.cpp burns 2.4× the CPU time: 85.8 vs 35.1 CPU-s per request. Its
     12 worker threads keep ~10 cores busy while decoding; LiteRT-LM's decode
     saturates at ~4 cores.
3. **Fastest track** (each framework tuned, accuracy-gated): LiteRT-LM with
   YNNPACK at 8 threads is the fastest end to end.
   - LiteRT-LM: 4.43 s end to end, 1,004 tok/s prefill, 74.5 tok/s decode,
     27.8 CPU-s, 2.85 GB peak RSS.
   - llama.cpp: 5.16 s end to end, 506 tok/s prefill, 81.5 tok/s decode,
     52.1 CPU-s, 4.35 GB peak RSS. This uses the same integer weights, with
     the INT2 tensors stored as Q4_0 so llama.cpp's repacked Arm kernels
     apply.
   - llama.cpp still decodes ~9% faster; LiteRT-LM prefills 2× faster and
     uses half the CPU time and 1.5 GB less memory.
4. **Accuracy is not equal, even with identical weights.**
   - Full MMLU (14,042 items): LiteRT-LM 56.9% vs llama.cpp 50.1%. On the
     items only one of them answered correctly, McNemar p = 2e-67.
   - GSM8K (250 items): 77.2% vs 46.8%.
   - The same llama.cpp build with the stock ggml-org Q4_0 GGUF, a different
     QAT checkpoint, scores 60.7% on the 2,000-item MMLU subset. So
     llama.cpp's Gemma 4 implementation is not the problem.
   - Cause: the mobile checkpoint was trained with static int8 activation
     rounding (SRQ). LiteRT-LM executes it; llama.cpp cannot express it.
   - The PyTorch reference diagnostic, on 120 items where the frameworks
     disagree, supports this:
     - With SRQ on, the reference matches LiteRT-LM's answer 72% of the time
       and llama.cpp's 23%.
     - With SRQ off, it matches llama.cpp 57% and LiteRT-LM 33%.
     - The reference's own accuracy drops from 53% to 38% when SRQ is
       removed.
     - The match is not perfect: the int8 KV cache and kernel-level
       differences remain.
   - **Practical consequence:** this checkpoint is meant for LiteRT-LM.
     llama.cpp users should use the Q4_0 QAT release, at a different
     memory/speed point (see `llama.cpp/stock-q4_0`).
5. **Reproducibility.**
   - Greedy decoding is bit-identical in every configuration of both
     frameworks, across repeats, processes and thread counts.
   - Seeded sampling is reproducible per request in llama.cpp. In LiteRT-LM
     v0.17.1 it is reproducible only per process: the sampler (and its RNG)
     is created once per engine from the first request's settings. The same
     prompt sampled twice in one process diverges after a median of 10
     tokens, and later requests cannot change the sampling settings.

## Caveats

**Environment**
- **Virtualized host.** Colima Linux VM with 12 vCPUs and 31 GiB on an
  18-core Apple M5 Max (6 + 12 cores by `hw.perflevel`). macOS schedules the
  vCPUs; frequency, pinning and SMT cannot be controlled from inside the VM.
  Metal and the Neural Engine are unreachable. These numbers are not
  comparable with native macOS runs or with the LiteRT-LM model card's macOS
  CPU figures, which use 4 threads and a 2,048-token context.
- **Host sleep during one timing process.** macOS slept from 11:37:29 to
  11:44:14 (local time; `pmset -g log`) during round 0 of
  `llama.cpp/fastest`. The VM's monotonic clock stops while asleep, so it
  shows up only as one slow request (TTFT 3.4 s vs ~2.0 s). The outlier
  check flags it, and the medians are unaffected. The suite runner now holds
  a `caffeinate` assertion.

**Numerics**
- **Identical weights ≠ identical numerics.** LiteRT-LM applies the
  checkpoint's static per-tensor int8 activation quantization and an int8 KV
  cache with fixed scales. llama.cpp quantizes activations dynamically per
  32/256-element block and uses a Q8_0 KV cache in the matched track (f16 in
  fastest/reference). Greedy outputs of the two frameworks diverge after a
  median of 6 tokens on the reproducibility prompts, and from the first
  token on the 1,024-token timing prompt.
- **Kernel coverage drives the llama.cpp prefill result.** llama.cpp has no
  repacked/i8mm path for Q2_K on Arm. The INT2 layers (the double-wide MLPs
  of layers 15–34, most of the compute) therefore run slowly in the
  storage-matched GGUF. Storing the same values as Q4_0 doubles llama.cpp
  prefill but adds 1.4 GB of peak RSS: +0.95 GB of anonymous repacked
  weights, and a larger file (2.7 GB vs 2.3 GB).

**Threads and CPU use**
- **Thread count changes the ratio.** The matched headline uses all 12
  vCPUs, per the protocol. LiteRT-LM XNNPACK decode peaks at 6 threads
  (45.5 tok/s) and drops to 37 tok/s at 12, while llama.cpp keeps improving
  to 12 threads. At 4 threads the decode ratio is 1.2×, not 1.9× (thread
  scaling chart). Do not quote one ratio without the thread count.
- **CPU time includes busy-waiting.** llama.cpp's thread pool polls between
  operations, which is a real cost on a shared device. Its polling level was
  left at the default.

**Memory**
- RSS includes mmap'd model pages (file-backed, shared, reclaimable).
- LiteRT-LM touches its weights lazily: RSS is 0.44 GB after load and
  2.7 GB after the first request.
- llama.cpp with Q4_0 repacking holds both the mmap'd original and an
  anonymous repacked copy.
- Container `memory.peak` also counts page cache.

**Caches and load time**
- XNNPACK weight caches were primed before measuring.
- YNNPACK writes no weight cache and repacks on every load (2.9 s vs 0.2 s).
- Cold start was not measured.

**Workload definition**
- **Prefill length is a sweet spot for LiteRT-LM.** The `.litertlm` ships
  fixed prefill signatures (128 and 1,024 tokens). The 1,024-token workload
  matches one exactly; other prompt lengths are chunked and padded, and may
  look less favourable.
- **Fixed-length decode.** Both frameworks decode exactly 256 tokens with
  stop tokens ignored: LiteRT-LM through its engine benchmark parameter,
  llama.cpp by not stopping. LiteRT-LM token counts come from its
  BenchmarkInfo; one streamed chunk per token was observed.
- **Greedy in LiteRT-LM** is the TOP_P sampler with `top_k = 1`; its GREEDY
  and TOP_K samplers are unimplemented on CPU at v0.17.1.

**Accuracy setup**
- **MMLU** is generative and zero-shot, with thinking disabled and the model
  turn pre-filled with "The answer is". It is a validity gate, not a
  leaderboard score; Google does not publish MMLU or GSM8K for Gemma 4 E2B.
- **GSM8K** used a 512-token cap; 83 (llama.cpp) and 55 (LiteRT-LM) of 250
  answers hit it. On answers that did not hit the cap: 67% vs 96%. A
  1,024–2,048-token cap is recommended for future runs.
- **Thread count during accuracy.** Accuracy ran two processes side by side
  at 6 threads each. Outputs were verified bit-identical to 12-thread runs
  on 65 items per framework, and the reproducibility suite confirms thread
  invariance.

**Fastest configuration selection**
- The fastest configs were picked on this machine from short sweeps (2
  measured requests per point), minimising median end-to-end latency for this
  workload. A different objective or prompt length could pick differently.

**Native-tool cross-check (agreement within ~5%)**

| | Native tool | Harness |
| --- | --- | --- |
| llama.cpp prefill, 1,024 tokens (tok/s) | 229.7 (`llama-bench`) | 240.5 |
| llama.cpp decode at depth 0 (tok/s) | 88.7 | — |
| llama.cpp decode at depth 1,024 (tok/s) | 74.3 | 70.2 |
| LiteRT-LM prefill (tok/s) | 1,099 (`litert_lm_advanced_main`, zero-padded prompt) | 1,014 |
| LiteRT-LM decode (tok/s) | 37.5 | 37.2 |
| LiteRT-LM TTFT (s) | 0.96 | 1.01 |

The harness's llama.cpp decode also includes greedy arg-max over the
262k-entry vocabulary each step. Run directories:
`results/apple-m5-max/{llama.cpp/native-matched-d*,litert-lm/native-matched}/`.

**Native tool fix**
- `litert_lm_main` at v0.17.1 ignores `--benchmark_prefill_tokens` and
  `--benchmark_decode_tokens`. The native bench script now uses
  `litert_lm_advanced_main` with `--max_num_tokens`.

**History**
- This is the first report, so the history has one point per configuration.
  Trend detection starts with the next run.
