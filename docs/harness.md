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
| [perf.py](../docker/harness/aeb/perf.py) | framework images | One timed process: load, warm-up, measured requests, resource sampling. |
| [monitor.py](../docker/harness/aeb/monitor.py) | framework images | 50 ms `/proc` + cgroup sampler for the driver process. |
| [accuracy.py](../docker/harness/aeb/accuracy.py) | framework images | MMLU / GSM8K validity gate through the same drivers. |
| [repro.py](../docker/harness/aeb/repro.py) | framework images | Output reproducibility (greedy and seeded) across repeats, processes and thread counts. |
| [workload.py](../docker/harness/aeb/workload.py) | framework images | Token-exact chat prompt from a public-domain passage. |
| [prep/](../docker/harness/aeb/prep) | tools image | Matched-weights GGUF conversion and `.litertlm` parity proof; `hf_reference.py` is an optional diagnostic that reruns the PyTorch reference of the QAT checkpoint with/without its activation quantization (`make diagnose-activations`, image `docker/tools/hf.Dockerfile`). |
| [datasets.py](../docker/harness/aeb/datasets.py) | tools image | Pinned MMLU / GSM8K staging into `./datasets`. |
| [report.py](../docker/harness/aeb/report.py) | tools image | Report, charts and performance history. |
| [scripts/suite.py](../scripts/suite.py) | host | Runs a [suite](../suites) one container at a time, interleaved. |

Runtime modules use only the Python standard library, so the framework images
need nothing beyond `python3`.

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

```bash
make MACHINE=apple-m5-max images tools     # framework + tools images
make MACHINE=apple-m5-max models           # stock GGUF + .litertlm
make MACHINE=apple-m5-max matched-models   # QAT checkpoint -> matched GGUFs (+ proofs)
make MACHINE=apple-m5-max datasets         # MMLU + GSM8K, pinned revisions
make MACHINE=apple-m5-max suite-perf       # prime caches, 3 interleaved rounds
make MACHINE=apple-m5-max suite-accuracy   # validity gate
make MACHINE=apple-m5-max suite-repro      # output reproducibility
make MACHINE=apple-m5-max report           # reports/<date>-<suite>-<machine>/
```

`suites/<suite>.json` defines the configurations. The `machines.<machine>`
section holds the per-machine thread count and the tuned "fastest" settings
together with how they were selected; add one before running a new machine.
