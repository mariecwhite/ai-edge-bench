## Key findings

1. **LiteRT-LM + YNNPACK has the lowest end-to-end latency among the
   three tested configurations:** 4.28 s for
   1,024 -> 256 tokens, versus 6.15 s for stock llama.cpp Q4_0 and 18.33 s for
   ONNX Runtime GenAI Q4_K_M. These are independently quantized reference
   configurations, not a matched-weight comparison or an exhaustive tuning
   result.
2. **llama.cpp decodes slightly faster; LiteRT-LM prefills much faster.**
   Decode is 77.75 versus 76.29 tok/s. Prefill is 357.7 versus 1,092.9 tok/s.
   ONNX Runtime reaches 155.9 tok/s prefill and 21.66 tok/s decode.
3. **ONNX Runtime's accuracy is close to stock llama.cpp on these subsets.**
   MMLU: 60.25% versus 60.65%; GSM8K: 77.60% versus 78.40%.
   LiteRT-LM scores 57.20% and 76.00%. Inspect the confidence intervals and
   differing model artifacts; this does not establish numerical equivalence.
4. **Resource use differs substantially.** Median peak request RSS is
   2,839 MB for LiteRT-LM, 4,365 MB for llama.cpp and 6,658 MB for ONNX Runtime.
   CPU time per request is 26.5, 64.6 and 217.1 CPU-seconds respectively.
5. **Completion and reproducibility:** all nine timing processes completed,
   giving 15 measured requests per configuration; all six accuracy jobs
   completed (2,000 MMLU and 250 GSM8K items each). Greedy outputs are
   reproducible across repeats, processes and tested thread counts. ONNX
   Runtime and llama.cpp seeded sampling is reproducible per request;
   LiteRT-LM remains reproducible per process only.

## Caveats

**Scope and model equivalence**
- This is an open/reference comparison of independently published Gemma 4 E2B
  artifacts, not a matched-weight or matched-quantization comparison. The
  original matched-weight study is preserved in the 2026-10-01 report.
- LiteRT-LM uses the mobile QAT `.litertlm`, llama.cpp uses the stock ggml-org
  Q4_0 QAT GGUF, and ONNX Runtime GenAI uses the Mobius Q4_K_M/default package.
  Labels such as Q4_0 and Q4_K_M do not
  establish numerical equivalence. Accuracy must be interpreted alongside
  speed.
- LiteRT-LM YNNPACK uses the previously selected 8 threads; the other
  configurations use 12. ONNX Runtime has not had a tuning sweep, so
  its results must not be described as its fastest possible configuration.

**Environment and reproducibility**
- All runs execute in the same Apple M5 Max Linux/ARM VM, with 12 vCPUs and
  about 31 GiB RAM. Metal and the Apple Neural Engine are not available.
  macOS controls vCPU placement and frequency; these are not native macOS
  results.
- Timing processes execute sequentially, with three interleaved rounds, two
  warm-ups and five measured requests per process. The suite runner holds a
  macOS no-idle-sleep assertion. Accuracy and reproducibility follow timing;
  neither runs concurrently with timed requests.
- The shared harness is mounted read-only with the suite runner's `--dev`
  option into every engine image. This uses the same on-disk harness source
  across the original and new runtime images; each run records its harness
  digest and driver build metadata.
- Model packages are pinned to the revisions documented in `docs/harness.md`.
  ONNX package hashes include graph, weight and tokenizer assets, not
  only their JSON configuration.

**Workload and measurement**
- The workload is a realistic 1,024-token chat prompt, including BOS, followed
  by exactly 256 greedy generated tokens with stop tokens ignored, and a
  4,096-token context limit. Prompt identity is checked using token-ID hashes.
- The LiteRT-LM artifact has a 1,024-token prefill signature. That prompt
  length may favour it relative to lengths requiring padding or chunking.
- Weight caches are primed before timing; cold-start performance is not
  measured. Model-load times are reported separately.
- Peak RSS includes file-backed model pages. The ONNX package contains
  multimodal assets, but this suite sends text only.
- ONNX Runtime enforces a 4,096-token request limit; its internal cache
  allocation policies need not match llama.cpp or LiteRT-LM. A request limit
  alone is not proof of identical reserved KV-cache memory.

**Accuracy and output reproducibility**
- Accuracy uses the same deterministic 2,000-item MMLU subset and 250-item
  GSM8K subset for all configurations. MMLU is generative, zero-shot and
  answer-prefilled; GSM8K is greedy zero-shot chain-of-thought. Neither score
  is directly comparable with published results using different setups.
- GSM8K retains the harness's 512-token generation cap. Truncated generations
  are reported; this cap can penalize models that need longer answers.
- Accuracy runs execute sequentially at each configuration's thread count,
  avoiding assumptions about thread-invariant outputs in new runtimes.
- LiteRT-LM v0.17.1 initializes its sampler once per engine and reuses its RNG
  across requests. Seeded output reproducibility must be interpreted per
  process, not assumed per request.
