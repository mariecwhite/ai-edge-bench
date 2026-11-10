# Kernel Scripts

Execution scripts, microkernel extractors, and report generators for Tier 1 (`kernel/`):

- [extract_litert_lm_kernels.py](extract_litert_lm_kernels.py) — Reproducible pipeline that profiles `.litertlm` models across **Text (`p1_d1` to `p16384_d128`)**, **Vision (`vis_70` to `vis_1120`, 27-layer ViT + Adapter)**, and **Audio (`audio_204`, Conformer Encoder + Adapter)** user-journey and boundary (minimum/maximum) input shapes for both **XNNPACK** (`--enable_ynnpack=false`) and **YNNPACK** (`--enable_ynnpack=true`), capturing:
  1. **TFLite / LiteRT operator-level & delegate-node profiles** (`--enable_profiling=true`), including invocation counts, total latency (ms), and exact execution time proportions (`%`) for every operator and delegate kernel across all subgraphs (`tf_lite_prefill_decode`, `tf_lite_embedder`, `vision_encoder`, `vision_adapter`, `audio_encoder`, `audio_adapter`).
  2. **Hardware ISA microkernel profiles** via Linux `perf` (`--with-perf`, `perf record -e cycles:u`), attributing CPU cycle shares (`%`) to individual XNNPACK (`xnn_*_ukernel*`) and YNNPACK (`ynn::*`) symbols and their ISA targets (`AMX-INT8`, `AMX-BF16`, `AVX-512 VNNI`, `AVX-512F`, `FMA3`, `Scalar`).
  3. **JSON + Markdown reports** ([kernels_gemma4_31b_sapphire_rapids.json](../reports/gemma4-31b-sapphire-rapids/kernels_gemma4_31b_sapphire_rapids.json) and [kernels_gemma4_31b_sapphire_rapids.md](../reports/gemma4-31b-sapphire-rapids/kernels_gemma4_31b_sapphire_rapids.md)) with per-shape kernel tables, cross-shape time-proportion breakdowns, delegate operator parameter catalogs, and uprobe hardware microkernel parameter tables.
- [extract_tflite_params.py](extract_tflite_params.py) — Zero-dependency TFLite v3 FlatBuffer parser that extracts every operator's exact input/weight/output tensor shapes, data types (`f32`, `i8`, `i4`, `i2`, `i32`, `bool`), per-channel quantization metadata, and operator options (`M, N, K`, `adj_x/adj_y`, `stride`, `filter`, `axes`, `perm`, `new_shape`) from `.litertlm` files, recursively inlining `STABLEHLO_COMPOSITE` decomposition subgraphs (`subgraph_operator_params.json`).
- [litert_subgraph_profiler.c](litert_subgraph_profiler.c) — `LD_PRELOAD` LiteRT C-API profiler hook (`LiteRtCreateCompiledModel`, `LiteRtRunCompiledModel`, `LiteRtGetProfileSummary`) used with `/opt/litert-lm/bin/litert_lm_advanced_main` to benchmark and profile standalone multimodal subgraphs (`vision_encoder`, `vision_adapter`, `audio_encoder`, `audio_adapter`) embedded inside `.litertlm` flatbuffer bundles with `enable_ynnpack = false/true` and `enable_profiling = true`.

## Usage (`extract_litert_lm_kernels.py`)

```bash
# 1. Run live profiling on a remote Sapphire Rapids machine:
python3 kernel/scripts/extract_litert_lm_kernels.py \
    --ssh user@sapphire-rapids-host \
    --model /models/litert-community/gemma-4-31B-it-litert-lm/gemma-4-31B-it.litertlm \
    --threads 28 \
    --with-perf \
    --out-dir kernel/reports/gemma4-31b-sapphire-rapids

# 2. Re-generate JSON and Markdown reports from saved raw logs (offline / reproducible):
python3 kernel/scripts/extract_litert_lm_kernels.py \
    --from-logs kernel/reports/gemma4-31b-sapphire-rapids/raw \
    --out-dir kernel/reports/gemma4-31b-sapphire-rapids
```

## Planned Additional Scripts

- `run_suite.py` — Reads a `kernel/suites/*.json` definition, compiles or invokes the corresponding `kernel/probes/` binaries with `taskset` / `numactl` CPU affinity, and writes JSONL results to `kernel/results/<machine>/<suite>/`.
- `report.py` — Aggregates `kernel/results/` into Markdown/HTML tables and roofline charts in `kernel/reports/` (GB/s achieved vs. STREAM Triad ceiling; GFLOPS/GIOPS achieved vs. ISA peak).
