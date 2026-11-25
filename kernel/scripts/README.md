# Kernel Scripts

Execution scripts, microkernel benchmark harnesses, extractors, and report generators for Tier 1 (`kernel/`):

- [bench_litert_microkernels.c](bench_litert_microkernels.c) — Repeatable standalone C/OpenMP microkernel benchmark harness (x86_64 Intel Sapphire Rapids) that resolves the exact load-bearing XNNPACK (`xnn_*_ukernel*`) and YNNPACK (`ynn::*`) microkernels directly inside `litert_lm_advanced_main` via `dl_iterate_phdr` (or direct-linked with `-DAEB_DIRECT_LINK`) and benchmarks all **7 matched load-bearing kernel pairs** across their canonical Gemma 4 31B `(M, N, K)` shapes on `1T`, `28T`, and `56T` with 7-trial statistics (`median`, `mean`, `min`, `max`, `stdev`, `cv_pct`).
- [bench_litert_microkernels_arm64.c](bench_litert_microkernels_arm64.c) — Repeatable standalone ARM64 / Apple Silicon (`pthreads` + `QOS_CLASS_USER_INTERACTIVE` + lock-free adaptive dynamic tile dispatch) microkernel benchmark harness linking upstream XNNPACK, KleidiAI SME2, and YNNPACK SME2/SME/NEON microkernels directly, benchmarking all **7 matched Gemma 4 31B kernel pairs** plus controlled **NEON I8MM / DotProd / FP32 FMA baselines** across `1T`, `6T`, and `18T` with 19 built-in pre-timing numerical smoke checks and 7-trial statistics.
- [verify_litert_microkernels.c](verify_litert_microkernels.c) — Standalone C/OpenMP numerical precision and quantization fidelity verification harness (x86_64) that verifies each microkernel against:
  1. **Level A (Exact Packed-Layout Microkernel Reference)**: Bit-exact / ULP verification of the SIMD/AMX microkernel against a scalar `INT64` / `FP64` reference operating on the exact same quantized inputs and packed weight layout (`max_abs_err`, `max_rel_err`, `max_ulp_diff`, `sqnr_db`, `cosine_sim`, `exact_match_pct`, `nan_inf_count`, `status`).
  2. **Level B (FP64 Golden Operator Reference)**: End-to-end quantization + microkernel fidelity against unquantized `FP64` matrix multiplication $C_{\text{fp64}} = A_{\text{fp64}} B_{\text{fp64}}^T + \text{bias}_{\text{fp64}}$ (plus Softmax simplex & shift-invariance checks and 32-bit pack/transpose bit-exactness checks) across 4 input distributions (`normal`, `saturation`, `outliers`, `tails`) in [ukernel_precision_sapphire_rapids.json](../reports/2026-11-10-gemma4-31b-sapphire-rapids/ukernel_precision_sapphire_rapids.json).
- [verify_litert_microkernels_arm64.c](verify_litert_microkernels_arm64.c) — Standalone ARM64 / Apple Silicon two-level (Level A + Level B) numerical precision verifier testing all 14 load-bearing XNNPACK (KleidiAI SME2 / NEON) and YNNPACK (SME2 / SME / NEON) microkernels across `normal`, `saturation`, `outliers`, and `tails` distributions ([ukernel_precision_apple_m5_max.json](../reports/2026-11-25-gemma4-31b-apple-m5-max/ukernel_precision_apple_m5_max.json)).
- [generate_kernel_report.py](generate_kernel_report.py) — Architecture-aware report generator supporting both **Intel Sapphire Rapids (`x86_64`)** and **Apple M5 Max (`arm64`)**: combines empirical hardware ceilings (`ceilings_*.json`), microkernel benchmark measurements (`ukernel_benchmarks_*.json`), and numerical precision results (`ukernel_precision_*.json`) to emit unified `kernel_report_*.json` and `kernel_report_*.md` reports (e.g., [2026-11-25 Apple M5 Max](../reports/2026-11-25-gemma4-31b-apple-m5-max/kernel_report_gemma4_31b_apple_m5_max.md) and [2026-11-17 Sapphire Rapids](../reports/2026-11-17-gemma4-31b-sapphire-rapids/kernel_report_gemma4_31b_sapphire_rapids.md)).
- [update_dashboard.py](update_dashboard.py) — Aggregates all dated `kernel/reports/YYYY-MM-DD-<suite>/` reports over time and generates the interactive HTML/SVG dashboard ([../dashboards/index.html](../dashboards/index.html)), structured time-series JSON ([../dashboards/data.json](../dashboards/data.json)), and GitHub Markdown dashboard ([../dashboards/README.md](../dashboards/README.md)).
- [extract_litert_lm_kernels.py](extract_litert_lm_kernels.py) — Reproducible pipeline that profiles `.litertlm` models across **Text (`p1_d1` to `p16384_d128`)**, **Vision (`vis_70` to `vis_1120`, 27-layer ViT + Adapter)**, and **Audio (`audio_204`, Conformer Encoder + Adapter)** user-journey and boundary (minimum/maximum) input shapes for both **XNNPACK** (`--enable_ynnpack=false`) and **YNNPACK** (`--enable_ynnpack=true`), capturing:
  1. **TFLite / LiteRT operator-level & delegate-node profiles** (`--enable_profiling=true`), including invocation counts, total latency (ms), and exact execution time proportions (`%`) for every operator and delegate kernel across all subgraphs (`tf_lite_prefill_decode`, `tf_lite_embedder`, `vision_encoder`, `vision_adapter`, `audio_encoder`, `audio_adapter`).
  2. **Hardware ISA microkernel profiles** via Linux `perf` (`--with-perf`, `perf record -e cycles:u`), attributing CPU cycle shares (`%`) to individual XNNPACK (`xnn_*_ukernel*`) and YNNPACK (`ynn::*`) symbols and their ISA targets (`AMX-INT8`, `AMX-BF16`, `AVX-512 VNNI`, `AVX-512F`, `FMA3`, `Scalar`).
  3. **JSON + Markdown reports** ([kernels_gemma4_31b_sapphire_rapids.json](../reports/2026-11-10-gemma4-31b-sapphire-rapids/kernels_gemma4_31b_sapphire_rapids.json) and [kernels_gemma4_31b_sapphire_rapids.md](../reports/2026-11-10-gemma4-31b-sapphire-rapids/kernels_gemma4_31b_sapphire_rapids.md)) with per-shape kernel tables, cross-shape time-proportion breakdowns, delegate operator parameter catalogs, and uprobe hardware microkernel parameter tables.
- [extract_tflite_params.py](extract_tflite_params.py) — Zero-dependency TFLite v3 FlatBuffer parser that extracts every operator's exact input/weight/output tensor shapes, data types (`f32`, `i8`, `i4`, `i2`, `i32`, `bool`), per-channel quantization metadata, and operator options (`M, N, K`, `adj_x/adj_y`, `stride`, `filter`, `axes`, `perm`, `new_shape`) from `.litertlm` files, recursively inlining `STABLEHLO_COMPOSITE` decomposition subgraphs (`subgraph_operator_params.json`).
- [litert_subgraph_profiler.c](litert_subgraph_profiler.c) — `LD_PRELOAD` LiteRT C-API profiler hook (`LiteRtCreateCompiledModel`, `LiteRtRunCompiledModel`, `LiteRtGetProfileSummary`) used with `/opt/litert-lm/bin/litert_lm_advanced_main` to benchmark and profile standalone multimodal subgraphs (`vision_encoder`, `vision_adapter`, `audio_encoder`, `audio_adapter`) embedded inside `.litertlm` flatbuffer bundles with `enable_ynnpack = false/true` and `enable_profiling = true`.

## Usage (`bench_litert_microkernels.c`, `verify_litert_microkernels.c` & `generate_kernel_report.py`)

```bash
# 1. Compile and run the standalone microkernel benchmark suite on the target Sapphire Rapids host:
gcc -shared -fPIC -O3 -march=sapphirerapids -fopenmp \
    kernel/scripts/bench_litert_microkernels.c -o /tmp/libbench_ukernels.so -lm

LD_LIBRARY_PATH=/opt/litert-lm/lib:/opt/litert-lm/bin \
  LD_PRELOAD=/tmp/libbench_ukernels.so \
  AEB_BENCH_THREADS=28,56 \
  AEB_BENCH_OUTPUT=kernel/reports/YYYY-MM-DD-gemma4-31b-sapphire-rapids/ukernel_benchmarks_sapphire_rapids.json \
  /opt/litert-lm/bin/litert_lm_advanced_main

# 2. Compile and run the standalone microkernel numerical precision verification suite:
gcc -shared -fPIC -O3 -march=sapphirerapids -fopenmp \
    kernel/scripts/verify_litert_microkernels.c -o /tmp/libverify_ukernels.so -lm

LD_LIBRARY_PATH=/opt/litert-lm/lib:/opt/litert-lm/bin \
  LD_PRELOAD=/tmp/libverify_ukernels.so \
  AEB_VERIFY_OUTPUT=kernel/reports/YYYY-MM-DD-gemma4-31b-sapphire-rapids/ukernel_precision_sapphire_rapids.json \
  /opt/litert-lm/bin/litert_lm_advanced_main

# 3. Generate the JSON and Markdown kernel report (roofline + precision):
python3 kernel/scripts/generate_kernel_report.py \
    --report-dir kernel/reports/YYYY-MM-DD-gemma4-31b-sapphire-rapids
```

## Usage (`extract_litert_lm_kernels.py`)

```bash
# 1. Run live profiling on a remote Sapphire Rapids machine:
python3 kernel/scripts/extract_litert_lm_kernels.py \
    --ssh user@sapphire-rapids-host \
    --model /models/litert-community/gemma-4-31B-it-litert-lm/gemma-4-31B-it.litertlm \
    --threads 28 \
    --with-perf \
    --out-dir kernel/reports/YYYY-MM-DD-gemma4-31b-sapphire-rapids

# 2. Re-generate JSON and Markdown reports from saved raw logs (offline / reproducible):
python3 kernel/scripts/extract_litert_lm_kernels.py \
    --from-logs kernel/reports/2026-11-10-gemma4-31b-sapphire-rapids/raw \
    --out-dir kernel/reports/2026-11-10-gemma4-31b-sapphire-rapids
```
