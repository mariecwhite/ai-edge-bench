# Layer Runner & Roofline Scripts (`layer/scripts/`)

This directory contains orchestration and analysis scripts for the Layer tier:

- **Subgraph Benchmark Runner:** Executes LiteRT Tensor API and competitor subgraphs across thread counts and sequence/context lengths, capturing latency distributions (`p50`, `p90`, `p99`), cache spill counters, and peak buffer footprint.
- **Whole-Subgraph Roofline Calculator:** Computes theoretical minimum subgraph latency $T_{\text{min}}$ from total subgraph FLOPs ($F_{\text{subgraph}}$), boundary-only memory bytes ($B_{\text{min\_subgraph}}$), and Level 1 hardware probe ceilings ($C_{\text{prac}}, BW_{\text{prac}}$).
- **Numerical Drift & Sensitivity Evaluator:** Compares subgraph output tensors against `FP32` golden references (cosine similarity, $L_2$ relative error, KL-divergence) to qualify mixed-precision and fusion schedules.
