# Level 1 Kernel Benchmark Suite Dashboard — Gemma 4 31B on Intel Sapphire Rapids

- **Interactive HTML Dashboard**: [`kernel/dashboards/index.html`](index.html)
- **Raw Time-Series JSON**: [`kernel/dashboards/data.json`](data.json)
- **Last Updated**: `2026-11-17 09:46 UTC`
- **Tracked Reports**:
  - **`2026-11-10`** (`google/XNNPACK` @ `28d1fc6d`, threads `[1, 28]`): [2026-11-10-gemma4-31b-sapphire-rapids/kernel_report_gemma4_31b_sapphire_rapids.md](../reports/2026-11-10-gemma4-31b-sapphire-rapids/kernel_report_gemma4_31b_sapphire_rapids.md)
  - **`2026-11-13`** (`google/XNNPACK` @ `6e5b0566`, threads `[1, 28, 56]`): [2026-11-13-gemma4-31b-sapphire-rapids/kernel_report_gemma4_31b_sapphire_rapids.md](../reports/2026-11-13-gemma4-31b-sapphire-rapids/kernel_report_gemma4_31b_sapphire_rapids.md)
  - **`2026-11-17`** (`google/XNNPACK` @ `328e0a0e`, threads `[1, 28, 56]`): [2026-11-17-gemma4-31b-sapphire-rapids/kernel_report_gemma4_31b_sapphire_rapids.md](../reports/2026-11-17-gemma4-31b-sapphire-rapids/kernel_report_gemma4_31b_sapphire_rapids.md)

---

## 1. Key Kernel Improvements Over Time (`2026-11-10` → `2026-11-13` → `2026-11-17`)

> [!IMPORTANT]
> **Summary of Major Kernel & Harness Advances Across Tracked Snapshots**:
> 1. **YNNPACK 4-Bit Prefill GEMM (`Pair 2`, `AVX-512 VNNI 12x16x8` → `AMX-INT8+GFNI 32x32x64`)**:
>    - **28-Core Throughput**: Jumped from **`13.89–14.51 TOPS`** (`2026-11-10`) to **`35.84–36.41 TOPS`** (`2026-11-13` / `2026-11-17`) — a **`2.47×–2.59×` (`+147%` to `+159%`) speedup**, surpassing XNNPACK's `16x64c4__avx512amx` (`22.39–22.70 TOPS` @ 28C) by **`1.58×–1.60×`**.
>    - **56-Core Full-Socket Throughput**: Reaches **`71.69–73.76 TOPS`** on `128×21504×5376` (**`5.75×–5.93×` faster** than the `12.44–12.46 TOPS` AVX-512 VNNI baseline and **`1.58×–1.63×` faster** than XNNPACK AMX at `45.20–45.27 TOPS`).
>    - **Single-Core (`1T`) Throughput**: Increased **`2.56×` (`+156%`)** on YNNPACK (`491.8 GOPS` → `1,259.7 GOPS`) and **`+8.2%–10.7%`** on XNNPACK (`748.9` → `810.5 GOPS` on `M=128`; `711.7` → `788.1 GOPS` on `M=1024`) from GFNI nibble-unpack optimizations.
> 2. **YNNPACK FP32 Softmax Reductions (`Pair 4`, `max_k1 + sum_k1`)**:
>    - **28-Core Bandwidth**: Improved **`3.36×–4.45×` (`+236%` to `+345%`)** from `437.9–581.7 GB/s` (`2026-11-10`) to **`1,470.0–2,325.5 GB/s`** (`2026-11-17`) at 28C (and **`3,314.5–3,957.8 GB/s` = `74.1%` of L2 Read ceiling** at 56C) after correcting `k = row_len` element counts and 4-row `min_max_accumulator<N=4>` batching.
> 3. **FP32 Attention `BatchMatMul` (`Pair 3`, 2D N-Major Tile Partitioning & Zero-Init Accumulators)**:
>    - **28-Core Throughput**: Improved **`1.34×` (`+34.3%`)** on YNNPACK `5x64_avx512` (`2.34` → `3.14 TFLOPS` on `1024×1024×256`, reaching **`6.40–6.61 TFLOPS` = `90%–93%` of FP32 FMA ceiling** on 56C) and **`1.29×` (`+28.7%`)** on XNNPACK `7x32__avx512f_broadcast` (`2.51` → `3.23 TFLOPS` on 28C, **`6.35–6.42 TFLOPS`** on 56C).
> 4. **8-Bit Vision ViT GEMM (`Pair 5`, `2520×1152×1152` & `2520×4304×1152`)**:
>    - **28-Core Throughput**: Improved **`1.48×` (`+48.1%`)** on YNNPACK `32x32x64_amxint8` (`21.80` → `32.30 TOPS` on `2520×1152×1152`, **`67.05 TOPS`** at 56C) and **`1.18×` (`+17.7%`)** on XNNPACK `16x64c4__avx512amx` (`20.24` → `23.83 TOPS` on 28C, **`81.20 TOPS`** at 56C).

```mermaid
%%{init: {"xyChart": {"width": 1100, "height": 420}, "themeVariables": {"xyChart": {"plotColorPalette": "#ea580c, #d97706, #dc2626, #2563eb"}}}%%
xychart-beta
    title "28-Core GEMM Throughput Progression Over Time (TOPS, Higher is Better)"
    x-axis "Benchmark Date" ["2026-11-10", "2026-11-13", "2026-11-17"]
    y-axis "Throughput (TOPS, 28 Cores)" 0 --> 42
    line "YNN P2: INT4 128x21504x5376" [14.51, 36.41, 35.84]
    line "YNN P2: INT4 1024x21504x5376" [12.96, 33.5, 32.3]
    line "YNN P5: INT8 2520x1152x1152" [21.8, 30.28, 32.3]
    line "XNN P2: INT4 128x21504x5376" [32.53, 22.62, 22.7]
```

```mermaid
%%{init: {"xyChart": {"width": 1100, "height": 420}, "themeVariables": {"xyChart": {"plotColorPalette": "#9333ea, #ea580c, #dc2626, #16a34a"}}}%%
xychart-beta
    title "Cumulative Kernel Speedup Ratio vs. 2026-11-10 Baseline (28 Cores, x)"
    x-axis "Benchmark Date" ["2026-11-10", "2026-11-13", "2026-11-17"]
    y-axis "Speedup Ratio vs. 2026-11-10 (x)" 0 --> 5
    line "YNN P4: Softmax 1024x4096 (4.45x)" [1.0, 4.08, 4.45]
    line "YNN P2: INT4 128x8192x5376 (2.59x)" [1.0, 2.61, 2.59]
    line "YNN P5: INT8 2520x1152x1152 (1.48x)" [1.0, 1.39, 1.48]
    line "YNN P3: FP32 BMM 1024x1024x256 (1.34x)" [1.0, 1.34, 1.34]
```

---

## 2. Complete Multi-Core (`28C` & `56C`) Progression Table Across All 7 Kernel Pairs

| Kernel Pair | Shape `(M×N×K)` | Regime | Backend | `2026-11-10` (`28C`) | `2026-11-13` (`28C`) | `2026-11-17` (`28C`) | **28C Speedup (`Nov 10 → Nov 17`)** | `2026-11-13` (`56C`) | `2026-11-17` (`56C`) | **Latest `56C` ISA Util** |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| `P1: INT4 GEMV (Decode)` | `1×4096×5376` | `L3_LLC_Warm` | **YNNPACK** | `4,499.31 GOPS` | `1,440.54 GOPS` | `1,418.25 GOPS` | **`0.32×` (`-68.5%`)** | `2,385.64 GOPS` | **`2,356.53 GOPS`** | `11.1%` |
| `P1: INT4 GEMV (Decode)` | `1×4096×5376` | `L3_LLC_Warm` | **XNNPACK** | `1,842.31 GOPS` | `1,132.51 GOPS` | `1,123.74 GOPS` | **`0.61×` (`-39.0%`)** | `1,693.10 GOPS` | **`1,650.77 GOPS`** | `7.8%` |
| `P1: INT4 GEMV (Decode)` | `1×5376×21504` | `DDR5_DRAM_Streaming` | **YNNPACK** | `912.63 GOPS` | `835.68 GOPS` | `862.73 GOPS` | `0.95×` (`-5.5%`) | `934.98 GOPS` | **`980.16 GOPS`** | `102.8%` |
| `P1: INT4 GEMV (Decode)` | `1×5376×21504` | `DDR5_DRAM_Streaming` | **XNNPACK** | `927.78 GOPS` | `850.62 GOPS` | `840.42 GOPS` | `0.91×` (`-9.4%`) | `919.63 GOPS` | **`789.86 GOPS`** | `82.9%` |
| `P1: INT4 GEMV (Decode)` | `1×8192×5376` | `DDR5_DRAM_Streaming` | **YNNPACK** | `874.29 GOPS` | `829.06 GOPS` | `827.96 GOPS` | `0.95×` (`-5.3%`) | `966.94 GOPS` | **`988.38 GOPS`** | `103.8%` |
| `P1: INT4 GEMV (Decode)` | `1×8192×5376` | `DDR5_DRAM_Streaming` | **XNNPACK** | `905.21 GOPS` | `797.81 GOPS` | `793.78 GOPS` | `0.88×` (`-12.3%`) | `939.54 GOPS` | **`100.55 GOPS`** | `10.6%` |
| `P1: INT4 GEMV (Decode)` | `1×8192×5376` | `L3_LLC_Warm` | **YNNPACK** | `4,537.57 GOPS` | `1,438.30 GOPS` | `1,407.14 GOPS` | **`0.31×` (`-69.0%`)** | `2,866.97 GOPS` | **`2,745.32 GOPS`** | `12.9%` |
| `P1: INT4 GEMV (Decode)` | `1×8192×5376` | `L3_LLC_Warm` | **XNNPACK** | `2,179.74 GOPS` | `1,333.49 GOPS` | `1,344.74 GOPS` | **`0.62×` (`-38.3%`)** | `2,191.49 GOPS` | **`2,233.35 GOPS`** | `10.5%` |
| `P1: INT4 GEMV (Decode)` | `1×21504×5376` | `DDR5_DRAM_Streaming` | **YNNPACK** | `898.36 GOPS` | `869.53 GOPS` | `855.82 GOPS` | `0.95×` (`-4.7%`) | `1,004.20 GOPS` | **`978.50 GOPS`** | `102.7%` |
| `P1: INT4 GEMV (Decode)` | `1×21504×5376` | `DDR5_DRAM_Streaming` | **XNNPACK** | `907.05 GOPS` | `836.62 GOPS` | `802.79 GOPS` | `0.89×` (`-11.5%`) | `1,016.74 GOPS` | **`984.76 GOPS`** | `103.4%` |
| `P2: INT4 GEMM (Prefill)` | `128×5376×21504` | `DDR5_DRAM_to_L2_Tiled` | **YNNPACK** | `14.01 TOPS` | `35.98 TOPS` | `35.03 TOPS` | **`2.50×` (`+150.1%`)** | `60.76 TOPS` | **`58.13 TOPS`** | `26.4%` |
| `P2: INT4 GEMM (Prefill)` | `128×5376×21504` | `DDR5_DRAM_to_L2_Tiled` | **XNNPACK** | `33.92 TOPS` | `23.54 TOPS` | `23.36 TOPS` | **`0.69×` (`-31.1%`)** | `46.88 TOPS` | **`46.39 TOPS`** | `21.0%` |
| `P2: INT4 GEMM (Prefill)` | `128×8192×5376` | `L3_LLC_to_L2_Tiled` | **YNNPACK** | `13.89 TOPS` | `36.30 TOPS` | `35.92 TOPS` | **`2.59×` (`+158.5%`)** | `70.30 TOPS` | **`69.66 TOPS`** | `31.6%` |
| `P2: INT4 GEMM (Prefill)` | `128×8192×5376` | `L3_LLC_to_L2_Tiled` | **XNNPACK** | `25.60 TOPS` | `22.22 TOPS` | `22.39 TOPS` | `0.87×` (`-12.5%`) | `43.01 TOPS` | **`43.04 TOPS`** | `19.5%` |
| `P2: INT4 GEMM (Prefill)` | `128×21504×5376` | `DDR5_DRAM_to_L2_Tiled` | **YNNPACK** | `14.51 TOPS` | `36.41 TOPS` | `35.84 TOPS` | **`2.47×` (`+146.9%`)** | `73.76 TOPS` | **`71.69 TOPS`** | `32.5%` |
| `P2: INT4 GEMM (Prefill)` | `128×21504×5376` | `DDR5_DRAM_to_L2_Tiled` | **XNNPACK** | `32.53 TOPS` | `22.62 TOPS` | `22.70 TOPS` | **`0.70×` (`-30.2%`)** | `45.20 TOPS` | **`45.27 TOPS`** | `20.5%` |
| `P2: INT4 GEMM (Prefill)` | `1024×8192×5376` | `Compute_Bound_Tiled` | **YNNPACK** | `13.34 TOPS` | `34.34 TOPS` | `33.65 TOPS` | **`2.52×` (`+152.3%`)** | `64.50 TOPS` | **`63.27 TOPS`** | `28.7%` |
| `P2: INT4 GEMM (Prefill)` | `1024×8192×5376` | `Compute_Bound_Tiled` | **XNNPACK** | `27.66 TOPS` | `22.59 TOPS` | `22.58 TOPS` | **`0.82×` (`-18.4%`)** | `44.37 TOPS` | **`44.54 TOPS`** | `20.2%` |
| `P2: INT4 GEMM (Prefill)` | `1024×21504×5376` | `Compute_Bound_Tiled` | **YNNPACK** | `12.96 TOPS` | `33.50 TOPS` | `32.30 TOPS` | **`2.49×` (`+149.3%`)** | `61.59 TOPS` | **`59.60 TOPS`** | `27.0%` |
| `P2: INT4 GEMM (Prefill)` | `1024×21504×5376` | `Compute_Bound_Tiled` | **XNNPACK** | `26.56 TOPS` | `22.22 TOPS` | `22.06 TOPS` | **`0.83×` (`-16.9%`)** | `43.74 TOPS` | **`43.47 TOPS`** | `19.7%` |
| `P3: FP32 Attention BMM` | `128×1024×256` | `L2_Cache_Resident` | **YNNPACK** | `2,469.33 GFLOPS` | `3,276.56 GFLOPS` | `3,300.09 GFLOPS` | **`1.34×` (`+33.6%`)** | `6,219.61 GFLOPS` | **`6,106.18 GFLOPS`** | `86.3%` |
| `P3: FP32 Attention BMM` | `128×1024×256` | `L2_Cache_Resident` | **XNNPACK** | `2,729.31 GFLOPS` | `3,207.92 GFLOPS` | `3,057.12 GFLOPS` | `1.12×` (`+12.0%`) | `6,362.87 GFLOPS` | **`6,248.07 GFLOPS`** | `88.3%` |
| `P3: FP32 Attention BMM` | `1024×1024×256` | `L3_LLC_Resident` | **YNNPACK** | `2,341.05 GFLOPS` | `3,138.08 GFLOPS` | `3,144.36 GFLOPS` | **`1.34×` (`+34.3%`)** | `6,613.14 GFLOPS` | **`6,397.22 GFLOPS`** | `90.5%` |
| `P3: FP32 Attention BMM` | `1024×1024×256` | `L3_LLC_Resident` | **XNNPACK** | `2,511.19 GFLOPS` | `3,201.33 GFLOPS` | `3,231.01 GFLOPS` | **`1.29×` (`+28.7%`)** | `6,420.97 GFLOPS` | **`6,352.04 GFLOPS`** | `89.8%` |
| `P3: FP32 Attention BMM` | `1024×4096×512` | `L3_LLC_Resident` | **YNNPACK** | `3,343.29 GFLOPS` | `3,119.39 GFLOPS` | `3,134.39 GFLOPS` | `0.94×` (`-6.2%`) | `6,204.81 GFLOPS` | **`6,225.24 GFLOPS`** | `88.0%` |
| `P3: FP32 Attention BMM` | `1024×4096×512` | `L3_LLC_Resident` | **XNNPACK** | `3,717.83 GFLOPS` | `2,909.87 GFLOPS` | `2,999.07 GFLOPS` | **`0.81×` (`-19.3%`)** | `5,852.43 GFLOPS` | **`5,881.58 GFLOPS`** | `83.2%` |
| `P3: FP32 Attention BMM` | `2520×2520×72` | `L3_LLC_Resident` | **YNNPACK** | `2,240.25 GFLOPS` | `2,623.30 GFLOPS` | `2,711.73 GFLOPS` | **`1.21×` (`+21.0%`)** | `4,942.49 GFLOPS` | **`4,451.90 GFLOPS`** | `63.0%` |
| `P3: FP32 Attention BMM` | `2520×2520×72` | `L3_LLC_Resident` | **XNNPACK** | `3,774.38 GFLOPS` | `2,784.00 GFLOPS` | `2,818.22 GFLOPS` | **`0.75×` (`-25.3%`)** | `5,739.29 GFLOPS` | **`5,500.29 GFLOPS`** | `77.8%` |
| `P4: FP32 Softmax` | `1024×1024×1` | `L2_L3_Resident` | **YNNPACK** | `581.66 GB/s` | `2,061.27 GB/s` | `2,048.66 GB/s` | **`3.52×` (`+252.2%`)** | `3,534.80 GB/s` | **`3,520.22 GB/s`** | `66.0%` |
| `P4: FP32 Softmax` | `1024×1024×1` | `L2_L3_Resident` | **XNNPACK** | `1,741.66 GB/s` | `1,067.33 GB/s` | `1,041.86 GB/s` | **`0.60×` (`-40.2%`)** | `1,639.62 GB/s` | **`1,965.28 GB/s`** | `71.5%` |
| `P4: FP32 Softmax` | `1024×4096×1` | `L3_LLC_Resident` | **YNNPACK** | `522.28 GB/s` | `2,131.07 GB/s` | `2,325.47 GB/s` | **`4.45×` (`+345.3%`)** | `4,010.57 GB/s` | **`3,957.85 GB/s`** | `74.1%` |
| `P4: FP32 Softmax` | `1024×4096×1` | `L3_LLC_Resident` | **XNNPACK** | `1,450.52 GB/s` | `1,035.77 GB/s` | `1,029.08 GB/s` | **`0.71×` (`-29.1%`)** | `2,010.54 GB/s` | **`1,964.62 GB/s`** | `71.5%` |
| `P4: FP32 Softmax` | `2520×2520×1` | `L3_LLC_Resident` | **YNNPACK** | `437.88 GB/s` | `1,482.38 GB/s` | `1,470.02 GB/s` | **`3.36×` (`+235.7%`)** | `3,332.85 GB/s` | **`3,314.50 GB/s`** | `62.1%` |
| `P4: FP32 Softmax` | `2520×2520×1` | `L3_LLC_Resident` | **XNNPACK** | `1,560.26 GB/s` | `961.16 GB/s` | `987.87 GB/s` | **`0.63×` (`-36.7%`)** | `2,036.39 GB/s` | **`1,995.69 GB/s`** | `72.6%` |
| `P5: INT8 GEMM (Multimodal)` | `280×5376×1152` | `L3_LLC_Resident` | **YNNPACK** | `81.99 TOPS` | `60.01 TOPS` | `46.00 TOPS` | **`0.56×` (`-43.9%`)** | `118.36 TOPS` | **`99.21 TOPS`** | `45.0%` |
| `P5: INT8 GEMM (Multimodal)` | `280×5376×1152` | `L3_LLC_Resident` | **XNNPACK** | `47.50 TOPS` | `42.06 TOPS` | `29.28 TOPS` | **`0.62×` (`-38.4%`)** | `84.25 TOPS` | **`46.20 TOPS`** | `20.9%` |
| `P5: INT8 GEMM (Multimodal)` | `2520×1152×1152` | `L3_LLC_Resident` | **YNNPACK** | `21.80 TOPS` | `30.28 TOPS` | `32.30 TOPS` | **`1.48×` (`+48.1%`)** | `65.03 TOPS` | **`67.05 TOPS`** | `30.4%` |
| `P5: INT8 GEMM (Multimodal)` | `2520×1152×1152` | `L3_LLC_Resident` | **XNNPACK** | `20.24 TOPS` | `23.72 TOPS` | `23.83 TOPS` | **`1.18×` (`+17.7%`)** | `74.12 TOPS` | **`81.20 TOPS`** | `36.8%` |
| `P5: INT8 GEMM (Multimodal)` | `2520×4304×1152` | `L3_LLC_Resident` | **YNNPACK** | `23.63 TOPS` | `32.08 TOPS` | `27.81 TOPS` | **`1.18×` (`+17.7%`)** | `64.08 TOPS` | **`58.98 TOPS`** | `26.7%` |
| `P5: INT8 GEMM (Multimodal)` | `2520×4304×1152` | `L3_LLC_Resident` | **XNNPACK** | `25.17 TOPS` | `26.72 TOPS` | `28.22 TOPS` | `1.12×` (`+12.1%`) | `48.51 TOPS` | **`53.39 TOPS`** | `24.2%` |
| `P6: INT2 GEMM (Audio/Embed)` | `204×1024×1024` | `L2_Cache_Resident` | **YNNPACK** | `7,035.89 GOPS` | `7,189.23 GOPS` | `7,291.94 GOPS` | `1.04×` (`+3.6%`) | `13.84 TOPS` | **`682.63 GOPS`** | `2.5%` |
| `P6: INT2 GEMM (Audio/Embed)` | `204×1024×1024` | `L2_Cache_Resident` | **XNNPACK** | `5,690.16 GOPS` | `5,058.74 GOPS` | `5,011.36 GOPS` | `0.88×` (`-11.9%`) | `10.05 TOPS` | **`9,961.94 GOPS`** | `70.4%` |
| `P6: INT2 GEMM (Audio/Embed)` | `204×1024×4096` | `L2_L3_Resident` | **YNNPACK** | `7,529.66 GOPS` | `7,497.96 GOPS` | `7,362.69 GOPS` | `0.98×` (`-2.2%`) | `14.92 TOPS` | **`14.17 TOPS`** | `50.9%` |
| `P6: INT2 GEMM (Audio/Embed)` | `204×1024×4096` | `L2_L3_Resident` | **XNNPACK** | `8,180.97 GOPS` | `5,305.48 GOPS` | `5,040.39 GOPS` | **`0.62×` (`-38.4%`)** | `10.58 TOPS` | **`10.06 TOPS`** | `71.1%` |
| `P6: INT2 GEMM (Audio/Embed)` | `204×4096×1024` | `L2_L3_Resident` | **YNNPACK** | `9,973.45 GOPS` | `7,394.74 GOPS` | `7,417.25 GOPS` | **`0.74×` (`-25.6%`)** | `13.89 TOPS` | **`11.71 TOPS`** | `42.1%` |
| `P6: INT2 GEMM (Audio/Embed)` | `204×4096×1024` | `L2_L3_Resident` | **XNNPACK** | `8,423.48 GOPS` | `5,076.20 GOPS` | `5,056.94 GOPS` | **`0.60×` (`-40.0%`)** | `10.12 TOPS` | **`10.12 TOPS`** | `71.5%` |
| `P7: 32-Bit Pack/Transpose` | `1×1024×256` | `L2_Cache_Copy` | **YNNPACK** | `746.86 GB/s` | `309.26 GB/s` | `306.15 GB/s` | **`0.41×` (`-59.0%`)** | `383.25 GB/s` | **`381.90 GB/s`** | `13.9%` |
| `P7: 32-Bit Pack/Transpose` | `1×1024×256` | `L2_Cache_Copy` | **XNNPACK** | `912.99 GB/s` | `572.09 GB/s` | `591.50 GB/s` | **`0.65×` (`-35.2%`)** | `1,254.95 GB/s` | **`1,235.96 GB/s`** | `45.0%` |
| `P7: 32-Bit Pack/Transpose` | `1×2520×1152` | `L3_LLC_Copy` | **YNNPACK** | `1,315.26 GB/s` | `795.98 GB/s` | `792.39 GB/s` | **`0.60×` (`-39.8%`)** | `1,172.29 GB/s` | **`917.53 GB/s`** | `33.4%` |
| `P7: 32-Bit Pack/Transpose` | `1×2520×1152` | `L3_LLC_Copy` | **XNNPACK** | `1,585.56 GB/s` | `1,007.20 GB/s` | `1,003.32 GB/s` | **`0.63×` (`-36.7%`)** | `1,505.54 GB/s` | **`1,505.79 GB/s`** | `54.8%` |
| `P7: 32-Bit Pack/Transpose` | `1×4096×512` | `L3_LLC_Copy` | **YNNPACK** | `650.70 GB/s` | `289.74 GB/s` | `299.15 GB/s` | **`0.46×` (`-54.0%`)** | `475.06 GB/s` | **`493.12 GB/s`** | `17.9%` |
| `P7: 32-Bit Pack/Transpose` | `1×4096×512` | `L3_LLC_Copy` | **XNNPACK** | `1,163.57 GB/s` | `674.23 GB/s` | `679.39 GB/s` | **`0.58×` (`-41.6%`)** | `1,122.28 GB/s` | **`1,108.66 GB/s`** | `40.4%` |

---

## 3. Updating This Dashboard

Whenever a new dated report directory (`kernel/reports/YYYY-MM-DD-gemma4-31b-sapphire-rapids/`) is added or updated, regenerate both the interactive HTML dashboard and this Markdown summary via:

```bash
python3 kernel/scripts/update_dashboard.py
# or via Make:
make -C kernel dashboard
```
