# Empirical Hardware Ceiling & Microkernel Roofline Report — Gemma 4 31B on Intel Sapphire Rapids

- **Model**: Gemma 4 31B (`gemma-4-31b-it`, 4-bit Decoder / 8-bit Vision ViT / 2-bit & 8-bit Audio Conformer)
- **Target Microarchitecture**: Intel Sapphire Rapids (`x86_64`, AVX-512F, AVX-512 VNNI, AVX-VNNI, AMX-INT8, AMX-BF16)
- **Execution Configurations**:
  - **Single-Core (`1T`)**: Pinned to physical core `0` (`sched_setaffinity`)
  - **Multi-Core (`28T`)**: Pinned across all 28 physical cores of NUMA Node 0 (`0..27`, 1 thread per physical core, no SMT sibling contention, zero cross-socket UPI traffic)
- **Repeatability Methodology**:
  - Timing clock: `CLOCK_MONOTONIC_RAW`
  - Thread synchronization: `#pragma omp barrier` immediately before and after every timed loop (eliminating OpenMP fork/join jitter)
  - Warmup + Multi-Trial Statistics: Explicit frequency/cache warmup pass followed by **7 independent timed trials** per probe and per microkernel shape, reporting `median`, `mean`, `min`, `max`, `stdev`, and Coefficient of Variation (`cv_pct`)
- **Companion End-to-End Profiling Report**: [kernels_gemma4_31b_sapphire_rapids.md](kernels_gemma4_31b_sapphire_rapids.md)

---

## 1. Executive Summary & Key Roofline Findings

> [!IMPORTANT]
> **Why XNNPACK Beats YNNPACK in Prefill (`2.16×` End-to-End), Matches in Decode (`~6.0 tok/s`), and Trails Slightly in Audio (`0.93×`) — Explained by Hardware Ceilings**:
> 1. **Decode (`M = 1` 4-Bit GEMV, 83%–87% of Decode Time) is Strictly Clamped by DDR5 DRAM Bandwidth (`246.60 GB/s` Ceiling)**:
>    - At `M = 1`, arithmetic intensity is only **$3.99\text{ OPs/Byte}$**, placing both backends deep in the **DDR5 DRAM bandwidth-bound regime** ($BW_{\text{prac}}(\text{DDR5 Read}) = \mathbf{246.60\text{ GB/s}}$ on 28 cores $\Rightarrow \mathbf{984.9\text{ GOPS}}$ ceiling).
>    - On cold DDR5-streaming weights (`1×21504×5376` and `1×5376×21504`), **XNNPACK achieves `227.1–232.1 GB/s` (`92.1%–94.1%` of practical DDR5 ceiling)** and **YNNPACK achieves `224.9–228.3 GB/s` (`91.2%–92.6%` of practical DDR5 ceiling)**. Because both saturate the memory bus above 91%, AMX vs. AVX-512 VNNI compute differences are completely hidden in full-model decode.
>    - Conversely, when weights fit warm in L3 cache (`1×8192×5376`, `L3_LLC_Warm`), YNNPACK's `1x32x8_avx512vnni` (**4.54 TOPS**) is **2.08× faster** than XNNPACK's `1x64c4__avx512amx` (**2.18 TOPS**) because `M = 1` utilizes only 1 of 16 rows (`6.25%`) of the AMX tile while incurring stack spill/reload overhead for `tileloadd`.
> 2. **Prefill (`M = 128..1024` 4-Bit GEMM, 46%–65% of Prefill Time) is Governed by an `8.98×` ISA Ceiling Gap (`AMX-INT8` vs. `AVX-512 VNNI`)**:
>    - Our compute ceiling probes measure **$C_{\text{prac}}(\text{AMX-INT8}) = \mathbf{188.40\text{ TOPS}}$** (`7.70 TOPS`/core) vs. **$C_{\text{prac}}(\text{AVX-512 VNNI}) = \mathbf{20.97\text{ TOPS}}$** (`0.95 TOPS`/core) on 28 cores — an **8.98× hardware ceiling advantage** for AMX-INT8.
>    - YNNPACK's `ynn::dot_uint8_int4_int32_12x16x8_1x16x8_avx512vnni` achieves **14.51 TOPS** at `128×21504×5376`, which is **69.2% of the practical AVX-512 VNNI ceiling** (`20.97 TOPS`) — meaning the remaining `~30%` is consumed by `vpshufb`/`vpandd`/`vpsrlw` 4-bit nibble unpacking instructions on vector ports 0/1/5.
>    - Even a **100% ideal AVX-512 VNNI kernel (`20.97 TOPS`)** can never reach XNNPACK's `16x64c4__avx512amx` (**32.53–33.92 TOPS**, which exceeds the *entire* AVX-512 VNNI ceiling by **1.55×–1.62×** by overlapping AVX-512 nibble unpacking with AMX `tdpbssd` tile multiplies).
> 3. **8-Bit Vision/Audio Linear GEMM (`pair5_int8_gemm`) — Both Use AMX-INT8, and YNNPACK's `32×32` 2×2 Tile Blocking Beats XNNPACK's `16×64` 1×4 Tile Blocking in Raw `dot` (`81.99 TOPS` vs. `47.50 TOPS`)**:
>    - On `280×5376×1152`, YNNPACK's `dot_int8_int8_int32_32x32x64_16x16x4_amxint8` reaches **43.5% of the 28-core AMX-INT8 ceiling (`81.99 TOPS`)** vs. XNNPACK's **25.2% (`47.50 TOPS`)** (`1.81×` faster on 1-core, `1.73×` faster on 28-core) because `2×2` tile blocking (`32×32`) halves `A`-tile loads compared to `1×4` (`16×64`) and omits in-kernel FP32 dequantization. However, in end-to-end execution, YNNPACK loses part of this advantage to its unfused `dequantize_dot_f32_avx512` epilogue pass.
> 4. **2-Bit Audio Conformer GEMM (`pair6_int2_gemm`) — XNNPACK Uses 256-Bit `AVX-VNNI` (`10.67 TOPS` Ceiling) While YNNPACK Uses 512-Bit `AVX-512 VNNI` (`20.97 TOPS` Ceiling)**:
>    - XNNPACK's `5x8c8__avxvnni` reaches **79.0% of the 256-bit AVX-VNNI ceiling (`8.42 TOPS`)**, proving the kernel is tightly coded for 256-bit `ymm` registers, but leaves half of Sapphire Rapids' 512-bit vector width unused. YNNPACK's 512-bit `4x64x16_avx512vnni` reaches **9.97 TOPS** (`1.50×` faster on 1-core, `1.18×–1.24×` faster on 28-core).
> 5. **FP32 Attention `BatchMatMul` & `Softmax` (`pair3` & `pair4`)**:
>    - Both `xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast` and `ynn::dot_fp32_5x64x1_1x16x1_avx512` achieve **71%–83% of the 1-core AVX-512 FP32 FMA ceiling (`235.72 GFLOPS`)** and **60%–67% of the 28-core ceiling (`5,616.26 GFLOPS`)** on large shapes (`1024×4096×512`). On non-multiple-of-64 `N` shapes (`2520×2520×72`), XNNPACK's `nr=32` tile width (`79` panels across 28 threads) achieves **67.2% (`3.77 TFLOPS`)** vs. YNNPACK's `nr=64` (`39` panels across 28 threads, **39.9% / `2.24 TFLOPS`**) due to multi-thread wave tail imbalance.
>    - In Softmax (`2520×2520`), XNNPACK's fused 2-pass `rmax` + `raddstoreexpminusmax` achieves **1,560.26 GB/s** (**70.4% of 28-core L2 Read+Write bandwidth**) while computing the degree-5 polynomial `exp(x - max)` in-flight, beating YNNPACK's un-unrolled `max_k1 + sum_k1` reductions (**437.88 GB/s**) by **2.38×** before YNNPACK even computes `exp`.

---

## 2. Visual Summary: Throughput & Practical Roofline Utilization

> **Series Legend**: **Left Bar (Blue `#2b6cd9`)** = **XNNPACK** | **Right Bar (Orange `#e67e22`)** = **YNNPACK**

```mermaid
%%{init: {"xyChart": {"width": 1250, "height": 480, "xAxis": {"labelRotation": 0, "labelFontSize": 11, "showTick": false}}, "themeVariables": {"xyChart": {"plotColorPalette": "#2b6cd9, #e67e22"}}}}%%
xychart-beta
    title "28-Core Compute Throughput: XNNPACK (Blue) vs YNNPACK (Orange) [TOPS / TFLOPS]"
    x-axis "Kernel & Shape" ["⠀⠀⠀P2: INT4 128x21504", "​​‌", "​​‍", "​​⁠", "⠀⠀⠀P2: INT4 128x5376", "​‌​", "​‌‌", "​‌‍", "⠀⠀⠀P2: INT4 1024x21504", "​‌⁠", "​‍​", "​‍‌", "⠀⠀⠀P5: INT8 280x5376", "​‍‍", "​‍⁠", "​⁠​", "⠀⠀⠀P5: INT8 2520x1152", "​⁠‌", "​⁠‍", "​⁠⁠", "⠀⠀⠀P6: INT2 204x4096", "‌​​", "‌​‌", "‌​‍", "⠀⠀⠀P3: FP32 1024x4096", "‌​⁠"]
    y-axis "TOPS / TFLOPS (28 Cores)" 0 --> 85
    bar "XNNPACK" [32.53, -85, -85, -85, 33.92, -85, -85, -85, 26.56, -85, -85, -85, 47.5, -85, -85, -85, 20.24, -85, -85, -85, 8.42, -85, -85, -85, 3.72, -85]
    bar "YNNPACK" [-85, 14.51, -85, -85, -85, 14.01, -85, -85, -85, 12.96, -85, -85, -85, 81.99, -85, -85, -85, 21.8, -85, -85, -85, 9.97, -85, -85, -85, 3.34]
```

```mermaid
%%{init: {"xyChart": {"width": 1250, "height": 480, "xAxis": {"labelRotation": 0, "labelFontSize": 11, "showTick": false}}, "themeVariables": {"xyChart": {"plotColorPalette": "#2b6cd9, #e67e22"}}}}%%
xychart-beta
    title "28-Core Practical ISA Roofline Utilization (%): XNNPACK (Blue) vs YNNPACK (Orange)"
    x-axis "Kernel & Shape" ["⠀⠀⠀P1: INT4 (DRAM)", "​​‌", "​​‍", "​​⁠", "⠀⠀⠀P1: INT4 (L2/L3)", "​‌​", "​‌‌", "​‌‍", "⠀⠀⠀P2: INT4 (M=128)", "​‌⁠", "​‍​", "​‍‌", "⠀⠀⠀P3: FP32 BMM", "​‍‍", "​‍⁠", "​⁠​", "⠀⠀⠀P4: FP32 Softmax", "​⁠‌", "​⁠‍", "​⁠⁠", "⠀⠀⠀P5: INT8 (M=280)", "‌​​", "‌​‌", "‌​‍", "⠀⠀⠀P6: INT2 (Audio)", "‌​⁠", "‌‌​", "‌‌‌", "⠀⠀⠀P7: Pack/Transpose", "‌‌‍"]
    y-axis "Roofline Utilization (%)" 0 --> 100
    bar "XNNPACK" [94.13, -100, -100, -100, 12.2, -100, -100, -100, 18.0, -100, -100, -100, 66.2, -100, -100, -100, 70.4, -100, -100, -100, 25.21, -100, -100, -100, 78.96, -100, -100, -100, 52.5, -100]
    bar "YNNPACK" [-100, 92.59, -100, -100, -100, 25.4, -100, -100, -100, 66.79, -100, -100, -100, 59.53, -100, -100, -100, 19.76, -100, -100, -100, 43.52, -100, -100, -100, 47.55, -100, -100, -100, 29.36]
```

---

## 3. Empirical Hardware Ceilings on Intel Sapphire Rapids (`cpu_ceilings.c`)

All ceilings below were measured directly on the target Sapphire Rapids machine using [cpu_ceilings.c](../../probes/cpu_ceilings.c) across **7 independent barrier-synchronized trials** after explicit warmup.

### 3.1 Practical Compute Ceilings: $C_{\text{prac}}(\text{op}, \text{dtype})$

| Instruction Set / Compute Unit | Instruction | Precision | Ops / Inst | 1-Core Median | 1-Core `cv_pct` | 28-Core (NUMA 0) Median | 28-Core `cv_pct` | Relative to AMX-INT8 (28C) | Load-Bearing Kernels Governed |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **AMX-INT8 (`TMUL`)** | `tdpbssd` (`16×16×64`) | `INT8 / u8×s8→s32` | `32,768` | **7,701.36 GOPS** | `3.11%` | **188,402.15 GOPS (188.40 TOPS)** | `0.71%` | `1.00×` (`100%`) | `xnn_qd8_f32_qc4w_*__avx512amx`, `xnn_qd8_f32_qc8w_*__avx512amx`, `ynn::dot_int8_int8_int32_*_amxint8` |
| **AMX-BF16 (`TMUL`)** | `tdpbf16ps` (`16×16×32`) | `BF16→FP32` | `16,384` | **3,866.86 GFLOPS** | `1.89%` | **83,899.87 GFLOPS (83.90 TFLOPS)** | `6.82%` | `0.445×` (`44.5%`) | Reference BF16 TMUL ceiling |
| **AVX-512 VNNI (512-bit)** | `vpdpbusd %zmm` | `u8×s8→s32` | `128` | **948.07 GOPS** | `1.62%` | **20,973.59 GOPS (20.97 TOPS)** | `5.79%` | **`0.111×` (`11.1%`)** | `ynn::dot_uint8_int4_int32_*_avx512vnni`, `ynn::dot_uint8_int2_int32_*_avx512vnni` |
| **AVX-VNNI (256-bit)** | `vpdpbusd %ymm` | `u8×s8→s32` | `64` | **484.59 GOPS** | `0.49%` | **10,667.71 GOPS (10.67 TOPS)** | `4.67%` | **`0.057×` (`5.7%`)** | `xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni` |
| **AVX-512 FP32 FMA (512-bit)** | `vfmadd231ps %zmm` | `FP32` | `32` | **235.72 GFLOPS** | `1.91%` | **5,616.26 GFLOPS (5.62 TFLOPS)** | `3.32%` | `0.030×` (`3.0%`) | `xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast`, `ynn::dot_fp32_*_avx512` |

### 3.2 Practical Memory-Hierarchy Bandwidth Ceilings: $BW_{\text{prac}}(\text{WorkingSet})$

| Memory Hierarchy Level | Working Set | Access Pattern | 1-Core Median (`GB/s`) | 1-Core `cv_pct` | 28-Core (NUMA 0) Median (`GB/s`) | 28-Core `cv_pct` | Primary Kernel Regimes Governed |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| **L1D Cache** (`48 KB`/core) | `16 KB` / core | Streaming Read (`vmovaps`) | **284.72 GB/s** | `1.61%` | **6,691.22 GB/s** | `2.12%` | Microkernel register-tile A/B streaming |
| **L1D Cache** (`48 KB`/core) | `16 KB` / core | Read + Write Copy | **484.87 GB/s** | `0.21%` | **5,560.20 GB/s** | `3.49%` | AMX stack-spill `tilestored` / `tileloadd` scratchpad |
| **L2 Cache** (`2 MB`/core) | `512 KB` / core | Streaming Read (`vmovaps`) | **190.36 GB/s** | `2.22%` | **4,473.83 GB/s** | `5.80%` | L2-tiled GEMM panels (`M=128..204`), Small Attention |
| **L2 Cache** (`2 MB`/core) | `512 KB` / core | Read + Write Copy | **97.37 GB/s** | `0.22%` | **2,216.39 GB/s** | `2.38%` | Multi-threaded KV packing/transpose & Softmax rows |
| **L3 Shared LLC** (`52.5 MB`) | `32 MB` shared (`8 MB`/T) | Streaming Read (`vmovaps`) | **33.03 GB/s** | `0.04%` | **780.35 GB/s** | `1.51%` | Single-layer weights (`11–44 MB`) & KV cache (`p1024`–`p4096`) |
| **L3 Shared LLC** (`52.5 MB`) | `32 MB` shared (`8 MB`/T) | Read + Write Copy | **28.73 GB/s** | `0.60%` | **535.86 GB/s** | `1.76%` | Large L3-resident tensor transposes & dequantize passes |
| **DDR5 DRAM** (Socket 0) | `896 MB` (`32 MB`/T) | Streaming Read (`vmovaps`) | **15.89 GB/s** | `3.64%` | **246.60 GB/s** | `0.15%` | **Full-model `M=1` Decode weight streaming (`18+ GB`)** |
| **DDR5 DRAM** (Socket 0) | `896 MB` (`32 MB`/T) | Read + Write Copy | **14.39 GB/s** | `1.19%` | **134.61 GB/s** | `0.62%` | Cold DRAM buffer copies |
| **DDR5 DRAM** (Socket 0) | `896 MB` (`32 MB`/T) | STREAM Triad (`2R + 1W`) | **15.49 GB/s** | `0.21%` | **151.58 GB/s** | `0.63%` | Cold DRAM elementwise / residual fusion |

---

## 4. Per-Kernel Roofline Comparison: XNNPACK vs. YNNPACK

### 4.1 Pair 1: 4-Bit Quantized Linear GEMV (`M = 1`, Decode Bottleneck — 83.2% XNNPACK / 87.3% YNNPACK Decode)

| Shape `(M×N×K)` | Regime | Backend | Microkernel (`ISA`) | AI (`OP/B`) | 1C Latency (`ms`) | 1C Perf | 1C Ceiling | 1C Roofline Util | 28C Latency (`ms`) | 28C Perf | 28C `cv_pct` | 28C ISA Ceiling | **28C ISA Roofline Util** | **28C Machine Peak Util** |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1×21504×5376` | `DDR5_DRAM_Streaming` | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx` (`AMX-INT8`) | `3.99` | `5.031` | `45.95 GOPS` (`11.5 GB/s`) | `63.46 GOPS` | `72.4%` | `0.255` | **`907.05 GOPS`** (`227.1 GB/s`) | `2.73%` | `984.92 GOPS` | **`92.1%`** | **`92.1%`** |
| `1×21504×5376` | `DDR5_DRAM_Streaming` | **YNNPACK** | `ynn::dot_uint8_int4_int32_1x32x8_1x16x8_avx512vnni` (`AVX512-VNNI`) | `3.99` | `4.817` | `48.00 GOPS` (`12.0 GB/s`) | `63.46 GOPS` | `75.6%` | `0.257` | **`898.36 GOPS`** (`224.9 GB/s`) | `2.79%` | `984.92 GOPS` | **`91.2%`** | **`91.2%`** |
| `1×5376×21504` | `DDR5_DRAM_Streaming` | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx` (`AMX-INT8`) | `4.00` | `4.907` | `47.12 GOPS` (`11.8 GB/s`) | `63.51 GOPS` | `74.2%` | `0.249` | **`927.78 GOPS`** (`232.1 GB/s`) | `3.37%` | `985.66 GOPS` | **`94.1%`** | **`94.1%`** |
| `1×5376×21504` | `DDR5_DRAM_Streaming` | **YNNPACK** | `ynn::dot_uint8_int4_int32_1x32x8_1x16x8_avx512vnni` (`AVX512-VNNI`) | `4.00` | `4.768` | `48.49 GOPS` (`12.1 GB/s`) | `63.51 GOPS` | `76.3%` | `0.253` | **`912.63 GOPS`** (`228.3 GB/s`) | `2.79%` | `985.66 GOPS` | **`92.6%`** | **`92.6%`** |
| `1×8192×5376` | `DDR5_DRAM_Streaming` | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx` (`AMX-INT8`) | `3.99` | `1.877` | `46.94 GOPS` (`11.8 GB/s`) | `63.45 GOPS` | `74.0%` | `0.097` | **`905.21 GOPS`** (`226.7 GB/s`) | `8.49%` | `984.67 GOPS` | **`91.9%`** | **`91.9%`** |
| `1×8192×5376` | `DDR5_DRAM_Streaming` | **YNNPACK** | `ynn::dot_uint8_int4_int32_1x32x8_1x16x8_avx512vnni` (`AVX512-VNNI`) | `3.99` | `1.840` | `47.87 GOPS` (`12.0 GB/s`) | `63.45 GOPS` | `75.5%` | `0.101` | **`874.29 GOPS`** (`218.9 GB/s`) | `6.36%` | `984.67 GOPS` | **`88.8%`** | **`88.8%`** |
| `1×8192×5376` | `L3_LLC_Warm` | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx` (`AMX-INT8`) | `3.99` | `0.901` | `97.71 GOPS` (`24.5 GB/s`) | `131.89 GOPS` | `74.1%` | `0.040` | **`2179.74 GOPS`** (`545.9 GB/s`) | `1.73%` | `17864.00 GOPS` | **`12.2%`** | **`12.2%`** |
| `1×8192×5376` | `L3_LLC_Warm` | **YNNPACK** | `ynn::dot_uint8_int4_int32_1x32x8_1x16x8_avx512vnni` (`AVX512-VNNI`) | `3.99` | `0.725` | `121.54 GOPS` (`30.4 GB/s`) | `131.89 GOPS` | `92.2%` | `0.019` | **`4537.57 GOPS`** (`1136.4 GB/s`) | `3.01%` | `17864.00 GOPS` | **`25.4%`** | **`25.4%`** |
| `1×4096×5376` | `L3_LLC_Warm` | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx` (`AMX-INT8`) | `3.99` | `0.451` | `97.62 GOPS` (`24.4 GB/s`) | `131.86 GOPS` | `74.0%` | `0.024` | **`1842.31 GOPS`** (`461.5 GB/s`) | `3.40%` | `17859.53 GOPS` | **`10.3%`** | **`10.3%`** |
| `1×4096×5376` | `L3_LLC_Warm` | **YNNPACK** | `ynn::dot_uint8_int4_int32_1x32x8_1x16x8_avx512vnni` (`AVX512-VNNI`) | `3.99` | `0.362` | `121.79 GOPS` (`30.5 GB/s`) | `131.86 GOPS` | `92.4%` | `0.010` | **`4499.31 GOPS`** (`1127.0 GB/s`) | `3.51%` | `17859.53 GOPS` | **`25.2%`** | **`25.2%`** |

### 4.2 Pair 2: 4-Bit Quantized Linear Batched GEMM (`M ∈ {128, 1024}`, Prefill Bottleneck — 45.7% XNNPACK / 64.9% YNNPACK Prefill)

| Shape `(M×N×K)` | Regime | Backend | Microkernel (`ISA`) | AI (`OP/B`) | 1C Latency (`ms`) | 1C Perf | 1C Ceiling | 1C Roofline Util | 28C Latency (`ms`) | 28C Perf | 28C `cv_pct` | 28C ISA Ceiling | **28C ISA Roofline Util** | **28C Machine Peak Util** |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `128×21504×5376` | `DDR5_DRAM_to_L2_Tiled` | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx` (`AMX-INT8`) | `425.82` | `20.849` | `1419.50 GOPS` (`3.3 GB/s`) | `7701.36 GOPS` | `18.4%` | `0.910` | **`32527.90 GOPS`** (`76.4 GB/s`) | `2.89%` | `188402.15 GOPS` | **`17.3%`** | **`17.3%`** |
| `128×21504×5376` | `DDR5_DRAM_to_L2_Tiled` | **YNNPACK** | `ynn::dot_uint8_int4_int32_12x16x8_1x16x8_avx512vnni` (`AVX512-VNNI`) | `425.82` | `46.198` | `640.61 GOPS` (`1.5 GB/s`) | `948.07 GOPS` | `67.6%` | `2.039` | **`14512.43 GOPS`** (`34.1 GB/s`) | `1.93%` | `20973.59 GOPS` | **`69.2%`** | **`7.7%`** |
| `128×5376×21504` | `DDR5_DRAM_to_L2_Tiled` | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx` (`AMX-INT8`) | `467.48` | `20.178` | `1466.68 GOPS` (`3.1 GB/s`) | `7701.36 GOPS` | `19.0%` | `0.873` | **`33918.61 GOPS`** (`72.6 GB/s`) | `8.46%` | `188402.15 GOPS` | **`18.0%`** | **`18.0%`** |
| `128×5376×21504` | `DDR5_DRAM_to_L2_Tiled` | **YNNPACK** | `ynn::dot_uint8_int4_int32_12x16x8_1x16x8_avx512vnni` (`AVX512-VNNI`) | `467.48` | `52.966` | `558.75 GOPS` (`1.2 GB/s`) | `948.07 GOPS` | `58.9%` | `2.113` | **`14007.96 GOPS`** (`30.0 GB/s`) | `2.57%` | `20973.59 GOPS` | **`66.8%`** | **`7.4%`** |
| `128×8192×5376` | `L3_LLC_to_L2_Tiled` | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx` (`AMX-INT8`) | `419.08` | `9.325` | `1209.07 GOPS` (`2.9 GB/s`) | `7701.36 GOPS` | `15.7%` | `0.440` | **`25603.06 GOPS`** (`61.1 GB/s`) | `7.27%` | `188402.15 GOPS` | **`13.6%`** | **`13.6%`** |
| `128×8192×5376` | `L3_LLC_to_L2_Tiled` | **YNNPACK** | `ynn::dot_uint8_int4_int32_12x16x8_1x16x8_avx512vnni` (`AVX512-VNNI`) | `419.08` | `20.211` | `557.83 GOPS` (`1.3 GB/s`) | `948.07 GOPS` | `58.8%` | `0.811` | **`13894.89 GOPS`** (`33.2 GB/s`) | `1.52%` | `20973.59 GOPS` | **`66.2%`** | **`7.4%`** |
| `1024×21504×5376` | `Compute_Bound_Tiled` | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx` (`AMX-INT8`) | `1563.93` | `209.847` | `1128.25 GOPS` (`0.7 GB/s`) | `7701.36 GOPS` | `14.7%` | `8.915` | **`26556.31 GOPS`** (`17.0 GB/s`) | `1.76%` | `188402.15 GOPS` | **`14.1%`** | **`14.1%`** |
| `1024×21504×5376` | `Compute_Bound_Tiled` | **YNNPACK** | `ynn::dot_uint8_int4_int32_12x16x8_1x16x8_avx512vnni` (`AVX512-VNNI`) | `1563.93` | `464.298` | `509.93 GOPS` (`0.3 GB/s`) | `948.07 GOPS` | `53.8%` | `18.273` | **`12957.18 GOPS`** (`8.3 GB/s`) | `1.40%` | `20973.59 GOPS` | **`61.8%`** | **`6.9%`** |
| `1024×8192×5376` | `Compute_Bound_Tiled` | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx` (`AMX-INT8`) | `1476.67` | `76.700` | `1175.94 GOPS` (`0.8 GB/s`) | `7701.36 GOPS` | `15.3%` | `3.261` | **`27659.48 GOPS`** (`18.7 GB/s`) | `4.89%` | `188402.15 GOPS` | **`14.7%`** | **`14.7%`** |
| `1024×8192×5376` | `Compute_Bound_Tiled` | **YNNPACK** | `ynn::dot_uint8_int4_int32_12x16x8_1x16x8_avx512vnni` (`AVX512-VNNI`) | `1476.67` | `170.243` | `529.80 GOPS` (`0.4 GB/s`) | `948.07 GOPS` | `55.9%` | `6.764` | **`13335.09 GOPS`** (`9.0 GB/s`) | `5.10%` | `20973.59 GOPS` | **`63.6%`** | **`7.1%`** |

### 4.3 Pair 3: FP32 Attention `BatchMatMul` (`Q·K^T` and `Scores·V` — 36.2% XNNPACK / 17.5% YNNPACK Prefill)

| Shape `(M×N×K)` | Regime | Backend | Microkernel (`ISA`) | AI (`OP/B`) | 1C Latency (`ms`) | 1C Perf | 1C Ceiling | 1C Roofline Util | 28C Latency (`ms`) | 28C Perf | 28C `cv_pct` | 28C ISA Ceiling | **28C ISA Roofline Util** | **28C Machine Peak Util** |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `128×1024×256` | `L2_Cache_Resident` | **XNNPACK** | `xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast` (`AVX512-FP32`) | `39.38` | `0.347` | `193.26 GFLOPS` (`4.9 GB/s`) | `235.72 GFLOPS` | `82.0%` | `0.025` | **`2729.31 GFLOPS`** (`69.3 GB/s`) | `2.14%` | `5616.26 GFLOPS` | **`48.6%`** | **`48.6%`** |
| `128×1024×256` | `L2_Cache_Resident` | **YNNPACK** | `ynn::dot_fp32_5x64x1_1x16x1_avx512` (`AVX512-FP32`) | `39.38` | `0.341` | `196.61 GFLOPS` (`5.0 GB/s`) | `235.72 GFLOPS` | `83.4%` | `0.027` | **`2469.33 GFLOPS`** (`62.7 GB/s`) | `7.97%` | `5616.26 GFLOPS` | **`44.0%`** | **`44.0%`** |
| `1024×1024×256` | `L3_LLC_Resident` | **XNNPACK** | `xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast` (`AVX512-FP32`) | `85.33` | `2.860` | `187.72 GFLOPS` (`2.2 GB/s`) | `235.72 GFLOPS` | `79.6%` | `0.214` | **`2511.19 GFLOPS`** (`29.4 GB/s`) | `1.18%` | `5616.26 GFLOPS` | **`44.7%`** | **`44.7%`** |
| `1024×1024×256` | `L3_LLC_Resident` | **YNNPACK** | `ynn::dot_fp32_5x64x1_1x16x1_avx512` (`AVX512-FP32`) | `85.33` | `3.012` | `178.26 GFLOPS` (`2.1 GB/s`) | `235.72 GFLOPS` | `75.6%` | `0.229` | **`2341.05 GFLOPS`** (`27.4 GB/s`) | `4.74%` | `5616.26 GFLOPS` | **`41.7%`** | **`41.7%`** |
| `1024×4096×512` | `L3_LLC_Resident` | **XNNPACK** | `xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast` (`AVX512-FP32`) | `157.54` | `25.530` | `168.23 GFLOPS` (`1.1 GB/s`) | `235.72 GFLOPS` | `71.4%` | `1.155` | **`3717.83 GFLOPS`** (`23.6 GB/s`) | `0.81%` | `5616.26 GFLOPS` | **`66.2%`** | **`66.2%`** |
| `1024×4096×512` | `L3_LLC_Resident` | **YNNPACK** | `ynn::dot_fp32_5x64x1_1x16x1_avx512` (`AVX512-FP32`) | `157.54` | `23.768` | `180.70 GFLOPS` (`1.1 GB/s`) | `235.72 GFLOPS` | `76.7%` | `1.285` | **`3343.29 GFLOPS`** (`21.2 GB/s`) | `1.51%` | `5616.26 GFLOPS` | **`59.5%`** | **`59.5%`** |
| `2520×2520×72` | `L3_LLC_Resident` | **XNNPACK** | `xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast` (`AVX512-FP32`) | `34.05` | `5.119` | `178.63 GFLOPS` (`5.2 GB/s`) | `235.72 GFLOPS` | `75.8%` | `0.242` | **`3774.38 GFLOPS`** (`110.8 GB/s`) | `14.18%` | `5616.26 GFLOPS` | **`67.2%`** | **`67.2%`** |
| `2520×2520×72` | `L3_LLC_Resident` | **YNNPACK** | `ynn::dot_fp32_5x64x1_1x16x1_avx512` (`AVX512-FP32`) | `34.05` | `5.828` | `156.90 GFLOPS` (`4.6 GB/s`) | `235.72 GFLOPS` | `66.6%` | `0.408` | **`2240.25 GFLOPS`** (`65.8 GB/s`) | `1.21%` | `5616.26 GFLOPS` | **`39.9%`** | **`39.9%`** |

### 4.4 Pair 4: FP32 Attention `Softmax` (`rmax + raddstoreexpminusmax` vs. `max_k1 + sum_k1`)

| Shape `(M×N×K)` | Regime | Backend | Microkernel (`ISA`) | AI (`OP/B`) | 1C Latency (`ms`) | 1C Perf | 1C Ceiling | 1C Roofline Util | 28C Latency (`ms`) | 28C Perf | 28C `cv_pct` | 28C ISA Ceiling | **28C ISA Roofline Util** | **28C Machine Peak Util** |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1024×1024×1` | `L2_L3_Resident` | **XNNPACK** | `xnn_f32_rmax + xnn_f32_raddstoreexpminusmax__avx512f_rr2_p5` (`AVX512-FP32`) | `1.00` | `0.311` | `40.45 GB/s` (`40.5 GB/s`) | `97.37 GB/s` | `41.5%` | `0.007` | **`1741.66 GB/s`** (`1741.7 GB/s`) | `4.78%` | `2216.39 GB/s` | **`78.6%`** | **`78.6%`** |
| `1024×1024×1` | `L2_L3_Resident` | **YNNPACK** | `ynn::max_k1_fp32_avx512 + ynn::sum_k1_fp32_avx512` (`AVX512-FP32`) | `0.25` | `0.474` | `17.71 GB/s` (`17.7 GB/s`) | `97.37 GB/s` | `18.2%` | `0.014` | **`581.66 GB/s`** (`581.7 GB/s`) | `10.69%` | `2216.39 GB/s` | **`26.2%`** | **`26.2%`** |
| `2520×2520×1` | `L3_LLC_Resident` | **XNNPACK** | `xnn_f32_rmax + xnn_f32_raddstoreexpminusmax__avx512f_rr2_p5` (`AVX512-FP32`) | `1.00` | `1.767` | `43.13 GB/s` (`43.1 GB/s`) | `97.37 GB/s` | `44.3%` | `0.049` | **`1560.26 GB/s`** (`1560.3 GB/s`) | `3.23%` | `2216.39 GB/s` | **`70.4%`** | **`70.4%`** |
| `2520×2520×1` | `L3_LLC_Resident` | **YNNPACK** | `ynn::max_k1_fp32_avx512 + ynn::sum_k1_fp32_avx512` (`AVX512-FP32`) | `0.25` | `3.351` | `15.16 GB/s` (`15.2 GB/s`) | `97.37 GB/s` | `15.6%` | `0.116` | **`437.88 GB/s`** (`437.9 GB/s`) | `6.23%` | `2216.39 GB/s` | **`19.8%`** | **`19.8%`** |
| `1024×4096×1` | `L3_LLC_Resident` | **XNNPACK** | `xnn_f32_rmax + xnn_f32_raddstoreexpminusmax__avx512f_rr2_p5` (`AVX512-FP32`) | `1.00` | `1.110` | `45.33 GB/s` (`45.3 GB/s`) | `97.37 GB/s` | `46.5%` | `0.035` | **`1450.52 GB/s`** (`1450.5 GB/s`) | `10.64%` | `2216.39 GB/s` | **`65.5%`** | **`65.5%`** |
| `1024×4096×1` | `L3_LLC_Resident` | **YNNPACK** | `ynn::max_k1_fp32_avx512 + ynn::sum_k1_fp32_avx512` (`AVX512-FP32`) | `0.25` | `2.133` | `15.73 GB/s` (`15.7 GB/s`) | `97.37 GB/s` | `16.1%` | `0.064` | **`522.28 GB/s`** (`522.3 GB/s`) | `11.78%` | `2216.39 GB/s` | **`23.6%`** | **`23.6%`** |

### 4.5 Pair 5: 8-Bit Dynamic-Quantized Linear GEMM (Vision ViT & Audio Conformer — 38%–47% of Vision/Audio)

| Shape `(M×N×K)` | Regime | Backend | Microkernel (`ISA`) | AI (`OP/B`) | 1C Latency (`ms`) | 1C Perf | 1C Ceiling | 1C Roofline Util | 28C Latency (`ms`) | 28C Perf | 28C `cv_pct` | 28C ISA Ceiling | **28C ISA Roofline Util** | **28C Machine Peak Util** |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `280×5376×1152` | `L3_LLC_Resident` | **XNNPACK** | `xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx` (`AMX-INT8`) | `276.64` | `2.462` | `1408.52 GOPS` (`5.1 GB/s`) | `7701.36 GOPS` | `18.3%` | `0.073` | **`47501.19 GOPS`** (`171.7 GB/s`) | `6.61%` | `188402.15 GOPS` | **`25.2%`** | **`25.2%`** |
| `280×5376×1152` | `L3_LLC_Resident` | **YNNPACK** | `ynn::dot_int8_int8_int32_32x32x64_16x16x4_amxint8` (`AMX-INT8`) | `276.64` | `1.357` | `2555.28 GOPS` (`9.2 GB/s`) | `7701.36 GOPS` | `33.2%` | `0.042` | **`81985.79 GOPS`** (`296.4 GB/s`) | `5.38%` | `188402.15 GOPS` | **`43.5%`** | **`43.5%`** |
| `2520×1152×1152` | `L3_LLC_Resident` | **XNNPACK** | `xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx` (`AMX-INT8`) | `422.20` | `5.206` | `1284.84 GOPS` (`3.0 GB/s`) | `7701.36 GOPS` | `16.7%` | `0.330` | **`20242.42 GOPS`** (`48.0 GB/s`) | `0.53%` | `188402.15 GOPS` | **`10.7%`** | **`10.7%`** |
| `2520×1152×1152` | `L3_LLC_Resident` | **YNNPACK** | `ynn::dot_int8_int8_int32_32x32x64_16x16x4_amxint8` (`AMX-INT8`) | `422.20` | `4.887` | `1368.72 GOPS` (`3.2 GB/s`) | `7701.36 GOPS` | `17.8%` | `0.307` | **`21803.46 GOPS`** (`51.6 GB/s`) | `1.40%` | `188402.15 GOPS` | **`11.6%`** | **`11.6%`** |
| `2520×4304×1152` | `L3_LLC_Resident` | **XNNPACK** | `xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx` (`AMX-INT8`) | `487.64` | `19.816` | `1261.10 GOPS` (`2.6 GB/s`) | `7701.36 GOPS` | `16.4%` | `0.993` | **`25170.13 GOPS`** (`51.6 GB/s`) | `4.95%` | `188402.15 GOPS` | **`13.4%`** | **`13.4%`** |
| `2520×4304×1152` | `L3_LLC_Resident` | **YNNPACK** | `ynn::dot_int8_int8_int32_32x32x64_16x16x4_amxint8` (`AMX-INT8`) | `487.64` | `18.349` | `1361.90 GOPS` (`2.8 GB/s`) | `7701.36 GOPS` | `17.7%` | `1.058` | **`23628.80 GOPS`** (`48.5 GB/s`) | `12.30%` | `188402.15 GOPS` | **`12.5%`** | **`12.5%`** |

### 4.6 Pair 6: 2-Bit Weight-Only Linear GEMM (Audio Conformer Encoder — 29%–41% of Audio)

| Shape `(M×N×K)` | Regime | Backend | Microkernel (`ISA`) | AI (`OP/B`) | 1C Latency (`ms`) | 1C Perf | 1C Ceiling | 1C Roofline Util | 28C Latency (`ms`) | 28C Perf | 28C `cv_pct` | 28C ISA Ceiling | **28C ISA Roofline Util** | **28C Machine Peak Util** |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `204×1024×1024` | `L2_Cache_Resident` | **XNNPACK** | `xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni` (`AVX-VNNI-256`) | `629.21` | `1.212` | `352.87 GOPS` (`0.6 GB/s`) | `484.59 GOPS` | `72.8%` | `0.075` | **`5690.16 GOPS`** (`9.0 GB/s`) | `11.49%` | `10667.71 GOPS` | **`53.3%`** | **`3.0%`** |
| `204×1024×1024` | `L2_Cache_Resident` | **YNNPACK** | `ynn::dot_uint8_int2_int32_4x64x16_1x16x16_avx512vnni` (`AVX512-VNNI`) | `629.21` | `0.776` | `551.42 GOPS` (`1.7 GB/s`) | `948.07 GOPS` | `58.2%` | `0.061` | **`7035.89 GOPS`** (`21.5 GB/s`) | `4.65%` | `20973.59 GOPS` | **`33.5%`** | **`3.7%`** |
| `204×4096×1024` | `L2_L3_Resident` | **XNNPACK** | `xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni` (`AVX-VNNI-256`) | `817.60` | `4.883` | `350.43 GOPS` (`0.4 GB/s`) | `484.59 GOPS` | `72.3%` | `0.203` | **`8423.48 GOPS`** (`10.3 GB/s`) | `2.62%` | `10667.71 GOPS` | **`79.0%`** | **`4.5%`** |
| `204×4096×1024` | `L2_L3_Resident` | **YNNPACK** | `ynn::dot_uint8_int2_int32_4x64x16_1x16x16_avx512vnni` (`AVX512-VNNI`) | `817.60` | `3.245` | `527.34 GOPS` (`1.4 GB/s`) | `948.07 GOPS` | `55.6%` | `0.172` | **`9973.45 GOPS`** (`26.8 GB/s`) | `2.60%` | `20973.59 GOPS` | **`47.5%`** | **`5.3%`** |
| `204×1024×4096` | `L2_L3_Resident` | **XNNPACK** | `xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni` (`AVX-VNNI-256`) | `817.60` | `4.658` | `367.37 GOPS` (`0.5 GB/s`) | `484.59 GOPS` | `75.8%` | `0.209` | **`8180.97 GOPS`** (`10.0 GB/s`) | `2.16%` | `10667.71 GOPS` | **`76.7%`** | **`4.3%`** |
| `204×1024×4096` | `L2_L3_Resident` | **YNNPACK** | `ynn::dot_uint8_int2_int32_4x64x16_1x16x16_avx512vnni` (`AVX512-VNNI`) | `817.60` | `3.123` | `547.94 GOPS` (`0.9 GB/s`) | `948.07 GOPS` | `57.8%` | `0.227` | **`7529.66 GOPS`** (`12.0 GB/s`) | `2.55%` | `20973.59 GOPS` | **`35.9%`** | **`4.0%`** |

### 4.7 Pair 7: Matrix Packing / 32-Bit Transpose (`packw_gemm_goi` vs. `transpose_x32_avx512`)

| Shape `(M×N×K)` | Regime | Backend | Microkernel (`ISA`) | AI (`OP/B`) | 1C Latency (`ms`) | 1C Perf | 1C Ceiling | 1C Roofline Util | 28C Latency (`ms`) | 28C Perf | 28C `cv_pct` | 28C ISA Ceiling | **28C ISA Roofline Util** | **28C Machine Peak Util** |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1×1024×256` | `L2_Cache_Copy` | **XNNPACK** | `xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4_prfm` (`AVX512F`) | `0.00` | `0.038` | `55.01 GB/s` (`55.0 GB/s`) | `97.37 GB/s` | `56.5%` | `0.002` | **`912.99 GB/s`** (`913.0 GB/s`) | `1.56%` | `2216.39 GB/s` | **`41.2%`** | **`41.2%`** |
| `1×1024×256` | `L2_Cache_Copy` | **YNNPACK** | `ynn::transpose_x32_avx512` (`AVX512F`) | `0.00` | `0.057` | `36.81 GB/s` (`36.8 GB/s`) | `97.37 GB/s` | `37.8%` | `0.003` | **`746.86 GB/s`** (`746.9 GB/s`) | `10.95%` | `2216.39 GB/s` | **`33.7%`** | **`33.7%`** |
| `1×4096×512` | `L3_LLC_Copy` | **XNNPACK** | `xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4_prfm` (`AVX512F`) | `0.00` | `0.633` | `26.51 GB/s` (`26.5 GB/s`) | `28.73 GB/s` | `92.3%` | `0.014` | **`1163.57 GB/s`** (`1163.6 GB/s`) | `2.29%` | `2216.39 GB/s` | **`52.5%`** | **`52.5%`** |
| `1×4096×512` | `L3_LLC_Copy` | **YNNPACK** | `ynn::transpose_x32_avx512` (`AVX512F`) | `0.00` | `0.925` | `18.14 GB/s` (`18.1 GB/s`) | `28.73 GB/s` | `63.1%` | `0.026` | **`650.70 GB/s`** (`650.7 GB/s`) | `0.24%` | `2216.39 GB/s` | **`29.4%`** | **`29.4%`** |
| `1×2520×1152` | `L3_LLC_Copy` | **XNNPACK** | `xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4_prfm` (`AVX512F`) | `0.00` | `0.844` | `27.51 GB/s` (`27.5 GB/s`) | `28.73 GB/s` | `95.8%` | `0.015` | **`1585.56 GB/s`** (`1585.6 GB/s`) | `4.10%` | `2216.39 GB/s` | **`71.5%`** | **`71.5%`** |
| `1×2520×1152` | `L3_LLC_Copy` | **YNNPACK** | `ynn::transpose_x32_avx512` (`AVX512F`) | `0.00` | `1.331` | `17.45 GB/s` (`17.4 GB/s`) | `28.73 GB/s` | `60.7%` | `0.018` | **`1315.26 GB/s`** (`1315.3 GB/s`) | `23.07%` | `2216.39 GB/s` | **`59.3%`** | **`59.3%`** |

---

## 5. Architectural Diagnosis: Comparing Roofline Ceilings vs. Achieved Performance

### 5.1 Why Both Backends Hit >91% of Roofline in Decode (`M = 1`)
- **Roofline Math**: For `M = 1` 4-bit GEMV (`N = 5376, K = 21504`), each weight byte (`2 × INT4` weights) performs `4` MACs = `8` INT8 OPs, giving $AI = 4.0\text{ OPs/Byte}$. Against the 28-core DDR5 DRAM streaming read ceiling ($BW_{\text{prac}} = 246.60\text{ GB/s}$), the maximum achievable throughput on this machine is:
  $$\text{GOPS}_{\text{ceil}}(M=1) = 3.997 \times 246.60\text{ GB/s} = \mathbf{985.7\text{ GOPS}}$$
- **Measured Performance**:
  - **XNNPACK (`1x64c4__avx512amx`)**: `927.78 GOPS` / `232.12 GB/s` (**94.1% of practical DDR5 roofline**).
  - **YNNPACK (`dot_uint8_int4_int32_1x32x8_1x16x8_avx512vnni`)**: `912.63 GOPS` / `228.33 GB/s` (**92.6% of practical DDR5 roofline**).
- **Conclusion**: Neither XNNPACK nor YNNPACK can speed up Gemma 4 31B `M = 1` cold-weight decode by more than `~6–8%` via compute microkernel changes alone, because both are already saturating >91% of Socket 0's DDR5 memory channels. Further `M = 1` decode gains require **reducing bytes read per token** (e.g., 2-bit/3-bit quantization, speculative decoding, or multi-socket NUMA weight sharding across both sockets to double DRAM bandwidth to `~493 GB/s`).

### 5.2 Why YNNPACK Hits a Hard Wall in 4-Bit Prefill (`M = 128..1024`)
- **Roofline Math**: At `M = 128`, $AI = 425.8\text{ OPs/Byte}$, which is well above the ridge point for both AVX-512 VNNI ($AI_{\text{ridge}} = 20,974 / 780.4 = 26.9\text{ OPs/Byte}$) and AMX-INT8 ($AI_{\text{ridge}} = 188,402 / 780.4 = 241.4\text{ OPs/Byte}$).
- **Measured Performance vs. Ceiling**:
  - **YNNPACK (`12x16x8_avx512vnni`)**: Achieves **14,512.43 GOPS** (`14.51 TOPS`), which is **69.2% of the 28-core AVX-512 VNNI ceiling (`20,973.59 GOPS`)**, but only **7.7% of the machine's 28-core AMX-INT8 ceiling (`188,402.15 GOPS`)**.
  - **XNNPACK (`16x64c4__avx512amx`)**: Achieves **32,527.90–33,918.61 GOPS** (`32.53–33.92 TOPS`), which is **18.0% of the AMX-INT8 ceiling** and **161.7% of the entire AVX-512 VNNI ceiling**.
- **Why XNNPACK Only Reaches 18% of Pure `tdpbssd` AMX-INT8 Peak on 4-Bit Weights**:
  - Sapphire Rapids AMX `tdpbssd` only accepts unpacked 8-bit integers (`int8`/`uint8`), not packed 4-bit nibbles.
  - Inside `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx`, for every `64×64` weight block (`2,048` bytes of 4-bit weights), the CPU must:
    1. Load `2,048` bytes into ZMM registers (`vmovdqa64`),
    2. Unpack low/high nibbles into `4,096` bytes of `int8` via `vpslld`/`vpandq`,
    3. Spill all `4,096` bytes to the stack (`vmovdqa64 %zmm, 0x1c0(%rsp)..0x580(%rsp)`), and
    4. Reload them into `tmm` tiles via `tileloadd` (`4,096` bytes read from L1D per `64×64` step).
  - Because `4,096` bytes of `tileloadd` + `4,096` bytes of stack stores = `8,192` bytes of L1D traffic per `4 × tdpbssd` (`131,072` OPs, or `16 OPs/B` of L1D traffic), the kernel is bounded by **L1D cache read+write bandwidth (`5,560 GB/s`) + AVX-512 unpack port pressure**, not raw TMUL pipes!
  - Even with this L1D stack-bounce overhead, XNNPACK's AMX kernel is **2.24× to 2.42× faster** than YNNPACK's AVX-512 VNNI kernel.

### 5.3 Why YNNPACK's `32×32` AMX-INT8 Kernel Beats XNNPACK's `16×64` AMX-INT8 Kernel on 8-Bit Weights (`pair5_int8_gemm`)
- When weights are already 8-bit (`qc8w`), no 4-bit nibble unpacking or stack spilling is needed — `tileloadd` streams directly from the pre-packed weight buffer!
- On `280×5376×1152`, **YNNPACK (`dot_int8_int8_int32_32x32x64_16x16x4_amxint8`) achieves `81.99 TOPS` (43.5% of 28-core AMX-INT8 ceiling)** vs. **XNNPACK (`16x64c4__avx512amx`) at `47.50 TOPS` (25.2% of ceiling)**.
- **Root Cause**:
  1. YNNPACK uses a **`2×2` tile register blocking (`32×32` output using 4 accumulator tiles `tmm0..tmm3`, 2 `A` tiles, 2 `B` tiles)**: each `64`-byte `A` or `B` tile load feeds **2** `tdpbssd` instructions (`4` tile loads for `4` `tdpbssd` instructions).
  2. XNNPACK uses a **`1×4` tile register blocking (`16×64` output using 4 accumulator tiles, 1 `A` tile, 4 `B` tiles)**: `5` tile loads for `4` `tdpbssd` instructions, plus XNNPACK spills the `16×64` `int32` tiles to the stack on completion to perform AVX-512 FP32 dequantization.
  3. **Actionable Takeaway for YNNPACK**: If YNNPACK fuses its `dequantize_dot_f32_avx512` epilogue directly into `dot_int8_int8_int32_32x32x64_16x16x4_amxint8` (avoiding the separate memory trip over the `M×N` `int32` output buffer) AND applies the same `32×32` AMX tile blocking to 4-bit (`uint8_int4`) weights, YNNPACK will surpass XNNPACK in both 8-bit and 4-bit prefill!

### 5.4 Cache Level Transitions: Aggregate L2 Partitioning (`56 MB`) and In-L1D Row Reuse
- **Why Warm `1×8192×5376` (`22 MB`) and `1×4096×5376` (`11 MB`) Transition from L3 on 1 Core to Private L2 on 28 Cores**:
  - On a **single core (`1C`)**, a core has only `2 MB` of private L2 cache, so an `11–22 MB` weight matrix spills into the `52.5 MB` Shared L3 LLC (`33.03 GB/s` 1C Read ceiling $\Rightarrow 131.9\text{ GOPS}$ ceiling). YNNPACK reaches `30.4–30.5 GB/s` (`92.2%–92.4%` of the 1C L3 ceiling).
  - On **28 cores (`28C`)**, each physical core has its own `2 MB` private L2 cache, yielding **`28 × 2 MB = 56 MB` of aggregate L2 capacity** on Socket 0 (larger than the `52.5 MB` shared L3!). Under static `N`-partitioning, each core owns only `22 MB / 28 = 786 KB` (or `11 MB / 28 = 393 KB`), which fits **100% inside its private `2 MB` L2 cache**. Consequently, on 28 cores the warm benchmark achieves **`1,136.4 GB/s` (`4,537.57 GOPS`)** — which would appear to be **`145.6%`** if mistakenly normalized against the 28-core L3 ceiling (`780.35 GB/s`), but is **`25.4%`** of the true 28-core L2 Read ceiling (`4,473.83 GB/s` / `17,863.3 GOPS`).
- **Why Row-Wise Multi-Pass Softmax (`2520×2520` = `25.4 MB`, `1024×4096` = `16.8 MB`) Operates at L2/L1D Speeds Rather Than Cold L3 Speeds**:
  - Although the full `M×N` attention matrix (`16.8–25.4 MB`) resides in L3 on 1 core, Softmax executes **row by row** (`N = 2520` = `10.1 KB` or `N = 4096` = `16.4 KB` per row), which fits comfortably inside the **`48 KB` L1D cache**.
  - Pass 1 (`xnn_f32_rmax`) streams the row from L3/L2 into L1D, and Pass 2 (`xnn_f32_raddstoreexpminusmax`) immediately re-reads the hot row from **L1D (`284.72 GB/s`)** rather than L3 (`33.03 GB/s`). Counting all logical pass bytes (`43.1–45.3 GB/s` on 1C) against the single-core L3 ceiling (`33.03 GB/s`) would yield an apparent **`130.6%–137.2%`**, whereas against the L2/L1D Read+Write Copy ceiling (`97.37 GB/s` 1C, `2,216.39 GB/s` 28C) it is **`44.3%–46.6%` (1C)** and **`65.5%–70.4%` (28C)**.

---

## 6. Reproducing the Probes and Benchmarks

All source code and automation scripts are checked into the repository and wired into [kernel/Makefile](../../Makefile):

```bash
# 1. Compile and run the repeatable 7-trial hardware ceiling probe
gcc -O3 -march=sapphirerapids -fopenmp kernel/probes/cpu_ceilings.c -o /tmp/cpu_ceilings -lm
/tmp/cpu_ceilings --threads 28 --trials 7 --output kernel/reports/gemma4-31b-sapphire-rapids/ceilings_sapphire_rapids.json

# 2. Compile and run the 7-pair standalone microkernel benchmark harness
gcc -shared -fPIC -O3 -march=sapphirerapids -fopenmp kernel/scripts/bench_litert_microkernels.c -o /tmp/libbench_ukernels.so -lm
LD_LIBRARY_PATH=/opt/litert-lm/lib:/opt/litert-lm/bin \
  LD_PRELOAD=/tmp/libbench_ukernels.so \
  AEB_BENCH_OUTPUT=kernel/reports/gemma4-31b-sapphire-rapids/ukernel_benchmarks_sapphire_rapids.json \
  /opt/litert-lm/bin/litert_lm_advanced_main

# 3. Generate the JSON & Markdown roofline comparison report
python3 kernel/scripts/generate_roofline_report.py
```
