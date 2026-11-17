# Microkernel Performance, Roofline & Numerical Precision Report — Gemma 4 31B on Intel Sapphire Rapids

- **Benchmark Date**: `2026-11-13`
- **Model**: Gemma 4 31B (`gemma-4-31b-it`, 4-bit Decoder / 8-bit Vision ViT / 2-bit & 8-bit Audio Conformer)
- **Target Microarchitecture**: Intel Sapphire Rapids (`x86_64`, AVX-512F, AVX-512 VNNI, AVX-VNNI, GFNI, AMX-INT8, AMX-BF16; `56` physical cores / `105 MiB` shared L3 LLC / `2 MiB` private L2 per core / `48 KiB` L1D per core on Socket 0)
- **Upstream Microkernel Revision**: `google/XNNPACK` `master` @ [`6e5b0566`](https://github.com/google/XNNPACK/commit/6e5b056669823d8501f9921a2c9059d6ddbb2ceb) (includes YNNPACK AMX-INT8+GFNI 4-bit `32×32` and XNNPACK GFNI-optimized AMX 4-bit `1x64c4`/`16x64c4`)
- **Execution Configurations**:
  - **Single-Core (`1T`)**: Pinned to physical core `0` (`sched_setaffinity`)
  - **Half-Socket (`28T`)**: Pinned across physical cores `0..27` of NUMA Node 0 (1 thread per physical core, for mid-socket scaling comparison)
  - **Full NUMA Node 0 (`56T`)**: Pinned across all `56` physical cores of Socket 0 (`0..55`, 1 thread per physical core, no SMT sibling contention, zero cross-socket UPI traffic)
- **Repeatability & Verification Methodology**:
  - **Performance & Roofline**: `CLOCK_MONOTONIC_RAW` with `#pragma omp barrier` synchronization before and after every timed loop, `OMP_WAIT_POLICY=ACTIVE`, explicit warmup pass, and **7 independent timed trials** per probe and microkernel shape (`median`, `mean`, `min`, `max`, `stdev`, `cv_pct`)
  - **Numerical Precision & Quantization Fidelity**: Baseline two-level verification from `2026-11-10` ([verify_litert_microkernels.c](../../scripts/verify_litert_microkernels.c)) retained in Section 5 (not re-run for `6e5b0566`; built-in standalone numerical smoke checks verified exact `INT32` dot outputs and finite non-zero `FP32` outputs prior to timing)
- **Previous Kernel Report (`2026-11-10`)**: [2026-11-10-gemma4-31b-sapphire-rapids/kernel_report_gemma4_31b_sapphire_rapids.md](../2026-11-10-gemma4-31b-sapphire-rapids/kernel_report_gemma4_31b_sapphire_rapids.md)
- **Companion End-to-End Profiling Report (`2026-11-10` baseline)**: [kernels_gemma4_31b_sapphire_rapids.md](../2026-11-10-gemma4-31b-sapphire-rapids/kernels_gemma4_31b_sapphire_rapids.md)

---

## 1. Executive Summary & Key Roofline Findings

> [!IMPORTANT]
> **Performance Roofline Summary Across All 7 Load-Bearing Microkernel Pairs (`1T` / `28T` / `56T`, Upstream `google/XNNPACK` `6e5b0566`)**:
> 1. **Decode (`M = 1` 4-Bit GEMV, 83%–87% of Decode Time) — Cold Weights Saturate Socket 0 DDR5 (`245.44 GB/s`), While Warm Cache Exposes an `M = 1` AMX Cost-Model Gap**:
>    - At `M = 1`, arithmetic intensity is only **$3.99\text{ OPs/Byte}$**, placing cold-weight decode deep in the **DDR5 DRAM bandwidth-bound regime** ($BW_{\text{prac}}(\text{DDR5 Read}) = \mathbf{245.44\text{ GB/s}}$ on 56 cores $\Rightarrow \mathbf{981.0\text{ GOPS}}$ ceiling).
>    - On cold DDR5-streaming weights (`1×21504×5376` and `1×5376×21504`), **XNNPACK achieves `230.1–254.6 GB/s` (`93.7%–103.7%` of practical DDR5 ceiling)** and **YNNPACK achieves `233.9–251.4 GB/s` (`95.3%–102.4%` of practical DDR5 ceiling)** at 56C (matching prior VNNI at `256.9 GB/s`).
>    - Conversely, when weights fit warm in L3/L2 cache (`1×8192×5376`, `L3_LLC_Warm`), **a `get_dot_kernel()` cost-model regression is exposed**: upstream `6e5b0566` dispatches YNNPACK's `32x32x64_amxint8` even at `M = 1` (**2.87 TOPS** @ 56C vs. XNNPACK `1x64c4__avx512amx` at **2.19 TOPS**), whereas YNNPACK's prior `1x32x8_avx512vnni` reaches **5.35 TOPS** (**1.87× faster** than YNNPACK AMX and **2.44× faster** than XNNPACK AMX) because `M = 1` utilizes only 1 of 16 AMX tile rows (`6.25%`) while still paying full GFNI unpack + stack-bounce (`vmovdqa64` → `tileloadd`) overhead.
> 2. **Prefill (`M = 128..1024` 4-Bit GEMM, 46%–65% of Prefill Time) — Upstream YNNPACK AMX-INT8+GFNI (`6e5b0566`) Surpasses Both Prior VNNI (`5.93×`) and XNNPACK AMX (`1.63×`)**:
>    - Our compute ceiling probes measure **$C_{\text{prac}}(\text{AMX-INT8}) = \mathbf{223.84\text{ TOPS}}$** (`4.06 TOPS`/core) vs. **$C_{\text{prac}}(\text{AVX-512 VNNI}) = \mathbf{27.62\text{ TOPS}}$** (`0.50 TOPS`/core) on 56 cores — an **8.10× hardware ceiling advantage** for AMX-INT8.
>    - YNNPACK's new `ynn::dot_uint8_int4_int32_32x32x64_16x16x8_amxint8` (GFNI `vgf2p8affineqb` nibble unpack + `2×2` AMX `tdpbusd` tile blocking) achieves **73.76 TOPS** at `128×21504×5376` (`33.0%` of 56C AMX-INT8 ceiling) and **61.59–64.50 TOPS** at `M = 1024`.
>    - This represents a **5.93× speedup** over YNNPACK's prior AVX-512 VNNI kernel (`12.44 TOPS`, `45.0%` of VNNI ceiling) and a **1.63× speedup** over XNNPACK's GFNI-optimized `16x64c4__avx512amx` (**45.20 TOPS**, `20.2%` of AMX ceiling; **43.74 TOPS** at `M = 1024`) because `2×2` tile blocking (`32×32`) amortizes every unpacked `B`-tile across **2** `tdpbusd` instructions instead of 1 (and omits XNNPACK's fused `FP32` epilogue).
> 3. **8-Bit Vision/Audio Linear GEMM (`pair5_int8_gemm`) — Both Use AMX-INT8, and YNNPACK's `32×32` `2×2` Tile Blocking Beats XNNPACK's `16×64` `1×4` Tile Blocking**:
>    - On `280×5376×1152`, YNNPACK's `dot_int8_int8_int32_32x32x64_16x16x4_amxint8` reaches **118.36 TOPS (`52.9%` of 56C AMX-INT8 ceiling)** vs. XNNPACK's **84.25 TOPS (`37.6%`)** (`1.24×` faster on 1C, `1.40×` faster on 56C), and on `2520×4304×1152` YNNPACK leads **64.08 TOPS** vs. **48.51 TOPS**.
> 4. **2-Bit Audio Conformer GEMM (`pair6_int2_gemm`) — 512-Bit `AVX-512 VNNI` (`27.62 TOPS` Ceiling) vs. 256-Bit `AVX-VNNI` (`14.06 TOPS` Ceiling)**:
>    - XNNPACK's 256-bit `5x8c8__avxvnni` hits **10.12 TOPS (`72.0%` of the 256-bit AVX-VNNI ceiling)** on `204×4096×1024`, leaving half of Sapphire Rapids' 512-bit vector width unused. YNNPACK's 512-bit `4x64x16_avx512vnni` reaches **13.89–14.92 TOPS** (`1.47×` faster on 1C, `1.37×–1.41×` faster on 56C).
> 5. **FP32 Attention `BatchMatMul` & `Softmax` (`pair3` & `pair4`)**:
>    - With 2D N-major tile partitioning and zero-init accumulators (`C_in = NULL`), both XNNPACK `7x32` and YNNPACK `5x64` reach **82.5%–93.2% of the 56-core AVX-512 FP32 FMA ceiling (`7.09 TFLOPS`)** (`5.85` vs. `6.20 TFLOPS` on `1024×4096×512`, and `5.74` vs. `4.94 TFLOPS` on `2520×2520×72`).
>    - In Softmax (`1024×4096` and `2520×2520`), XNNPACK's fused 2-pass `rmax + raddstoreexpminusmax` achieves **2,010.5–2,036.4 GB/s (`74.8%–75.7%` of 56C L2 Read+Write Copy)** while computing polynomial `exp(x - max)` in-flight; YNNPACK's read-only reductions (`max_k1 + sum_k1`, without `exp`) achieve **3,332.8–4,010.6 GB/s (`59.6%–71.8%` of 56C L2 Read-Only)** after correcting the prior harness's element-count (`k = row_len`) and 4-row batching parameters.
> 6. **32-Bit Pack / Transpose (`pair7_pack_transpose`)**:
>    - XNNPACK's `packw_gemm_goi_x32` reaches **1,122.3–1,505.5 GB/s (`41.7%–56.0%` of 56C L2 Copy)** vs. YNNPACK's `transpose_x32_avx512` at **475.1–1,172.3 GB/s** (`1.28×–2.36×` advantage for XNNPACK's software-prefetched `u4_prfm` packing loop).

- **Socket 0 Scaling (`1T` → `28T` → `56T`)**: In-register compute ceilings scale nearly linearly across all 56 physical cores (AMX-INT8: `4.06` → `113.90` → **`223.84 TOPS`**, `55.2×`; L2 Read: `99.9` → `2,802.4` → **`5,587.4 GB/s`**, `55.9×`), whereas Socket 0's 8-channel DDR5 streaming read ceiling saturates by 28 cores (`252.19 GB/s` @ 28C vs. **`245.44 GB/s`** @ 56C).

---

## 2. Visual Summary: Throughput, Roofline Utilization, 4-Bit Evolution & Baseline Precision

> **Series Legend**: **Left Bar (Blue `#2b6cd9`)** = **XNNPACK** | **Right Bar (Orange `#e67e22`)** = **YNNPACK**

```mermaid
%%{init: {"xyChart": {"width": 1250, "height": 480, "xAxis": {"labelRotation": 0, "labelFontSize": 11, "showTick": false}}, "themeVariables": {"xyChart": {"plotColorPalette": "#2b6cd9, #e67e22"}}}}%%
xychart-beta
    title "56-Core Compute Throughput: XNNPACK (Blue) vs YNNPACK (Orange) [TOPS / TFLOPS]"
    x-axis "Kernel & Shape" ["⠀⠀⠀P2: INT4 128x21504", "​​‌", "​​‍", "​​⁠", "⠀⠀⠀P2: INT4 128x5376", "​‌​", "​‌‌", "​‌‍", "⠀⠀⠀P2: INT4 1024x21504", "​‌⁠", "​‍​", "​‍‌", "⠀⠀⠀P5: INT8 280x5376", "​‍‍", "​‍⁠", "​⁠​", "⠀⠀⠀P5: INT8 2520x1152", "​⁠‌", "​⁠‍", "​⁠⁠", "⠀⠀⠀P6: INT2 204x4096", "‌​​", "‌​‌", "‌​‍", "⠀⠀⠀P3: FP32 1024x4096", "‌​⁠"]
    y-axis "TOPS / TFLOPS (56 Cores)" 0 --> 140
    bar "XNNPACK" [45.2, -140, -140, -140, 46.88, -140, -140, -140, 43.74, -140, -140, -140, 84.25, -140, -140, -140, 74.12, -140, -140, -140, 10.12, -140, -140, -140, 5.85, -140]
    bar "YNNPACK" [-140, 73.76, -140, -140, -140, 60.76, -140, -140, -140, 61.59, -140, -140, -140, 118.36, -140, -140, -140, 65.03, -140, -140, -140, 13.89, -140, -140, -140, 6.2]
```

```mermaid
%%{init: {"xyChart": {"width": 1250, "height": 480, "xAxis": {"labelRotation": 0, "labelFontSize": 11, "showTick": false}}, "themeVariables": {"xyChart": {"plotColorPalette": "#2b6cd9, #e67e22"}}}}%%
xychart-beta
    title "56-Core Practical ISA Roofline Utilization (%): XNNPACK (Blue) vs YNNPACK (Orange)"
    x-axis "Kernel & Shape" ["⠀⠀⠀P1: INT4 (DRAM)", "​​‌", "​​‍", "​​⁠", "⠀⠀⠀P1: INT4 (L2/L3)", "​‌​", "​‌‌", "​‌‍", "⠀⠀⠀P2: INT4 (M=128)", "​‌⁠", "​‍​", "​‍‌", "⠀⠀⠀P3: FP32 BMM", "​‍‍", "​‍⁠", "​⁠​", "⠀⠀⠀P4: FP32 Softmax", "​⁠‌", "​⁠‍", "​⁠⁠", "⠀⠀⠀P5: INT8 (M=280)", "‌​​", "‌​‌", "‌​‍", "⠀⠀⠀P6: INT2 (Audio)", "‌​⁠", "‌‌​", "‌‌‌", "⠀⠀⠀P7: Pack/Transpose", "‌‌‍"]
    y-axis "Roofline Utilization (%)" 0 --> 100
    bar "XNNPACK" [93.74, -100, -100, -100, 9.82, -100, -100, -100, 20.19, -100, -100, -100, 82.53, -100, -100, -100, 74.75, -100, -100, -100, 37.64, -100, -100, -100, 71.97, -100, -100, -100, 55.98, -100]
    bar "YNNPACK" [-100, 95.31, -100, -100, -100, 12.85, -100, -100, -100, 32.95, -100, -100, -100, 87.5, -100, -100, -100, 71.78, -100, -100, -100, 52.88, -100, -100, -100, 50.28, -100, -100, -100, 43.59]
```

> **4-Bit Evolution Legend**: **Left (Teal `#16a085`)** = Prior YNNPACK AVX-512 VNNI | **Middle (Blue `#2b6cd9`)** = XNNPACK AMX-INT8+GFNI (`1x64c4`/`16x64c4`) | **Right (Orange `#e67e22`)** = New YNNPACK AMX-INT8+GFNI (`32x32x64`)

```mermaid
%%{init: {"xyChart": {"width": 1250, "height": 480, "xAxis": {"labelRotation": 0, "labelFontSize": 11, "showTick": false}}, "themeVariables": {"xyChart": {"plotColorPalette": "#16a085, #2b6cd9, #e67e22"}}}}%%
xychart-beta
    title "56-Core 4-Bit Throughput Evolution: Prior YNN VNNI vs XNN AMX vs New YNN AMX [TOPS]"
    x-axis "4-Bit Shape (M x N x K)" ["​​‌", "M=1 L3 8192", "​​‍", "​​⁠", "​‌​", "​‌‌", "M=128 21504", "​‌‍", "​‌⁠", "​‍​", "​‍‌", "M=128 5376", "​‍‍", "​‍⁠", "​⁠​", "​⁠‌", "M=128 8192", "​⁠‍", "​⁠⁠", "‌​​", "‌​‌", "M=1024 21504", "‌​‍", "‌​⁠", "‌‌​", "‌‌‌", "M=1024 8192", "‌‌‍"]
    y-axis "TOPS (56 Cores)" 0 --> 90
    bar "YNNPACK VNNI (Prior)" [5.35, -90, -90, -90, -90, 12.44, -90, -90, -90, -90, 12.59, -90, -90, -90, -90, 11.33, -90, -90, -90, -90, 12.26, -90, -90, -90, -90, 12.5, -90, -90]
    bar "XNNPACK AMX (GFNI)" [-90, 2.19, -90, -90, -90, -90, 45.2, -90, -90, -90, -90, 46.88, -90, -90, -90, -90, 43.01, -90, -90, -90, -90, 43.74, -90, -90, -90, -90, 44.37, -90]
    bar "YNNPACK AMX (GFNI)" [-90, -90, 2.87, -90, -90, -90, -90, 73.76, -90, -90, -90, -90, 60.76, -90, -90, -90, -90, 70.3, -90, -90, -90, -90, 61.59, -90, -90, -90, -90, 64.5]
```

```mermaid
%%{init: {"xyChart": {"width": 1250, "height": 480, "xAxis": {"labelRotation": 0, "labelFontSize": 11, "showTick": false}}, "themeVariables": {"xyChart": {"plotColorPalette": "#2b6cd9, #e67e22"}}}}%%
xychart-beta
    title "Baseline Numerical Fidelity vs. FP64 Golden Reference (Level B SQNR in dB, 200 dB = Bit-Exact)"
    x-axis "Kernel & Shape" ["⠀⠀⠀P1: INT4 GEMV", "​​‌", "​​‍", "​​⁠", "⠀⠀⠀P2: INT4 GEMM", "​‌​", "​‌‌", "​‌‍", "⠀⠀⠀P3: FP32 BMM", "​‌⁠", "​‍​", "​‍‌", "⠀⠀⠀P4: FP32 Softmax", "​‍‍", "​‍⁠", "​⁠​", "⠀⠀⠀P5: INT8 GEMM", "​⁠‌", "​⁠‍", "​⁠⁠", "⠀⠀⠀P6: INT2 GEMM", "‌​​", "‌​‌", "‌​‍", "⠀⠀⠀P7: Pack/Transpose", "‌​⁠"]
    y-axis "SQNR vs FP64 Golden (dB)" 0 --> 200
    bar "XNNPACK" [21.31, -200, -200, -200, 19.41, -200, -200, -200, 134.83, -200, -200, -200, 141.54, -200, -200, -200, 41.85, -200, -200, -200, 5.76, -200, -200, -200, 200.0, -200]
    bar "YNNPACK" [-200, 20.57, -200, -200, -200, 20.44, -200, -200, -200, 134.18, -200, -200, -200, 143.57, -200, -200, -200, 41.98, -200, -200, -200, 6.16, -200, -200, -200, 200.0]
```

---

## 3. Empirical Hardware Ceilings on Intel Sapphire Rapids (`cpu_ceilings.c`)

All ceilings below were measured directly on the target 56-core Sapphire Rapids socket using [cpu_ceilings.c](../../probes/cpu_ceilings.c) across **7 independent barrier-synchronized trials** (`OMP_WAIT_POLICY=ACTIVE`, NUMA Node 0 physical cores `0..55`) after explicit warmup.

### 3.1 Practical Compute Ceilings: $C_{\text{prac}}(\text{op}, \text{dtype})$

| Instruction Set / Compute Unit | Instruction | Precision | Ops / Inst | 1-Core Median (`cv%`) | 28-Core (NUMA 0) Median (`cv%`) | 56-Core (NUMA 0) Median (`cv%`) | Rel. to AMX-INT8 (56C) | Load-Bearing Kernels Governed |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **AMX-INT8 (`TMUL`)** | `tdpbssd` (`16×16×64`) | `INT8 / u8×s8→s32` | `32,768` | **4,055.13 GOPS** (`0.07%`) | **113,896.09 GOPS (113.90 TOPS)** (`0.06%`) | **223,844.55 GOPS (223.84 TOPS)** (`1.12%`) | `1.000×` (`100.0%`) | `xnn_qd8_f32_qc4w_*__avx512amx`, `xnn_qd8_f32_qc8w_*__avx512amx`, `ynn::dot_uint8_int4_*_amxint8`, `ynn::dot_int8_int8_*_amxint8` |
| **AMX-BF16 (`TMUL`)** | `tdpbf16ps` (`16×16×32`) | `BF16→FP32` | `16,384` | **2,027.60 GFLOPS** (`0.05%`) | **56,965.18 GFLOPS (56.97 TFLOPS)** (`0.03%`) | **111,851.68 GFLOPS (111.85 TFLOPS)** (`0.85%`) | `0.500×` (`50.0%`) | Reference BF16 TMUL ceiling |
| **AVX-512 VNNI (512-bit)** | `vpdpbusd %zmm` | `u8×s8→s32` | `128` | **500.75 GOPS** (`0.04%`) | **14,040.06 GOPS (14.04 TOPS)** (`0.28%`) | **27,618.92 GOPS (27.62 TOPS)** (`0.26%`) | `0.123×` (`12.3%`) | `ynn::dot_uint8_int2_int32_*_avx512vnni`, prior `ynn::dot_uint8_int4_*_avx512vnni` |
| **AVX-VNNI (256-bit)** | `vpdpbusd %ymm` | `u8×s8→s32` | `64` | **253.36 GOPS** (`0.09%`) | **7,096.48 GOPS (7.10 TOPS)** (`0.07%`) | **14,063.79 GOPS (14.06 TOPS)** (`0.91%`) | `0.063×` (`6.3%`) | `xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni` |
| **AVX-512 FP32 FMA (512-bit)** | `vfmadd231ps %zmm` | `FP32` | `32` | **127.01 GFLOPS** (`0.09%`) | **3,560.80 GFLOPS (3.56 TFLOPS)** (`0.27%`) | **7,091.48 GFLOPS (7.09 TFLOPS)** (`2.07%`) | `0.032×` (`3.2%`) | `xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast`, `ynn::dot_fp32_*_avx512` |

### 3.2 Practical Memory-Hierarchy Bandwidth Ceilings: $BW_{\text{prac}}(\text{WorkingSet})$

| Memory Hierarchy Level | Working Set | Access Pattern | 1-Core Median (`cv%`) | 28-Core (NUMA 0) Median (`cv%`) | 56-Core (NUMA 0) Median (`cv%`) | Primary Kernel Regimes Governed |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| **L1D Cache** (`48 KiB`/core) | `16 KB` / core | Streaming Read (`vmovaps`) | **149.11 GB/s** (`0.40%`) | **4,033.38 GB/s** (`1.11%`) | **8,095.45 GB/s** (`0.51%`) | Microkernel register-tile A/B streaming |
| **L1D Cache** (`48 KiB`/core) | `16 KB` / core | Read + Write Copy | **246.47 GB/s** (`0.09%`) | **6,888.32 GB/s** (`0.04%`) | **13,773.29 GB/s** (`0.01%`) | AMX stack-spill `vmovdqa64` / `tileloadd` / `tilestored` scratchpad |
| **L2 Cache** (`2 MiB`/core) | `512 KB` / core | Streaming Read (`vmovaps`) | **99.92 GB/s** (`0.09%`) | **2,802.38 GB/s** (`0.52%`) | **5,587.42 GB/s** (`1.53%`) | L2-tiled GEMM panels (`M=128..204`), L2-sliced warm GEMV & Softmax reductions |
| **L2 Cache** (`2 MiB`/core) | `512 KB` / core | Read + Write Copy | **51.27 GB/s** (`0.05%`) | **1,439.00 GB/s** (`0.05%`) | **2,689.62 GB/s** (`6.77%`) | Multi-threaded KV packing/transpose & fused Softmax rows |
| **L3 Shared LLC** (`105 MiB` Socket 0) | `16 MB` (1C) / `32 MB` shared (`8 MB`/T) | Streaming Read (`vmovaps`) | **22.26 GB/s** (`0.03%`) | **776.02 GB/s** (`0.30%`) | **1,257.98 GB/s** (`1.38%`) | Single-layer weights (`11–58 MB`) & KV cache (`p1024`–`p4096`) |
| **L3 Shared LLC** (`105 MiB` Socket 0) | `16 MB` (1C) / `32 MB` shared (`8 MB`/T) | Read + Write Copy | **19.94 GB/s** (`0.18%`) | **481.98 GB/s** (`0.14%`) | **680.53 GB/s** (`3.08%`) | Large L3-resident tensor transposes & dequantize passes |
| **DDR5 DRAM** (Socket 0) | `256 MB` (1C) / `896 MB` (28C) / `1,792 MB` (56C) | Streaming Read (`vmovaps`) | **14.82 GB/s** (`1.53%`) | **252.19 GB/s** (`2.29%`) | **245.44 GB/s** (`1.36%`) | **Full-model `M=1` Decode weight streaming (`18+ GB`)** |
| **DDR5 DRAM** (Socket 0) | `256 MB` (1C) / `896 MB` (28C) / `1,792 MB` (56C) | Read + Write Copy | **13.20 GB/s** (`0.51%`) | **136.38 GB/s** (`0.29%`) | **156.75 GB/s** (`0.74%`) | Cold DRAM buffer copies |
| **DDR5 DRAM** (Socket 0) | `256 MB` (1C) / `896 MB` (28C) / `1,792 MB` (56C) | STREAM Triad (`2R + 1W`) | **13.62 GB/s** (`0.25%`) | **155.57 GB/s** (`0.19%`) | **200.49 GB/s** (`0.07%`) | Cold DRAM elementwise / residual fusion |

---

## 4. Per-Kernel Roofline Comparison: XNNPACK vs. YNNPACK

### 4.1 Pair 1: 4-Bit Quantized Linear GEMV (`M = 1`, Decode Bottleneck — 83.2% XNNPACK / 87.3% YNNPACK Decode)

| Shape `(M×N×K)` | Regime | Backend | Microkernel (`ISA`) | AI (`OP/B`) | 1C Perf (`Util`) | 28C Perf (`cv%`, `ISA Util`) | 56C Lat (`ms`) | **56C Perf** | 56C `cv%` | 56C ISA Ceiling | **56C ISA Roofline Util** | **56C Machine Peak Util** |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1×21504×5376` | `DDR5_DRAM_Streaming` | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx` (`AMX-INT8`) | `3.99` | `39.69 GOPS` (`67.0%`) | `836.62 GOPS` (`2.30%`, `83.1%`) | `0.227` | **`1,016.74 GOPS`** (`254.6 GB/s`) | `3.41%` | `980.29 GOPS` | **`103.7%`** | **`103.7%`** |
| `1×21504×5376` | `DDR5_DRAM_Streaming` | **YNNPACK** | `ynn::dot_uint8_int4_int32_32x32x64_16x16x8_amxint8` (`AMX-INT8`) | `3.99` | `44.49 GOPS` (`75.2%`) | `869.53 GOPS` (`2.11%`, `86.3%`) | `0.230` | **`1,004.20 GOPS`** (`251.4 GB/s`) | `3.03%` | `980.29 GOPS` | **`102.4%`** | **`102.4%`** |
| `1×5376×21504` | `DDR5_DRAM_Streaming` | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx` (`AMX-INT8`) | `4.00` | `40.34 GOPS` (`68.1%`) | `850.62 GOPS` (`1.66%`, `84.4%`) | `0.251` | **`919.63 GOPS`** (`230.1 GB/s`) | `1.08%` | `981.02 GOPS` | **`93.7%`** | **`93.7%`** |
| `1×5376×21504` | `DDR5_DRAM_Streaming` | **YNNPACK** | `ynn::dot_uint8_int4_int32_32x32x64_16x16x8_amxint8` (`AMX-INT8`) | `4.00` | `43.80 GOPS` (`73.9%`) | `835.68 GOPS` (`2.89%`, `82.9%`) | `0.247` | **`934.98 GOPS`** (`233.9 GB/s`) | `2.38%` | `981.02 GOPS` | **`95.3%`** | **`95.3%`** |
| `1×8192×5376` | `DDR5_DRAM_Streaming` | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx` (`AMX-INT8`) | `3.99` | `39.54 GOPS` (`66.8%`) | `797.81 GOPS` (`5.22%`, `79.2%`) | `0.094` | **`939.54 GOPS`** (`235.3 GB/s`) | `4.22%` | `980.04 GOPS` | **`95.9%`** | **`95.9%`** |
| `1×8192×5376` | `DDR5_DRAM_Streaming` | **YNNPACK** | `ynn::dot_uint8_int4_int32_32x32x64_16x16x8_amxint8` (`AMX-INT8`) | `3.99` | `44.33 GOPS` (`74.9%`) | `829.06 GOPS` (`3.57%`, `82.3%`) | `0.091` | **`966.94 GOPS`** (`242.2 GB/s`) | `3.72%` | `980.04 GOPS` | **`98.7%`** | **`98.7%`** |
| `1×8192×5376` | `L3_LLC_Warm` | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx` (`AMX-INT8`) | `3.99` | `51.60 GOPS` (`58.0%`) | `1,333.49 GOPS` (`1.14%`, `11.9%`) | `0.040` | **`2,191.49 GOPS`** (`548.8 GB/s`) | `1.38%` | `22.31 TOPS` | **`9.8%`** | **`9.8%`** |
| `1×8192×5376` | `L3_LLC_Warm` | **YNNPACK** | `ynn::dot_uint8_int4_int32_32x32x64_16x16x8_amxint8` (`AMX-INT8`) | `3.99` | `52.90 GOPS` (`59.5%`) | `1,438.30 GOPS` (`3.37%`, `12.8%`) | `0.031` | **`2,866.97 GOPS`** (`718.0 GB/s`) | `0.59%` | `22.31 TOPS` | **`12.8%`** | **`12.8%`** |
| `1×4096×5376` | `L3_LLC_Warm` | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx` (`AMX-INT8`) | `3.99` | `51.59 GOPS` (`58.1%`) | `1,132.51 GOPS` (`0.91%`, `10.1%`) | `0.026` | **`1,693.10 GOPS`** (`424.1 GB/s`) | `1.36%` | `22.30 TOPS` | **`7.6%`** | **`7.6%`** |
| `1×4096×5376` | `L3_LLC_Warm` | **YNNPACK** | `ynn::dot_uint8_int4_int32_32x32x64_16x16x8_amxint8` (`AMX-INT8`) | `3.99` | `53.70 GOPS` (`60.4%`) | `1,440.54 GOPS` (`1.22%`, `12.9%`) | `0.018` | **`2,385.64 GOPS`** (`597.6 GB/s`) | `0.91%` | `22.30 TOPS` | **`10.7%`** | **`10.7%`** |

### 4.2 Pair 2: 4-Bit Quantized Linear Batched GEMM (`M ∈ {128, 1024}`, Prefill Bottleneck — 45.7% XNNPACK / 64.9% YNNPACK Prefill)

| Shape `(M×N×K)` | Regime | Backend | Microkernel (`ISA`) | AI (`OP/B`) | 1C Perf (`Util`) | 28C Perf (`cv%`, `ISA Util`) | 56C Lat (`ms`) | **56C Perf** | 56C `cv%` | 56C ISA Ceiling | **56C ISA Roofline Util** | **56C Machine Peak Util** |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `128×21504×5376` | `DDR5_DRAM_to_L2_Tiled` | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx` (`AMX-INT8`) | `425.82` | `815.33 GOPS` (`20.1%`) | `22.62 TOPS` (`0.28%`, `19.9%`) | `0.655` | **`45.20 TOPS`** (`106.2 GB/s`) | `2.64%` | `223.84 TOPS` | **`20.2%`** | **`20.2%`** |
| `128×21504×5376` | `DDR5_DRAM_to_L2_Tiled` | **YNNPACK** | `ynn::dot_uint8_int4_int32_32x32x64_16x16x8_amxint8` (`AMX-INT8`) | `425.82` | `1,319.05 GOPS` (`32.5%`) | `36.41 TOPS` (`0.32%`, `32.0%`) | `0.401` | **`73.76 TOPS`** (`173.2 GB/s`) | `1.36%` | `223.84 TOPS` | **`33.0%`** | **`33.0%`** |
| `128×5376×21504` | `DDR5_DRAM_to_L2_Tiled` | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx` (`AMX-INT8`) | `467.48` | `830.62 GOPS` (`20.5%`) | `23.54 TOPS` (`0.20%`, `20.7%`) | `0.631` | **`46.88 TOPS`** (`100.3 GB/s`) | `0.49%` | `223.84 TOPS` | **`20.9%`** | **`20.9%`** |
| `128×5376×21504` | `DDR5_DRAM_to_L2_Tiled` | **YNNPACK** | `ynn::dot_uint8_int4_int32_32x32x64_16x16x8_amxint8` (`AMX-INT8`) | `467.48` | `1,132.43 GOPS` (`27.9%`) | `35.98 TOPS` (`0.90%`, `31.6%`) | `0.487` | **`60.76 TOPS`** (`130.0 GB/s`) | `12.68%` | `223.84 TOPS` | **`27.1%`** | **`27.1%`** |
| `128×8192×5376` | `L3_LLC_to_L2_Tiled` | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx` (`AMX-INT8`) | `419.08` | `813.80 GOPS` (`20.1%`) | `22.22 TOPS` (`0.34%`, `19.5%`) | `0.262` | **`43.01 TOPS`** (`102.6 GB/s`) | `0.53%` | `223.84 TOPS` | **`19.2%`** | **`19.2%`** |
| `128×8192×5376` | `L3_LLC_to_L2_Tiled` | **YNNPACK** | `ynn::dot_uint8_int4_int32_32x32x64_16x16x8_amxint8` (`AMX-INT8`) | `419.08` | `1,310.54 GOPS` (`32.3%`) | `36.30 TOPS` (`0.30%`, `31.9%`) | `0.160` | **`70.30 TOPS`** (`167.8 GB/s`) | `1.82%` | `223.84 TOPS` | **`31.4%`** | **`31.4%`** |
| `1024×21504×5376` | `Compute_Bound_Tiled` | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx` (`AMX-INT8`) | `1563.93` | `774.34 GOPS` (`19.1%`) | `22.22 TOPS` (`1.24%`, `19.5%`) | `5.413` | **`43.74 TOPS`** (`28.0 GB/s`) | `0.25%` | `223.84 TOPS` | **`19.5%`** | **`19.5%`** |
| `1024×21504×5376` | `Compute_Bound_Tiled` | **YNNPACK** | `ynn::dot_uint8_int4_int32_32x32x64_16x16x8_amxint8` (`AMX-INT8`) | `1563.93` | `1,065.88 GOPS` (`26.3%`) | `33.50 TOPS` (`0.32%`, `29.4%`) | `3.844` | **`61.59 TOPS`** (`39.4 GB/s`) | `10.30%` | `223.84 TOPS` | **`27.5%`** | **`27.5%`** |
| `1024×8192×5376` | `Compute_Bound_Tiled` | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx` (`AMX-INT8`) | `1476.67` | `799.34 GOPS` (`19.7%`) | `22.59 TOPS` (`0.56%`, `19.8%`) | `2.033` | **`44.37 TOPS`** (`30.1 GB/s`) | `0.42%` | `223.84 TOPS` | **`19.8%`** | **`19.8%`** |
| `1024×8192×5376` | `Compute_Bound_Tiled` | **YNNPACK** | `ynn::dot_uint8_int4_int32_32x32x64_16x16x8_amxint8` (`AMX-INT8`) | `1476.67` | `1,093.57 GOPS` (`27.0%`) | `34.34 TOPS` (`0.07%`, `30.1%`) | `1.398` | **`64.50 TOPS`** (`43.7 GB/s`) | `0.77%` | `223.84 TOPS` | **`28.8%`** | **`28.8%`** |

#### 4.2.1 4-Bit Microkernel Evolution (`6e5b0566`): Prior YNNPACK AVX-512 VNNI vs. XNNPACK AMX-INT8+GFNI vs. New YNNPACK AMX-INT8+GFNI

| Pair & Shape `(M×N×K)` | Regime | 1C: VNNI / XNN AMX / YNN AMX | 28C: VNNI / XNN AMX / YNN AMX | 56C Prior YNN VNNI | 56C XNNPACK AMX | **56C New YNNPACK AMX** | **YNN AMX / Prior VNNI (56C)** | **YNN AMX / XNN AMX (56C)** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `P1: 1×21504×5376` | `DDR5_DRAM_Streaming` | `41.6` / `39.7` / `44.5 GOPS` | `804.60 GOPS` / `836.62 GOPS` / `869.53 GOPS` | `1,009.52 GOPS` | `1,016.74 GOPS` | **`1,004.20 GOPS`** | **`0.99×`** | **`0.99×`** |
| `P1: 1×5376×21504` | `DDR5_DRAM_Streaming` | `42.4` / `40.3` / `43.8 GOPS` | `814.74 GOPS` / `850.62 GOPS` / `835.68 GOPS` | `1,026.93 GOPS` | `919.63 GOPS` | **`934.98 GOPS`** | **`0.91×`** | **`1.02×`** |
| `P1: 1×8192×5376` | `DDR5_DRAM_Streaming` | `42.5` / `39.5` / `44.3 GOPS` | `775.76 GOPS` / `797.81 GOPS` / `829.06 GOPS` | `1,009.89 GOPS` | `939.54 GOPS` | **`966.94 GOPS`** | **`0.96×`** | **`1.03×`** |
| `P1: 1×8192×5376` | `L3_LLC_Warm` | `81.1` / `51.6` / `52.9 GOPS` | `2,706.08 GOPS` / `1,333.49 GOPS` / `1,438.30 GOPS` | `5,347.69 GOPS` | `2,191.49 GOPS` | **`2,866.97 GOPS`** | **`0.54×`** | **`1.31×`** |
| `P1: 1×4096×5376` | `L3_LLC_Warm` | `81.1` / `51.6` / `53.7 GOPS` | `2,690.15 GOPS` / `1,132.51 GOPS` / `1,440.54 GOPS` | `4,440.23 GOPS` | `1,693.10 GOPS` | **`2,385.64 GOPS`** | **`0.54×`** | **`1.41×`** |
| `P2: 128×21504×5376` | `DDR5_DRAM_to_L2_Tiled` | `220.9` / `815.3` / `1319.0 GOPS` | `6,203.51 GOPS` / `22.62 TOPS` / `36.41 TOPS` | `12.44 TOPS` | `45.20 TOPS` | **`73.76 TOPS`** | **`5.93×`** | **`1.63×`** |
| `P2: 128×5376×21504` | `DDR5_DRAM_to_L2_Tiled` | `224.4` / `830.6` / `1132.4 GOPS` | `6,309.52 GOPS` / `23.54 TOPS` / `35.98 TOPS` | `12.59 TOPS` | `46.88 TOPS` | **`60.76 TOPS`** | **`4.83×`** | **`1.30×`** |
| `P2: 128×8192×5376` | `L3_LLC_to_L2_Tiled` | `221.5` / `813.8` / `1310.5 GOPS` | `6,220.07 GOPS` / `22.22 TOPS` / `36.30 TOPS` | `11.33 TOPS` | `43.01 TOPS` | **`70.30 TOPS`** | **`6.20×`** | **`1.63×`** |
| `P2: 1024×21504×5376` | `Compute_Bound_Tiled` | `218.0` / `774.3` / `1065.9 GOPS` | `6,192.87 GOPS` / `22.22 TOPS` / `33.50 TOPS` | `12.26 TOPS` | `43.74 TOPS` | **`61.59 TOPS`** | **`5.03×`** | **`1.41×`** |
| `P2: 1024×8192×5376` | `Compute_Bound_Tiled` | `224.2` / `799.3` / `1093.6 GOPS` | `6,278.95 GOPS` / `22.59 TOPS` / `34.34 TOPS` | `12.50 TOPS` | `44.37 TOPS` | **`64.50 TOPS`** | **`5.16×`** | **`1.45×`** |

### 4.3 Pair 3: FP32 Attention `BatchMatMul` (`Q·K^T` and `Scores·V` — 36.2% XNNPACK / 17.5% YNNPACK Prefill)

| Shape `(M×N×K)` | Regime | Backend | Microkernel (`ISA`) | AI (`OP/B`) | 1C Perf (`Util`) | 28C Perf (`cv%`, `ISA Util`) | 56C Lat (`ms`) | **56C Perf** | 56C `cv%` | 56C ISA Ceiling | **56C ISA Roofline Util** | **56C Machine Peak Util** |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `128×1024×256` | `L2_Cache_Resident` | **XNNPACK** | `xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast` (`AVX512-FP32`) | `39.38` | `115.12 GFLOPS` (`90.6%`) | `3,207.92 GFLOPS` (`4.29%`, `90.1%`) | `0.011` | **`6,362.87 GFLOPS`** (`161.6 GB/s`) | `6.91%` | `7,091.48 GFLOPS` | **`89.7%`** | **`89.7%`** |
| `128×1024×256` | `L2_Cache_Resident` | **YNNPACK** | `ynn::dot_fp32_5x64x1_1x16x1_avx512` (`AVX512-FP32`) | `39.38` | `117.74 GFLOPS` (`92.7%`) | `3,276.56 GFLOPS` (`4.32%`, `92.0%`) | `0.011` | **`6,219.61 GFLOPS`** (`157.9 GB/s`) | `6.55%` | `7,091.48 GFLOPS` | **`87.7%`** | **`87.7%`** |
| `1024×1024×256` | `L3_LLC_Resident` | **XNNPACK** | `xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast` (`AVX512-FP32`) | `85.33` | `116.10 GFLOPS` (`91.4%`) | `3,201.33 GFLOPS` (`6.92%`, `89.9%`) | `0.084` | **`6,420.97 GFLOPS`** (`75.2 GB/s`) | `1.70%` | `7,091.48 GFLOPS` | **`90.5%`** | **`90.5%`** |
| `1024×1024×256` | `L3_LLC_Resident` | **YNNPACK** | `ynn::dot_fp32_5x64x1_1x16x1_avx512` (`AVX512-FP32`) | `85.33` | `113.22 GFLOPS` (`89.1%`) | `3,138.08 GFLOPS` (`0.65%`, `88.1%`) | `0.081` | **`6,613.14 GFLOPS`** (`77.5 GB/s`) | `2.02%` | `7,091.48 GFLOPS` | **`93.2%`** | **`93.2%`** |
| `1024×4096×512` | `L3_LLC_Resident` | **XNNPACK** | `xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast` (`AVX512-FP32`) | `157.54` | `102.21 GFLOPS` (`80.5%`) | `2,909.87 GFLOPS` (`3.38%`, `81.7%`) | `0.734` | **`5,852.43 GFLOPS`** (`37.1 GB/s`) | `0.70%` | `7,091.48 GFLOPS` | **`82.5%`** | **`82.5%`** |
| `1024×4096×512` | `L3_LLC_Resident` | **YNNPACK** | `ynn::dot_fp32_5x64x1_1x16x1_avx512` (`AVX512-FP32`) | `157.54` | `113.17 GFLOPS` (`89.1%`) | `3,119.39 GFLOPS` (`4.36%`, `87.6%`) | `0.692` | **`6,204.81 GFLOPS`** (`39.4 GB/s`) | `0.40%` | `7,091.48 GFLOPS` | **`87.5%`** | **`87.5%`** |
| `2520×2520×72` | `L3_LLC_Resident` | **XNNPACK** | `xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast` (`AVX512-FP32`) | `34.05` | `109.37 GFLOPS` (`86.1%`) | `2,784.00 GFLOPS` (`0.33%`, `78.2%`) | `0.159` | **`5,739.29 GFLOPS`** (`168.5 GB/s`) | `1.15%` | `7,091.48 GFLOPS` | **`80.9%`** | **`80.9%`** |
| `2520×2520×72` | `L3_LLC_Resident` | **YNNPACK** | `ynn::dot_fp32_5x64x1_1x16x1_avx512` (`AVX512-FP32`) | `34.05` | `105.31 GFLOPS` (`82.9%`) | `2,623.30 GFLOPS` (`3.44%`, `73.7%`) | `0.185` | **`4,942.49 GFLOPS`** (`145.1 GB/s`) | `6.38%` | `7,091.48 GFLOPS` | **`69.7%`** | **`69.7%`** |

### 4.4 Pair 4: FP32 Attention `Softmax` (`rmax + raddstoreexpminusmax` vs. `max_k1 + sum_k1` Reductions)

| Shape `(M×N×K)` | Regime | Backend | Microkernel (`ISA`) | AI (`OP/B`) | 1C Perf (`Util`) | 28C Perf (`cv%`, `ISA Util`) | 56C Lat (`ms`) | **56C Perf** | 56C `cv%` | 56C ISA Ceiling | **56C ISA Roofline Util** | **56C Machine Peak Util** |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1024×1024×1` | `L2_L3_Resident` | **XNNPACK** | `xnn_f32_rmax + xnn_f32_raddstoreexpminusmax__avx512f_rr2_p5` (`AVX512-FP32`) | `1.00` | `25.73 GB/s` (`50.2%`) | `1,067.33 GB/s` (`19.01%`, `74.2%`) | `0.008` | **`1,639.62 GB/s`** | `23.75%` | `2,689.62 GB/s` | **`61.0%`** | **`61.0%`** |
| `1024×1024×1` | `L2_L3_Resident` | **YNNPACK** | `ynn::max_k1_fp32_avx512 + ynn::sum_k1_fp32_avx512` (`AVX512-FP32`) | `0.25` | `22.12 GB/s` (`99.4%`) | `2,061.27 GB/s` (`12.79%`, `73.5%`) | `0.002` | **`3,534.80 GB/s`** | `4.62%` | `5,587.42 GB/s` | **`63.3%`** | **`63.3%`** |
| `2520×2520×1` | `L3_LLC_Resident` | **XNNPACK** | `xnn_f32_rmax + xnn_f32_raddstoreexpminusmax__avx512f_rr2_p5` (`AVX512-FP32`) | `1.00` | `25.54 GB/s` (`49.8%`) | `961.16 GB/s` (`17.48%`, `66.8%`) | `0.037` | **`2,036.39 GB/s`** | `5.27%` | `2,689.62 GB/s` | **`75.7%`** | **`75.7%`** |
| `2520×2520×1` | `L3_LLC_Resident` | **YNNPACK** | `ynn::max_k1_fp32_avx512 + ynn::sum_k1_fp32_avx512` (`AVX512-FP32`) | `0.25` | `21.40 GB/s` (`96.1%`) | `1,482.38 GB/s` (`14.53%`, `52.9%`) | `0.015` | **`3,332.85 GB/s`** | `2.08%` | `5,587.42 GB/s` | **`59.6%`** | **`59.6%`** |
| `1024×4096×1` | `L3_LLC_Resident` | **XNNPACK** | `xnn_f32_rmax + xnn_f32_raddstoreexpminusmax__avx512f_rr2_p5` (`AVX512-FP32`) | `1.00` | `25.52 GB/s` (`49.8%`) | `1,035.77 GB/s` (`3.66%`, `72.0%`) | `0.025` | **`2,010.54 GB/s`** | `5.54%` | `2,689.62 GB/s` | **`74.8%`** | **`74.8%`** |
| `1024×4096×1` | `L3_LLC_Resident` | **YNNPACK** | `ynn::max_k1_fp32_avx512 + ynn::sum_k1_fp32_avx512` (`AVX512-FP32`) | `0.25` | `22.01 GB/s` (`98.9%`) | `2,131.07 GB/s` (`8.77%`, `76.0%`) | `0.008` | **`4,010.57 GB/s`** | `8.47%` | `5,587.42 GB/s` | **`71.8%`** | **`71.8%`** |

### 4.5 Pair 5: 8-Bit Dynamic-Quantized Linear GEMM (Vision ViT & Audio Conformer — 38%–47% of Vision/Audio)

| Shape `(M×N×K)` | Regime | Backend | Microkernel (`ISA`) | AI (`OP/B`) | 1C Perf (`Util`) | 28C Perf (`cv%`, `ISA Util`) | 56C Lat (`ms`) | **56C Perf** | 56C `cv%` | 56C ISA Ceiling | **56C ISA Roofline Util** | **56C Machine Peak Util** |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `280×5376×1152` | `L3_LLC_Resident` | **XNNPACK** | `xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx` (`AMX-INT8`) | `276.64` | `1,115.84 GOPS` (`27.5%`) | `42.06 TOPS` (`10.20%`, `36.9%`) | `0.041` | **`84.25 TOPS`** (`304.6 GB/s`) | `5.89%` | `223.84 TOPS` | **`37.6%`** | **`37.6%`** |
| `280×5376×1152` | `L3_LLC_Resident` | **YNNPACK** | `ynn::dot_int8_int8_int32_32x32x64_16x16x4_amxint8` (`AMX-INT8`) | `276.64` | `1,388.55 GOPS` (`34.2%`) | `60.01 TOPS` (`2.70%`, `52.7%`) | `0.029` | **`118.36 TOPS`** (`427.9 GB/s`) | `10.33%` | `223.84 TOPS` | **`52.9%`** | **`52.9%`** |
| `2520×1152×1152` | `L3_LLC_Resident` | **XNNPACK** | `xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx` (`AMX-INT8`) | `422.20` | `1,005.67 GOPS` (`24.8%`) | `23.72 TOPS` (`3.93%`, `20.8%`) | `0.090` | **`74.12 TOPS`** (`175.6 GB/s`) | `4.14%` | `223.84 TOPS` | **`33.1%`** | **`33.1%`** |
| `2520×1152×1152` | `L3_LLC_Resident` | **YNNPACK** | `ynn::dot_int8_int8_int32_32x32x64_16x16x4_amxint8` (`AMX-INT8`) | `422.20` | `1,004.66 GOPS` (`24.8%`) | `30.28 TOPS` (`2.82%`, `26.6%`) | `0.103` | **`65.03 TOPS`** (`154.0 GB/s`) | `1.57%` | `223.84 TOPS` | **`29.1%`** | **`29.1%`** |
| `2520×4304×1152` | `L3_LLC_Resident` | **XNNPACK** | `xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx` (`AMX-INT8`) | `487.64` | `1,004.99 GOPS` (`24.8%`) | `26.72 TOPS` (`1.13%`, `23.5%`) | `0.515` | **`48.51 TOPS`** (`99.5 GB/s`) | `1.70%` | `223.84 TOPS` | **`21.7%`** | **`21.7%`** |
| `2520×4304×1152` | `L3_LLC_Resident` | **YNNPACK** | `ynn::dot_int8_int8_int32_32x32x64_16x16x4_amxint8` (`AMX-INT8`) | `487.64` | `994.24 GOPS` (`24.5%`) | `32.08 TOPS` (`0.56%`, `28.2%`) | `0.390` | **`64.08 TOPS`** (`131.4 GB/s`) | `0.96%` | `223.84 TOPS` | **`28.6%`** | **`28.6%`** |

### 4.6 Pair 6: 2-Bit Weight-Only Linear GEMM (Audio Conformer Encoder — 29%–41% of Audio)

| Shape `(M×N×K)` | Regime | Backend | Microkernel (`ISA`) | AI (`OP/B`) | 1C Perf (`Util`) | 28C Perf (`cv%`, `ISA Util`) | 56C Lat (`ms`) | **56C Perf** | 56C `cv%` | 56C ISA Ceiling | **56C ISA Roofline Util** | **56C Machine Peak Util** |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `204×1024×1024` | `L2_Cache_Resident` | **XNNPACK** | `xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni` (`AVX-VNNI-256`) | `629.21` | `182.09 GOPS` (`71.9%`) | `5,058.74 GOPS` (`0.31%`, `71.3%`) | `0.043` | **`10.05 TOPS`** (`16.0 GB/s`) | `0.46%` | `14.06 TOPS` | **`71.5%`** | **`4.5%`** |
| `204×1024×1024` | `L2_Cache_Resident` | **YNNPACK** | `ynn::dot_uint8_int2_int32_4x64x16_1x16x16_avx512vnni` (`AVX512-VNNI`) | `629.21` | `268.44 GOPS` (`53.6%`) | `7,189.23 GOPS` (`2.89%`, `51.2%`) | `0.031` | **`13.84 TOPS`** (`42.3 GB/s`) | `2.65%` | `27.62 TOPS` | **`50.1%`** | **`6.2%`** |
| `204×4096×1024` | `L2_L3_Resident` | **XNNPACK** | `xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni` (`AVX-VNNI-256`) | `817.60` | `181.61 GOPS` (`71.7%`) | `5,076.20 GOPS` (`0.16%`, `71.5%`) | `0.169` | **`10.12 TOPS`** (`12.4 GB/s`) | `0.25%` | `14.06 TOPS` | **`72.0%`** | **`4.5%`** |
| `204×4096×1024` | `L2_L3_Resident` | **YNNPACK** | `ynn::dot_uint8_int2_int32_4x64x16_1x16x16_avx512vnni` (`AVX512-VNNI`) | `817.60` | `266.53 GOPS` (`53.2%`) | `7,394.74 GOPS` (`0.20%`, `52.7%`) | `0.123` | **`13.89 TOPS`** (`37.3 GB/s`) | `0.46%` | `27.62 TOPS` | **`50.3%`** | **`6.2%`** |
| `204×1024×4096` | `L2_L3_Resident` | **XNNPACK** | `xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni` (`AVX-VNNI-256`) | `817.60` | `191.05 GOPS` (`75.4%`) | `5,305.48 GOPS` (`0.20%`, `74.8%`) | `0.162` | **`10.58 TOPS`** (`12.9 GB/s`) | `0.33%` | `14.06 TOPS` | **`75.2%`** | **`4.7%`** |
| `204×1024×4096` | `L2_L3_Resident` | **YNNPACK** | `ynn::dot_uint8_int2_int32_4x64x16_1x16x16_avx512vnni` (`AVX512-VNNI`) | `817.60` | `273.20 GOPS` (`54.6%`) | `7,497.96 GOPS` (`0.52%`, `53.4%`) | `0.115` | **`14.92 TOPS`** (`23.7 GB/s`) | `0.88%` | `27.62 TOPS` | **`54.0%`** | **`6.7%`** |

### 4.7 Pair 7: Matrix Packing / 32-Bit Transpose (`packw_gemm_goi` vs. `transpose_x32_avx512`)

| Shape `(M×N×K)` | Regime | Backend | Microkernel (`ISA`) | AI (`OP/B`) | 1C Perf (`Util`) | 28C Perf (`cv%`, `ISA Util`) | 56C Lat (`ms`) | **56C Perf** | 56C `cv%` | 56C ISA Ceiling | **56C ISA Roofline Util** | **56C Machine Peak Util** |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1×1024×256` | `L2_Cache_Copy` | **XNNPACK** | `xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4_prfm` (`AVX512F`) | `0.00` | `28.40 GB/s` (`55.4%`) | `572.09 GB/s` (`4.72%`, `39.8%`) | `0.002` | **`1,254.95 GB/s`** | `3.41%` | `2,689.62 GB/s` | **`46.7%`** | **`46.7%`** |
| `1×1024×256` | `L2_Cache_Copy` | **YNNPACK** | `ynn::transpose_x32_avx512` (`AVX512F`) | `0.00` | `12.23 GB/s` (`23.9%`) | `309.26 GB/s` (`2.66%`, `21.5%`) | `0.005` | **`383.25 GB/s`** | `2.22%` | `2,689.62 GB/s` | **`14.2%`** | **`14.2%`** |
| `1×4096×512` | `L3_LLC_Copy` | **XNNPACK** | `xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4_prfm` (`AVX512F`) | `0.00` | `16.92 GB/s` (`84.8%`) | `674.23 GB/s` (`1.77%`, `46.9%`) | `0.015` | **`1,122.28 GB/s`** | `6.72%` | `2,689.62 GB/s` | **`41.7%`** | **`41.7%`** |
| `1×4096×512` | `L3_LLC_Copy` | **YNNPACK** | `ynn::transpose_x32_avx512` (`AVX512F`) | `0.00` | `9.17 GB/s` (`46.0%`) | `289.74 GB/s` (`0.63%`, `20.1%`) | `0.035` | **`475.06 GB/s`** | `0.38%` | `2,689.62 GB/s` | **`17.7%`** | **`17.7%`** |
| `1×2520×1152` | `L3_LLC_Copy` | **XNNPACK** | `xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4_prfm` (`AVX512F`) | `0.00` | `18.77 GB/s` (`94.1%`) | `1,007.20 GB/s` (`2.88%`, `70.0%`) | `0.015` | **`1,505.54 GB/s`** | `1.45%` | `2,689.62 GB/s` | **`56.0%`** | **`56.0%`** |
| `1×2520×1152` | `L3_LLC_Copy` | **YNNPACK** | `ynn::transpose_x32_avx512` (`AVX512F`) | `0.00` | `9.32 GB/s` (`46.7%`) | `795.98 GB/s` (`10.82%`, `55.3%`) | `0.020` | **`1,172.29 GB/s`** | `3.99%` | `2,689.62 GB/s` | **`43.6%`** | **`43.6%`** |

---

## 5. Numerical Precision & Quantization Fidelity Verification (`verify_litert_microkernels.c`)

> [!NOTE]
> **Baseline Precision Results Retained (`2026-11-10`)**: The full two-level numerical precision sweep below ([ukernel_precision_sapphire_rapids.json](../2026-11-10-gemma4-31b-sapphire-rapids/ukernel_precision_sapphire_rapids.json)) is retained from the `2026-11-10` verification run and was **not re-run** for upstream `google/XNNPACK` commit `6e5b0566` (this benchmark update covers microkernel performance and hardware roofline analysis only). In particular, Pairs 1 and 2 below reflect the prior YNNPACK AVX-512 VNNI 4-bit symbols (`1x32x8` / `12x16x8`). Prior to timing `6e5b0566`, the standalone benchmark binary executed built-in numerical smoke checks confirming exact `INT32` accumulator output (`C[0] == C[last] == K`) for `ynn::dot_uint8_int4_int32_32x32x64_16x16x8_amxint8` at both `M = 1` and `M = 32` and finite non-zero `FP32` outputs for `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_{1,16}x64c4__avx512amx`.

To verify that SIMD/AMX optimizations and low-bit quantization schemes do not introduce silent numerical regressions, [verify_litert_microkernels.c](../../scripts/verify_litert_microkernels.c) tests all 14 load-bearing XNNPACK and YNNPACK microkernels across **two reference levels** and **four synthetic input distributions**.

### 5.1 Two-Level Reference Verification Methodology & Input Distributions

1. **Level A — Exact Packed-Layout Microkernel Reference**:
   - Feeds the **identical quantized input buffer and packed weight buffer** into both the target SIMD/AMX microkernel and an exact scalar reference (`INT64` accumulator + `FP64` post-scaling for quantized GEMMs, sequential IEEE-754 `FP32` FMA for `FP32` GEMMs, exact `FP64` $\exp(x - x_{\max})$ for Softmax, and scalar 32-bit pack/transpose).
   - **Purpose**: Isolates hardware/microkernel implementation fidelity (catching register clobbers, AMX tile stride/layout mismatches, `INT32` accumulator overflow, `FP32` post-scaling rounding, or polynomial approximation error).
2. **Level B — Unquantized `FP64` Golden Operator Reference**:
   - Generates high-precision `FP64` tensors $A_{\text{fp64}}$, $B_{\text{fp64}}$, and $\text{bias}_{\text{fp64}}$, quantizes and packs them using each backend's exact quantization scheme (`QD8`, `QC8W`, `QC4W`, `QC2W`), executes the microkernel, and compares the dequantized output against the unquantized `FP64` golden reference:
     $$C_{\text{fp64}}[m, n] = \sum_{k=0}^{K-1} A_{\text{fp64}}[m, k] \cdot B_{\text{fp64}}[n, k] + \text{bias}_{\text{fp64}}[n]$$
   - **Purpose**: Measures end-to-end operator and quantization noise (`SQNR (dB)`, `Cosine Similarity`, `RMSE`, `Max Abs Error`).
3. **Four Deterministic Input Distributions (`NUM_DISTS = 4`)**:
   - **`normal`**: Standard Gaussian $\mathcal{N}(0, 1)$ activations and $\mathcal{N}(0, 0.25)$ weights.
   - **`saturation`**: Alternating $\pm 10.0$ activations and $\pm 1.0$ weights to stress maximum integer accumulator magnitude (`INT32` overflow check) and quantization clipping boundaries.
   - **`outliers`**: Heavy-tailed outlier channels ($\mathcal{N}(0, 15.0)$ every 16th channel alongside $\mathcal{N}(0, 0.5)$ background channels) to stress per-row dynamic activation quantization and per-channel weight scales.
   - **`tails`**: Asymmetric non-zero-mean activations ($\mathcal{N}(1.25, 0.4)$ + sub-threshold $\mathcal{N}(-0.02, 0.005)$ tail channels) and unaligned `N = 53` tail dimensions for packing/transposition.

### 5.2 Level A Verification Summary: SIMD / AMX Microkernel vs. Exact Packed-Layout Reference (Baseline)

| Kernel Pair | Backend | Microkernel Symbol | Max Abs Error | Max Rel Error | RMSE | Max ULP | Mean ULP | Level A SQNR | Exact Match (`%`) | `NaN`/`Inf` | Status |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Pair 1: 4-Bit Decode GEMV (`M=1`) | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx` | `5.13e-05` | `3.61e-07` | `1.32e-05` | `2` | `0.4453` | **`141.93 dB`** | `100.00%` | `0` | **`PASS`** |
| Pair 1: 4-Bit Decode GEMV (`M=1`) | **YNNPACK** | `ynn::dot_uint8_int4_int32_1x32x8_1x16x8_avx512vnni` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0` | `0.0000` | **`200.00 dB`** | `100.00%` | `0` | **`PASS`** |
| Pair 2: 4-Bit Prefill GEMM (`M=16/12`) | **XNNPACK** | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx` | `5.13e-05` | `3.48e-07` | `1.31e-05` | `2` | `0.4470` | **`141.93 dB`** | `100.00%` | `0` | **`PASS`** |
| Pair 2: 4-Bit Prefill GEMM (`M=16/12`) | **YNNPACK** | `ynn::dot_uint8_int4_int32_12x16x8_1x16x8_avx512vnni` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0` | `0.0000` | **`200.00 dB`** | `100.00%` | `0` | **`PASS`** |
| Pair 3: FP32 Attention `BatchMatMul` | **XNNPACK** | `xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0` | `0.0000` | **`200.00 dB`** | `100.00%` | `0` | **`PASS`** |
| Pair 3: FP32 Attention `BatchMatMul` | **YNNPACK** | `ynn::dot_fp32_5x64x1_1x16x1_avx512` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0` | `0.0000` | **`200.00 dB`** | `100.00%` | `0` | **`PASS`** |
| Pair 4: FP32 Attention `Softmax` | **XNNPACK** | `xnn_f32_rmax + xnn_f32_raddstoreexpminusmax (avx512f_p5)` | `9.56e-08` | `6.24e-07` | `1.40e-09` | `5` | `0.0648` | **`144.81 dB`** | `100.00%` | `0` | **`PASS`** |
| Pair 4: FP32 Attention `Softmax` | **YNNPACK** | `ynn::max_k1_fp32_avx512 + ynn::sum_k1_fp32_avx512` | `2.85e-08` | `5.33e-07` | `5.25e-10` | `4` | `0.0094` | **`153.60 dB`** | `100.00%` | `0` | **`PASS`** |
| Pair 5: 8-Bit Multimodal GEMM | **XNNPACK** | `xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx` | `2.03e-05` | `2.42e-05` | `5.36e-06` | `4` | `0.2021` | **`148.41 dB`** | `99.95%` | `0` | **`PASS`** |
| Pair 5: 8-Bit Multimodal GEMM | **YNNPACK** | `ynn::dot_int8_int8_int32_32x32x64_16x16x4_amxint8` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0` | `0.0000` | **`200.00 dB`** | `100.00%` | `0` | **`PASS`** |
| Pair 6: 2-Bit Weight-Only GEMM | **XNNPACK** | `xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0` | `0.0000` | **`200.00 dB`** | `100.00%` | `0` | **`PASS`** |
| Pair 6: 2-Bit Weight-Only GEMM | **YNNPACK** | `ynn::dot_uint8_int2_int32_4x64x16_1x16x16_avx512vnni` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0` | `0.0000` | **`200.00 dB`** | `100.00%` | `0` | **`PASS`** |
| Pair 7: 32-Bit Pack / Transpose | **XNNPACK** | `xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4_prfm` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0` | `0.0000` | **`200.00 dB`** | `100.00%` | `0` | **`PASS`** |
| Pair 7: 32-Bit Pack / Transpose | **YNNPACK** | `ynn::transpose_x32_avx512` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0` | `0.0000` | **`200.00 dB`** | `100.00%` | `0` | **`PASS`** |

### 5.3 Level B Verification Summary: End-to-End Operator & Quantization Fidelity vs. `FP64` Golden Reference (Baseline)

| Kernel Pair | Backend | Quantization / Arithmetic Scheme | SQNR (`normal`) | SQNR (`saturation`) | SQNR (`outliers`) | SQNR (`tails`) | **Worst-Case SQNR** | **Worst-Case Cosine Sim** | Aggregate RMSE | Max Abs Error | Status |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Pair 1: 4-Bit Decode GEMV (`M=1`) | **XNNPACK** | Per-row `QD8` act × Per-channel `QC4W` (`INT4`) | `21.31 dB` | `200.00 dB` | `25.58 dB` | `22.55 dB` | **`21.31 dB`** | **`0.996326`** | `5.71e-01` | `5.63e+00` | **`PASS`** |
| Pair 1: 4-Bit Decode GEMV (`M=1`) | **YNNPACK** | Per-row `QD8` act × Per-channel `QC4W` (`INT4`) | `20.57 dB` | `200.00 dB` | `25.44 dB` | `21.46 dB` | **`20.57 dB`** | **`0.995609`** | `4.42e-01` | `4.01e+00` | **`PASS`** |
| Pair 2: 4-Bit Prefill GEMM (`M=16/12`) | **XNNPACK** | Per-row `QD8` act × Per-channel `QC4W` (`INT4`) | `19.41 dB` | `200.00 dB` | `25.66 dB` | `22.41 dB` | **`19.41 dB`** | **`0.994275`** | `4.49e-01` | `8.85e+00` | **`PASS`** |
| Pair 2: 4-Bit Prefill GEMM (`M=16/12`) | **YNNPACK** | Per-row `QD8` act × Per-channel `QC4W` (`INT4`) | `20.44 dB` | `200.00 dB` | `25.79 dB` | `22.39 dB` | **`20.44 dB`** | **`0.995580`** | `4.53e-01` | `5.28e+00` | **`PASS`** |
| Pair 3: FP32 Attention `BatchMatMul` | **XNNPACK** | IEEE-754 `FP32` FMA (`K=64`) | `136.49 dB` | `200.00 dB` | `136.63 dB` | `134.83 dB` | **`134.83 dB`** | **`1.000000`** | `1.13e-06` | `2.01e-05` | **`PASS`** |
| Pair 3: FP32 Attention `BatchMatMul` | **YNNPACK** | IEEE-754 `FP32` FMA (`K=64`) | `135.90 dB` | `200.00 dB` | `134.72 dB` | `134.18 dB` | **`134.18 dB`** | **`1.000000`** | `1.68e-06` | `3.76e-05` | **`PASS`** |
| Pair 4: FP32 Attention `Softmax` | **XNNPACK** | IEEE-754 `FP32` (`rr2_p5` poly vs `expf`, `K=1024`) | `141.54 dB` | `200.00 dB` | `200.00 dB` | `141.62 dB` | **`141.54 dB`** | **`1.000000`** | `2.97e-10` | `5.93e-08` | **`PASS`** |
| Pair 4: FP32 Attention `Softmax` | **YNNPACK** | IEEE-754 `FP32` (`rr2_p5` poly vs `expf`, `K=1024`) | `143.57 dB` | `200.00 dB` | `200.00 dB` | `144.88 dB` | **`143.57 dB`** | **`1.000000`** | `2.27e-10` | `4.65e-08` | **`PASS`** |
| Pair 5: 8-Bit Multimodal GEMM | **XNNPACK** | Per-row `QD8`/`INT8` act × Per-channel `QC8W` (`INT8`) | `41.85 dB` | `200.00 dB` | `47.08 dB` | `46.26 dB` | **`41.85 dB`** | **`0.999968`** | `3.71e-02` | `4.76e-01` | **`PASS`** |
| Pair 5: 8-Bit Multimodal GEMM | **YNNPACK** | Per-row `QD8`/`INT8` act × Per-channel `QC8W` (`INT8`) | `41.98 dB` | `200.00 dB` | `44.72 dB` | `43.85 dB` | **`41.98 dB`** | **`0.999968`** | `4.79e-02` | `8.24e-01` | **`PASS`** |
| Pair 6: 2-Bit Weight-Only GEMM | **XNNPACK** | Per-tensor `INT8`/`UINT8` act × Per-channel `QC2W` (`INT2`) | `5.76 dB` | `200.00 dB` | `11.18 dB` | `7.18 dB` | **`5.76 dB`** | **`0.874698`** | `2.30e+00` | `2.52e+01` | **`PASS`** |
| Pair 6: 2-Bit Weight-Only GEMM | **YNNPACK** | Per-tensor `INT8`/`UINT8` act × Per-channel `QC2W` (`INT2`) | `6.16 dB` | `200.00 dB` | `11.77 dB` | `9.44 dB` | **`6.16 dB`** | **`0.896820`** | `2.22e+00` | `3.51e+01` | **`PASS`** |
| Pair 7: 32-Bit Pack / Transpose | **XNNPACK** | Lossless 32-bit `uint32_t` panel pack / transpose | `200.00 dB` | `200.00 dB` | `200.00 dB` | `200.00 dB` | **`200.00 dB`** | **`1.000000`** | `0.00e+00` | `0.00e+00` | **`PASS`** |
| Pair 7: 32-Bit Pack / Transpose | **YNNPACK** | Lossless 32-bit `uint32_t` panel pack / transpose | `200.00 dB` | `200.00 dB` | `200.00 dB` | `200.00 dB` | **`200.00 dB`** | **`1.000000`** | `0.00e+00` | `0.00e+00` | **`PASS`** |

### 5.4 Operator-Specific Invariant Verification (`Softmax` & `32-Bit Pack/Transpose`, Baseline)

| Kernel Pair | Backend | Microkernel Symbol | Invariant 1 | Measured Value | Invariant 2 | Measured Value | Guard / Canary Check | Status |
| :--- | :--- | :--- | :--- | ---: | :--- | ---: | ---: | :---: |
| **Pair 4: FP32 Softmax** | **XNNPACK** | `xnn_f32_rmax + xnn_f32_raddstoreexpminusmax (avx512f_p5)` | Max Simplex Sum Error $\max \lvert\sum_i P_i - 1\rvert$ | `1.62e-07` | Shift-Invariance Error $\lVert\text{Softmax}(x+1000) - \text{Softmax}(x)\rVert_\infty$ | `6.74e-06` | `0 NaN / 0 Inf` | **`PASS`** |
| **Pair 4: FP32 Softmax** | **YNNPACK** | `ynn::max_k1_fp32_avx512 + ynn::sum_k1_fp32_avx512` | Max Simplex Sum Error $\max \lvert\sum_i P_i - 1\rvert$ | `1.34e-07` | Shift-Invariance Error $\lVert\text{Softmax}(x+1000) - \text{Softmax}(x)\rVert_\infty$ | `6.74e-06` | `0 NaN / 0 Inf` | **`PASS`** |
| **Pair 7: 32-Bit Pack/Transpose** | **XNNPACK** | `xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4_prfm` | Aligned `64×64` Bit-Exact Match | `100.00%` | Unaligned Tail `53×64` Bit-Exact Match (incl. zero-pad lanes) | `100.00%` | `0 corrupted bytes` | **`PASS`** |
| **Pair 7: 32-Bit Pack/Transpose** | **YNNPACK** | `ynn::transpose_x32_avx512` | Aligned `64×64` Bit-Exact Match | `100.00%` | Unaligned Tail `53×64` Bit-Exact Match | `100.00%` | `0 corrupted bytes` | **`PASS`** |

### 5.5 Key Numerical Precision & Quantization Insights (Baseline)

1. **Why YNNPACK Achieves `0 ULP` in Level A Quantized Dot Kernels While XNNPACK Shows `2–4 Max ULP` (`141.9–148.4 dB` SQNR)**:
   - YNNPACK's quantized microkernels (`dot_uint8_int4_int32`, `dot_int8_int8_int32`, `dot_uint8_int2_int32`) output raw `INT32` accumulators directly and defer `FP32` dequantization to a separate pass (`ynn::dequantize_dot_f32_avx512`), yielding **`0 ULP` (`100.0%` exact match, `200.0 dB` SQNR)** against the `INT64` reference.
   - XNNPACK's AMX microkernels fuse `INT32 -> FP32` conversion (`vcvtdq2ps`), activation `inv_scale[m]` (`vmulps`), and weight `w_scale[n]` / `w_bias[n]` (`vfmadd213ps`) in-kernel, introducing a benign **`0.46–0.77 Mean ULP` (`2–4 Max ULP`, `141.9–148.4 dB` SQNR)** relative to `FP64` post-scaling while eliminating a full memory pass over the output matrix.
   - In **Level B (`FP64` Golden Reference)**, both backends achieve virtually identical end-to-end fidelity (**`21.3 dB`** on `M=1` `INT4`, **`19.4 dB`** on `M=16/12` `INT4`, **`41.8 dB`** on `INT8`), confirming that XNNPACK's in-kernel `FP32` epilogue adds zero measurable degradation relative to quantization noise.
2. **Empirical Quantization Scaling Law Across Bit-Widths (`~6 dB/bit`)**:
   - **`FP32` (`Pair 3` & `Pair 4`)**: `134.2–143.6 dB` SQNR (`Cosine Similarity = 1.000000`).
   - **`INT8` (`Pair 5`, `QD8 × QC8W`)**: `41.8–44.2 dB` SQNR (`Cosine Similarity = 0.999967–0.999980`).
   - **`INT4` (`Pair 1` & `Pair 2`, `QD8 × QC4W`)**: `19.4–24.9 dB` SQNR (`Cosine Similarity = 0.994275–0.998436`).
   - **`INT2` (`Pair 6`, `QS8/UINT8 × QC2W`)**: `5.8–6.2 dB` SQNR on dense Gaussian (`normal`), rising to `11.2–11.8 dB` (`Cosine Similarity = 0.968`) on `outliers`.
3. **Softmax Polynomial Approximation (`avx512f_rr2_p5`) vs. Standard `expf` (`Pair 4`)**:
   - XNNPACK's degree-5 minimax polynomial (`avx512f_rr2_p5`) achieves **`Max ULP = 5` (`SQNR = 144.81 dB`)** in Level A and **`141.54 dB` SQNR**, **`Cosine Similarity = 1.000000`**, and shift-invariance error **`6.74e-06`** under $+1000.0$ logit shifts in Level B.

---

## 6. Architectural Diagnosis: Comparing Roofline Ceilings vs. Achieved Performance

### 6.1 Decode (`M = 1`): Cold DDR5 Bandwidth Saturation vs. Warm-Cache AMX Cost-Model Gap
- **Cold DDR5 Streaming Roofline (`1×5376×21504`, `DDR5_DRAM_Streaming`)**:
  - Each weight byte (`2 × INT4` weights) performs `4` MACs = `8` INT8 OPs, giving $AI = 3.997\text{ OPs/Byte}$. Against the 56-core DDR5 streaming read ceiling ($BW_{\text{prac}} = 245.44\text{ GB/s}$), the practical throughput ceiling is:
    $$\text{GOPS}_{\text{ceil}}(M=1) = 3.997 \times 245.44\text{ GB/s} = \mathbf{981.0\text{ GOPS}}$$
  - **Measured (56C)**: XNNPACK `1x64c4__avx512amx` achieves **`919.63 GOPS` (`230.08 GB/s`, `93.7%`)**, YNNPACK `32x32x64_amxint8` achieves **`934.98 GOPS` (`233.92 GB/s`, `95.3%`)**, and prior YNNPACK `1x32x8_avx512vnni` achieves **`1026.93 GOPS` (`256.92 GB/s`, `104.7%`)**. On `1×21504×5376`, all three reach `251–255 GB/s` (`~102%–104%` of the 56C streaming probe `245.44 GB/s`, right at Socket 0's `252.19 GB/s` 28C / `255.37 GB/s` max DDR5 ceiling, because interleaved GEMV compute slightly reduces 56-core memory-controller queue contention).
  - **Takeaway**: Cold full-model `M = 1` decode already saturates `>93%` of Socket 0's 8-channel DDR5 bus across all three kernels; further decode speedup requires reducing bytes fetched per token (lower-bit quantization, speculative decoding, or multi-socket NUMA weight sharding).
- **Warm L3/L2 Cache (`1×8192×5376`, `L3_LLC_Warm`) — Actionable Upstream Dispatch Finding**:
  - When weights reside in cache (`22 MB` / `56` cores = `402 KB`/core in private L2), AMX-INT8 suffers from two penalties at `M = 1`: (1) `tdpbusd`/`tdpbssd` operates on 16-row tiles, leaving `15/16` (`93.75%`) of compute lanes unused, and (2) both AMX kernels still unpack 4-bit nibbles via GFNI into a stack buffer (`vmovdqa64`) and reload via `tileloadd`.
  - Consequently, prior YNNPACK `1x32x8_avx512vnni` (**`5,347.7 GOPS` / `1,339.2 GB/s`**) outperforms both YNNPACK `32x32x64_amxint8` (**`2,867.0 GOPS`**, `1.87×` slower) and XNNPACK `1x64c4__avx512amx` (**`2,191.5 GOPS`**, `2.44×` slower).
  - **Root Cause & Fix**: Upstream `6e5b0566` assigns `dot_uint8_int4_int32_32x32x64_16x16x8_amxint8` the same `x86_amx_uint8` cost model as 8-bit AMX without accounting for 4-bit GFNI unpack + stack-spill overhead. Adding a dedicated `x86_amx_uint8_int4` cost entry in `ynnpack/kernels/dot/get_dot_kernel.cc` so `M = 1` (and very small `M`) keeps dispatching `1x32x8_avx512vnni` while `M >= 4..8` switches to `32x32x64_amxint8` recovers the full `1.87×` warm-cache GEMV speed without sacrificing any prefill throughput.

### 6.2 Prefill (`M = 128..1024`): Why YNNPACK's New `32×32` AMX+GFNI Kernel Beats Both Prior VNNI and XNNPACK's `16×64` AMX+GFNI Kernel
- **Roofline Math**: At `M = 128`, $AI = 419\text{–}467\text{ OPs/Byte}$, comfortably above both the AVX-512 VNNI ridge point ($AI_{\text{ridge}} = 22.0\text{ OPs/Byte}$ against 56C L3) and the AMX-INT8 ridge point ($AI_{\text{ridge}} = 177.9\text{ OPs/Byte}$ against 56C L3).
- **Measured Performance (56C, `128×21504×5376`)**:
  - **Prior YNNPACK (`12x16x8_avx512vnni`)**: **`12.44 TOPS`** (`45.0%` of 56C AVX-512 VNNI ceiling, only `5.6%` of AMX-INT8 peak).
  - **XNNPACK (`16x64c4__avx512amx`, GFNI-optimized)**: **`45.20 TOPS`** (`20.2%` of 56C AMX-INT8 ceiling).
  - **New YNNPACK (`32x32x64_amxint8`, GFNI-optimized)**: **`73.76 TOPS`** (**`33.0%` of 56C AMX-INT8 ceiling** — **`5.93×`** faster than prior YNNPACK VNNI and **`1.63×`** faster than XNNPACK AMX!).
- **Microarchitectural Explanation (`2×2` vs. `1×4` Tile Blocking under GFNI Unpack + L1D Stack Bounce)**:
  - Sapphire Rapids AMX (`tdpbssd`/`tdpbusd`) requires unpacked 8-bit tiles, so both backends unpack 4-bit weights in ZMM registers using GFNI (`vgf2p8affineqb`), store unpacked `16×64` tiles to a stack scratchpad (`vmovdqa64`), and reload them into `tmm` tiles (`tileloadd`).
  - **XNNPACK `16×64` (`1×4` tile blocking)** loads `1` A-tile and unpacks/spills/loads **`4` B-tiles (`4,096 B` written + `4,096 B` read = `8,192 B` L1D traffic)** for every `4 × tdpbssd` (`131,072` OPs = `16 OPs/B` of L1D scratchpad traffic).
  - **YNNPACK `32×32` (`2×2` tile blocking)** loads `2` A-tiles and unpacks/spills/loads only **`2` B-tiles (`2,048 B` written + `2,048 B` read = `4,096 B` L1D traffic)** for the same `4 × tdpbusd` (`131,072` OPs = **`32 OPs/B` — cutting GFNI unpack instructions and L1D stack-bounce traffic in half per MAC!**).
  - **End-to-End Caveat**: Note that YNNPACK's standalone `dot` kernel outputs raw `INT32` (requiring a separate `dequantize_dot_f32_avx512` epilogue pass in full-graph execution), whereas XNNPACK fuses `INT32 -> FP32` scale/bias dequantization inside the microkernel. Also note that at `M = 128`, YNNPACK's `73.76 TOPS` already consumes `173.2 GB/s` of logical weight bandwidth (`71%` of Socket 0's `245.4 GB/s` DDR5 ceiling for cold layers).

### 6.3 8-Bit Multimodal GEMM (`pair5_int8_gemm`) & 2-Bit Audio GEMM (`pair6_int2_gemm`)
- **Pair 5 (`INT8` AMX)**: Without 4-bit unpacking, `tileloadd` reads packed `INT8` weights directly from cache. YNNPACK's `2×2` blocking (`32×32`) leads XNNPACK's `1×4` blocking (`16×64`) on `280×5376×1152` (**`118.36`** vs. `84.25 TOPS`, reaching **`52.9%`** of 56C AMX-INT8 peak) and `2520×4304×1152` (**`64.08`** vs. `48.51 TOPS`), while on narrow-`N` `2520×1152×1152` (`18` `nr=64` panels vs. `36` `nr=32` panels) XNNPACK reaches **`74.12 TOPS`** vs. YNNPACK **`65.03 TOPS`** at 56C (whereas YNNPACK leads at 28C).
- **Pair 6 (`INT2` Audio)**: XNNPACK's `5x8c8__avxvnni` hits **`72.0%`** of the 256-bit AVX-VNNI ceiling (`10.12 TOPS`), demonstrating tight 256-bit codegen but capping out at half the machine's 512-bit vector width. YNNPACK's 512-bit `4x64x16_avx512vnni` reaches **`13.89–14.92 TOPS`** (`1.37×–1.41×` faster).

### 6.4 Cache-Hierarchy Transitions, Panel Granularity & Harness Methodology Notes
1. **Aggregate Private L2 Capacity (`56 MiB` @ 28C, `112 MiB` @ 56C) Exceeds Shared L3 (`105 MiB`)**:
   - On `1C`, `11–22 MB` warm GEMV weights (`pair1`) and `8–51 MB` Softmax buffers (`pair4` YNNPACK `in`+`out`) exceed the `2 MiB` private L2 and stream from the `105 MiB` shared L3 (`22.26 GB/s` 1C Read ceiling — YNNPACK Softmax reductions hit **`22.01 GB/s` = `98.9%` of 1C L3 Read!**).
   - On `28C` and `56C`, static row/column partitioning shrinks per-core working sets below `2 MiB` (into each core's private L2), transitioning the true bandwidth ceiling from shared L3 (`1,258.0 GB/s`) to aggregate private L2 (`5,587.4 GB/s` Read-Only / `2,689.6 GB/s` Read+Write Copy at 56C).
2. **Pair 4 Softmax Ceiling & Harness Correction**:
   - XNNPACK (`rmax + raddstoreexpminusmax`, `2R + 1W`) is normalized against **L2/L1D Read+Write Copy** (`51.27` / `2,689.62 GB/s`), achieving `74.8%–75.7%` utilization at 56C while computing the minimax `exp` polynomial.
   - YNNPACK (`max_k1 + sum_k1`, reductions only, `2R + ~0W`) is normalized against **L3 (1C) / L2-Sliced (56C) Read-Only** (`22.26` / `5,587.42 GB/s`), achieving `59.6%–71.8%` at 56C. Note that this update corrects a bug in the earlier harness that passed byte length (`row_len * 4`) instead of element count (`row_len`) to `ynn::max_k1`/`sum_k1` and now batches 4 rows per call (`min_max_accumulator<N=4>`).
3. **2D N-Major GEMM Tiling & Pair 7 Panel Granularity at `56T`**:
   - Switching the GEMM benchmark runners to N-major 2D `(tile_n, tile_m)` OpenMP partitioning with exact `cur_nc` tail clamping eliminates the 1D wave-tail imbalance seen previously on `pair3` `2520×2520×72` (both `7x32` and `5x64` now exceed `4.94 TFLOPS` on 56C).
   - In `pair7_pack_transpose`, `1024×256` has only `1024 / 32 = 32` `nr=32` panels (`< 56` cores, leaving 24 cores idle at `56T`), whereas `2520×1152` (`79` panels) and `4096×512` (`128` panels) engage all `56` cores.
4. **Compiler & Ceiling Probe Hygiene**:
   - `cpu_ceilings.c` allocates each thread's bandwidth probe arrays contiguously with a `+1024`-byte inter-array stride offset to prevent 4K load-after-store aliasing stalls across arbitrary `malloc` arena alignments, yielding `0.01%–2.3%` CV across all compute and DDR5 probes (`0.01%–0.09%` on L1D Copy).
   - When compiling AMX microkernels directly with GCC 15 (`-O2 -march=sapphirerapids`), GCC 15's `<amxtileintrin.h>` omits `"memory"` clobbers on `_tile_loadd_internal` / `_tile_stream_loadd_internal`, allowing Dead Store Elimination (DSE) to drop stack-buffer stores (`vmovdqa64`) preceding `tileloadd`; adding the missing `"memory"` clobber (matching `_tile_stored_internal` and Clang's intrinsic semantics) preserves all GFNI unpacks and stack stores, verified via both `objdump -d` and built-in numerical smoke checks.

---

## 7. Reproducing the Probes, Benchmarks, and Reports

All source code and automation scripts are checked into the repository and wired into [kernel/Makefile](../../Makefile):

```bash
# 1. Compile and run the repeatable 7-trial hardware ceiling probe (1T, 28T, 56T)
gcc -O3 -march=sapphirerapids -fopenmp kernel/probes/cpu_ceilings.c -o /tmp/cpu_ceilings -lm
OMP_WAIT_POLICY=ACTIVE /tmp/cpu_ceilings \
  --threads 28,56 --trials 7 \
  --output kernel/reports/2026-11-13-gemma4-31b-sapphire-rapids/ceilings_sapphire_rapids.json

# 2a. Benchmark microkernels directly from upstream https://github.com/google/XNNPACK
#     Compile target XNNPACK/YNNPACK microkernel objects + bridge and link with -DAEB_DIRECT_LINK:
#     gcc -O2 -march=sapphirerapids -fopenmp -DAEB_DIRECT_LINK \
#         kernel/scripts/bench_litert_microkernels.c <ukernel_objs> -o /tmp/bench_ukernels -lm -lstdc++
OMP_WAIT_POLICY=ACTIVE /tmp/bench_ukernels \
  --threads 28,56 --trials 7 \
  --xnnpack-commit 6e5b056669823d8501f9921a2c9059d6ddbb2ceb \
  --output kernel/reports/2026-11-13-gemma4-31b-sapphire-rapids/ukernel_benchmarks_sapphire_rapids.json

# 2b. Or run in LD_PRELOAD mode against an installed LiteRT-LM binary:
gcc -shared -fPIC -O3 -march=sapphirerapids -fopenmp kernel/scripts/bench_litert_microkernels.c -o /tmp/libbench_ukernels.so -lm
OMP_WAIT_POLICY=ACTIVE LD_LIBRARY_PATH=/opt/litert-lm/lib:/opt/litert-lm/bin \
  LD_PRELOAD=/tmp/libbench_ukernels.so \
  AEB_BENCH_THREADS=28,56 \
  AEB_BENCH_OUTPUT=kernel/reports/2026-11-13-gemma4-31b-sapphire-rapids/ukernel_benchmarks_sapphire_rapids.json \
  /opt/litert-lm/bin/litert_lm_advanced_main

# 3. (Optional) Compile and run the 7-pair microkernel numerical precision verifier (Level A & Level B)
gcc -shared -fPIC -O3 -march=sapphirerapids -fopenmp kernel/scripts/verify_litert_microkernels.c -o /tmp/libverify_ukernels.so -lm
LD_LIBRARY_PATH=/opt/litert-lm/lib:/opt/litert-lm/bin \
  LD_PRELOAD=/tmp/libverify_ukernels.so \
  AEB_VERIFY_OUTPUT=kernel/reports/2026-11-13-gemma4-31b-sapphire-rapids/ukernel_precision_sapphire_rapids.json \
  /opt/litert-lm/bin/litert_lm_advanced_main

# 4. Generate the unified JSON & Markdown kernel report
python3 kernel/scripts/generate_kernel_report.py --report-dir kernel/reports/2026-11-13-gemma4-31b-sapphire-rapids
```
