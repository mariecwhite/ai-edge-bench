#!/usr/bin/env python3
# Generates the Empirical Hardware Ceiling & Microkernel Roofline Report
# (`roofline_gemma4_31b_sapphire_rapids.json` and `roofline_gemma4_31b_sapphire_rapids.md`)
# from `ceilings_sapphire_rapids.json` and `ukernel_benchmarks_sapphire_rapids.json`.

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPORT_DIR = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "gemma4-31b-sapphire-rapids"
)
CEILINGS_PATH = REPORT_DIR / "ceilings_sapphire_rapids.json"
BENCH_PATH = REPORT_DIR / "ukernel_benchmarks_sapphire_rapids.json"
OUT_JSON_PATH = REPORT_DIR / "roofline_gemma4_31b_sapphire_rapids.json"
OUT_MD_PATH = REPORT_DIR / "roofline_gemma4_31b_sapphire_rapids.md"


def resolve_ceilings(
    ceilings: Dict[str, Any], isa: str, mem_regime: str, is_pack_or_red: bool
) -> Tuple[str, float, float, str, float, float]:
  """Returns (compute_ceil_name, c_1c, c_28c, bw_ceil_name, bw_1c, bw_28c)."""
  comp = ceilings["compute_ceilings_gops"]
  bw = ceilings["bandwidth_ceilings_gbps"]

  if isa == "AMX-INT8":
    c_name = "AMX-INT8 (tdpbssd)"
    c_1c = comp["amx_int8_tdpbssd"]["single_core"]
    c_28c = comp["amx_int8_tdpbssd"]["numa_node0_28c"]
  elif isa == "AVX512-VNNI":
    c_name = "AVX-512 VNNI (vpdpbusd 512b)"
    c_1c = comp["avx512_vnni_int8_vpdpbusd"]["single_core"]
    c_28c = comp["avx512_vnni_int8_vpdpbusd"]["numa_node0_28c"]
  elif isa == "AVX-VNNI-256":
    c_name = "AVX-VNNI (vpdpbusd 256b)"
    c_1c = comp["avx_vnni_256_int8_vpdpbusd"]["single_core"]
    c_28c = comp["avx_vnni_256_int8_vpdpbusd"]["numa_node0_28c"]
  else:
    c_name = "AVX-512 FP32 FMA (vfmadd231ps)"
    c_1c = comp["avx512_fp32_fma"]["single_core"]
    c_28c = comp["avx512_fp32_fma"]["numa_node0_28c"]

  if mem_regime == "DDR5_DRAM_Streaming":
    b_name = "DDR5 DRAM Read"
    b_1c = bw["ddr5_dram"]["read_only"]["single_core"]
    b_28c = bw["ddr5_dram"]["read_only"]["numa_node0_28c"]
  elif mem_regime == "L3_LLC_Warm":
    # On 1 core, 11-22 MB exceeds the 2 MB private L2 and streams from 52.5 MB L3.
    # On 28 cores, static N-partitioning gives 393-786 KB/core, which fits 100%
    # inside each core's 2 MB private L2 cache (56 MB aggregate L2 across 28 cores).
    b_name = "L3 (1C) / L2-Sliced (28C) Read"
    b_1c = bw["l3_llc"]["read_only"]["single_core"]
    b_28c = bw["l2_512kb_per_core"]["read_only"]["numa_node0_28c"]
  elif mem_regime == "L2_Cache_Copy":
    b_name = "L2 Cache Read+Write Copy"
    b_1c = bw["l2_512kb_per_core"]["read_write_copy"]["single_core"]
    b_28c = bw["l2_512kb_per_core"]["read_write_copy"]["numa_node0_28c"]
  elif mem_regime == "L3_LLC_Copy":
    b_name = "L3 (1C) / L2-Sliced (28C) Copy"
    b_1c = bw["l3_llc"]["read_write_copy"]["single_core"]
    b_28c = bw["l2_512kb_per_core"]["read_write_copy"]["numa_node0_28c"]
  elif mem_regime in ("L2_L3_Resident", "L2_Cache_Resident", "L3_LLC_Resident") and is_pack_or_red:
    # Row-wise Softmax operates on 4-16 KB rows; Pass 1 warms the row into L1D/L2
    # so subsequent row passes are serviced from L1D/L2 rather than L3.
    b_name = "L2/L1D Row-Fused Copy"
    b_1c = bw["l2_512kb_per_core"]["read_write_copy"]["single_core"]
    b_28c = bw["l2_512kb_per_core"]["read_write_copy"]["numa_node0_28c"]
  elif "L2" in mem_regime:
    b_name = "L2 Cache Read"
    b_1c = bw["l2_512kb_per_core"]["read_only"]["single_core"]
    b_28c = bw["l2_512kb_per_core"]["read_only"]["numa_node0_28c"]
  else:
    b_name = "L3 Shared LLC Read"
    b_1c = bw["l3_llc"]["read_only"]["single_core"]
    b_28c = bw["l3_llc"]["read_only"]["numa_node0_28c"]

  return c_name, c_1c, c_28c, b_name, b_1c, b_28c


def _invisible_cat(index: int) -> str:
  """Returns a unique non-whitespace zero-width string that Mermaid will not trim or deduplicate."""
  alphabet = ("\u200b", "\u200c", "\u200d", "\u2060")
  return (
      f"{alphabet[(index >> 4) & 3]}"
      f"{alphabet[(index >> 2) & 3]}"
      f"{alphabet[index & 3]}"
  )


def build_side_by_side_chart(
    title: str,
    y_label: str,
    y_max: int,
    labels: List[str],
    xnn_vals: List[float],
    ynn_vals: List[float],
    gap_slots: int = 2,
) -> str:
  """Builds a Mermaid xychart-beta with side-by-side XNNPACK/YNNPACK bars and aligned labels."""
  sentinel = -y_max
  x_cats: List[str] = []
  x_bar: List[float] = []
  y_bar: List[float] = []
  inv_idx = 1

  for idx, lbl in enumerate(labels):
    # Prefix with 3 Braille Pattern Blank (U+2800, category So, never trimmed/collapsed by SVG/JS)
    # so text-anchor="middle" on slot 0 centers the visible label right between slot 0 (XNN) and slot 1 (YNN).
    x_cats.append(f'"\u2800\u2800\u2800{lbl}"')
    x_bar.append(round(xnn_vals[idx], 2))
    y_bar.append(sentinel)

    x_cats.append(f'"{_invisible_cat(inv_idx)}"')
    inv_idx += 1
    x_bar.append(sentinel)
    y_bar.append(round(ynn_vals[idx], 2))

    if idx + 1 < len(labels):
      for _ in range(gap_slots):
        x_cats.append(f'"{_invisible_cat(inv_idx)}"')
        inv_idx += 1
        x_bar.append(sentinel)
        y_bar.append(sentinel)

  lines = [
      "```mermaid",
      '%%{init: {"xyChart": {"width": 1250, "height": 480, "xAxis": {"labelRotation": 0, "labelFontSize": 11, "showTick": false}}, "themeVariables": {"xyChart": {"plotColorPalette": "#2b6cd9, #e67e22"}}}}%%',
      "xychart-beta",
      f'    title "{title}"',
      f'    x-axis "Kernel & Shape" [{", ".join(x_cats)}]',
      f'    y-axis "{y_label}" 0 --> {y_max}',
      f'    bar "XNNPACK" [{", ".join(str(v) for v in x_bar)}]',
      f'    bar "YNNPACK" [{", ".join(str(v) for v in y_bar)}]',
      "```",
  ]
  return "\n".join(lines)


def main() -> None:
  ceilings = json.loads(CEILINGS_PATH.read_text())
  bench_data = json.loads(BENCH_PATH.read_text())

  amx_1c = ceilings["compute_ceilings_gops"]["amx_int8_tdpbssd"]["single_core"]
  amx_28c = ceilings["compute_ceilings_gops"]["amx_int8_tdpbssd"][
      "numa_node0_28c"
  ]

  enriched_rows: List[Dict[str, Any]] = []
  for row in bench_data["benchmarks"]:
    pair_id = row["pair_id"]
    isa = row["isa"]
    mem_regime = row["memory_regime"]
    ai = row["arithmetic_intensity_ops_per_byte"]
    is_pack_or_red = pair_id in ("pair4_fp32_softmax", "pair7_pack_transpose")

    c_name, c_1c, c_28c, b_name, b_1c, b_28c = resolve_ceilings(
        ceilings, isa, mem_regime, is_pack_or_red
    )

    if pair_id == "pair7_pack_transpose":
      # Pure memory copy/transpose kernel: ceiling is BW_prac
      ceil_1c = b_1c
      ceil_28c = b_28c
      bound_1c = "Memory-Bound"
      bound_28c = "Memory-Bound"
      meas_1c = row["single_core"]["gbps"]
      meas_28c = row["numa_node0_28c"]["gbps"]
      unit = "GB/s"
    elif pair_id == "pair4_fp32_softmax":
      # Softmax is memory-hierarchy / vector-polynomial bound; compare GB/s vs BW ceiling
      ceil_1c = b_1c
      ceil_28c = b_28c
      bound_1c = "Memory/Exp-Poly"
      bound_28c = "Memory/Exp-Poly"
      meas_1c = row["single_core"]["gbps"]
      meas_28c = row["numa_node0_28c"]["gbps"]
      unit = "GB/s"
    else:
      mem_roof_1c = ai * b_1c
      mem_roof_28c = ai * b_28c
      ceil_1c = min(c_1c, mem_roof_1c)
      ceil_28c = min(c_28c, mem_roof_28c)
      bound_1c = "Memory-Bound" if mem_roof_1c < c_1c else "Compute-Bound"
      bound_28c = "Memory-Bound" if mem_roof_28c < c_28c else "Compute-Bound"
      meas_1c = row["single_core"]["gops"]
      meas_28c = row["numa_node0_28c"]["gops"]
      unit = "GOPS" if "FP32" not in isa else "GFLOPS"

    util_1c = (meas_1c / ceil_1c * 100.0) if ceil_1c > 0 else 0.0
    util_28c = (meas_28c / ceil_28c * 100.0) if ceil_28c > 0 else 0.0

    # Also compute utilization relative to machine INT8 peak (AMX-INT8) for quantized GEMMs
    if pair_id in ("pair2_int4_gemm", "pair5_int8_gemm", "pair6_int2_gemm"):
      mach_ceil_1c = min(amx_1c, ai * b_1c)
      mach_ceil_28c = min(amx_28c, ai * b_28c)
      mach_util_1c = meas_1c / mach_ceil_1c * 100.0
      mach_util_28c = meas_28c / mach_ceil_28c * 100.0
    else:
      mach_util_1c = util_1c
      mach_util_28c = util_28c

    enriched = dict(row)
    enriched["roofline"] = {
        "unit": unit,
        "compute_ceiling_name": c_name,
        "compute_ceiling_1c": round(c_1c, 2),
        "compute_ceiling_28c": round(c_28c, 2),
        "bandwidth_ceiling_name": b_name,
        "bandwidth_ceiling_1c_gbps": round(b_1c, 2),
        "bandwidth_ceiling_28c_gbps": round(b_28c, 2),
        "effective_ceiling_1c": round(ceil_1c, 2),
        "effective_ceiling_28c": round(ceil_28c, 2),
        "regime_1c": bound_1c,
        "regime_28c": bound_28c,
        "isa_roofline_utilization_1c_pct": round(util_1c, 2),
        "isa_roofline_utilization_28c_pct": round(util_28c, 2),
        "machine_roofline_utilization_1c_pct": round(mach_util_1c, 2),
        "machine_roofline_utilization_28c_pct": round(mach_util_28c, 2),
    }
    enriched_rows.append(enriched)

  out_json = {
      "model": "Gemma 4 31B",
      "microarchitecture": "Intel Sapphire Rapids",
      "hardware_ceilings": ceilings,
      "kernel_roofline_benchmarks": enriched_rows,
  }
  OUT_JSON_PATH.write_text(json.dumps(out_json, indent=2) + "\n")

  # Build Markdown report
  comp = ceilings["compute_ceilings_gops"]
  bw = ceilings["bandwidth_ceilings_gbps"]

  # Prepare charts
  # Chart 1: 28-Core Throughput (TOPS / TFLOPS) across Compute-Bound GEMM Pairs (Pairs 2, 3, 5, 6)
  gemm_labels = [
      "P2: INT4 128x21504",
      "P2: INT4 128x5376",
      "P2: INT4 1024x21504",
      "P5: INT8 280x5376",
      "P5: INT8 2520x1152",
      "P6: INT2 204x4096",
      "P3: FP32 1024x4096",
  ]
  # Map (pair_id, M, N, K, backend) -> row
  lookup: Dict[Tuple[str, int, int, int, str, str], Dict[str, Any]] = {}
  for r in enriched_rows:
    s = r["shape"]
    key = (
        r["pair_id"],
        s["M"],
        s["N"],
        s["K"],
        r["memory_regime"],
        r["backend"],
    )
    lookup[key] = r

  chart1_specs = [
      ("pair2_int4_gemm", 128, 21504, 5376, "DDR5_DRAM_to_L2_Tiled"),
      ("pair2_int4_gemm", 128, 5376, 21504, "DDR5_DRAM_to_L2_Tiled"),
      ("pair2_int4_gemm", 1024, 21504, 5376, "Compute_Bound_Tiled"),
      ("pair5_int8_gemm", 280, 5376, 1152, "L3_LLC_Resident"),
      ("pair5_int8_gemm", 2520, 1152, 1152, "L3_LLC_Resident"),
      ("pair6_int2_gemm", 204, 4096, 1024, "L2_L3_Resident"),
      ("pair3_fp32_attn_bmm", 1024, 4096, 512, "L3_LLC_Resident"),
  ]
  c1_xnn = [
      lookup[(*sp, "XNNPACK")]["numa_node0_28c"]["gops"] / 1000.0
      for sp in chart1_specs
  ]
  c1_ynn = [
      lookup[(*sp, "YNNPACK")]["numa_node0_28c"]["gops"] / 1000.0
      for sp in chart1_specs
  ]
  chart1_md = build_side_by_side_chart(
      "28-Core Compute Throughput: XNNPACK (Blue) vs YNNPACK (Orange) [TOPS / TFLOPS]",
      "TOPS / TFLOPS (28 Cores)",
      85,
      gemm_labels,
      c1_xnn,
      c1_ynn,
  )

  # Chart 2: 28-Core Machine Roofline Utilization (%) across All 7 Pairs
  util_labels = [
      "P1: INT4 (DRAM)",
      "P1: INT4 (L2/L3)",
      "P2: INT4 (M=128)",
      "P3: FP32 BMM",
      "P4: FP32 Softmax",
      "P5: INT8 (M=280)",
      "P6: INT2 (Audio)",
      "P7: Pack/Transpose",
  ]
  chart2_specs = [
      ("pair1_int4_gemv", 1, 5376, 21504, "DDR5_DRAM_Streaming"),
      ("pair1_int4_gemv", 1, 8192, 5376, "L3_LLC_Warm"),
      ("pair2_int4_gemm", 128, 5376, 21504, "DDR5_DRAM_to_L2_Tiled"),
      ("pair3_fp32_attn_bmm", 1024, 4096, 512, "L3_LLC_Resident"),
      ("pair4_fp32_softmax", 2520, 2520, 1, "L3_LLC_Resident"),
      ("pair5_int8_gemm", 280, 5376, 1152, "L3_LLC_Resident"),
      ("pair6_int2_gemm", 204, 4096, 1024, "L2_L3_Resident"),
      ("pair7_pack_transpose", 1, 4096, 512, "L3_LLC_Copy"),
  ]
  c2_xnn = [
      lookup[(*sp, "XNNPACK")]["roofline"]["isa_roofline_utilization_28c_pct"]
      for sp in chart2_specs
  ]
  c2_ynn = [
      lookup[(*sp, "YNNPACK")]["roofline"]["isa_roofline_utilization_28c_pct"]
      for sp in chart2_specs
  ]
  chart2_md = build_side_by_side_chart(
      "28-Core Practical ISA Roofline Utilization (%): XNNPACK (Blue) vs YNNPACK (Orange)",
      "Roofline Utilization (%)",
      100,
      util_labels,
      c2_xnn,
      c2_ynn,
  )

  md = rf"""# Empirical Hardware Ceiling & Microkernel Roofline Report — Gemma 4 31B on Intel Sapphire Rapids

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
>    - At `M = 1`, arithmetic intensity is only **$3.99\text{{ OPs/Byte}}$**, placing both backends deep in the **DDR5 DRAM bandwidth-bound regime** ($BW_{{\text{{prac}}}}(\text{{DDR5 Read}}) = \mathbf{{246.60\text{{ GB/s}}}}$ on 28 cores $\Rightarrow \mathbf{{984.9\text{{ GOPS}}}}$ ceiling).
>    - On cold DDR5-streaming weights (`1×21504×5376` and `1×5376×21504`), **XNNPACK achieves `227.1–232.1 GB/s` (`92.1%–94.1%` of practical DDR5 ceiling)** and **YNNPACK achieves `224.9–228.3 GB/s` (`91.2%–92.6%` of practical DDR5 ceiling)**. Because both saturate the memory bus above 91%, AMX vs. AVX-512 VNNI compute differences are completely hidden in full-model decode.
>    - Conversely, when weights fit warm in L3 cache (`1×8192×5376`, `L3_LLC_Warm`), YNNPACK's `1x32x8_avx512vnni` (**4.54 TOPS**) is **2.08× faster** than XNNPACK's `1x64c4__avx512amx` (**2.18 TOPS**) because `M = 1` utilizes only 1 of 16 rows (`6.25%`) of the AMX tile while incurring stack spill/reload overhead for `tileloadd`.
> 2. **Prefill (`M = 128..1024` 4-Bit GEMM, 46%–65% of Prefill Time) is Governed by an `8.98×` ISA Ceiling Gap (`AMX-INT8` vs. `AVX-512 VNNI`)**:
>    - Our compute ceiling probes measure **$C_{{\text{{prac}}}}(\text{{AMX-INT8}}) = \mathbf{{188.40\text{{ TOPS}}}}$** (`7.70 TOPS`/core) vs. **$C_{{\text{{prac}}}}(\text{{AVX-512 VNNI}}) = \mathbf{{20.97\text{{ TOPS}}}}$** (`0.95 TOPS`/core) on 28 cores — an **8.98× hardware ceiling advantage** for AMX-INT8.
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

{chart1_md}

{chart2_md}

---

## 3. Empirical Hardware Ceilings on Intel Sapphire Rapids (`cpu_ceilings.c`)

All ceilings below were measured directly on the target Sapphire Rapids machine using [cpu_ceilings.c](../../probes/cpu_ceilings.c) across **7 independent barrier-synchronized trials** after explicit warmup.

### 3.1 Practical Compute Ceilings: $C_{{\text{{prac}}}}(\text{{op}}, \text{{dtype}})$

| Instruction Set / Compute Unit | Instruction | Precision | Ops / Inst | 1-Core Median | 1-Core `cv_pct` | 28-Core (NUMA 0) Median | 28-Core `cv_pct` | Relative to AMX-INT8 (28C) | Load-Bearing Kernels Governed |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **AMX-INT8 (`TMUL`)** | `tdpbssd` (`16×16×64`) | `INT8 / u8×s8→s32` | `32,768` | **7,701.36 GOPS** | `3.11%` | **188,402.15 GOPS (188.40 TOPS)** | `0.71%` | `1.00×` (`100%`) | `xnn_qd8_f32_qc4w_*__avx512amx`, `xnn_qd8_f32_qc8w_*__avx512amx`, `ynn::dot_int8_int8_int32_*_amxint8` |
| **AMX-BF16 (`TMUL`)** | `tdpbf16ps` (`16×16×32`) | `BF16→FP32` | `16,384` | **3,866.86 GFLOPS** | `1.89%` | **83,899.87 GFLOPS (83.90 TFLOPS)** | `6.82%` | `0.445×` (`44.5%`) | Reference BF16 TMUL ceiling |
| **AVX-512 VNNI (512-bit)** | `vpdpbusd %zmm` | `u8×s8→s32` | `128` | **948.07 GOPS** | `1.62%` | **20,973.59 GOPS (20.97 TOPS)** | `5.79%` | **`0.111×` (`11.1%`)** | `ynn::dot_uint8_int4_int32_*_avx512vnni`, `ynn::dot_uint8_int2_int32_*_avx512vnni` |
| **AVX-VNNI (256-bit)** | `vpdpbusd %ymm` | `u8×s8→s32` | `64` | **484.59 GOPS** | `0.49%` | **10,667.71 GOPS (10.67 TOPS)** | `4.67%` | **`0.057×` (`5.7%`)** | `xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni` |
| **AVX-512 FP32 FMA (512-bit)** | `vfmadd231ps %zmm` | `FP32` | `32` | **235.72 GFLOPS** | `1.91%` | **5,616.26 GFLOPS (5.62 TFLOPS)** | `3.32%` | `0.030×` (`3.0%`) | `xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast`, `ynn::dot_fp32_*_avx512` |

### 3.2 Practical Memory-Hierarchy Bandwidth Ceilings: $BW_{{\text{{prac}}}}(\text{{WorkingSet}})$

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

"""

  # Group enriched_rows by pair_id
  pair_titles = {
      "pair1_int4_gemv": (
          "4.1 Pair 1: 4-Bit Quantized Linear GEMV (`M = 1`, Decode Bottleneck — 83.2% XNNPACK / 87.3% YNNPACK Decode)"
      ),
      "pair2_int4_gemm": (
          "4.2 Pair 2: 4-Bit Quantized Linear Batched GEMM (`M ∈ {128, 1024}`, Prefill Bottleneck — 45.7% XNNPACK / 64.9% YNNPACK Prefill)"
      ),
      "pair3_fp32_attn_bmm": (
          "4.3 Pair 3: FP32 Attention `BatchMatMul` (`Q·K^T` and `Scores·V` — 36.2% XNNPACK / 17.5% YNNPACK Prefill)"
      ),
      "pair4_fp32_softmax": (
          "4.4 Pair 4: FP32 Attention `Softmax` (`rmax + raddstoreexpminusmax` vs. `max_k1 + sum_k1`)"
      ),
      "pair5_int8_gemm": (
          "4.5 Pair 5: 8-Bit Dynamic-Quantized Linear GEMM (Vision ViT & Audio Conformer — 38%–47% of Vision/Audio)"
      ),
      "pair6_int2_gemm": (
          "4.6 Pair 6: 2-Bit Weight-Only Linear GEMM (Audio Conformer Encoder — 29%–41% of Audio)"
      ),
      "pair7_pack_transpose": (
          "4.7 Pair 7: Matrix Packing / 32-Bit Transpose (`packw_gemm_goi` vs. `transpose_x32_avx512`)"
      ),
  }

  for pid, ptitle in pair_titles.items():
    prows = [r for r in enriched_rows if r["pair_id"] == pid]
    md += f"### {ptitle}\n\n"
    md += "| Shape `(M×N×K)` | Regime | Backend | Microkernel (`ISA`) | AI (`OP/B`) | 1C Latency (`ms`) | 1C Perf | 1C Ceiling | 1C Roofline Util | 28C Latency (`ms`) | 28C Perf | 28C `cv_pct` | 28C ISA Ceiling | **28C ISA Roofline Util** | **28C Machine Peak Util** |\n"
    md += "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    for r in prows:
      s = r["shape"]
      rf = r["roofline"]
      sc = r["single_core"]
      mc = r["numa_node0_28c"]
      unit = rf["unit"]
      p1 = sc["gbps"] if unit == "GB/s" else sc["gops"]
      p28 = mc["gbps"] if unit == "GB/s" else mc["gops"]
      shape_str = f"`{s['M']}×{s['N']}×{s['K']}`"
      k_short = r["kernel"]
      md += (
          f"| {shape_str} | `{r['memory_regime']}` | **{r['backend']}** | "
          f"`{k_short}` (`{r['isa']}`) | `{r['arithmetic_intensity_ops_per_byte']:.2f}` | "
          f"`{sc['latency_ms']:.3f}` | `{p1:.2f} {unit}` (`{sc['gbps']:.1f} GB/s`) | "
          f"`{rf['effective_ceiling_1c']:.2f} {unit}` | `{rf['isa_roofline_utilization_1c_pct']:.1f}%` | "
          f"`{mc['latency_ms']:.3f}` | **`{p28:.2f} {unit}`** (`{mc['gbps']:.1f} GB/s`) | "
          f"`{mc['cv_pct']:.2f}%` | `{rf['effective_ceiling_28c']:.2f} {unit}` | "
          f"**`{rf['isa_roofline_utilization_28c_pct']:.1f}%`** | "
          f"**`{rf['machine_roofline_utilization_28c_pct']:.1f}%`** |\n"
      )
    md += "\n"

  md += r"""---

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
"""

  OUT_MD_PATH.write_text(md)
  print(f"Wrote {OUT_JSON_PATH} and {OUT_MD_PATH}")


if __name__ == "__main__":
  main()
