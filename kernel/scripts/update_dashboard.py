#!/usr/bin/env python3
"""Generate and update the AI Edge Bench Level-1 Kernel Dashboard.

Aggregates dated kernel benchmark reports under `kernel/reports/YYYY-MM-DD-<suite>/`
(`ceilings_*.json`, `ukernel_benchmarks_*.json`, `kernel_report_*.json`) and emits:
  1. `kernel/dashboards/data.json`   — structured time-series JSON across all report dates
  2. `kernel/dashboards/index.html`  — interactive self-contained HTML/SVG dashboard
  3. `kernel/dashboards/README.md`   — GitHub-native Markdown dashboard with Mermaid charts

Usage:
  python3 kernel/scripts/update_dashboard.py
  python3 kernel/dashboards/update_dashboard.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

KERNEL_ROOT = Path(__file__).resolve().parents[1]
DATE_DIR_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)$")

PAIR_META = {
    "pair1_int4_gemv": {
        "short": "P1: INT4 GEMV (Decode)",
        "title": "Pair 1: 4-Bit Linear GEMV (Decode, M=1)",
        "metric": "gops",
        "unit": "GOPS",
        "chart_group": "memory",
        "order": 1,
    },
    "pair2_int4_gemm": {
        "short": "P2: INT4 GEMM (Prefill)",
        "title": "Pair 2: 4-Bit Linear GEMM (Prefill, M=128..1024)",
        "metric": "gops",
        "unit": "GOPS",
        "chart_group": "compute",
        "order": 2,
    },
    "pair3_fp32_attn_bmm": {
        "short": "P3: FP32 Attention BMM",
        "title": "Pair 3: FP32 Attention BatchMatMul (SDPA)",
        "metric": "gops",
        "unit": "GFLOPS",
        "chart_group": "compute",
        "order": 3,
    },
    "pair4_fp32_softmax": {
        "short": "P4: FP32 Softmax",
        "title": "Pair 4: FP32 Attention Softmax / Reductions",
        "metric": "gbps",
        "unit": "GB/s",
        "chart_group": "memory",
        "order": 4,
    },
    "pair5_int8_gemm": {
        "short": "P5: INT8 GEMM (Multimodal)",
        "title": "Pair 5: 8-Bit Linear GEMM (Vision ViT & Audio)",
        "metric": "gops",
        "unit": "GOPS",
        "chart_group": "compute",
        "order": 5,
    },
    "pair6_int2_gemm": {
        "short": "P6: INT2 GEMM (Audio/Embed)",
        "title": "Pair 6: 2-Bit Weight-Only GEMM (Audio Conformer)",
        "metric": "gops",
        "unit": "GOPS",
        "chart_group": "compute",
        "order": 6,
    },
    "pair7_pack_transpose": {
        "short": "P7: 32-Bit Pack/Transpose",
        "title": "Pair 7: Matrix Packing / 32-Bit Transpose",
        "metric": "gbps",
        "unit": "GB/s",
        "chart_group": "memory",
        "order": 7,
    },
}

# Headline representative shapes featured by default in the overview charts
HEADLINE_SHAPES = {
    ("pair1_int4_gemv", 1, 21504, 5376, "DDR5_DRAM_Streaming"),
    ("pair1_int4_gemv", 1, 8192, 5376, "L3_LLC_Warm"),
    ("pair2_int4_gemm", 128, 21504, 5376, "DDR5_DRAM_to_L2_Tiled"),
    ("pair2_int4_gemm", 128, 8192, 5376, "L3_LLC_to_L2_Tiled"),
    ("pair2_int4_gemm", 1024, 21504, 5376, "Compute_Bound_Tiled"),
    ("pair3_fp32_attn_bmm", 1024, 1024, 256, "L3_LLC_Resident"),
    ("pair3_fp32_attn_bmm", 1024, 4096, 512, "L3_LLC_Resident"),
    ("pair4_fp32_softmax", 1024, 4096, 1, "L3_LLC_Resident"),
    ("pair5_int8_gemm", 280, 5376, 1152, "L3_LLC_Resident"),
    ("pair5_int8_gemm", 2520, 1152, 1152, "L3_LLC_Resident"),
    ("pair6_int2_gemm", 204, 4096, 1024, "L2_L3_Resident"),
    ("pair7_pack_transpose", 1, 2520, 1152, "L3_LLC_Copy"),
}

PAIR_COLORS = {
    "pair1_int4_gemv": {"YNNPACK": "#d97706", "XNNPACK": "#2563eb", "YNNPACK_VNNI_BASELINE": "#0d9488"},
    "pair2_int4_gemm": {"YNNPACK": "#ea580c", "XNNPACK": "#1d4ed8", "YNNPACK_VNNI_BASELINE": "#0f766e"},
    "pair3_fp32_attn_bmm": {"YNNPACK": "#16a34a", "XNNPACK": "#0284c7", "YNNPACK_VNNI_BASELINE": "#0d9488"},
    "pair4_fp32_softmax": {"YNNPACK": "#9333ea", "XNNPACK": "#4f46e5", "YNNPACK_VNNI_BASELINE": "#0d9488"},
    "pair5_int8_gemm": {"YNNPACK": "#dc2626", "XNNPACK": "#0369a1", "YNNPACK_VNNI_BASELINE": "#0d9488"},
    "pair6_int2_gemm": {"YNNPACK": "#ca8a04", "XNNPACK": "#4338ca", "YNNPACK_VNNI_BASELINE": "#0d9488"},
    "pair7_pack_transpose": {"YNNPACK": "#db2777", "XNNPACK": "#0f766e", "YNNPACK_VNNI_BASELINE": "#0d9488"},
}

BACKEND_STYLE = {
    "YNNPACK": ("circle", "solid"),
    "XNNPACK": ("square", "dashed"),
    "YNNPACK_VNNI_BASELINE": ("triangle", "dotted"),
}


def _tc_key(t: int) -> str:
  return "single_core" if t == 1 else f"numa_node0_{t}c"


def _tc_sfx(t: int) -> str:
  return f"{t}c"


def _shape_str(s: dict[str, int]) -> str:
  return f"{s['M']}×{s['N']}×{s['K']}"


def _extract_point(row: dict[str, Any], report_id: str, date_str: str, commit: str, report_rel_url: str) -> dict[str, Any]:
  roof = row.get("roofline", {})
  by_threads: dict[str, Any] = {}
  for t in (1, 28, 56):
    k = _tc_key(t)
    sfx = _tc_sfx(t)
    if k not in row:
      continue
    m = row[k]
    gops = m.get("gops", 0.0)
    gbps = m.get("gbps", 0.0)
    pair_id = row["pair_id"]
    is_bw_primary = pair_id in ("pair4_fp32_softmax", "pair7_pack_transpose")
    val_med = gbps if is_bw_primary else gops
    val_min = m.get("min_gops", val_med)
    val_max = m.get("max_gops", val_med)
    by_threads[str(t)] = {
        "latency_ms": m.get("latency_ms"),
        "gops": gops,
        "gbps": gbps,
        "value": val_med,
        "min": val_min,
        "max": val_max,
        "cv_pct": m.get("cv_pct", 0.0),
        "effective_ceiling": roof.get(f"effective_ceiling_{sfx}"),
        "compute_ceiling": roof.get(f"compute_ceiling_{sfx}"),
        "bandwidth_ceiling_gbps": roof.get(f"bandwidth_ceiling_{sfx}_gbps"),
        "regime": roof.get(f"regime_{sfx}"),
        "isa_util_pct": roof.get(f"isa_roofline_utilization_{sfx}_pct"),
        "machine_util_pct": roof.get(f"machine_roofline_utilization_{sfx}_pct"),
    }
  return {
      "report_id": report_id,
      "date": date_str,
      "xnnpack_commit": commit,
      "kernel": row.get("kernel", ""),
      "isa": row.get("isa", ""),
      "report_url": report_rel_url,
      "threads": by_threads,
  }


def collect_kernel_dashboard_data(reports_dir: Path) -> dict[str, Any]:
  suites_map: dict[str, list[tuple[str, Path]]] = {}
  for p in sorted(reports_dir.iterdir()):
    if not p.is_dir():
      continue
    m = DATE_DIR_RE.match(p.name)
    if not m:
      continue
    date_str, suite_id = m.group(1), m.group(2)
    suites_map.setdefault(suite_id, []).append((date_str, p))

  suites_out: list[dict[str, Any]] = []
  for suite_id, dated_dirs in sorted(suites_map.items()):
    reports_meta: list[dict[str, Any]] = []
    series_map: dict[tuple[str, int, int, int, str, str], dict[str, Any]] = {}

    for date_str, rdir in dated_dirs:
      report_files = sorted(rdir.glob("kernel_report_*.json"))
      if not report_files:
        continue
      rj_path = report_files[0]
      rmd_files = sorted(rdir.glob("kernel_report_*.md"))
      rmd_rel = f"../reports/{rdir.name}/{rmd_files[0].name}" if rmd_files else ""

      rdata = json.loads(rj_path.read_text(encoding="utf-8"))
      commit = rdata.get("xnnpack_commit") or "28d1fc6d (LiteRT-LM v0.19.1)"
      commit_short = commit[:8] if len(commit) >= 8 else commit
      t_cfgs = rdata.get("thread_configs", [1, 28])
      ceilings = rdata.get("hardware_ceilings", {})

      reports_meta.append({
          "report_id": rdir.name,
          "date": date_str,
          "xnnpack_commit": commit,
          "commit_short": commit_short,
          "thread_configs": t_cfgs,
          "report_url": rmd_rel,
          "hardware_ceilings": ceilings,
      })

      all_rows = list(rdata.get("kernel_roofline_benchmarks", [])) + list(
          rdata.get("baseline_comparisons", [])
      )
      for row in all_rows:
        s = row["shape"]
        pair_id = row["pair_id"]
        backend = row["backend"]
        reg = row["memory_regime"]
        key = (pair_id, s["M"], s["N"], s["K"], reg, backend)
        pmeta = PAIR_META.get(pair_id, {})
        if key not in series_map:
          shape_label = _shape_str(s)
          is_headline = (pair_id, s["M"], s["N"], s["K"], reg) in HEADLINE_SHAPES
          color = PAIR_COLORS.get(pair_id, {}).get(backend, "#2563eb")
          marker, linestyle = BACKEND_STYLE.get(backend, ("circle", "solid"))
          reg_short = "Warm L3" if "Warm" in reg else ("Cold DRAM" if "DRAM_Streaming" in reg else reg.replace("_", " "))
          series_id = f"{pmeta.get('short', pair_id)} · {backend} · {shape_label}"
          if pair_id == "pair1_int4_gemv" and s["N"] == 8192:
            series_id += f" ({reg_short})"
          series_map[key] = {
              "id": series_id,
              "pair_id": pair_id,
              "pair_short": pmeta.get("short", pair_id),
              "pair_title": pmeta.get("title", row.get("category", pair_id)),
              "pair_order": pmeta.get("order", 99),
              "chart_group": pmeta.get("chart_group", "compute"),
              "backend": backend,
              "shape": s,
              "shape_str": shape_label,
              "memory_regime": reg,
              "arithmetic_intensity": row.get("arithmetic_intensity_ops_per_byte", 0.0),
              "primary_metric": pmeta.get("metric", "gops"),
              "unit": pmeta.get("unit", "GOPS"),
              "is_headline": is_headline,
              "color": color,
              "marker": marker,
              "linestyle": linestyle,
              "points": [],
          }
        pt = _extract_point(row, rdir.name, date_str, commit_short, rmd_rel)
        series_map[key]["points"].append(pt)

    # Enrich each series with per-thread baseline ratios and deltas
    series_list = list(series_map.values())
    series_list.sort(
        key=lambda x: (
            x["pair_order"],
            x["shape"]["M"],
            x["shape"]["N"],
            x["shape"]["K"],
            x["memory_regime"],
            0 if x["backend"] == "YNNPACK" else (1 if x["backend"] == "XNNPACK" else 2),
        )
    )

    for s in series_list:
      pts = s["points"]
      summary_by_thread: dict[str, Any] = {}
      for t_str in ("1", "28", "56"):
        t_pts = [p for p in pts if t_str in p["threads"]]
        if not t_pts:
          continue
        first_v = t_pts[0]["threads"][t_str]["value"]
        last_v = t_pts[-1]["threads"][t_str]["value"]
        prev_v = t_pts[-2]["threads"][t_str]["value"] if len(t_pts) >= 2 else None

        # Annotate speedup_vs_first on each point
        for p in t_pts:
          cur_v = p["threads"][t_str]["value"]
          p["threads"][t_str]["speedup_vs_first"] = (
              round(cur_v / first_v, 4) if first_v and first_v > 0 else 1.0
          )

        delta_first_abs = (last_v - first_v) if (first_v is not None and len(t_pts) >= 2) else None
        delta_first_pct = (
            ((last_v - first_v) / first_v * 100.0)
            if (first_v and first_v > 0 and len(t_pts) >= 2)
            else None
        )
        speedup_first_x = (
            (last_v / first_v) if (first_v and first_v > 0 and len(t_pts) >= 2) else None
        )
        delta_prev_abs = (last_v - prev_v) if prev_v is not None else None
        delta_prev_pct = (
            ((last_v - prev_v) / prev_v * 100.0) if (prev_v and prev_v > 0) else None
        )

        summary_by_thread[t_str] = {
            "first_date": t_pts[0]["date"],
            "first_value": first_v,
            "latest_date": t_pts[-1]["date"],
            "latest_value": last_v,
            "latest_kernel": t_pts[-1]["kernel"],
            "latest_isa": t_pts[-1]["isa"],
            "latest_isa_util_pct": t_pts[-1]["threads"][t_str]["isa_util_pct"],
            "latest_machine_util_pct": t_pts[-1]["threads"][t_str]["machine_util_pct"],
            "delta_vs_first_abs": delta_first_abs,
            "delta_vs_first_pct": delta_first_pct,
            "speedup_vs_first_x": speedup_first_x,
            "delta_vs_prev_abs": delta_prev_abs,
            "delta_vs_prev_pct": delta_prev_pct,
        }
      s["summary_by_thread"] = summary_by_thread

    suites_out.append({
        "id": suite_id,
        "model": "Gemma 4 31B (gemma-4-31B-it)",
        "microarchitecture": "Intel Sapphire Rapids (Socket 0, AMX-INT8 / AVX-512 VNNI / GFNI)",
        "reports": reports_meta,
        "series": series_list,
    })

  return {
      "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
      "suites": suites_out,
  }


def generate_markdown_dashboard(data: dict[str, Any]) -> str:
  suite = data["suites"][0]
  reports = suite["reports"]
  series = suite["series"]

  by_key: dict[tuple[str, int, int, int, str, str], dict[str, Any]] = {}
  for s in series:
    sh = s["shape"]
    by_key[(s["pair_id"], sh["M"], sh["N"], sh["K"], s["memory_regime"], s["backend"])] = s

  def get_pt(s: dict[str, Any], date_str: str, t_str: str = "28") -> dict[str, Any] | None:
    for p in s["points"]:
      if p["date"] == date_str and t_str in p["threads"]:
        return p["threads"][t_str]
    return None

  # Build Mermaid progression charts across 2026-11-10 -> 2026-11-13 -> 2026-11-17 (28C)
  dates = [r["date"] for r in reports]
  p2_128_21k_y = by_key[("pair2_int4_gemm", 128, 21504, 5376, "DDR5_DRAM_to_L2_Tiled", "YNNPACK")]
  p2_128_21k_x = by_key[("pair2_int4_gemm", 128, 21504, 5376, "DDR5_DRAM_to_L2_Tiled", "XNNPACK")]
  p2_128_8k_y = by_key[("pair2_int4_gemm", 128, 8192, 5376, "L3_LLC_to_L2_Tiled", "YNNPACK")]
  p2_1024_21k_y = by_key[("pair2_int4_gemm", 1024, 21504, 5376, "Compute_Bound_Tiled", "YNNPACK")]
  p5_2520_1k_y = by_key[("pair5_int8_gemm", 2520, 1152, 1152, "L3_LLC_Resident", "YNNPACK")]
  p3_1024_1k_y = by_key[("pair3_fp32_attn_bmm", 1024, 1024, 256, "L3_LLC_Resident", "YNNPACK")]
  p4_1024_4k_y = by_key[("pair4_fp32_softmax", 1024, 4096, 1, "L3_LLC_Resident", "YNNPACK")]

  def vals_tops(s: dict[str, Any], t_str: str = "28") -> list[float]:
    out = []
    for d in dates:
      pt = get_pt(s, d, t_str)
      out.append(round((pt["value"] / 1000.0) if pt else 0.0, 2))
    return out

  def vals_speedup(s: dict[str, Any], t_str: str = "28") -> list[float]:
    out = []
    for d in dates:
      pt = get_pt(s, d, t_str)
      out.append(round(pt["speedup_vs_first"] if pt else 1.0, 2))
    return out

  date_axis = ", ".join(f'"{d}"' for d in dates)

  md = f"""# Level 1 Kernel Benchmark Suite Dashboard — Gemma 4 31B on Intel Sapphire Rapids

- **Interactive HTML Dashboard**: [`kernel/dashboards/index.html`](index.html)
- **Raw Time-Series JSON**: [`kernel/dashboards/data.json`](data.json)
- **Last Updated**: `{data['generated_at']}`
- **Tracked Reports**:
"""
  for r in reports:
    md += f"  - **`{r['date']}`** (`google/XNNPACK` @ `{r['commit_short']}`, threads `{r['thread_configs']}`): [{r['report_id']}/kernel_report_gemma4_31b_sapphire_rapids.md]({r['report_url']})\n"

  md += f"""
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
%%{{init: {{"xyChart": {{"width": 1100, "height": 420}}, "themeVariables": {{"xyChart": {{"plotColorPalette": "#ea580c, #d97706, #dc2626, #2563eb"}}}}}}%%
xychart-beta
    title "28-Core GEMM Throughput Progression Over Time (TOPS, Higher is Better)"
    x-axis "Benchmark Date" [{date_axis}]
    y-axis "Throughput (TOPS, 28 Cores)" 0 --> 42
    line "YNN P2: INT4 128x21504x5376" {vals_tops(p2_128_21k_y)}
    line "YNN P2: INT4 1024x21504x5376" {vals_tops(p2_1024_21k_y)}
    line "YNN P5: INT8 2520x1152x1152" {vals_tops(p5_2520_1k_y)}
    line "XNN P2: INT4 128x21504x5376" {vals_tops(p2_128_21k_x)}
```

```mermaid
%%{{init: {{"xyChart": {{"width": 1100, "height": 420}}, "themeVariables": {{"xyChart": {{"plotColorPalette": "#9333ea, #ea580c, #dc2626, #16a34a"}}}}}}%%
xychart-beta
    title "Cumulative Kernel Speedup Ratio vs. 2026-11-10 Baseline (28 Cores, x)"
    x-axis "Benchmark Date" [{date_axis}]
    y-axis "Speedup Ratio vs. 2026-11-10 (x)" 0 --> 5
    line "YNN P4: Softmax 1024x4096 (4.45x)" {vals_speedup(p4_1024_4k_y)}
    line "YNN P2: INT4 128x8192x5376 (2.59x)" {vals_speedup(p2_128_8k_y)}
    line "YNN P5: INT8 2520x1152x1152 (1.48x)" {vals_speedup(p5_2520_1k_y)}
    line "YNN P3: FP32 BMM 1024x1024x256 (1.34x)" {vals_speedup(p3_1024_1k_y)}
```

---

## 2. Complete Multi-Core (`28C` & `56C`) Progression Table Across All 7 Kernel Pairs

| Kernel Pair | Shape `(M×N×K)` | Regime | Backend | `2026-11-10` (`28C`) | `2026-11-13` (`28C`) | `2026-11-17` (`28C`) | **28C Speedup (`Nov 10 → Nov 17`)** | `2026-11-13` (`56C`) | `2026-11-17` (`56C`) | **Latest `56C` ISA Util** |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
"""

  def fmt_v(val: float | None, unit: str) -> str:
    if val is None:
      return "—"
    if unit in ("GOPS", "GFLOPS") and val >= 10000.0:
      tu = "TOPS" if unit == "GOPS" else "TFLOPS"
      return f"{val / 1000.0:.2f} {tu}"
    return f"{val:,.2f} {unit}"

  for s in series:
    if s["backend"] == "YNNPACK_VNNI_BASELINE":
      continue
    unit = s["unit"]
    pt10_28 = get_pt(s, "2026-11-10", "28")
    pt13_28 = get_pt(s, "2026-11-13", "28")
    pt17_28 = get_pt(s, "2026-11-17", "28")
    pt13_56 = get_pt(s, "2026-11-13", "56")
    pt17_56 = get_pt(s, "2026-11-17", "56")

    v10_28 = pt10_28["value"] if pt10_28 else None
    v13_28 = pt13_28["value"] if pt13_28 else None
    v17_28 = pt17_28["value"] if pt17_28 else None
    v13_56 = pt13_56["value"] if pt13_56 else None
    v17_56 = pt17_56["value"] if pt17_56 else None
    u17_56 = pt17_56["isa_util_pct"] if pt17_56 else None

    if v10_28 and v17_28:
      ratio = v17_28 / v10_28
      pct = (ratio - 1.0) * 100.0
      sign = "+" if pct >= 0 else ""
      sp_str = f"**`{ratio:.2f}×` (`{sign}{pct:.1f}%`)**" if abs(pct) >= 15.0 else f"`{ratio:.2f}×` (`{sign}{pct:.1f}%`)"
    else:
      sp_str = "—"

    util_str = f"`{u17_56:.1f}%`" if u17_56 is not None else "—"
    md += (
        f"| `{s['pair_short']}` | `{s['shape_str']}` | `{s['memory_regime']}` | **{s['backend']}** | "
        f"`{fmt_v(v10_28, unit)}` | `{fmt_v(v13_28, unit)}` | `{fmt_v(v17_28, unit)}` | {sp_str} | "
        f"`{fmt_v(v13_56, unit)}` | **`{fmt_v(v17_56, unit)}`** | {util_str} |\n"
    )

  md += """
---

## 3. Updating This Dashboard

Whenever a new dated report directory (`kernel/reports/YYYY-MM-DD-gemma4-31b-sapphire-rapids/`) is added or updated, regenerate both the interactive HTML dashboard and this Markdown summary via:

```bash
python3 kernel/scripts/update_dashboard.py
# or via Make:
make -C kernel dashboard
```
"""
  return md


def render_html_dashboard(data: dict[str, Any]) -> str:
  generated_at = data.get("generated_at", "")
  suite = data["suites"][0]
  n_reports = len(suite["reports"])
  n_series = len([s for s in suite["series"] if s["backend"] != "YNNPACK_VNNI_BASELINE"])
  data_json = json.dumps(data).replace("</", "<\\/")

  return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Edge Bench — Level 1 Kernel Benchmark Dashboard Over Time</title>
  <style>
    :root {{
      --bg: #f6f8fa;
      --surface: #ffffff;
      --surface-alt: #f0f3f6;
      --fg: #1f2328;
      --muted: #59636e;
      --border: #d1d9e0;
      --border-subtle: #e6ebf1;
      --accent: #0969da;
      --accent-soft: #ddf4ff;
      --success: #1a7f37;
      --success-bg: #dafbe1;
      --danger: #cf222e;
      --danger-bg: #ffebe9;
      --code-bg: #eff2f5;
      --shadow: 0 1px 3px rgba(31, 35, 40, 0.06), 0 4px 12px rgba(31, 35, 40, 0.04);
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
      --font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 0 0 3.5rem 0;
      font-family: var(--font-sans);
      font-size: 14.5px;
      line-height: 1.5;
      color: var(--fg);
      background: var(--bg);
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{
      font-family: var(--font-mono);
      font-size: 0.86em;
      background: var(--code-bg);
      padding: 0.14em 0.38em;
      border-radius: 4px;
    }}
    header.top-bar {{
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 1.15rem 2rem;
      position: sticky;
      top: 0;
      z-index: 20;
      box-shadow: 0 1px 2px rgba(31, 35, 40, 0.04);
    }}
    .top-bar-inner {{
      max-width: 1560px;
      margin: 0 auto;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
    }}
    .brand-title h1 {{
      margin: 0;
      font-size: 1.35rem;
      font-weight: 700;
      letter-spacing: -0.01em;
    }}
    .brand-subtitle {{
      font-size: 0.86rem;
      color: var(--muted);
      margin-top: 0.15rem;
    }}
    .header-meta {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
      font-size: 0.82rem;
      color: var(--muted);
    }}
    .meta-pill {{
      background: var(--surface-alt);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 0.3rem 0.75rem;
      font-weight: 500;
      color: var(--fg);
    }}
    main {{
      max-width: 1560px;
      margin: 1.5rem auto 0;
      padding: 0 2rem;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1rem;
      margin-bottom: 1.25rem;
    }}
    .kpi-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.95rem 1.15rem;
      box-shadow: var(--shadow);
    }}
    .kpi-label {{
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
      font-weight: 600;
    }}
    .kpi-value {{
      font-size: 1.5rem;
      font-weight: 700;
      margin-top: 0.2rem;
      color: var(--fg);
    }}
    .kpi-sub {{
      font-size: 0.81rem;
      color: var(--muted);
      margin-top: 0.15rem;
    }}
    .suite-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      margin-bottom: 2rem;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .suite-header {{
      padding: 1.25rem 1.5rem 1rem;
      border-bottom: 1px solid var(--border-subtle);
      background: linear-gradient(180deg, #ffffff 0%, #fafbfc 100%);
    }}
    .suite-title-row {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
    }}
    .suite-title-row h2 {{
      margin: 0;
      font-size: 1.25rem;
      font-weight: 700;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.5rem;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 0.18rem 0.55rem;
      font-size: 0.76rem;
      font-weight: 600;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--surface-alt);
      color: var(--fg);
    }}
    .badge-machine {{
      background: var(--accent-soft);
      border-color: #80ccff;
      color: #0550ae;
    }}
    .report-links-bar {{
      margin-top: 0.65rem;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.45rem;
      font-size: 0.82rem;
    }}
    .report-link-pill {{
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      padding: 0.2rem 0.6rem;
      border-radius: 5px;
      background: var(--surface-alt);
      border: 1px solid var(--border);
      font-weight: 500;
      font-size: 0.8rem;
    }}
    .controls-bar {{
      padding: 0.8rem 1.5rem;
      background: var(--surface-alt);
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.85rem 1.35rem;
      font-size: 0.82rem;
    }}
    .control-group {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.35rem;
    }}
    .control-label {{
      font-weight: 600;
      color: var(--muted);
      margin-right: 0.2rem;
    }}
    .ctrl-btn {{
      appearance: none;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--fg);
      font-size: 0.78rem;
      font-weight: 500;
      padding: 0.25rem 0.6rem;
      border-radius: 5px;
      cursor: pointer;
      transition: all 0.1s ease;
    }}
    .ctrl-btn:hover {{ border-color: var(--accent); }}
    .ctrl-btn.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
      font-weight: 600;
    }}
    .chart-section {{ padding: 1.25rem 1.5rem 0.75rem; }}
    .charts-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 1.25rem;
    }}
    .charts-grid.stacked {{ grid-template-columns: 1fr; }}
    @media (max-width: 1180px) {{
      .charts-grid {{ grid-template-columns: 1fr; }}
    }}
    .chart-panel {{
      background: #ffffff;
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 0.75rem 0.85rem 0.5rem;
      box-shadow: 0 1px 2px rgba(31, 35, 40, 0.03);
    }}
    .chart-panel-header {{
      display: flex;
      flex-wrap: wrap;
      align-items: baseline;
      justify-content: space-between;
      gap: 0.5rem;
      margin-bottom: 0.35rem;
      padding: 0 0.25rem;
    }}
    .chart-panel-title {{
      font-size: 0.98rem;
      font-weight: 700;
      color: var(--fg);
    }}
    .chart-panel-subtitle {{
      font-size: 0.78rem;
      color: var(--muted);
    }}
    .chart-svg {{
      width: 100%;
      height: auto;
      display: block;
      overflow: visible;
      user-select: none;
    }}
    .legend-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.38rem 0.6rem;
      margin-top: 0.9rem;
      padding: 0.65rem 0.85rem;
      background: var(--surface-alt);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 0.38rem;
      font-size: 0.78rem;
      padding: 0.18rem 0.48rem;
      border-radius: 4px;
      cursor: pointer;
      border: 1px solid transparent;
      background: #ffffff;
      transition: all 0.12s ease;
    }}
    .legend-item:hover {{
      border-color: var(--accent);
      background: var(--accent-soft);
    }}
    .legend-item.muted {{
      opacity: 0.35;
      text-decoration: line-through;
    }}
    .legend-swatch {{
      width: 22px;
      height: 12px;
      display: inline-block;
      vertical-align: middle;
    }}
    .table-section {{ padding: 0.5rem 1.5rem 1.5rem; }}
    .table-section h3 {{
      font-size: 1.02rem;
      margin: 0.75rem 0 0.6rem;
      color: var(--fg);
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
    }}
    .table-wrap {{
      width: 100%;
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12.8px;
      white-space: nowrap;
    }}
    th, td {{
      padding: 0.45rem 0.65rem;
      border-bottom: 1px solid var(--border-subtle);
      border-right: 1px solid var(--border-subtle);
      text-align: left;
      vertical-align: middle;
    }}
    th:last-child, td:last-child {{ border-right: none; }}
    tr:last-child td {{ border-bottom: none; }}
    thead th {{
      background: var(--surface-alt);
      font-weight: 600;
      color: var(--fg);
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    tbody tr:hover {{ background: #f0f6fc; }}
    tbody tr.group-row td {{
      background: #eef2f7;
      font-weight: 700;
      font-size: 12.5px;
      color: var(--fg);
      border-top: 1.5px solid var(--border);
      border-bottom: 1px solid var(--border);
      padding: 0.4rem 0.65rem;
    }}
    tbody tr.best-row {{ background: #f0fdf4; }}
    tbody tr.best-row:hover {{ background: #dcfce7; }}
    tbody tr.best-row td:first-child {{ border-left: 3.5px solid #16a34a; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .sub-range {{
      display: block;
      font-size: 10.8px;
      color: var(--muted);
      font-weight: 400;
    }}
    .delta-pill {{
      display: inline-block;
      padding: 0.1rem 0.42rem;
      border-radius: 4px;
      font-size: 11.3px;
      font-weight: 600;
    }}
    .delta-good {{ background: var(--success-bg); color: var(--success); }}
    .delta-bad {{ background: var(--danger-bg); color: var(--danger); }}
    .delta-neutral {{ background: var(--code-bg); color: var(--muted); }}
    .chart-tooltip {{
      position: fixed;
      pointer-events: none;
      z-index: 100;
      background: rgba(24, 28, 34, 0.96);
      color: #f0f6fc;
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 8px;
      padding: 0.65rem 0.85rem;
      font-size: 12.2px;
      line-height: 1.45;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
      max-width: 390px;
      display: none;
    }}
    .tt-title {{
      font-weight: 700;
      font-size: 13px;
      margin-bottom: 0.3rem;
      display: flex;
      align-items: center;
      gap: 0.45rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.15);
      padding-bottom: 0.25rem;
    }}
    .tt-grid {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 0.15rem 0.75rem;
    }}
    .tt-key {{ color: #9198a1; }}
    .tt-val {{ text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; }}
    .update-banner {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.9rem 1.25rem;
      font-size: 0.86rem;
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
    }}
  </style>
</head>
<body>
  <header class="top-bar">
    <div class="top-bar-inner">
      <div class="brand-title">
        <h1>AI Edge Bench — Level 1 Kernel Benchmark Suite Dashboard Over Time</h1>
        <div class="brand-subtitle">
          XNNPACK vs. YNNPACK microkernel throughput, practical ISA roofline utilization (%), and cumulative speedups over time on Intel Sapphire Rapids
        </div>
      </div>
      <div class="header-meta">
        <span class="meta-pill">Updated: {generated_at}</span>
        <span>Update command: <code>make -C kernel dashboard</code></span>
      </div>
    </div>
  </header>

  <main>
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Kernel Suite Coverage</div>
        <div class="kpi-value">7 Pairs · {n_series} Series</div>
        <div class="kpi-sub">{n_reports} dated report snapshots (2026-11-10 → 2026-11-17)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">4-Bit Prefill GEMM Gain (Pair 2)</div>
        <div class="kpi-value" style="color:var(--success)">+159% (2.59×)</div>
        <div class="kpi-sub">YNNPACK VNNI → AMX+GFNI (13.9 → 35.9 TOPS @ 28C, 73.8 TOPS @ 56C)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">FP32 Softmax Reduction Gain (Pair 4)</div>
        <div class="kpi-value" style="color:var(--success)">+345% (4.45×)</div>
        <div class="kpi-sub">YNNPACK AVX-512 (522 → 2,325 GB/s @ 28C, 4,011 GB/s @ 56C)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">8-Bit &amp; FP32 Attention Gains (Pairs 3 &amp; 5)</div>
        <div class="kpi-value" style="color:var(--success)">+48% ViT · +34% BMM</div>
        <div class="kpi-sub">YNNPACK INT8 ViT (21.8 → 32.3 TOPS @ 28C) &amp; FP32 BMM (90%+ Roofline)</div>
      </div>
    </div>

    <div id="suites-root"></div>

    <div class="update-banner">
      <div>
        <strong>Updating this dashboard:</strong> Run <code>python3 kernel/scripts/update_dashboard.py</code>
        (or <code>make -C kernel dashboard</code>) to refresh <code>kernel/dashboards/index.html</code>,
        <code>kernel/dashboards/data.json</code>, and <code>kernel/dashboards/README.md</code>.
      </div>
      <div>
        <a href="data.json">Raw Dashboard JSON (data.json)</a> · <a href="README.md">Markdown Dashboard (README.md)</a>
      </div>
    </div>
  </main>

  <div class="chart-tooltip" id="chart-tooltip"></div>

  <script id="dashboard-data" type="application/json">
{data_json}
  </script>

  <script>
  (() => {{
    const rawData = JSON.parse(document.getElementById("dashboard-data").textContent);
    const suite = (rawData.suites || [])[0];
    const tooltip = document.getElementById("chart-tooltip");
    if (!suite) return;

    const st = {{
      threads: "28",          // "1" | "28" | "56"
      pairFilter: "all",      // "all" | pair_id
      backendFilter: "all",   // "all" | "YNNPACK" | "XNNPACK" | "YNNPACK_VNNI_BASELINE"
      scopeFilter: "headline",// "headline" | "all"
      xSpacing: "ordinal",    // "time" | "ordinal"
      yScale: "linear",       // "linear" | "log"
      layout: "side-by-side", // "side-by-side" | "stacked"
      showWhiskers: true,
      hiddenSeries: new Set(),
      hoveredSeries: null,
    }};

    function dashArrayFor(ls) {{
      if (ls === "dashed") return "7,4";
      if (ls === "dotted") return "2.5,3.5";
      return "";
    }}

    function fmtVal(val, unit) {{
      if (val == null || !isFinite(val)) return "—";
      if ((unit === "GOPS" || unit === "GFLOPS") && val >= 10000) {{
        const tu = unit === "GOPS" ? "TOPS" : "TFLOPS";
        return `${{(val / 1000).toFixed(2)}} ${{tu}}`;
      }}
      return `${{Number(val).toLocaleString(undefined, {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }})}} ${{unit}}`;
    }}

    function createMarkerSVG(markerType, cx, cy, r, color, isHighlighted) {{
      const ns = "http://www.w3.org/2000/svg";
      const strokeW = isHighlighted ? "2.2" : "1.4";
      if (markerType === "square") {{
        const el = document.createElementNS(ns, "rect");
        const side = r * 1.75;
        el.setAttribute("x", cx - side / 2);
        el.setAttribute("y", cy - side / 2);
        el.setAttribute("width", side);
        el.setAttribute("height", side);
        el.setAttribute("fill", color);
        el.setAttribute("stroke", "#ffffff");
        el.setAttribute("stroke-width", strokeW);
        return el;
      }}
      if (markerType === "triangle") {{
        const el = document.createElementNS(ns, "polygon");
        const h = r * 2.0;
        const pts = `${{cx}},${{cy - h * 0.62}} ${{cx - h * 0.58}},${{cy + h * 0.45}} ${{cx + h * 0.58}},${{cy + h * 0.45}}`;
        el.setAttribute("points", pts);
        el.setAttribute("fill", color);
        el.setAttribute("stroke", "#ffffff");
        el.setAttribute("stroke-width", strokeW);
        return el;
      }}
      const el = document.createElementNS(ns, "circle");
      el.setAttribute("cx", cx);
      el.setAttribute("cy", cy);
      el.setAttribute("r", r);
      el.setAttribute("fill", color);
      el.setAttribute("stroke", "#ffffff");
      el.setAttribute("stroke-width", strokeW);
      return el;
    }}

    function visibleSeriesForChart(chartFilterFn) {{
      return (suite.series || []).filter(s => {{
        if (st.pairFilter !== "all" && s.pair_id !== st.pairFilter) return false;
        if (st.backendFilter !== "all" && s.backend !== st.backendFilter) return false;
        if (st.scopeFilter === "headline" && st.pairFilter === "all" && !s.is_headline) return false;
        if (chartFilterFn && !chartFilterFn(s)) return false;
        return s.points.some(p => p.threads && p.threads[st.threads]);
      }});
    }}

    function showTooltip(evt, s, pt, prevPt, mode) {{
      const tObj = pt.threads[st.threads];
      if (!tObj) return;
      let deltaHtml = "";
      if (prevPt && prevPt.threads && prevPt.threads[st.threads]) {{
        const prevVal = prevPt.threads[st.threads].value;
        const diff = tObj.value - prevVal;
        const pct = prevVal > 0 ? (diff / prevVal) * 100 : 0;
        const ratio = prevVal > 0 ? (tObj.value / prevVal) : 1;
        const sign = diff >= 0 ? "+" : "";
        const col = pct > 1.0 ? "#4ade80" : (pct < -1.0 ? "#f87171" : "#9ca3af");
        deltaHtml = `<div class="tt-key">Δ vs ${{prevPt.date}}</div><div class="tt-val" style="color:${{col}}">${{ratio.toFixed(2)}}× (${{sign}}${{pct.toFixed(1)}}%)</div>`;
      }}
      tooltip.innerHTML = `
        <div class="tt-title">
          <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${{s.color}}"></span>
          <span>${{s.id}}</span>
        </div>
        <div class="tt-grid">
          <div class="tt-key">Report Date / Commit</div><div class="tt-val">${{pt.date}} (${{pt.xnnpack_commit}})</div>
          <div class="tt-key">Threads</div><div class="tt-val">${{st.threads}}C</div>
          <div class="tt-key">Microkernel (${{pt.isa}})</div><div class="tt-val"><code>${{pt.kernel}}</code></div>
          <div class="tt-key">Median Throughput</div><div class="tt-val">${{fmtVal(tObj.value, s.unit)}}</div>
          <div class="tt-key">7-Trial Range (CV)</div><div class="tt-val">${{fmtVal(tObj.min, s.unit)}} – ${{fmtVal(tObj.max, s.unit)}} (${{tObj.cv_pct.toFixed(2)}}%)</div>
          <div class="tt-key">Speedup vs First Run</div><div class="tt-val">${{(tObj.speedup_vs_first || 1).toFixed(2)}}×</div>
          ${{deltaHtml}}
          <div class="tt-key">ISA Roofline Util</div><div class="tt-val">${{tObj.isa_util_pct != null ? tObj.isa_util_pct.toFixed(1) + "%" : "—"}}</div>
          <div class="tt-key">Machine Peak Util</div><div class="tt-val">${{tObj.machine_util_pct != null ? tObj.machine_util_pct.toFixed(1) + "%" : "—"}}</div>
          <div class="tt-key">Regime / Latency</div><div class="tt-val">${{tObj.regime || s.memory_regime}} · ${{tObj.latency_ms != null ? tObj.latency_ms.toFixed(4) + " ms" : "—"}}</div>
        </div>
      `;
      tooltip.style.display = "block";
      positionTooltip(evt);
    }}

    function positionTooltip(evt) {{
      const pad = 14;
      const rect = tooltip.getBoundingClientRect();
      let left = evt.clientX + pad;
      let top = evt.clientY + pad;
      if (left + rect.width > window.innerWidth - 12) left = evt.clientX - rect.width - pad;
      if (top + rect.height > window.innerHeight - 12) top = evt.clientY - rect.height - pad;
      tooltip.style.left = `${{Math.max(8, left)}}px`;
      tooltip.style.top = `${{Math.max(8, top)}}px`;
    }}

    function hideTooltip() {{ tooltip.style.display = "none"; }}

    function renderChart(svgEl, spec, seriesList, onHoverChange) {{
      const ns = "http://www.w3.org/2000/svg";
      svgEl.innerHTML = "";
      const W = 1180, H = 430;
      const padL = 78, padR = 265, padT = 24, padB = 56;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;
      svgEl.setAttribute("viewBox", `0 0 ${{W}} ${{H}}`);

      const reports = (suite.reports || []).filter(r => (r.thread_configs || []).includes(Number(st.threads)));
      const activeSeries = seriesList.filter(s => !st.hiddenSeries.has(s.id));

      const xByReport = {{}};
      const parsedTs = reports.map(r => Date.parse(r.date + "T00:00:00Z"));
      const minTs = Math.min(...parsedTs);
      const maxTs = Math.max(...parsedTs);

      if (reports.length === 1) {{
        xByReport[reports[0].report_id] = padL + plotW / 2;
      }} else if (st.xSpacing === "time" && maxTs > minTs) {{
        const innerPad = 36;
        reports.forEach((r, idx) => {{
          const frac = (parsedTs[idx] - minTs) / (maxTs - minTs);
          xByReport[r.report_id] = padL + innerPad + frac * (plotW - 2 * innerPad);
        }});
      }} else {{
        const innerPad = 36;
        reports.forEach((r, idx) => {{
          const frac = idx / (reports.length - 1);
          xByReport[r.report_id] = padL + innerPad + frac * (plotW - 2 * innerPad);
        }});
      }}

      function extractVal(s, pt) {{
        const tObj = pt.threads[st.threads];
        if (!tObj) return null;
        if (spec.mode === "tops") return tObj.value / 1000.0;
        if (spec.mode === "raw") return tObj.value;
        if (spec.mode === "util") return tObj.isa_util_pct;
        if (spec.mode === "speedup") return tObj.speedup_vs_first;
        return tObj.value;
      }}

      let minVal = Infinity, maxVal = -Infinity;
      for (const s of activeSeries) {{
        for (const p of s.points || []) {{
          if (!(p.report_id in xByReport)) continue;
          const v = extractVal(s, p);
          if (v == null || !isFinite(v)) continue;
          if (v < minVal) minVal = v;
          if (v > maxVal) maxVal = v;
        }}
      }}
      if (!isFinite(minVal) || !isFinite(maxVal)) {{ minVal = 0.5; maxVal = 10; }}

      const isLog = st.yScale === "log" && spec.mode !== "util";
      const yMin = isLog ? Math.max(0.05, minVal * 0.85) : 0;
      const yMax = maxVal * 1.14;

      function yPos(val) {{
        const c = Math.max(isLog ? yMin : 0, val);
        if (isLog) {{
          const frac = (Math.log10(c) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin) || 1);
          return padT + plotH * (1 - frac);
        }}
        const frac = (c - yMin) / (yMax - yMin || 1);
        return padT + plotH * (1 - frac);
      }}

      const nTicks = 6;
      const rawStep = (yMax - yMin) / nTicks;
      const mag = Math.pow(10, Math.floor(Math.log10(rawStep || 1)));
      const res = rawStep / mag;
      const niceStep = res > 5 ? 10 * mag : res > 2 ? 5 * mag : res > 1 ? 2 * mag : mag;
      const yTicks = [];
      if (isLog) {{
        for (const cand of [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000]) {{
          if (cand >= yMin * 0.9 && cand <= yMax * 1.05) yTicks.push(cand);
        }}
      }} else {{
        for (let v = 0; v <= yMax + niceStep * 0.1; v += niceStep) yTicks.push(v);
      }}

      for (const tv of yTicks) {{
        const y = yPos(tv);
        if (y < padT - 2 || y > padT + plotH + 2) continue;
        const line = document.createElementNS(ns, "line");
        line.setAttribute("x1", padL);
        line.setAttribute("y1", y);
        line.setAttribute("x2", padL + plotW);
        line.setAttribute("y2", y);
        line.setAttribute("stroke", "#e6ebf1");
        svgEl.appendChild(line);

        const txt = document.createElementNS(ns, "text");
        txt.setAttribute("x", padL - 8);
        txt.setAttribute("y", y + 4);
        txt.setAttribute("text-anchor", "end");
        txt.setAttribute("font-size", "11");
        txt.setAttribute("fill", "#59636e");
        txt.textContent = spec.mode === "speedup" ? `${{tv.toFixed(1)}}×` : (tv >= 10 ? Math.round(tv).toLocaleString() : tv.toFixed(1));
        svgEl.appendChild(txt);
      }}

      // Reference 1.0x line on speedup chart
      if (spec.mode === "speedup" && yPos(1.0) >= padT && yPos(1.0) <= padT + plotH) {{
        const refLine = document.createElementNS(ns, "line");
        refLine.setAttribute("x1", padL);
        refLine.setAttribute("y1", yPos(1.0));
        refLine.setAttribute("x2", padL + plotW);
        refLine.setAttribute("y2", yPos(1.0));
        refLine.setAttribute("stroke", "#8c959f");
        refLine.setAttribute("stroke-dasharray", "4,4");
        svgEl.appendChild(refLine);
      }}

      for (const r of reports) {{
        const x = xByReport[r.report_id];
        const vline = document.createElementNS(ns, "line");
        vline.setAttribute("x1", x);
        vline.setAttribute("y1", padT);
        vline.setAttribute("x2", x);
        vline.setAttribute("y2", padT + plotH);
        vline.setAttribute("stroke", "#eef2f6");
        vline.setAttribute("stroke-dasharray", "3,3");
        svgEl.appendChild(vline);

        const dTxt = document.createElementNS(ns, "text");
        dTxt.setAttribute("x", x);
        dTxt.setAttribute("y", padT + plotH + 20);
        dTxt.setAttribute("text-anchor", "middle");
        dTxt.setAttribute("font-size", "11.5");
        dTxt.setAttribute("font-weight", "600");
        dTxt.setAttribute("fill", "#1f2328");
        dTxt.textContent = `${{r.date}} (${{r.commit_short}})`;
        svgEl.appendChild(dTxt);
      }}

      const xAxis = document.createElementNS(ns, "line");
      xAxis.setAttribute("x1", padL);
      xAxis.setAttribute("y1", padT + plotH);
      xAxis.setAttribute("x2", padL + plotW);
      xAxis.setAttribute("y2", padT + plotH);
      xAxis.setAttribute("stroke", "#8c959f");
      svgEl.appendChild(xAxis);

      const yAxis = document.createElementNS(ns, "line");
      yAxis.setAttribute("x1", padL);
      yAxis.setAttribute("y1", padT);
      yAxis.setAttribute("x2", padL);
      yAxis.setAttribute("y2", padT + plotH);
      yAxis.setAttribute("stroke", "#8c959f");
      svgEl.appendChild(yAxis);

      const yTitle = document.createElementNS(ns, "text");
      yTitle.setAttribute("transform", `translate(18, ${{padT + plotH / 2}}) rotate(-90)`);
      yTitle.setAttribute("text-anchor", "middle");
      yTitle.setAttribute("font-size", "12");
      yTitle.setAttribute("font-weight", "600");
      yTitle.setAttribute("fill", "#1f2328");
      yTitle.textContent = spec.yLabel;
      svgEl.appendChild(yTitle);

      const endLabels = [];
      for (const s of activeSeries) {{
        const isHovered = st.hoveredSeries === s.id;
        const isDimmed = st.hoveredSeries && st.hoveredSeries !== s.id;
        const pts = (s.points || []).filter(p => (p.report_id in xByReport) && extractVal(s, p) != null);
        if (pts.length === 0) continue;

        const g = document.createElementNS(ns, "g");
        g.setAttribute("opacity", isDimmed ? "0.2" : "0.95");

        if (pts.length > 1) {{
          const coords = pts.map(p => `${{xByReport[p.report_id]}},${{yPos(extractVal(s, p))}}`).join(" ");
          const poly = document.createElementNS(ns, "polyline");
          poly.setAttribute("fill", "none");
          poly.setAttribute("stroke", s.color);
          poly.setAttribute("stroke-width", isHovered ? "3.2" : "2.1");
          const dash = dashArrayFor(s.linestyle);
          if (dash) poly.setAttribute("stroke-dasharray", dash);
          poly.setAttribute("points", coords);
          g.appendChild(poly);
        }}

        pts.forEach((pt, idx) => {{
          const val = extractVal(s, pt);
          const cx = xByReport[pt.report_id];
          const cy = yPos(val);
          const markerEl = createMarkerSVG(s.marker, cx, cy, isHovered ? 5.6 : 4.3, s.color, isHovered);
          g.appendChild(markerEl);

          const hit = document.createElementNS(ns, "circle");
          hit.setAttribute("cx", cx);
          hit.setAttribute("cy", cy);
          hit.setAttribute("r", 10);
          hit.setAttribute("fill", "transparent");
          hit.style.cursor = "pointer";
          const prevPt = idx > 0 ? pts[idx - 1] : null;
          hit.addEventListener("mouseenter", (e) => {{
            st.hoveredSeries = s.id;
            onHoverChange();
            showTooltip(e, s, pt, prevPt, spec.mode);
          }});
          hit.addEventListener("mousemove", (e) => positionTooltip(e));
          hit.addEventListener("mouseleave", () => {{
            st.hoveredSeries = null;
            onHoverChange();
            hideTooltip();
          }});
          g.appendChild(hit);
        }});

        const lastPt = pts[pts.length - 1];
        const lastVal = extractVal(s, lastPt);
        endLabels.push({{
          s,
          x: xByReport[lastPt.report_id],
          rawY: yPos(lastVal),
          y: yPos(lastVal),
          val: lastVal,
          isHovered,
          isDimmed,
        }});
        svgEl.appendChild(g);
      }}

      endLabels.sort((a, b) => a.rawY - b.rawY);
      const minGap = 14;
      for (let i = 1; i < endLabels.length; i++) {{
        if (endLabels[i].y - endLabels[i - 1].y < minGap) {{
          endLabels[i].y = endLabels[i - 1].y + minGap;
        }}
      }}
      for (const item of endLabels) {{
        const lg = document.createElementNS(ns, "g");
        lg.setAttribute("opacity", item.isDimmed ? "0.25" : "1.0");
        const labelX = padL + plotW + 10;
        const leader = document.createElementNS(ns, "path");
        const midX = (item.x + labelX) / 2;
        leader.setAttribute("d", `M ${{item.x + 5}} ${{item.rawY}} C ${{midX}} ${{item.rawY}}, ${{midX}} ${{item.y}}, ${{labelX - 3}} ${{item.y}}`);
        leader.setAttribute("fill", "none");
        leader.setAttribute("stroke", item.s.color);
        leader.setAttribute("stroke-width", "0.9");
        leader.setAttribute("opacity", "0.55");
        lg.appendChild(leader);

        const lbl = document.createElementNS(ns, "text");
        lbl.setAttribute("x", labelX);
        lbl.setAttribute("y", item.y + 3.5);
        lbl.setAttribute("font-size", "10.6");
        lbl.setAttribute("font-weight", item.isHovered ? "700" : "500");
        lbl.setAttribute("fill", item.s.color);
        const valSuffix = spec.mode === "speedup"
          ? `${{item.val.toFixed(2)}}×`
          : (spec.mode === "util" ? `${{item.val.toFixed(1)}}%` : `${{item.val.toFixed(2)}}`);
        lbl.textContent = `${{item.s.id}} (${{valSuffix}})`;
        lg.appendChild(lbl);
        svgEl.appendChild(lg);
      }}
    }}

    function renderAll() {{
      const root = document.getElementById("suites-root");
      root.innerHTML = "";
      const card = document.createElement("section");
      card.className = "suite-card";

      const reportLinks = (suite.reports || []).map(r => {{
        return `<a class="report-link-pill" href="${{r.report_url}}">📄 ${{r.date}} (${{r.commit_short}}) ↗</a>`;
      }}).join("");

      card.innerHTML = `
        <div class="suite-header">
          <div class="suite-title-row">
            <h2>
              <span>Gemma 4 31B Kernel Benchmark Suite</span>
              <span class="badge badge-machine">${{suite.microarchitecture}}</span>
            </h2>
            <span class="badge">${{(suite.reports || []).length}} Dated Reports · 7 Load-Bearing Kernel Pairs</span>
          </div>
          <div class="report-links-bar">
            <strong style="color:var(--muted)">Dated Kernel Reports:</strong>
            ${{reportLinks}}
          </div>
        </div>
        <div class="controls-bar" data-controls></div>
        <div class="chart-section">
          <div class="charts-grid" data-charts-grid>
            <div class="chart-panel">
              <div class="chart-panel-header">
                <div class="chart-panel-title">⚡ Compute-Bound GEMM Throughput Over Time (TOPS / TFLOPS ↑)</div>
                <div class="chart-panel-subtitle">Pairs 2 (INT4), 3 (FP32 BMM), 5 (INT8 ViT), and 6 (INT2 Audio)</div>
              </div>
              <svg class="chart-svg" data-chart-compute></svg>
            </div>
            <div class="chart-panel">
              <div class="chart-panel-header">
                <div class="chart-panel-title">🚀 Cumulative Speedup vs. First Report (× ↑)</div>
                <div class="chart-panel-subtitle">Normalized speedup trajectory relative to initial baseline (1.00×)</div>
              </div>
              <svg class="chart-svg" data-chart-speedup></svg>
            </div>
            <div class="chart-panel">
              <div class="chart-panel-header">
                <div class="chart-panel-title">🌊 Memory-Bound &amp; Reduction Throughput Over Time (GOPS / GB/s ↑)</div>
                <div class="chart-panel-subtitle">Pairs 1 (M=1 INT4 GEMV), 4 (FP32 Softmax), and 7 (32-Bit Pack/Transpose)</div>
              </div>
              <svg class="chart-svg" data-chart-memory></svg>
            </div>
            <div class="chart-panel">
              <div class="chart-panel-header">
                <div class="chart-panel-title">🎯 Practical ISA Roofline Utilization Over Time (% ↑)</div>
                <div class="chart-panel-subtitle">% of empirical compute or cache/DRAM ceiling achieved</div>
              </div>
              <svg class="chart-svg" data-chart-util></svg>
            </div>
          </div>
          <div class="legend-grid" data-legend></div>
        </div>
        <div class="table-section">
          <h3>
            <span>Per-Kernel Performance Progression &amp; Speedup Breakdown (${{st.threads}}C)</span>
            <span style="font-size:0.8rem;font-weight:400;color:var(--muted)">Click any report date header to open the full Markdown report</span>
          </h3>
          <div data-table></div>
        </div>
      `;

      const controlsEl = card.querySelector("[data-controls]");
      const chartsGridEl = card.querySelector("[data-charts-grid]");
      const svgCompute = card.querySelector("[data-chart-compute]");
      const svgSpeedup = card.querySelector("[data-chart-speedup]");
      const svgMemory = card.querySelector("[data-chart-memory]");
      const svgUtil = card.querySelector("[data-chart-util]");
      const legendEl = card.querySelector("[data-legend]");
      const tableEl = card.querySelector("[data-table]");

      function addCtrlGroup(label, options, curVal, onSelect) {{
        const grp = document.createElement("div");
        grp.className = "control-group";
        grp.innerHTML = `<span class="control-label">${{label}}:</span>`;
        for (const [val, text] of options) {{
          const b = document.createElement("button");
          b.className = "ctrl-btn" + (curVal === val ? " active" : "");
          b.textContent = text;
          b.onclick = () => {{ onSelect(val); refreshView(); }};
          grp.appendChild(b);
        }}
        controlsEl.appendChild(grp);
      }}

      function renderControls() {{
        controlsEl.innerHTML = "";
        addCtrlGroup("Threads", [
          ["28", "28C (All 3 Dates: Nov 10–17)"],
          ["1", "1C (All 3 Dates: Nov 10–17)"],
          ["56", "56C (Full Socket 0: Nov 13–17)"]
        ], st.threads, v => {{ st.threads = v; }});

        addCtrlGroup("Kernel Pair", [
          ["all", "All Pairs"],
          ["pair1_int4_gemv", "P1: INT4 GEMV"],
          ["pair2_int4_gemm", "P2: INT4 GEMM"],
          ["pair3_fp32_attn_bmm", "P3: FP32 BMM"],
          ["pair4_fp32_softmax", "P4: Softmax"],
          ["pair5_int8_gemm", "P5: INT8 GEMM"],
          ["pair6_int2_gemm", "P6: INT2 GEMM"],
          ["pair7_pack_transpose", "P7: Pack/Transpose"]
        ], st.pairFilter, v => {{ st.pairFilter = v; }});

        addCtrlGroup("Backend", [
          ["all", "All"],
          ["YNNPACK", "YNNPACK"],
          ["XNNPACK", "XNNPACK"],
          ["YNNPACK_VNNI_BASELINE", "YNN VNNI Baseline"]
        ], st.backendFilter, v => {{ st.backendFilter = v; }});

        addCtrlGroup("Chart Shapes", [
          ["headline", "Headline Shapes"],
          ["all", "All 26 Shapes"]
        ], st.scopeFilter, v => {{ st.scopeFilter = v; }});

        addCtrlGroup("Y-Scale", [
          ["linear", "Linear"],
          ["log", "Log"]
        ], st.yScale, v => {{ st.yScale = v; }});

        addCtrlGroup("Layout", [
          ["side-by-side", "2-Column Grid"],
          ["stacked", "Stacked Full-Width"]
        ], st.layout, v => {{ st.layout = v; }});
      }}

      function redrawCharts() {{
        chartsGridEl.className = "charts-grid" + (st.layout === "stacked" ? " stacked" : "");
        const compSeries = visibleSeriesForChart(s => s.chart_group === "compute");
        const memSeries = visibleSeriesForChart(s => s.chart_group === "memory");
        const allVis = visibleSeriesForChart(null);

        renderChart(svgCompute, {{ mode: "tops", yLabel: `Throughput (TOPS / TFLOPS, ${{st.threads}}C)` }}, compSeries, redrawCharts);
        renderChart(svgSpeedup, {{ mode: "speedup", yLabel: `Speedup Ratio vs. First Report (×, ${{st.threads}}C)` }}, allVis, redrawCharts);
        renderChart(svgMemory, {{ mode: "raw", yLabel: `Throughput / Bandwidth (GOPS / GB/s, ${{st.threads}}C)` }}, memSeries, redrawCharts);
        renderChart(svgUtil, {{ mode: "util", yLabel: `Practical ISA Roofline Utilization (%, ${{st.threads}}C)` }}, allVis, redrawCharts);
      }}

      function renderLegend() {{
        legendEl.innerHTML = "";
        const allVis = visibleSeriesForChart(null);
        for (const s of allVis) {{
          const item = document.createElement("div");
          const isHidden = st.hiddenSeries.has(s.id);
          item.className = "legend-item" + (isHidden ? " muted" : "");
          const dash = dashArrayFor(s.linestyle);
          const dashAttr = dash ? `stroke-dasharray="${{dash}}"` : "";
          item.innerHTML = `
            <svg class="legend-swatch" viewBox="0 0 24 12">
              <line x1="1" y1="6" x2="23" y2="6" stroke="${{s.color}}" stroke-width="2.2" ${{dashAttr}} />
              <circle cx="12" cy="6" r="3.2" fill="${{s.color}}" stroke="#fff" stroke-width="1" />
            </svg>
            <span><code>${{s.id}}</code></span>
          `;
          item.onclick = () => {{
            if (st.hiddenSeries.has(s.id)) st.hiddenSeries.delete(s.id);
            else st.hiddenSeries.add(s.id);
            renderLegend();
            redrawCharts();
          }};
          legendEl.appendChild(item);
        }}
      }}

      function renderTable() {{
        const reports = (suite.reports || []).filter(r => (r.thread_configs || []).includes(Number(st.threads)));
        const tableSeries = (suite.series || []).filter(s => {{
          if (st.pairFilter !== "all" && s.pair_id !== st.pairFilter) return false;
          if (st.backendFilter !== "all" && s.backend !== st.backendFilter) return false;
          return s.points.some(p => p.threads && p.threads[st.threads]);
        }});

        const byPair = {{}};
        for (const s of tableSeries) {{
          (byPair[s.pair_id] = byPair[s.pair_id] || []).push(s);
        }}

        const dateHeaders = reports.map(r => {{
          return `<th class="num"><a href="${{r.report_url}}">${{r.date}} (${{r.commit_short}}) ↗</a></th>`;
        }}).join("");

        let rowsHtml = "";
        const totalCols = 5 + reports.length + 3;
        for (const [pairId, group] of Object.entries(byPair)) {{
          rowsHtml += `<tr class="group-row"><td colspan="${{totalCols}}">${{group[0].pair_title}}</td></tr>`;
          for (const s of group) {{
            const ptByRep = {{}};
            for (const p of s.points || []) ptByRep[p.report_id] = p;
            const tSum = (s.summary_by_thread || {{}})[st.threads] || {{}};

            const dateCells = reports.map(r => {{
              const p = ptByRep[r.report_id];
              const tObj = p && p.threads ? p.threads[st.threads] : null;
              if (!tObj) return `<td class="num" style="color:var(--muted)">—</td>`;
              const valStr = fmtVal(tObj.value, s.unit);
              const utilStr = tObj.isa_util_pct != null ? `${{tObj.isa_util_pct.toFixed(1)}}% ISA` : "";
              return `<td class="num"><strong>${{valStr}}</strong><span class="sub-range">${{utilStr}} · CV ${{tObj.cv_pct.toFixed(1)}}%</span></td>`;
            }}).join("");

            function renderDeltaPill(pct, ratio) {{
              if (pct == null) return `<td class="num"><span class="delta-pill delta-neutral">n/a</span></td>`;
              const sign = pct > 0 ? "+" : "";
              const cls = pct > 2.0 ? "delta-good" : (pct < -2.0 ? "delta-bad" : "delta-neutral");
              const ratioPrefix = ratio != null ? `${{ratio.toFixed(2)}}× (` : "";
              const ratioSuffix = ratio != null ? `)` : "";
              return `<td class="num"><span class="delta-pill ${{cls}}">${{ratioPrefix}}${{sign}}${{pct.toFixed(1)}}%${{ratioSuffix}}</span></td>`;
            }}

            const isYnn = s.backend === "YNNPACK";
            rowsHtml += `
              <tr class="${{isYnn ? "best-row" : ""}}">
                <td>
                  <span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${{s.color}};margin-right:5px;"></span>
                  <strong>${{s.backend}}</strong>
                </td>
                <td><code>${{s.shape_str}}</code></td>
                <td><code>${{s.memory_regime}}</code></td>
                <td><code>${{tSum.latest_kernel || "—"}}</code></td>
                <td><code>${{tSum.latest_isa || "—"}}</code></td>
                ${{dateCells}}
                ${{renderDeltaPill(tSum.delta_vs_first_pct, tSum.speedup_vs_first_x)}}
                ${{renderDeltaPill(tSum.delta_vs_prev_pct, null)}}
                <td class="num">${{tSum.latest_isa_util_pct != null ? tSum.latest_isa_util_pct.toFixed(1) + "%" : "—"}}</td>
              </tr>
            `;
          }}
        }}

        tableEl.innerHTML = `
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Backend</th>
                  <th>Shape (M×N×K)</th>
                  <th>Memory Regime</th>
                  <th>Latest Microkernel Symbol</th>
                  <th>ISA</th>
                  ${{dateHeaders}}
                  <th class="num">Speedup vs First (${{reports[0]?.date || ""}})</th>
                  <th class="num">Δ vs Previous</th>
                  <th class="num">Latest ISA Roofline Util</th>
                </tr>
              </thead>
              <tbody>
                ${{rowsHtml}}
              </tbody>
            </table>
          </div>
        `;
      }}

      function refreshView() {{
        renderControls();
        redrawCharts();
        renderLegend();
        renderTable();
      }}

      refreshView();
      root.appendChild(card);
    }}

    renderAll();
  }})();
  </script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
      "--reports",
      type=Path,
      default=KERNEL_ROOT / "reports",
      help="Directory containing dated kernel report directories",
  )
  parser.add_argument(
      "-o",
      "--out",
      type=Path,
      default=KERNEL_ROOT / "dashboards" / "index.html",
      help="Output HTML dashboard path",
  )
  args = parser.parse_args(argv)

  out_html = args.out.resolve()
  out_html.parent.mkdir(parents=True, exist_ok=True)
  out_json = out_html.parent / "data.json"
  out_md = out_html.parent / "README.md"

  dashboard_data = collect_kernel_dashboard_data(args.reports.resolve())
  out_json.write_text(json.dumps(dashboard_data, indent=2) + "\n", encoding="utf-8")
  out_html.write_text(render_html_dashboard(dashboard_data), encoding="utf-8")
  out_md.write_text(generate_markdown_dashboard(dashboard_data), encoding="utf-8")

  suite = dashboard_data["suites"][0]
  print(
      f"Updated kernel dashboard: {out_html} "
      f"({len(suite['reports'])} report dates, {len(suite['series'])} tracked series)"
  )
  print(f"Updated kernel dashboard data: {out_json}")
  print(f"Updated kernel markdown dashboard: {out_md}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
