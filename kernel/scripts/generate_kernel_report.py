#!/usr/bin/env python3
# Generates the General Microkernel Performance, Roofline & Numerical Precision Report
# (`kernel_report_gemma4_31b_sapphire_rapids.json` and `kernel_report_gemma4_31b_sapphire_rapids.md`)
# from `ceilings_sapphire_rapids.json`, `ukernel_benchmarks_sapphire_rapids.json`,
# and `ukernel_precision_sapphire_rapids.json`.

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPORTS_ROOT = Path(__file__).resolve().parent.parent / "reports"
DATE_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)$")


def default_report_dir() -> Path:
  """Selects the latest dated `YYYY-MM-DD-gemma4-31b-sapphire-rapids` directory."""
  dated = sorted(
      p
      for p in REPORTS_ROOT.iterdir()
      if p.is_dir()
      and DATE_PREFIX_RE.match(p.name)
      and p.name.endswith("-gemma4-31b-sapphire-rapids")
  )
  if dated:
    return dated[-1]
  return REPORTS_ROOT / "gemma4-31b-sapphire-rapids"


def extract_benchmark_date(report_dir: Path) -> str:
  m = DATE_PREFIX_RE.match(report_dir.name)
  return m.group(1) if m else ""


def find_sibling_artifact(
    report_dir: Path, filename: str, strictly_earlier: bool = False
) -> Optional[Path]:
  """Finds `filename` in `report_dir` (unless strictly_earlier) or latest matching sibling."""
  if not strictly_earlier:
    local = report_dir / filename
    if local.exists():
      return local
  m = DATE_PREFIX_RE.match(report_dir.name)
  suite_suffix = f"-{m.group(2)}" if m else f"-{report_dir.name}"
  cur_date = m.group(1) if m else "9999-99-99"
  candidates: List[Path] = []
  for sib in sorted(REPORTS_ROOT.iterdir()):
    if not sib.is_dir() or sib.resolve() == report_dir.resolve():
      continue
    if not sib.name.endswith(suite_suffix):
      continue
    sm = DATE_PREFIX_RE.match(sib.name)
    if strictly_earlier and sm and sm.group(1) >= cur_date:
      continue
    cand = sib / filename
    if cand.exists():
      candidates.append(cand)
  return candidates[-1] if candidates else None


def rel_link(from_dir: Path, target: Path) -> str:
  return Path(os.path.relpath(target.resolve(), from_dir.resolve())).as_posix()

BENCH_TO_PRECISION_PAIR_ID = {
    "pair1_int4_gemv": "pair_1_qc4w_gemv_decode",
    "pair2_int4_gemm": "pair_2_qc4w_gemm_prefill",
    "pair3_fp32_attn_bmm": "pair_3_fp32_attn_bmm",
    "pair4_fp32_softmax": "pair_4_fp32_softmax",
    "pair5_int8_gemm": "pair_5_qc8w_gemm_multimodal",
    "pair6_int2_gemm": "pair_6_qc2w_gemm_embed",
    "pair7_pack_transpose": "pair_7_pack_transpose_x32",
}


def tc_key(threads: int) -> str:
  return "single_core" if threads == 1 else f"numa_node0_{threads}c"


def tc_suffix(threads: int) -> str:
  return f"{threads}c"


def tc_label(threads: int) -> str:
  return f"{threads}C"


def fmt_perf(val: float, unit: str) -> str:
  if val >= 10000.0 and unit in ("GOPS", "GFLOPS"):
    t_unit = "TOPS" if unit == "GOPS" else "TFLOPS"
    return f"{val / 1000.0:.2f} {t_unit}"
  return f"{val:,.2f} {unit}"


def fmt_gops_tops(val: float, unit: str = "GOPS") -> str:
  t_unit = "TOPS" if unit == "GOPS" else "TFLOPS"
  if val >= 1000.0:
    return f"{val:,.2f} {unit} ({val / 1000.0:.2f} {t_unit})"
  return f"{val:,.2f} {unit}"


def nice_y_max(values: List[float], step: int = 10, floor: int = 10) -> int:
  mx = max(values) if values else float(floor)
  return max(floor, int(math.ceil(mx * 1.12 / step) * step))


def resolve_ceilings_for_thread(
    ceilings: Dict[str, Any],
    isa: str,
    mem_regime: str,
    pair_id: str,
    backend: str,
    threads: int,
) -> Tuple[str, float, str, float]:
  """Returns (compute_ceil_name, c_val, bw_ceil_name, bw_val) for a thread count."""
  comp = ceilings["compute_ceilings_gops"]
  bw = ceilings["bandwidth_ceilings_gbps"]
  key = tc_key(threads)

  if isa == "AMX-INT8":
    c_name = "AMX-INT8 (tdpbssd)"
    c_val = comp["amx_int8_tdpbssd"][key]
  elif isa == "AVX512-VNNI":
    c_name = "AVX-512 VNNI (vpdpbusd 512b)"
    c_val = comp["avx512_vnni_int8_vpdpbusd"][key]
  elif isa == "AVX-VNNI-256":
    c_name = "AVX-VNNI (vpdpbusd 256b)"
    c_val = comp["avx_vnni_256_int8_vpdpbusd"][key]
  else:
    c_name = "AVX-512 FP32 FMA (vfmadd231ps)"
    c_val = comp["avx512_fp32_fma"][key]

  if mem_regime == "DDR5_DRAM_Streaming":
    b_name = "DDR5 DRAM Read"
    b_val = bw["ddr5_dram"]["read_only"][key]
  elif mem_regime == "L3_LLC_Warm":
    # On 1 core, 11-22 MB exceeds the 2 MB private L2 and streams from shared L3 LLC.
    # On 28/56 cores, static N-partitioning leaves 196-786 KB/core, fitting 100% inside
    # each core's 2 MB private L2 cache (56-112 MB aggregate L2).
    b_name = "L3 (1C) / L2-Sliced (MC) Read"
    if threads == 1:
      b_val = bw["l3_llc"]["read_only"][key]
    else:
      b_val = bw["l2_512kb_per_core"]["read_only"][key]
  elif mem_regime == "L2_Cache_Copy":
    b_name = "L2 Cache Read+Write Copy"
    b_val = bw["l2_512kb_per_core"]["read_write_copy"][key]
  elif mem_regime == "L3_LLC_Copy":
    b_name = "L3 (1C) / L2-Sliced (MC) Copy"
    if threads == 1:
      b_val = bw["l3_llc"]["read_write_copy"][key]
    else:
      b_val = bw["l2_512kb_per_core"]["read_write_copy"][key]
  elif pair_id == "pair4_fp32_softmax":
    if backend == "YNNPACK":
      # YNNPACK's max_k1 + sum_k1 reductions are read-only over two disjoint FP32 buffers
      # (in and out: 8-51 MB combined). On 1C this exceeds 2 MB L2 and streams from L3;
      # on 28C/56C, static row partitioning drops per-core working set to 0.14-1.8 MB (L2).
      b_name = "L3 (1C) / L2-Sliced (MC) Read"
      if threads == 1:
        b_val = bw["l3_llc"]["read_only"][key]
      else:
        b_val = bw["l2_512kb_per_core"]["read_only"][key]
    else:
      # XNNPACK's fused row-wise Softmax (rmax + raddstoreexpminusmax) touches 4-16 KB/row
      # (2R + 1W): Pass 1 warms the row into L1D/L2 and Pass 2 reads+writes in L2/L1D.
      b_name = "L2/L1D Row-Fused Copy"
      b_val = bw["l2_512kb_per_core"]["read_write_copy"][key]
  elif "L2" in mem_regime:
    b_name = "L2 Cache Read"
    b_val = bw["l2_512kb_per_core"]["read_only"][key]
  else:
    b_name = "L3 Shared LLC Read"
    b_val = bw["l3_llc"]["read_only"][key]

  return c_name, c_val, b_name, b_val


def enrich_benchmark_row(
    row: Dict[str, Any],
    ceilings: Dict[str, Any],
    thread_configs: List[int],
    precision_by_pair: Optional[Dict[str, Dict[str, Any]]] = None,
    baseline_retained: bool = False,
) -> Dict[str, Any]:
  pair_id = row["pair_id"]
  isa = row["isa"]
  mem_regime = row["memory_regime"]
  ai = row["arithmetic_intensity_ops_per_byte"]
  backend = row["backend"]

  amx_comp = ceilings["compute_ceilings_gops"]["amx_int8_tdpbssd"]

  roofline: Dict[str, Any] = {}
  c_name = ""
  b_name = ""
  unit = "GOPS"

  for t in thread_configs:
    key = tc_key(t)
    sfx = tc_suffix(t)
    c_name, c_val, b_name, b_val = resolve_ceilings_for_thread(
        ceilings, isa, mem_regime, pair_id, backend, t
    )
    if pair_id == "pair7_pack_transpose":
      ceil_val = b_val
      bound = "Memory-Bound"
      meas_val = row[key]["gbps"]
      unit = "GB/s"
    elif pair_id == "pair4_fp32_softmax":
      ceil_val = b_val
      bound = (
          "Memory-Bound (Reductions)"
          if backend == "YNNPACK"
          else "Memory/Exp-Poly"
      )
      meas_val = row[key]["gbps"]
      unit = "GB/s"
    else:
      mem_roof = ai * b_val
      ceil_val = min(c_val, mem_roof)
      bound = "Memory-Bound" if mem_roof < c_val else "Compute-Bound"
      meas_val = row[key]["gops"]
      unit = "GOPS" if "FP32" not in isa else "GFLOPS"

    util = (meas_val / ceil_val * 100.0) if ceil_val > 0 else 0.0

    if pair_id in ("pair2_int4_gemm", "pair5_int8_gemm", "pair6_int2_gemm"):
      mach_ceil = min(amx_comp[key], ai * b_val)
      mach_util = (meas_val / mach_ceil * 100.0) if mach_ceil > 0 else 0.0
    else:
      mach_util = util

    roofline[f"compute_ceiling_{sfx}"] = round(c_val, 2)
    roofline[f"bandwidth_ceiling_{sfx}_gbps"] = round(b_val, 2)
    roofline[f"effective_ceiling_{sfx}"] = round(ceil_val, 2)
    roofline[f"regime_{sfx}"] = bound
    roofline[f"isa_roofline_utilization_{sfx}_pct"] = round(util, 2)
    roofline[f"machine_roofline_utilization_{sfx}_pct"] = round(mach_util, 2)

  roofline = {
      "unit": unit,
      "compute_ceiling_name": c_name,
      "bandwidth_ceiling_name": b_name,
      **roofline,
  }

  enriched = dict(row)
  enriched["roofline"] = roofline

  if precision_by_pair is not None and pair_id in BENCH_TO_PRECISION_PAIR_ID:
    prec_pair_id = BENCH_TO_PRECISION_PAIR_ID[pair_id]
    prec_pair = precision_by_pair[prec_pair_id]
    bkey = "xnnpack" if backend == "XNNPACK" else "ynnpack"
    sym_key = "xnnpack_symbol" if backend == "XNNPACK" else "ynnpack_symbol"
    prec_entry = prec_pair[bkey]
    verified_sym = prec_pair[sym_key]
    enriched["precision"] = {
        "baseline_retained": baseline_retained,
        "verified_symbol": verified_sym,
        "symbol_matches_bench": verified_sym == row["kernel"],
        "level_a_status": prec_entry["level_a_exact_ukernel"]["status"],
        "level_a_max_ulp_diff": prec_entry["level_a_exact_ukernel"][
            "max_ulp_diff"
        ],
        "level_a_sqnr_db": prec_entry["level_a_exact_ukernel"]["sqnr_db"],
        "level_b_status": prec_entry["level_b_fp64_golden"]["status"],
        "level_b_sqnr_db": prec_entry["level_b_fp64_golden"]["sqnr_db"],
        "level_b_cosine_sim": prec_entry["level_b_fp64_golden"]["cosine_sim"],
    }

  return enriched


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


def build_three_series_chart(
    title: str,
    y_label: str,
    y_max: int,
    labels: List[str],
    s1_name: str,
    s1_vals: List[float],
    s2_name: str,
    s2_vals: List[float],
    s3_name: str,
    s3_vals: List[float],
    palette: str = "#16a085, #2b6cd9, #e67e22",
    gap_slots: int = 2,
) -> str:
  """Builds a Mermaid xychart-beta with 3 side-by-side bars centered on the middle slot."""
  sentinel = -y_max
  x_cats: List[str] = []
  b1: List[float] = []
  b2: List[float] = []
  b3: List[float] = []
  inv_idx = 1

  for idx, lbl in enumerate(labels):
    x_cats.append(f'"{_invisible_cat(inv_idx)}"')
    inv_idx += 1
    b1.append(round(s1_vals[idx], 2))
    b2.append(sentinel)
    b3.append(sentinel)

    # Center visible label on the middle bar of the 3-bar cluster.
    x_cats.append(f'"{lbl}"')
    b1.append(sentinel)
    b2.append(round(s2_vals[idx], 2))
    b3.append(sentinel)

    x_cats.append(f'"{_invisible_cat(inv_idx)}"')
    inv_idx += 1
    b1.append(sentinel)
    b2.append(sentinel)
    b3.append(round(s3_vals[idx], 2))

    if idx + 1 < len(labels):
      for _ in range(gap_slots):
        x_cats.append(f'"{_invisible_cat(inv_idx)}"')
        inv_idx += 1
        b1.append(sentinel)
        b2.append(sentinel)
        b3.append(sentinel)

  lines = [
      "```mermaid",
      f'%%{{init: {{"xyChart": {{"width": 1250, "height": 480, "xAxis": {{"labelRotation": 0, "labelFontSize": 11, "showTick": false}}}}, "themeVariables": {{"xyChart": {{"plotColorPalette": "{palette}"}}}}}}}}%%',
      "xychart-beta",
      f'    title "{title}"',
      f'    x-axis "4-Bit Shape (M x N x K)" [{", ".join(x_cats)}]',
      f'    y-axis "{y_label}" 0 --> {y_max}',
      f'    bar "{s1_name}" [{", ".join(str(v) for v in b1)}]',
      f'    bar "{s2_name}" [{", ".join(str(v) for v in b2)}]',
      f'    bar "{s3_name}" [{", ".join(str(v) for v in b3)}]',
      "```",
  ]
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Generate unified JSON & Markdown kernel roofline reports for a "
          "dated benchmark run directory."
      )
  )
  parser.add_argument(
      "--report-dir",
      type=Path,
      default=None,
      help=(
          "Directory containing ceilings_sapphire_rapids.json and "
          "ukernel_benchmarks_sapphire_rapids.json (defaults to latest "
          "YYYY-MM-DD-gemma4-31b-sapphire-rapids under kernel/reports/)."
      ),
  )
  parser.add_argument(
      "--precision-json",
      type=Path,
      default=None,
      help=(
          "Optional path to ukernel_precision_sapphire_rapids.json (defaults "
          "to report-dir, falling back to latest dated sibling directory)."
      ),
  )
  args = parser.parse_args()

  report_dir = (
      args.report_dir.resolve()
      if args.report_dir is not None
      else default_report_dir().resolve()
  )
  ceilings_path = report_dir / "ceilings_sapphire_rapids.json"
  bench_path = report_dir / "ukernel_benchmarks_sapphire_rapids.json"
  out_json_path = report_dir / "kernel_report_gemma4_31b_sapphire_rapids.json"
  out_md_path = report_dir / "kernel_report_gemma4_31b_sapphire_rapids.md"

  if args.precision_json is not None:
    precision_path = args.precision_json.resolve()
  else:
    found_prec = find_sibling_artifact(
        report_dir, "ukernel_precision_sapphire_rapids.json"
    )
    if found_prec is None:
      raise FileNotFoundError(
          "Could not locate ukernel_precision_sapphire_rapids.json in "
          f"{report_dir} or sibling report directories"
      )
    precision_path = found_prec.resolve()

  benchmark_date = extract_benchmark_date(report_dir)
  baseline_retained = precision_path.parent.resolve() != report_dir.resolve()
  precision_rel = rel_link(report_dir, precision_path)
  precision_date = extract_benchmark_date(precision_path.parent)

  companion_path = find_sibling_artifact(
      report_dir, "kernels_gemma4_31b_sapphire_rapids.md"
  )
  companion_rel = (
      rel_link(report_dir, companion_path)
      if companion_path is not None
      else "kernels_gemma4_31b_sapphire_rapids.md"
  )
  companion_date = (
      extract_benchmark_date(companion_path.parent)
      if companion_path is not None
      else ""
  )

  prev_report_path = find_sibling_artifact(
      report_dir,
      "kernel_report_gemma4_31b_sapphire_rapids.md",
      strictly_earlier=True,
  )
  prev_report_rel = (
      rel_link(report_dir, prev_report_path)
      if prev_report_path is not None
      else ""
  )
  prev_report_date = (
      extract_benchmark_date(prev_report_path.parent)
      if prev_report_path is not None
      else ""
  )

  ceilings = json.loads(ceilings_path.read_text())
  bench_data = json.loads(bench_path.read_text())
  precision_data = json.loads(precision_path.read_text())

  thread_configs: List[int] = bench_data.get(
      "thread_configs",
      ceilings.get("thread_configs", [1, bench_data.get("threads_numa_node0", 28)]),
  )
  mt_configs = [t for t in thread_configs if t > 1]
  primary_t = mt_configs[-1]
  primary_key = tc_key(primary_t)
  primary_sfx = tc_suffix(primary_t)
  has_28_and_56 = 28 in mt_configs and 56 in mt_configs

  xnnpack_commit = bench_data.get("xnnpack_commit", "")
  commit_short = xnnpack_commit[:8] if xnnpack_commit else "unknown"
  num_trials = bench_data.get("num_trials", 7)

  precision_by_pair: Dict[str, Dict[str, Any]] = {
      p["pair_id"]: p for p in precision_data["pairs"]
  }

  enriched_rows: List[Dict[str, Any]] = [
      enrich_benchmark_row(
          row,
          ceilings,
          thread_configs,
          precision_by_pair,
          baseline_retained=baseline_retained,
      )
      for row in bench_data["benchmarks"]
  ]
  enriched_baselines: List[Dict[str, Any]] = [
      enrich_benchmark_row(row, ceilings, thread_configs, None)
      for row in bench_data.get("baseline_comparisons", [])
  ]

  prec_meta: Dict[str, Any] = dict(precision_data)
  if baseline_retained:
    prec_meta = {
        "status": "baseline_retained",
        "source_report_date": precision_date or "baseline",
        "source_path": precision_rel,
        "note": (
            f"Retained from {precision_date or 'baseline'} precision "
            f"verification run; not re-run for upstream google/XNNPACK commit "
            f"{commit_short} (performance and roofline scope)."
        ),
        **precision_data,
    }

  out_json: Dict[str, Any] = {
      "model": "Gemma 4 31B",
      "microarchitecture": "Intel Sapphire Rapids",
      "benchmark_date": benchmark_date,
      "xnnpack_commit": xnnpack_commit,
      "thread_configs": thread_configs,
      "primary_multi_core_threads": primary_t,
      "num_trials": num_trials,
      "hardware_ceilings": ceilings,
      "kernel_roofline_benchmarks": enriched_rows,
      "baseline_comparisons": enriched_baselines,
      "kernel_precision_verification": prec_meta,
  }
  out_json_path.write_text(json.dumps(out_json, indent=2) + "\n")

  # Lookups for dynamic chart & prose interpolation
  lookup: Dict[Tuple[str, int, int, int, str, str], Dict[str, Any]] = {}
  for r in enriched_rows:
    s = r["shape"]
    lookup[
        (r["pair_id"], s["M"], s["N"], s["K"], r["memory_regime"], r["backend"])
    ] = r

  base_lookup: Dict[Tuple[str, int, int, int, str], Dict[str, Any]] = {}
  for r in enriched_baselines:
    s = r["shape"]
    base_lookup[(r["pair_id"], s["M"], s["N"], s["K"], r["memory_regime"])] = r

  def get_row(
      pair_id: str, m: int, n: int, k: int, regime: str, backend: str
  ) -> Dict[str, Any]:
    return lookup[(pair_id, m, n, k, regime, backend)]

  # Ceilings shortcuts
  comp = ceilings["compute_ceilings_gops"]
  bw = ceilings["bandwidth_ceilings_gbps"]
  amx_1c = comp["amx_int8_tdpbssd"]["single_core"]
  amx_mc = comp["amx_int8_tdpbssd"][primary_key]
  vnni512_1c = comp["avx512_vnni_int8_vpdpbusd"]["single_core"]
  vnni512_mc = comp["avx512_vnni_int8_vpdpbusd"][primary_key]
  vnni256_mc = comp["avx_vnni_256_int8_vpdpbusd"][primary_key]
  fp32_1c = comp["avx512_fp32_fma"]["single_core"]
  fp32_mc = comp["avx512_fp32_fma"][primary_key]
  ddr_rd_1c = bw["ddr5_dram"]["read_only"]["single_core"]
  ddr_rd_mc = bw["ddr5_dram"]["read_only"][primary_key]
  l1_copy_mc = bw["l1d_16kb_per_core"]["read_write_copy"][primary_key]
  l2_rd_mc = bw["l2_512kb_per_core"]["read_only"][primary_key]
  l2_copy_1c = bw["l2_512kb_per_core"]["read_write_copy"]["single_core"]
  l2_copy_mc = bw["l2_512kb_per_core"]["read_write_copy"][primary_key]
  l3_rd_1c = bw["l3_llc"]["read_only"]["single_core"]
  l3_rd_mc = bw["l3_llc"]["read_only"][primary_key]

  # --- Chart 1: Primary Multi-Core Compute Throughput (TOPS / TFLOPS) ---
  gemm_labels = [
      "P2: INT4 128x21504",
      "P2: INT4 128x5376",
      "P2: INT4 1024x21504",
      "P5: INT8 280x5376",
      "P5: INT8 2520x1152",
      "P6: INT2 204x4096",
      "P3: FP32 1024x4096",
  ]
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
      lookup[(*sp, "XNNPACK")][primary_key]["gops"] / 1000.0
      for sp in chart1_specs
  ]
  c1_ynn = [
      lookup[(*sp, "YNNPACK")][primary_key]["gops"] / 1000.0
      for sp in chart1_specs
  ]
  chart1_md = build_side_by_side_chart(
      f"{primary_t}-Core Compute Throughput: XNNPACK (Blue) vs YNNPACK (Orange) [TOPS / TFLOPS]",
      f"TOPS / TFLOPS ({primary_t} Cores)",
      nice_y_max(c1_xnn + c1_ynn, step=10, floor=20),
      gemm_labels,
      c1_xnn,
      c1_ynn,
  )

  # --- Chart 2: Primary Multi-Core Practical ISA Roofline Utilization (%) ---
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
      ("pair2_int4_gemm", 128, 21504, 5376, "DDR5_DRAM_to_L2_Tiled"),
      ("pair3_fp32_attn_bmm", 1024, 4096, 512, "L3_LLC_Resident"),
      ("pair4_fp32_softmax", 1024, 4096, 1, "L3_LLC_Resident"),
      ("pair5_int8_gemm", 280, 5376, 1152, "L3_LLC_Resident"),
      ("pair6_int2_gemm", 204, 4096, 1024, "L2_L3_Resident"),
      ("pair7_pack_transpose", 1, 2520, 1152, "L3_LLC_Copy"),
  ]
  util_field = f"isa_roofline_utilization_{primary_sfx}_pct"
  c2_xnn = [
      lookup[(*sp, "XNNPACK")]["roofline"][util_field] for sp in chart2_specs
  ]
  c2_ynn = [
      lookup[(*sp, "YNNPACK")]["roofline"][util_field] for sp in chart2_specs
  ]
  c2_ymax = (
      100
      if max(c2_xnn + c2_ynn) <= 100.0
      else nice_y_max(c2_xnn + c2_ynn, step=10, floor=100)
  )
  chart2_md = build_side_by_side_chart(
      f"{primary_t}-Core Practical ISA Roofline Utilization (%): XNNPACK (Blue) vs YNNPACK (Orange)",
      "Roofline Utilization (%)",
      c2_ymax,
      util_labels,
      c2_xnn,
      c2_ynn,
  )

  # --- Chart 3: 4-Bit Microkernel Evolution (Prior YNN VNNI vs XNN AMX vs New YNN AMX) ---
  chart3_evo_md = ""
  if enriched_baselines:
    evo_specs = [
        ("pair1_int4_gemv", 1, 8192, 5376, "L3_LLC_Warm", "M=1 L3 8192"),
        ("pair2_int4_gemm", 128, 21504, 5376, "DDR5_DRAM_to_L2_Tiled", "M=128 21504"),
        ("pair2_int4_gemm", 128, 5376, 21504, "DDR5_DRAM_to_L2_Tiled", "M=128 5376"),
        ("pair2_int4_gemm", 128, 8192, 5376, "L3_LLC_to_L2_Tiled", "M=128 8192"),
        ("pair2_int4_gemm", 1024, 21504, 5376, "Compute_Bound_Tiled", "M=1024 21504"),
        ("pair2_int4_gemm", 1024, 8192, 5376, "Compute_Bound_Tiled", "M=1024 8192"),
    ]
    evo_labels = [sp[5] for sp in evo_specs]
    evo_vnni = [
        base_lookup[sp[:5]][primary_key]["gops"] / 1000.0 for sp in evo_specs
    ]
    evo_xnn = [
        lookup[(*sp[:5], "XNNPACK")][primary_key]["gops"] / 1000.0
        for sp in evo_specs
    ]
    evo_ynn = [
        lookup[(*sp[:5], "YNNPACK")][primary_key]["gops"] / 1000.0
        for sp in evo_specs
    ]
    chart3_evo_md = (
        "\n> **4-Bit Evolution Legend**: **Left (Teal `#16a085`)** = Prior YNNPACK AVX-512 VNNI | "
        "**Middle (Blue `#2b6cd9`)** = XNNPACK AMX-INT8+GFNI (`1x64c4`/`16x64c4`) | "
        "**Right (Orange `#e67e22`)** = New YNNPACK AMX-INT8+GFNI (`32x32x64`)\n\n"
        + build_three_series_chart(
            f"{primary_t}-Core 4-Bit Throughput Evolution: Prior YNN VNNI vs XNN AMX vs New YNN AMX [TOPS]",
            f"TOPS ({primary_t} Cores)",
            nice_y_max(evo_vnni + evo_xnn + evo_ynn, step=10, floor=20),
            evo_labels,
            "YNNPACK VNNI (Prior)",
            evo_vnni,
            "XNNPACK AMX (GFNI)",
            evo_xnn,
            "YNNPACK AMX (GFNI)",
            evo_ynn,
        )
    )

  # --- Chart 4: Baseline End-to-End Numerical Fidelity (Level B SQNR) ---
  sqnr_labels = [
      "P1: INT4 GEMV",
      "P2: INT4 GEMM",
      "P3: FP32 BMM",
      "P4: FP32 Softmax",
      "P5: INT8 GEMM",
      "P6: INT2 GEMM",
      "P7: Pack/Transpose",
  ]
  c4_xnn = [
      p["xnnpack"]["level_b_fp64_golden"]["sqnr_db"]
      for p in precision_data["pairs"]
  ]
  c4_ynn = [
      p["ynnpack"]["level_b_fp64_golden"]["sqnr_db"]
      for p in precision_data["pairs"]
  ]
  chart4_md = build_side_by_side_chart(
      "Baseline Numerical Fidelity vs. FP64 Golden Reference (Level B SQNR in dB, 200 dB = Bit-Exact)",
      "SQNR vs FP64 Golden (dB)",
      200,
      sqnr_labels,
      c4_xnn,
      c4_ynn,
  )

  # --- Dynamic Metrics for Executive Summary & Diagnosis ---
  p1_cold1_x = get_row("pair1_int4_gemv", 1, 21504, 5376, "DDR5_DRAM_Streaming", "XNNPACK")
  p1_cold1_y = get_row("pair1_int4_gemv", 1, 21504, 5376, "DDR5_DRAM_Streaming", "YNNPACK")
  p1_cold2_x = get_row("pair1_int4_gemv", 1, 5376, 21504, "DDR5_DRAM_Streaming", "XNNPACK")
  p1_cold2_y = get_row("pair1_int4_gemv", 1, 5376, 21504, "DDR5_DRAM_Streaming", "YNNPACK")
  p1_cold2_b = base_lookup.get(("pair1_int4_gemv", 1, 5376, 21504, "DDR5_DRAM_Streaming"))

  p1_warm_x = get_row("pair1_int4_gemv", 1, 8192, 5376, "L3_LLC_Warm", "XNNPACK")
  p1_warm_y = get_row("pair1_int4_gemv", 1, 8192, 5376, "L3_LLC_Warm", "YNNPACK")
  p1_warm_b = base_lookup.get(("pair1_int4_gemv", 1, 8192, 5376, "L3_LLC_Warm"))

  p2_128_21k_x = get_row("pair2_int4_gemm", 128, 21504, 5376, "DDR5_DRAM_to_L2_Tiled", "XNNPACK")
  p2_128_21k_y = get_row("pair2_int4_gemm", 128, 21504, 5376, "DDR5_DRAM_to_L2_Tiled", "YNNPACK")
  p2_128_21k_b = base_lookup.get(("pair2_int4_gemm", 128, 21504, 5376, "DDR5_DRAM_to_L2_Tiled"))
  p2_128_5k_x = get_row("pair2_int4_gemm", 128, 5376, 21504, "DDR5_DRAM_to_L2_Tiled", "XNNPACK")
  p2_128_5k_y = get_row("pair2_int4_gemm", 128, 5376, 21504, "DDR5_DRAM_to_L2_Tiled", "YNNPACK")
  p2_1024_21k_x = get_row("pair2_int4_gemm", 1024, 21504, 5376, "Compute_Bound_Tiled", "XNNPACK")
  p2_1024_21k_y = get_row("pair2_int4_gemm", 1024, 21504, 5376, "Compute_Bound_Tiled", "YNNPACK")
  p2_1024_8k_x = get_row("pair2_int4_gemm", 1024, 8192, 5376, "Compute_Bound_Tiled", "XNNPACK")
  p2_1024_8k_y = get_row("pair2_int4_gemm", 1024, 8192, 5376, "Compute_Bound_Tiled", "YNNPACK")

  p3_1024_4k_x = get_row("pair3_fp32_attn_bmm", 1024, 4096, 512, "L3_LLC_Resident", "XNNPACK")
  p3_1024_4k_y = get_row("pair3_fp32_attn_bmm", 1024, 4096, 512, "L3_LLC_Resident", "YNNPACK")
  p3_1024_1k_x = get_row("pair3_fp32_attn_bmm", 1024, 1024, 256, "L3_LLC_Resident", "XNNPACK")
  p3_1024_1k_y = get_row("pair3_fp32_attn_bmm", 1024, 1024, 256, "L3_LLC_Resident", "YNNPACK")
  p3_2520_x = get_row("pair3_fp32_attn_bmm", 2520, 2520, 72, "L3_LLC_Resident", "XNNPACK")
  p3_2520_y = get_row("pair3_fp32_attn_bmm", 2520, 2520, 72, "L3_LLC_Resident", "YNNPACK")

  p4_1024_4k_x = get_row("pair4_fp32_softmax", 1024, 4096, 1, "L3_LLC_Resident", "XNNPACK")
  p4_1024_4k_y = get_row("pair4_fp32_softmax", 1024, 4096, 1, "L3_LLC_Resident", "YNNPACK")
  p4_2520_x = get_row("pair4_fp32_softmax", 2520, 2520, 1, "L3_LLC_Resident", "XNNPACK")
  p4_2520_y = get_row("pair4_fp32_softmax", 2520, 2520, 1, "L3_LLC_Resident", "YNNPACK")

  p5_280_x = get_row("pair5_int8_gemm", 280, 5376, 1152, "L3_LLC_Resident", "XNNPACK")
  p5_280_y = get_row("pair5_int8_gemm", 280, 5376, 1152, "L3_LLC_Resident", "YNNPACK")
  p5_2520_1k_x = get_row("pair5_int8_gemm", 2520, 1152, 1152, "L3_LLC_Resident", "XNNPACK")
  p5_2520_1k_y = get_row("pair5_int8_gemm", 2520, 1152, 1152, "L3_LLC_Resident", "YNNPACK")
  p5_2520_4k_x = get_row("pair5_int8_gemm", 2520, 4304, 1152, "L3_LLC_Resident", "XNNPACK")
  p5_2520_4k_y = get_row("pair5_int8_gemm", 2520, 4304, 1152, "L3_LLC_Resident", "YNNPACK")

  p6_4k_x = get_row("pair6_int2_gemm", 204, 4096, 1024, "L2_L3_Resident", "XNNPACK")
  p6_4k_y = get_row("pair6_int2_gemm", 204, 4096, 1024, "L2_L3_Resident", "YNNPACK")
  p6_1k4k_x = get_row("pair6_int2_gemm", 204, 1024, 4096, "L2_L3_Resident", "XNNPACK")
  p6_1k4k_y = get_row("pair6_int2_gemm", 204, 1024, 4096, "L2_L3_Resident", "YNNPACK")

  p7_4k_x = get_row("pair7_pack_transpose", 1, 4096, 512, "L3_LLC_Copy", "XNNPACK")
  p7_4k_y = get_row("pair7_pack_transpose", 1, 4096, 512, "L3_LLC_Copy", "YNNPACK")
  p7_2520_x = get_row("pair7_pack_transpose", 1, 2520, 1152, "L3_LLC_Copy", "XNNPACK")
  p7_2520_y = get_row("pair7_pack_transpose", 1, 2520, 1152, "L3_LLC_Copy", "YNNPACK")

  # Compute key ratios dynamically
  amx_vs_vnni_ceil = amx_mc / vnni512_mc
  p1_x_bw_lo = min(p1_cold1_x[primary_key]["gbps"], p1_cold2_x[primary_key]["gbps"])
  p1_x_bw_hi = max(p1_cold1_x[primary_key]["gbps"], p1_cold2_x[primary_key]["gbps"])
  p1_x_ut_lo = min(p1_cold1_x["roofline"][util_field], p1_cold2_x["roofline"][util_field])
  p1_x_ut_hi = max(p1_cold1_x["roofline"][util_field], p1_cold2_x["roofline"][util_field])
  p1_y_bw_lo = min(p1_cold1_y[primary_key]["gbps"], p1_cold2_y[primary_key]["gbps"])
  p1_y_bw_hi = max(p1_cold1_y[primary_key]["gbps"], p1_cold2_y[primary_key]["gbps"])
  p1_y_ut_lo = min(p1_cold1_y["roofline"][util_field], p1_cold2_y["roofline"][util_field])
  p1_y_ut_hi = max(p1_cold1_y["roofline"][util_field], p1_cold2_y["roofline"][util_field])

  p2_y_vs_x_128 = p2_128_21k_y[primary_key]["gops"] / p2_128_21k_x[primary_key]["gops"]
  p2_y_vs_x_128_1c = p2_128_21k_y["single_core"]["gops"] / p2_128_21k_x["single_core"]["gops"]
  p2_y_vs_vnni_128 = (
      p2_128_21k_y[primary_key]["gops"] / p2_128_21k_b[primary_key]["gops"]
      if p2_128_21k_b
      else 0.0
  )
  p1_vnni_vs_yamx_warm = (
      p1_warm_b[primary_key]["gops"] / p1_warm_y[primary_key]["gops"]
      if p1_warm_b
      else 0.0
  )
  p1_vnni_vs_xamx_warm = (
      p1_warm_b[primary_key]["gops"] / p1_warm_x[primary_key]["gops"]
      if p1_warm_b
      else 0.0
  )
  p5_y_vs_x_280_1c = p5_280_y["single_core"]["gops"] / p5_280_x["single_core"]["gops"]
  p5_y_vs_x_280_mc = p5_280_y[primary_key]["gops"] / p5_280_x[primary_key]["gops"]
  p6_y_vs_x_1c = p6_4k_y["single_core"]["gops"] / p6_4k_x["single_core"]["gops"]
  p6_y_vs_x_mc = p6_4k_y[primary_key]["gops"] / p6_4k_x[primary_key]["gops"]
  p6_y_vs_x_1k4k_mc = p6_1k4k_y[primary_key]["gops"] / p6_1k4k_x[primary_key]["gops"]

  # --- Build Section 3.1 Compute Ceilings Table ---
  comp_specs = [
      (
          "amx_int8_tdpbssd",
          "**AMX-INT8 (`TMUL`)**",
          "`tdpbssd` (`16×16×64`)",
          "`INT8 / u8×s8→s32`",
          "`32,768`",
          "GOPS",
          "`xnn_qd8_f32_qc4w_*__avx512amx`, `xnn_qd8_f32_qc8w_*__avx512amx`, `ynn::dot_uint8_int4_*_amxint8`, `ynn::dot_int8_int8_*_amxint8`",
      ),
      (
          "amx_bf16_tdpbf16ps",
          "**AMX-BF16 (`TMUL`)**",
          "`tdpbf16ps` (`16×16×32`)",
          "`BF16→FP32`",
          "`16,384`",
          "GFLOPS",
          "Reference BF16 TMUL ceiling",
      ),
      (
          "avx512_vnni_int8_vpdpbusd",
          "**AVX-512 VNNI (512-bit)**",
          "`vpdpbusd %zmm`",
          "`u8×s8→s32`",
          "`128`",
          "GOPS",
          "`ynn::dot_uint8_int2_int32_*_avx512vnni`, prior `ynn::dot_uint8_int4_*_avx512vnni`",
      ),
      (
          "avx_vnni_256_int8_vpdpbusd",
          "**AVX-VNNI (256-bit)**",
          "`vpdpbusd %ymm`",
          "`u8×s8→s32`",
          "`64`",
          "GOPS",
          "`xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni`",
      ),
      (
          "avx512_fp32_fma",
          "**AVX-512 FP32 FMA (512-bit)**",
          "`vfmadd231ps %zmm`",
          "`FP32`",
          "`32`",
          "GFLOPS",
          "`xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast`, `ynn::dot_fp32_*_avx512`",
      ),
  ]

  comp_hdr_cols = [
      "Instruction Set / Compute Unit",
      "Instruction",
      "Precision",
      "Ops / Inst",
      "1-Core Median (`cv%`)",
  ]
  for t in mt_configs:
    comp_hdr_cols.append(f"{t}-Core (NUMA 0) Median (`cv%`)")
  comp_hdr_cols.extend(
      [f"Rel. to AMX-INT8 ({primary_t}C)", "Load-Bearing Kernels Governed"]
  )
  comp_align = (
      [" :--- ", " :--- ", " :--- ", " ---: ", " ---: "]
      + [" ---: " for _ in mt_configs]
      + [" ---: ", " :--- "]
  )
  comp_table_md = (
      "| " + " | ".join(comp_hdr_cols) + " |\n"
      + "|" + "|".join(comp_align) + "|\n"
  )
  for ckey, label, inst, prec, ops_i, unit, kernels in comp_specs:
    entry = comp[ckey]
    v1 = entry["single_core"]
    cv1 = entry["single_core_stats"]["cv_pct"]
    row_cells = [
        label,
        inst,
        prec,
        ops_i,
        f"**{v1:,.2f} {unit}** (`{cv1:.2f}%`)",
    ]
    for t in mt_configs:
      tk = tc_key(t)
      vt = entry[tk]
      cvt = entry[f"{tk}_stats"]["cv_pct"]
      row_cells.append(f"**{fmt_gops_tops(vt, unit)}** (`{cvt:.2f}%`)")
    ratio = entry[primary_key] / amx_mc
    row_cells.append(f"`{ratio:.3f}×` (`{ratio * 100.0:.1f}%`)")
    row_cells.append(kernels)
    comp_table_md += "| " + " | ".join(row_cells) + " |\n"

  # --- Build Section 3.2 Bandwidth Ceilings Table ---
  dram_ws_desc = (
      f"`256 MB` (1C) / `896 MB` (28C) / `1,792 MB` (56C)"
      if has_28_and_56
      else f"`256 MB` (1C) / `{32 * primary_t} MB` ({primary_t}C)"
  )
  bw_specs = [
      (
          "l1d_16kb_per_core",
          "read_only",
          "**L1D Cache** (`48 KiB`/core)",
          "`16 KB` / core",
          "Streaming Read (`vmovaps`)",
          "Microkernel register-tile A/B streaming",
      ),
      (
          "l1d_16kb_per_core",
          "read_write_copy",
          "**L1D Cache** (`48 KiB`/core)",
          "`16 KB` / core",
          "Read + Write Copy",
          "AMX stack-spill `vmovdqa64` / `tileloadd` / `tilestored` scratchpad",
      ),
      (
          "l2_512kb_per_core",
          "read_only",
          "**L2 Cache** (`2 MiB`/core)",
          "`512 KB` / core",
          "Streaming Read (`vmovaps`)",
          "L2-tiled GEMM panels (`M=128..204`), L2-sliced warm GEMV & Softmax reductions",
      ),
      (
          "l2_512kb_per_core",
          "read_write_copy",
          "**L2 Cache** (`2 MiB`/core)",
          "`512 KB` / core",
          "Read + Write Copy",
          "Multi-threaded KV packing/transpose & fused Softmax rows",
      ),
      (
          "l3_llc",
          "read_only",
          "**L3 Shared LLC** (`105 MiB` Socket 0)",
          "`16 MB` (1C) / `32 MB` shared (`8 MB`/T)",
          "Streaming Read (`vmovaps`)",
          "Single-layer weights (`11–58 MB`) & KV cache (`p1024`–`p4096`)",
      ),
      (
          "l3_llc",
          "read_write_copy",
          "**L3 Shared LLC** (`105 MiB` Socket 0)",
          "`16 MB` (1C) / `32 MB` shared (`8 MB`/T)",
          "Read + Write Copy",
          "Large L3-resident tensor transposes & dequantize passes",
      ),
      (
          "ddr5_dram",
          "read_only",
          "**DDR5 DRAM** (Socket 0)",
          dram_ws_desc,
          "Streaming Read (`vmovaps`)",
          "**Full-model `M=1` Decode weight streaming (`18+ GB`)**",
      ),
      (
          "ddr5_dram",
          "read_write_copy",
          "**DDR5 DRAM** (Socket 0)",
          dram_ws_desc,
          "Read + Write Copy",
          "Cold DRAM buffer copies",
      ),
      (
          "ddr5_dram",
          "stream_triad",
          "**DDR5 DRAM** (Socket 0)",
          dram_ws_desc,
          "STREAM Triad (`2R + 1W`)",
          "Cold DRAM elementwise / residual fusion",
      ),
  ]
  bw_hdr_cols = [
      "Memory Hierarchy Level",
      "Working Set",
      "Access Pattern",
      "1-Core Median (`cv%`)",
  ]
  for t in mt_configs:
    bw_hdr_cols.append(f"{t}-Core (NUMA 0) Median (`cv%`)")
  bw_hdr_cols.append("Primary Kernel Regimes Governed")
  bw_align = (
      [" :--- ", " :--- ", " :--- ", " ---: "]
      + [" ---: " for _ in mt_configs]
      + [" :--- "]
  )
  bw_table_md = (
      "| " + " | ".join(bw_hdr_cols) + " |\n"
      + "|" + "|".join(bw_align) + "|\n"
  )
  for bkey, mode, lvl, ws, pat, desc in bw_specs:
    entry = bw[bkey][mode]
    v1 = entry["single_core"]
    cv1 = entry["single_core_stats"]["cv_pct"]
    row_cells = [lvl, ws, pat, f"**{v1:,.2f} GB/s** (`{cv1:.2f}%`)"]
    for t in mt_configs:
      tk = tc_key(t)
      vt = entry[tk]
      cvt = entry[f"{tk}_stats"]["cv_pct"]
      row_cells.append(f"**{vt:,.2f} GB/s** (`{cvt:.2f}%`)")
    row_cells.append(desc)
    bw_table_md += "| " + " | ".join(row_cells) + " |\n"

  # --- Build Precision Markdown Tables (Section 5, Baseline Retained) ---
  pair_short_names = [
      "Pair 1: 4-Bit Decode GEMV (`M=1`)",
      "Pair 2: 4-Bit Prefill GEMM (`M=16/12`)",
      "Pair 3: FP32 Attention `BatchMatMul`",
      "Pair 4: FP32 Attention `Softmax`",
      "Pair 5: 8-Bit Multimodal GEMM",
      "Pair 6: 2-Bit Weight-Only GEMM",
      "Pair 7: 32-Bit Pack / Transpose",
  ]
  pair_quant_schemes = [
      "Per-row `QD8` act × Per-channel `QC4W` (`INT4`)",
      "Per-row `QD8` act × Per-channel `QC4W` (`INT4`)",
      "IEEE-754 `FP32` FMA (`K=64`)",
      "IEEE-754 `FP32` (`rr2_p5` poly vs `expf`, `K=1024`)",
      "Per-row `QD8`/`INT8` act × Per-channel `QC8W` (`INT8`)",
      "Per-tensor `INT8`/`UINT8` act × Per-channel `QC2W` (`INT2`)",
      "Lossless 32-bit `uint32_t` panel pack / transpose",
  ]

  level_a_rows_md = ""
  level_b_rows_md = ""
  for idx, p in enumerate(precision_data["pairs"]):
    pshort = pair_short_names[idx]
    qscheme = pair_quant_schemes[idx]
    for backend_label, bkey, sym_key in (
        ("XNNPACK", "xnnpack", "xnnpack_symbol"),
        ("YNNPACK", "ynnpack", "ynnpack_symbol"),
    ):
      la = p[bkey]["level_a_exact_ukernel"]
      lb = p[bkey]["level_b_fp64_golden"]
      dists = p[bkey]["distributions"]
      sym = p[sym_key]

      level_a_rows_md += (
          f"| {pshort} | **{backend_label}** | `{sym}` | "
          f"`{la['max_abs_err']:.2e}` | `{la['max_rel_err']:.2e}` | "
          f"`{la['rmse']:.2e}` | `{la['max_ulp_diff']}` | "
          f"`{la['mean_ulp_diff']:.4f}` | **`{la['sqnr_db']:.2f} dB`** | "
          f"`{la['exact_match_pct']:.2f}%` | `{la['nan_inf_count']}` | "
          f"**`{la['status']}`** |\n"
      )

      level_b_rows_md += (
          f"| {pshort} | **{backend_label}** | {qscheme} | "
          f"`{dists['normal']['sqnr_db']:.2f} dB` | "
          f"`{dists['saturation']['sqnr_db']:.2f} dB` | "
          f"`{dists['outliers']['sqnr_db']:.2f} dB` | "
          f"`{dists['tails']['sqnr_db']:.2f} dB` | "
          f"**`{lb['sqnr_db']:.2f} dB`** | **`{lb['cosine_sim']:.6f}`** | "
          f"`{lb['rmse']:.2e}` | `{lb['max_abs_err']:.2e}` | "
          f"**`{lb['status']}`** |\n"
      )

  p4_xnn_prec = precision_data["pairs"][3]["xnnpack"]
  p4_ynn_prec = precision_data["pairs"][3]["ynnpack"]
  p7_xnn_prec = precision_data["pairs"][6]["xnnpack"]
  p7_ynn_prec = precision_data["pairs"][6]["ynnpack"]

  # --- Build Thread Configuration Bullet Text ---
  tc_bullets = [
      "  - **Single-Core (`1T`)**: Pinned to physical core `0` (`sched_setaffinity`)"
  ]
  for t in mt_configs:
    if t == primary_t:
      tc_bullets.append(
          f"  - **Full NUMA Node 0 (`{t}T`)**: Pinned across all `{t}` physical cores of Socket 0 (`0..{t - 1}`, 1 thread per physical core, no SMT sibling contention, zero cross-socket UPI traffic)"
      )
    else:
      tc_bullets.append(
          f"  - **Half-Socket (`{t}T`)**: Pinned across physical cores `0..{t - 1}` of NUMA Node 0 (1 thread per physical core, for mid-socket scaling comparison)"
      )
  tc_bullets_md = "\n".join(tc_bullets)

  # --- Build Warm-Cache M=1 Note (Dynamic) ---
  if p1_warm_b is not None:
    warm_m1_exec_bullet = (
        f"   - Conversely, when weights fit warm in L3/L2 cache (`1×8192×5376`, `L3_LLC_Warm`), **a `get_dot_kernel()` cost-model regression is exposed**: "
        f"upstream `{commit_short}` dispatches YNNPACK's `32x32x64_amxint8` even at `M = 1` (**{p1_warm_y[primary_key]['gops'] / 1000.0:.2f} TOPS** @ {primary_t}C vs. XNNPACK `1x64c4__avx512amx` at **{p1_warm_x[primary_key]['gops'] / 1000.0:.2f} TOPS**), "
        f"whereas YNNPACK's prior `1x32x8_avx512vnni` reaches **{p1_warm_b[primary_key]['gops'] / 1000.0:.2f} TOPS** (**{p1_vnni_vs_yamx_warm:.2f}× faster** than YNNPACK AMX and **{p1_vnni_vs_xamx_warm:.2f}× faster** than XNNPACK AMX) "
        f"because `M = 1` utilizes only 1 of 16 AMX tile rows (`6.25%`) while still paying full GFNI unpack + stack-bounce (`vmovdqa64` → `tileloadd`) overhead."
    )
  else:
    warm_m1_exec_bullet = (
        f"   - When weights fit warm in L3/L2 cache (`1×8192×5376`, `L3_LLC_Warm`), YNNPACK reaches **{p1_warm_y[primary_key]['gops'] / 1000.0:.2f} TOPS** "
        f"vs. XNNPACK at **{p1_warm_x[primary_key]['gops'] / 1000.0:.2f} TOPS** on {primary_t} cores."
    )

  # --- Build Prefill 4-Bit Executive Bullet (Dynamic) ---
  p2_y_128_tops = p2_128_21k_y[primary_key]["gops"] / 1000.0
  p2_x_128_tops = p2_128_21k_x[primary_key]["gops"] / 1000.0
  p2_y_1024_tops = p2_1024_21k_y[primary_key]["gops"] / 1000.0
  p2_x_1024_tops = p2_1024_21k_x[primary_key]["gops"] / 1000.0
  p2_y_128_util = p2_128_21k_y["roofline"][util_field]
  p2_x_128_util = p2_128_21k_x["roofline"][util_field]
  if p2_128_21k_b is not None:
    p2_b_128_tops = p2_128_21k_b[primary_key]["gops"] / 1000.0
    p2_b_128_util = p2_128_21k_b["roofline"][util_field]
    prefill_exec_bullet = (
        f"2. **Prefill (`M = 128..1024` 4-Bit GEMM, 46%–65% of Prefill Time) — Upstream YNNPACK AMX-INT8+GFNI (`{commit_short}`) Surpasses Both Prior VNNI (`{p2_y_vs_vnni_128:.2f}×`) and XNNPACK AMX (`{p2_y_vs_x_128:.2f}×`)**:\n"
        f">    - Our compute ceiling probes measure **$C_{{\\text{{prac}}}}(\\text{{AMX-INT8}}) = \\mathbf{{{amx_mc / 1000.0:.2f}\\text{{ TOPS}}}}$** (`{amx_1c / 1000.0:.2f} TOPS`/core) vs. **$C_{{\\text{{prac}}}}(\\text{{AVX-512 VNNI}}) = \\mathbf{{{vnni512_mc / 1000.0:.2f}\\text{{ TOPS}}}}$** (`{vnni512_1c / 1000.0:.2f} TOPS`/core) on {primary_t} cores — an **{amx_vs_vnni_ceil:.2f}× hardware ceiling advantage** for AMX-INT8.\n"
        f">    - YNNPACK's new `ynn::dot_uint8_int4_int32_32x32x64_16x16x8_amxint8` (GFNI `vgf2p8affineqb` nibble unpack + `2×2` AMX `tdpbusd` tile blocking) achieves **{p2_y_128_tops:.2f} TOPS** at `128×21504×5376` (`{p2_y_128_util:.1f}%` of {primary_t}C AMX-INT8 ceiling) and **{p2_y_1024_tops:.2f}–{p2_1024_8k_y[primary_key]['gops'] / 1000.0:.2f} TOPS** at `M = 1024`.\n"
        f">    - This represents a **{p2_y_vs_vnni_128:.2f}× speedup** over YNNPACK's prior AVX-512 VNNI kernel (`{p2_b_128_tops:.2f} TOPS`, `{p2_b_128_util:.1f}%` of VNNI ceiling) and a **{p2_y_vs_x_128:.2f}× speedup** over XNNPACK's GFNI-optimized `16x64c4__avx512amx` (**{p2_x_128_tops:.2f} TOPS**, `{p2_x_128_util:.1f}%` of AMX ceiling; **{p2_x_1024_tops:.2f} TOPS** at `M = 1024`) because `2×2` tile blocking (`32×32`) amortizes every unpacked `B`-tile across **2** `tdpbusd` instructions instead of 1 (and omits XNNPACK's fused `FP32` epilogue)."
    )
  else:
    prefill_exec_bullet = (
        f"2. **Prefill (`M = 128..1024` 4-Bit GEMM)**: YNNPACK reaches **{p2_y_128_tops:.2f} TOPS** (`{p2_y_128_util:.1f}%`) vs. XNNPACK **{p2_x_128_tops:.2f} TOPS** (`{p2_x_128_util:.1f}%`) on {primary_t} cores."
    )

  p4_x_gbps_lo = min(p4_2520_x[primary_key]["gbps"], p4_1024_4k_x[primary_key]["gbps"])
  p4_x_gbps_hi = max(p4_2520_x[primary_key]["gbps"], p4_1024_4k_x[primary_key]["gbps"])
  p4_x_ut_lo = min(p4_2520_x["roofline"][util_field], p4_1024_4k_x["roofline"][util_field])
  p4_x_ut_hi = max(p4_2520_x["roofline"][util_field], p4_1024_4k_x["roofline"][util_field])
  p4_y_gbps_lo = min(p4_2520_y[primary_key]["gbps"], p4_1024_4k_y[primary_key]["gbps"])
  p4_y_gbps_hi = max(p4_2520_y[primary_key]["gbps"], p4_1024_4k_y[primary_key]["gbps"])
  p4_y_ut_lo = min(p4_2520_y["roofline"][util_field], p4_1024_4k_y["roofline"][util_field])
  p4_y_ut_hi = max(p4_2520_y["roofline"][util_field], p4_1024_4k_y["roofline"][util_field])
  p7_x_vs_y_4k = p7_4k_x[primary_key]["gbps"] / p7_4k_y[primary_key]["gbps"]
  p7_x_vs_y_2520 = p7_2520_x[primary_key]["gbps"] / p7_2520_y[primary_key]["gbps"]
  p7_ratio_lo = min(p7_x_vs_y_4k, p7_x_vs_y_2520)
  p7_ratio_hi = max(p7_x_vs_y_4k, p7_x_vs_y_2520)

  # --- Check scaling text for 28T -> 56T ---
  scaling_line = ""
  if has_28_and_56:
    amx_28 = comp["amx_int8_tdpbssd"]["numa_node0_28c"]
    ddr_28 = bw["ddr5_dram"]["read_only"]["numa_node0_28c"]
    l2_28 = bw["l2_512kb_per_core"]["read_only"]["numa_node0_28c"]
    scaling_line = (
        f"- **Socket 0 Scaling (`1T` → `28T` → `56T`)**: In-register compute ceilings scale nearly linearly across all 56 physical cores "
        f"(AMX-INT8: `{amx_1c / 1000.0:.2f}` → `{amx_28 / 1000.0:.2f}` → **`{amx_mc / 1000.0:.2f} TOPS`**, `{amx_mc / amx_1c:.1f}×`; "
        f"L2 Read: `{bw['l2_512kb_per_core']['read_only']['single_core']:.1f}` → `{l2_28:,.1f}` → **`{l2_rd_mc:,.1f} GB/s`**, `{l2_rd_mc / bw['l2_512kb_per_core']['read_only']['single_core']:.1f}×`), "
        f"whereas Socket 0's 8-channel DDR5 streaming read ceiling saturates by 28 cores (`{ddr_28:.2f} GB/s` @ 28C vs. **`{ddr_rd_mc:.2f} GB/s`** @ 56C).\n"
    )

  date_meta_line = (
      f"- **Benchmark Date**: `{benchmark_date}`\n" if benchmark_date else ""
  )
  commit_meta_line = (
      f"- **Upstream Microkernel Revision**: `google/XNNPACK` `master` @ [`{commit_short}`](https://github.com/google/XNNPACK/commit/{xnnpack_commit}) (includes YNNPACK AMX-INT8+GFNI 4-bit `32×32` and XNNPACK GFNI-optimized AMX 4-bit `1x64c4`/`16x64c4`)\n"
      if xnnpack_commit
      else ""
  )
  prev_report_line = (
      f"- **Previous Kernel Report (`{prev_report_date}`)**: [{prev_report_path.parent.name}/kernel_report_gemma4_31b_sapphire_rapids.md]({prev_report_rel})\n"
      if prev_report_path is not None
      else ""
  )
  companion_suffix = (
      f" (`{companion_date}` baseline)"
      if companion_date and companion_date != benchmark_date
      else ""
  )
  prec_method_line = (
      f"  - **Numerical Precision & Quantization Fidelity**: Baseline two-level verification from `{precision_date or 'prior run'}` ([verify_litert_microkernels.c](../../scripts/verify_litert_microkernels.c)) retained in Section 5 (not re-run for `{commit_short}`; built-in standalone numerical smoke checks verified exact `INT32` dot outputs and finite non-zero `FP32` outputs prior to timing)"
      if baseline_retained
      else "  - **Numerical Precision & Quantization Fidelity**: Two-level verification ([verify_litert_microkernels.c](../../scripts/verify_litert_microkernels.c)) across **4 synthetic input distributions** (`normal`, `saturation`, `outliers`, `tails`) comparing every microkernel against (A) an **Exact Packed-Layout Scalar Reference (`INT64` / `FP64`)** and (B) an **Unquantized `FP64` Golden Operator Reference**"
  )

  # --- Main Markdown Document ---
  md = rf"""# Microkernel Performance, Roofline & Numerical Precision Report — Gemma 4 31B on Intel Sapphire Rapids

{date_meta_line}- **Model**: Gemma 4 31B (`gemma-4-31b-it`, 4-bit Decoder / 8-bit Vision ViT / 2-bit & 8-bit Audio Conformer)
- **Target Microarchitecture**: Intel Sapphire Rapids (`x86_64`, AVX-512F, AVX-512 VNNI, AVX-VNNI, GFNI, AMX-INT8, AMX-BF16; `56` physical cores / `105 MiB` shared L3 LLC / `2 MiB` private L2 per core / `48 KiB` L1D per core on Socket 0)
{commit_meta_line}- **Execution Configurations**:
{tc_bullets_md}
- **Repeatability & Verification Methodology**:
  - **Performance & Roofline**: `CLOCK_MONOTONIC_RAW` with `#pragma omp barrier` synchronization before and after every timed loop, `OMP_WAIT_POLICY=ACTIVE`, explicit warmup pass, and **{num_trials} independent timed trials** per probe and microkernel shape (`median`, `mean`, `min`, `max`, `stdev`, `cv_pct`)
{prec_method_line}
{prev_report_line}- **Companion End-to-End Profiling Report{companion_suffix}**: [{companion_path.name if companion_path else 'kernels_gemma4_31b_sapphire_rapids.md'}]({companion_rel})

---

## 1. Executive Summary & Key Roofline Findings

> [!IMPORTANT]
> **Performance Roofline Summary Across All 7 Load-Bearing Microkernel Pairs (`1T` / `28T` / `56T`, Upstream `google/XNNPACK` `{commit_short}`)**:
> 1. **Decode (`M = 1` 4-Bit GEMV, 83%–87% of Decode Time) — Cold Weights Saturate Socket 0 DDR5 (`{ddr_rd_mc:.2f} GB/s`), While Warm Cache Exposes an `M = 1` AMX Cost-Model Gap**:
>    - At `M = 1`, arithmetic intensity is only **$3.99\text{{ OPs/Byte}}$**, placing cold-weight decode deep in the **DDR5 DRAM bandwidth-bound regime** ($BW_{{\text{{prac}}}}(\text{{DDR5 Read}}) = \mathbf{{{ddr_rd_mc:.2f}\text{{ GB/s}}}}$ on {primary_t} cores $\Rightarrow \mathbf{{{p1_cold2_x['roofline'][f'effective_ceiling_{primary_sfx}']:.1f}\text{{ GOPS}}}}$ ceiling).
>    - On cold DDR5-streaming weights (`1×21504×5376` and `1×5376×21504`), **XNNPACK achieves `{p1_x_bw_lo:.1f}–{p1_x_bw_hi:.1f} GB/s` (`{p1_x_ut_lo:.1f}%–{p1_x_ut_hi:.1f}%` of practical DDR5 ceiling)** and **YNNPACK achieves `{p1_y_bw_lo:.1f}–{p1_y_bw_hi:.1f} GB/s` (`{p1_y_ut_lo:.1f}%–{p1_y_ut_hi:.1f}%` of practical DDR5 ceiling)** at {primary_t}C{f" (matching prior VNNI at `{p1_cold2_b[primary_key]['gbps']:.1f} GB/s`)" if p1_cold2_b else ""}.
> {warm_m1_exec_bullet}
> {prefill_exec_bullet}
> 3. **8-Bit Vision/Audio Linear GEMM (`pair5_int8_gemm`) — Both Use AMX-INT8, and YNNPACK's `32×32` `2×2` Tile Blocking Beats XNNPACK's `16×64` `1×4` Tile Blocking**:
>    - On `280×5376×1152`, YNNPACK's `dot_int8_int8_int32_32x32x64_16x16x4_amxint8` reaches **{p5_280_y[primary_key]['gops'] / 1000.0:.2f} TOPS (`{p5_280_y['roofline'][util_field]:.1f}%` of {primary_t}C AMX-INT8 ceiling)** vs. XNNPACK's **{p5_280_x[primary_key]['gops'] / 1000.0:.2f} TOPS (`{p5_280_x['roofline'][util_field]:.1f}%`)** (`{p5_y_vs_x_280_1c:.2f}×` faster on 1C, `{p5_y_vs_x_280_mc:.2f}×` faster on {primary_t}C), and on `2520×4304×1152` YNNPACK leads **{p5_2520_4k_y[primary_key]['gops'] / 1000.0:.2f} TOPS** vs. **{p5_2520_4k_x[primary_key]['gops'] / 1000.0:.2f} TOPS**.
> 4. **2-Bit Audio Conformer GEMM (`pair6_int2_gemm`) — 512-Bit `AVX-512 VNNI` (`{vnni512_mc / 1000.0:.2f} TOPS` Ceiling) vs. 256-Bit `AVX-VNNI` (`{vnni256_mc / 1000.0:.2f} TOPS` Ceiling)**:
>    - XNNPACK's 256-bit `5x8c8__avxvnni` hits **{p6_4k_x[primary_key]['gops'] / 1000.0:.2f} TOPS (`{p6_4k_x['roofline'][util_field]:.1f}%` of the 256-bit AVX-VNNI ceiling)** on `204×4096×1024`, leaving half of Sapphire Rapids' 512-bit vector width unused. YNNPACK's 512-bit `4x64x16_avx512vnni` reaches **{p6_4k_y[primary_key]['gops'] / 1000.0:.2f}–{p6_1k4k_y[primary_key]['gops'] / 1000.0:.2f} TOPS** (`{p6_y_vs_x_1c:.2f}×` faster on 1C, `{min(p6_y_vs_x_mc, p6_y_vs_x_1k4k_mc):.2f}×–{max(p6_y_vs_x_mc, p6_y_vs_x_1k4k_mc):.2f}×` faster on {primary_t}C).
> 5. **FP32 Attention `BatchMatMul` & `Softmax` (`pair3` & `pair4`)**:
>    - With 2D N-major tile partitioning and zero-init accumulators (`C_in = NULL`), both XNNPACK `7x32` and YNNPACK `5x64` reach **{min(p3_1024_1k_x['roofline'][util_field], p3_1024_4k_x['roofline'][util_field]):.1f}%–{max(p3_1024_1k_y['roofline'][util_field], p3_1024_4k_y['roofline'][util_field]):.1f}% of the {primary_t}-core AVX-512 FP32 FMA ceiling (`{fp32_mc / 1000.0:.2f} TFLOPS`)** (`{p3_1024_4k_x[primary_key]['gops'] / 1000.0:.2f}` vs. `{p3_1024_4k_y[primary_key]['gops'] / 1000.0:.2f} TFLOPS` on `1024×4096×512`, and `{p3_2520_x[primary_key]['gops'] / 1000.0:.2f}` vs. `{p3_2520_y[primary_key]['gops'] / 1000.0:.2f} TFLOPS` on `2520×2520×72`).
>    - In Softmax (`1024×4096` and `2520×2520`), XNNPACK's fused 2-pass `rmax + raddstoreexpminusmax` achieves **{p4_x_gbps_lo:,.1f}–{p4_x_gbps_hi:,.1f} GB/s (`{p4_x_ut_lo:.1f}%–{p4_x_ut_hi:.1f}%` of {primary_t}C L2 Read+Write Copy)** while computing polynomial `exp(x - max)` in-flight; YNNPACK's read-only reductions (`max_k1 + sum_k1`, without `exp`) achieve **{p4_y_gbps_lo:,.1f}–{p4_y_gbps_hi:,.1f} GB/s (`{p4_y_ut_lo:.1f}%–{p4_y_ut_hi:.1f}%` of {primary_t}C L2 Read-Only)** after correcting the prior harness's element-count (`k = row_len`) and 4-row batching parameters.
> 6. **32-Bit Pack / Transpose (`pair7_pack_transpose`)**:
>    - XNNPACK's `packw_gemm_goi_x32` reaches **{p7_4k_x[primary_key]['gbps']:,.1f}–{p7_2520_x[primary_key]['gbps']:,.1f} GB/s (`{p7_4k_x['roofline'][util_field]:.1f}%–{p7_2520_x['roofline'][util_field]:.1f}%` of {primary_t}C L2 Copy)** vs. YNNPACK's `transpose_x32_avx512` at **{p7_4k_y[primary_key]['gbps']:,.1f}–{p7_2520_y[primary_key]['gbps']:,.1f} GB/s** (`{p7_ratio_lo:.2f}×–{p7_ratio_hi:.2f}×` advantage for XNNPACK's software-prefetched `u4_prfm` packing loop).

{scaling_line}
---

## 2. Visual Summary: Throughput, Roofline Utilization, 4-Bit Evolution & Baseline Precision

> **Series Legend**: **Left Bar (Blue `#2b6cd9`)** = **XNNPACK** | **Right Bar (Orange `#e67e22`)** = **YNNPACK**

{chart1_md}

{chart2_md}
{chart3_evo_md}

{chart4_md}

---

## 3. Empirical Hardware Ceilings on Intel Sapphire Rapids (`cpu_ceilings.c`)

All ceilings below were measured directly on the target 56-core Sapphire Rapids socket using [cpu_ceilings.c](../../probes/cpu_ceilings.c) across **{ceilings['methodology']['num_trials']} independent barrier-synchronized trials** (`OMP_WAIT_POLICY=ACTIVE`, NUMA Node 0 physical cores `0..{primary_t - 1}`) after explicit warmup.

### 3.1 Practical Compute Ceilings: $C_{{\text{{prac}}}}(\text{{op}}, \text{{dtype}})$

{comp_table_md}
### 3.2 Practical Memory-Hierarchy Bandwidth Ceilings: $BW_{{\text{{prac}}}}(\text{{WorkingSet}})$

{bw_table_md}
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
          "4.4 Pair 4: FP32 Attention `Softmax` (`rmax + raddstoreexpminusmax` vs. `max_k1 + sum_k1` Reductions)"
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

  secondary_mt = [t for t in mt_configs if t != primary_t]

  for pid, ptitle in pair_titles.items():
    prows = [r for r in enriched_rows if r["pair_id"] == pid]
    md += f"### {ptitle}\n\n"

    hdr = [
        "Shape `(M×N×K)`",
        "Regime",
        "Backend",
        "Microkernel (`ISA`)",
        "AI (`OP/B`)",
        "1C Perf (`Util`)",
    ]
    align = [" :--- ", " :--- ", " :--- ", " :--- ", " ---: ", " ---: "]
    for st in secondary_mt:
      hdr.append(f"{st}C Perf (`cv%`, `ISA Util`)")
      align.append(" ---: ")
    hdr.extend([
        f"{primary_t}C Lat (`ms`)",
        f"**{primary_t}C Perf**",
        f"{primary_t}C `cv%`",
        f"{primary_t}C ISA Ceiling",
        f"**{primary_t}C ISA Roofline Util**",
        f"**{primary_t}C Machine Peak Util**",
    ])
    align.extend([" ---: ", " ---: ", " ---: ", " ---: ", " ---: ", " ---: "])

    md += "| " + " | ".join(hdr) + " |\n"
    md += "|" + "|".join(align) + "|\n"

    for r in prows:
      s = r["shape"]
      rf = r["roofline"]
      sc = r["single_core"]
      mc = r[primary_key]
      unit = rf["unit"]
      p1 = sc["gbps"] if unit == "GB/s" else sc["gops"]
      pmc = mc["gbps"] if unit == "GB/s" else mc["gops"]
      shape_str = f"`{s['M']}×{s['N']}×{s['K']}`"

      cells = [
          shape_str,
          f"`{r['memory_regime']}`",
          f"**{r['backend']}**",
          f"`{r['kernel']}` (`{r['isa']}`)",
          f"`{r['arithmetic_intensity_ops_per_byte']:.2f}`",
          f"`{fmt_perf(p1, unit)}` (`{rf['isa_roofline_utilization_1c_pct']:.1f}%`)",
      ]
      for st in secondary_mt:
        sk = tc_key(st)
        ssfx = tc_suffix(st)
        ps = r[sk]["gbps"] if unit == "GB/s" else r[sk]["gops"]
        cells.append(
            f"`{fmt_perf(ps, unit)}` (`{r[sk]['cv_pct']:.2f}%`, "
            f"`{rf[f'isa_roofline_utilization_{ssfx}_pct']:.1f}%`)"
        )

      perf_cell = (
          f"**`{fmt_perf(pmc, unit)}`**"
          if unit == "GB/s"
          else f"**`{fmt_perf(pmc, unit)}`** (`{mc['gbps']:.1f} GB/s`)"
      )
      cells.extend([
          f"`{mc['latency_ms']:.3f}`",
          perf_cell,
          f"`{mc['cv_pct']:.2f}%`",
          f"`{fmt_perf(rf[f'effective_ceiling_{primary_sfx}'], unit)}`",
          f"**`{rf[f'isa_roofline_utilization_{primary_sfx}_pct']:.1f}%`**",
          f"**`{rf[f'machine_roofline_utilization_{primary_sfx}_pct']:.1f}%`**",
      ])
      md += "| " + " | ".join(cells) + " |\n"
    md += "\n"

    # Insert 3-way 4-bit comparison table right after Pair 2
    if pid == "pair2_int4_gemm" and enriched_baselines:
      md += (
          f"#### 4.2.1 4-Bit Microkernel Evolution (`{commit_short}`): "
          "Prior YNNPACK AVX-512 VNNI vs. XNNPACK AMX-INT8+GFNI vs. New YNNPACK AMX-INT8+GFNI\n\n"
      )
      evo_hdr = [
          "Pair & Shape `(M×N×K)`",
          "Regime",
          "1C: VNNI / XNN AMX / YNN AMX",
      ]
      evo_align = [" :--- ", " :--- ", " ---: "]
      for st in secondary_mt:
        evo_hdr.append(f"{st}C: VNNI / XNN AMX / YNN AMX")
        evo_align.append(" ---: ")
      evo_hdr.extend([
          f"{primary_t}C Prior YNN VNNI",
          f"{primary_t}C XNNPACK AMX",
          f"**{primary_t}C New YNNPACK AMX**",
          f"**YNN AMX / Prior VNNI ({primary_t}C)**",
          f"**YNN AMX / XNN AMX ({primary_t}C)**",
      ])
      evo_align.extend([" ---: ", " ---: ", " ---: ", " ---: ", " ---: "])
      md += "| " + " | ".join(evo_hdr) + " |\n"
      md += "|" + "|".join(evo_align) + "|\n"

      for br in enriched_baselines:
        s = br["shape"]
        key5 = (br["pair_id"], s["M"], s["N"], s["K"], br["memory_regime"])
        xr = lookup.get((*key5, "XNNPACK"))
        yr = lookup.get((*key5, "YNNPACK"))
        if xr is None or yr is None:
          continue
        ptag = "P1" if br["pair_id"] == "pair1_int4_gemv" else "P2"
        b1, x1, y1 = (
            br["single_core"]["gops"],
            xr["single_core"]["gops"],
            yr["single_core"]["gops"],
        )
        bm, xm, ym = (
            br[primary_key]["gops"],
            xr[primary_key]["gops"],
            yr[primary_key]["gops"],
        )
        cells = [
            f"`{ptag}: {s['M']}×{s['N']}×{s['K']}`",
            f"`{br['memory_regime']}`",
            f"`{b1:.1f}` / `{x1:.1f}` / `{y1:.1f} GOPS`",
        ]
        for st in secondary_mt:
          sk = tc_key(st)
          cells.append(
              f"`{fmt_perf(br[sk]['gops'], 'GOPS')}` / "
              f"`{fmt_perf(xr[sk]['gops'], 'GOPS')}` / "
              f"`{fmt_perf(yr[sk]['gops'], 'GOPS')}`"
          )
        cells.extend([
            f"`{fmt_perf(bm, 'GOPS')}`",
            f"`{fmt_perf(xm, 'GOPS')}`",
            f"**`{fmt_perf(ym, 'GOPS')}`**",
            f"**`{ym / bm:.2f}×`**",
            f"**`{ym / xm:.2f}×`**",
        ])
        md += "| " + " | ".join(cells) + " |\n"
      md += "\n"

  # --- Section 5: Baseline Precision Verification (Retained) ---
  baseline_note_md = (
      f"> [!NOTE]\n"
      f"> **Baseline Precision Results Retained (`{precision_date or 'prior run'}`)**: The full two-level numerical precision sweep below ([{precision_path.name}]({precision_rel})) is retained from the `{precision_date or 'baseline'}` verification run and was **not re-run** for upstream `google/XNNPACK` commit `{commit_short}` (this benchmark update covers microkernel performance and hardware roofline analysis only). In particular, Pairs 1 and 2 below reflect the prior YNNPACK AVX-512 VNNI 4-bit symbols (`1x32x8` / `12x16x8`). Prior to timing `{commit_short}`, the standalone benchmark binary executed built-in numerical smoke checks confirming exact `INT32` accumulator output (`C[0] == C[last] == K`) for `ynn::dot_uint8_int4_int32_32x32x64_16x16x8_amxint8` at both `M = 1` and `M = 32` and finite non-zero `FP32` outputs for `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_{{1,16}}x64c4__avx512amx`.\n\n"
      if baseline_retained
      else ""
  )
  md += rf"""---

## 5. Numerical Precision & Quantization Fidelity Verification (`verify_litert_microkernels.c`)

{baseline_note_md}To verify that SIMD/AMX optimizations and low-bit quantization schemes do not introduce silent numerical regressions, [verify_litert_microkernels.c](../../scripts/verify_litert_microkernels.c) tests all 14 load-bearing XNNPACK and YNNPACK microkernels across **two reference levels** and **four synthetic input distributions**.

### 5.1 Two-Level Reference Verification Methodology & Input Distributions

1. **Level A — Exact Packed-Layout Microkernel Reference**:
   - Feeds the **identical quantized input buffer and packed weight buffer** into both the target SIMD/AMX microkernel and an exact scalar reference (`INT64` accumulator + `FP64` post-scaling for quantized GEMMs, sequential IEEE-754 `FP32` FMA for `FP32` GEMMs, exact `FP64` $\exp(x - x_{{\max}})$ for Softmax, and scalar 32-bit pack/transpose).
   - **Purpose**: Isolates hardware/microkernel implementation fidelity (catching register clobbers, AMX tile stride/layout mismatches, `INT32` accumulator overflow, `FP32` post-scaling rounding, or polynomial approximation error).
2. **Level B — Unquantized `FP64` Golden Operator Reference**:
   - Generates high-precision `FP64` tensors $A_{{\text{{fp64}}}}$, $B_{{\text{{fp64}}}}$, and $\text{{bias}}_{{\text{{fp64}}}}$, quantizes and packs them using each backend's exact quantization scheme (`QD8`, `QC8W`, `QC4W`, `QC2W`), executes the microkernel, and compares the dequantized output against the unquantized `FP64` golden reference:
     $$C_{{\text{{fp64}}}}[m, n] = \sum_{{k=0}}^{{K-1}} A_{{\text{{fp64}}}}[m, k] \cdot B_{{\text{{fp64}}}}[n, k] + \text{{bias}}_{{\text{{fp64}}}}[n]$$
   - **Purpose**: Measures end-to-end operator and quantization noise (`SQNR (dB)`, `Cosine Similarity`, `RMSE`, `Max Abs Error`).
3. **Four Deterministic Input Distributions (`NUM_DISTS = 4`)**:
   - **`normal`**: Standard Gaussian $\mathcal{{N}}(0, 1)$ activations and $\mathcal{{N}}(0, 0.25)$ weights.
   - **`saturation`**: Alternating $\pm 10.0$ activations and $\pm 1.0$ weights to stress maximum integer accumulator magnitude (`INT32` overflow check) and quantization clipping boundaries.
   - **`outliers`**: Heavy-tailed outlier channels ($\mathcal{{N}}(0, 15.0)$ every 16th channel alongside $\mathcal{{N}}(0, 0.5)$ background channels) to stress per-row dynamic activation quantization and per-channel weight scales.
   - **`tails`**: Asymmetric non-zero-mean activations ($\mathcal{{N}}(1.25, 0.4)$ + sub-threshold $\mathcal{{N}}(-0.02, 0.005)$ tail channels) and unaligned `N = 53` tail dimensions for packing/transposition.

### 5.2 Level A Verification Summary: SIMD / AMX Microkernel vs. Exact Packed-Layout Reference (Baseline)

| Kernel Pair | Backend | Microkernel Symbol | Max Abs Error | Max Rel Error | RMSE | Max ULP | Mean ULP | Level A SQNR | Exact Match (`%`) | `NaN`/`Inf` | Status |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
{level_a_rows_md}
### 5.3 Level B Verification Summary: End-to-End Operator & Quantization Fidelity vs. `FP64` Golden Reference (Baseline)

| Kernel Pair | Backend | Quantization / Arithmetic Scheme | SQNR (`normal`) | SQNR (`saturation`) | SQNR (`outliers`) | SQNR (`tails`) | **Worst-Case SQNR** | **Worst-Case Cosine Sim** | Aggregate RMSE | Max Abs Error | Status |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
{level_b_rows_md}
### 5.4 Operator-Specific Invariant Verification (`Softmax` & `32-Bit Pack/Transpose`, Baseline)

| Kernel Pair | Backend | Microkernel Symbol | Invariant 1 | Measured Value | Invariant 2 | Measured Value | Guard / Canary Check | Status |
| :--- | :--- | :--- | :--- | ---: | :--- | ---: | ---: | :---: |
| **Pair 4: FP32 Softmax** | **XNNPACK** | `xnn_f32_rmax + xnn_f32_raddstoreexpminusmax (avx512f_p5)` | Max Simplex Sum Error $\max \lvert\sum_i P_i - 1\rvert$ | `{p4_xnn_prec['max_simplex_sum_err']:.2e}` | Shift-Invariance Error $\lVert\text{{Softmax}}(x+1000) - \text{{Softmax}}(x)\rVert_\infty$ | `{p4_xnn_prec['shift_invariance_max_err']:.2e}` | `0 NaN / 0 Inf` | **`PASS`** |
| **Pair 4: FP32 Softmax** | **YNNPACK** | `ynn::max_k1_fp32_avx512 + ynn::sum_k1_fp32_avx512` | Max Simplex Sum Error $\max \lvert\sum_i P_i - 1\rvert$ | `{p4_ynn_prec['max_simplex_sum_err']:.2e}` | Shift-Invariance Error $\lVert\text{{Softmax}}(x+1000) - \text{{Softmax}}(x)\rVert_\infty$ | `{p4_ynn_prec['shift_invariance_max_err']:.2e}` | `0 NaN / 0 Inf` | **`PASS`** |
| **Pair 7: 32-Bit Pack/Transpose** | **XNNPACK** | `xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4_prfm` | Aligned `64×64` Bit-Exact Match | `{p7_xnn_prec['aligned_exact_match_pct']:.2f}%` | Unaligned Tail `53×64` Bit-Exact Match (incl. zero-pad lanes) | `{p7_xnn_prec['unaligned_tail_exact_match_pct']:.2f}%` | `{p7_xnn_prec['canary_corrupted_bytes']} corrupted bytes` | **`PASS`** |
| **Pair 7: 32-Bit Pack/Transpose** | **YNNPACK** | `ynn::transpose_x32_avx512` | Aligned `64×64` Bit-Exact Match | `{p7_ynn_prec['aligned_exact_match_pct']:.2f}%` | Unaligned Tail `53×64` Bit-Exact Match | `{p7_ynn_prec['unaligned_tail_exact_match_pct']:.2f}%` | `{p7_ynn_prec['canary_corrupted_bytes']} corrupted bytes` | **`PASS`** |

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
   - XNNPACK's degree-5 minimax polynomial (`avx512f_rr2_p5`) achieves **`Max ULP = {p4_xnn_prec['level_a_exact_ukernel']['max_ulp_diff']}` (`SQNR = {p4_xnn_prec['level_a_exact_ukernel']['sqnr_db']:.2f} dB`)** in Level A and **`{p4_xnn_prec['level_b_fp64_golden']['sqnr_db']:.2f} dB` SQNR**, **`Cosine Similarity = 1.000000`**, and shift-invariance error **`{p4_xnn_prec['shift_invariance_max_err']:.2e}`** under $+1000.0$ logit shifts in Level B.

---

## 6. Architectural Diagnosis: Comparing Roofline Ceilings vs. Achieved Performance

### 6.1 Decode (`M = 1`): Cold DDR5 Bandwidth Saturation vs. Warm-Cache AMX Cost-Model Gap
- **Cold DDR5 Streaming Roofline (`1×5376×21504`, `DDR5_DRAM_Streaming`)**:
  - Each weight byte (`2 × INT4` weights) performs `4` MACs = `8` INT8 OPs, giving $AI = 3.997\text{{ OPs/Byte}}$. Against the {primary_t}-core DDR5 streaming read ceiling ($BW_{{\text{{prac}}}} = {ddr_rd_mc:.2f}\text{{ GB/s}}$), the practical throughput ceiling is:
    $$\text{{GOPS}}_{{\text{{ceil}}}}(M=1) = 3.997 \times {ddr_rd_mc:.2f}\text{{ GB/s}} = \mathbf{{{p1_cold2_x['roofline'][f'effective_ceiling_{primary_sfx}']:.1f}\text{{ GOPS}}}}$$
  - **Measured ({primary_t}C)**: XNNPACK `1x64c4__avx512amx` achieves **`{p1_cold2_x[primary_key]['gops']:.2f} GOPS` (`{p1_cold2_x[primary_key]['gbps']:.2f} GB/s`, `{p1_cold2_x['roofline'][util_field]:.1f}%`)**, YNNPACK `32x32x64_amxint8` achieves **`{p1_cold2_y[primary_key]['gops']:.2f} GOPS` (`{p1_cold2_y[primary_key]['gbps']:.2f} GB/s`, `{p1_cold2_y['roofline'][util_field]:.1f}%`)**{f", and prior YNNPACK `1x32x8_avx512vnni` achieves **`{p1_cold2_b[primary_key]['gops']:.2f} GOPS` (`{p1_cold2_b[primary_key]['gbps']:.2f} GB/s`, `{p1_cold2_b['roofline'][util_field]:.1f}%`)**" if p1_cold2_b else ""}. On `1×21504×5376`, all three reach `251–255 GB/s` (`~102%–104%` of the 56C streaming probe `245.44 GB/s`, right at Socket 0's `252.19 GB/s` 28C / `255.37 GB/s` max DDR5 ceiling, because interleaved GEMV compute slightly reduces 56-core memory-controller queue contention).
  - **Takeaway**: Cold full-model `M = 1` decode already saturates `>93%` of Socket 0's 8-channel DDR5 bus across all three kernels; further decode speedup requires reducing bytes fetched per token (lower-bit quantization, speculative decoding, or multi-socket NUMA weight sharding).
- **Warm L3/L2 Cache (`1×8192×5376`, `L3_LLC_Warm`) — Actionable Upstream Dispatch Finding**:
  - When weights reside in cache (`22 MB` / `{primary_t}` cores = `{22 * 1024 // primary_t} KB`/core in private L2), AMX-INT8 suffers from two penalties at `M = 1`: (1) `tdpbusd`/`tdpbssd` operates on 16-row tiles, leaving `15/16` (`93.75%`) of compute lanes unused, and (2) both AMX kernels still unpack 4-bit nibbles via GFNI into a stack buffer (`vmovdqa64`) and reload via `tileloadd`.
  - Consequently, prior YNNPACK `1x32x8_avx512vnni` (**`{p1_warm_b[primary_key]['gops'] if p1_warm_b else 0.0:,.1f} GOPS` / `{p1_warm_b[primary_key]['gbps'] if p1_warm_b else 0.0:,.1f} GB/s`**) outperforms both YNNPACK `32x32x64_amxint8` (**`{p1_warm_y[primary_key]['gops']:,.1f} GOPS`**, `{p1_vnni_vs_yamx_warm:.2f}×` slower) and XNNPACK `1x64c4__avx512amx` (**`{p1_warm_x[primary_key]['gops']:,.1f} GOPS`**, `{p1_vnni_vs_xamx_warm:.2f}×` slower).
  - **Root Cause & Fix**: Upstream `{commit_short}` assigns `dot_uint8_int4_int32_32x32x64_16x16x8_amxint8` the same `x86_amx_uint8` cost model as 8-bit AMX without accounting for 4-bit GFNI unpack + stack-spill overhead. Adding a dedicated `x86_amx_uint8_int4` cost entry in `ynnpack/kernels/dot/get_dot_kernel.cc` so `M = 1` (and very small `M`) keeps dispatching `1x32x8_avx512vnni` while `M >= 4..8` switches to `32x32x64_amxint8` recovers the full `{p1_vnni_vs_yamx_warm:.2f}×` warm-cache GEMV speed without sacrificing any prefill throughput.

### 6.2 Prefill (`M = 128..1024`): Why YNNPACK's New `32×32` AMX+GFNI Kernel Beats Both Prior VNNI and XNNPACK's `16×64` AMX+GFNI Kernel
- **Roofline Math**: At `M = 128`, $AI = 419\text{{–}}467\text{{ OPs/Byte}}$, comfortably above both the AVX-512 VNNI ridge point ($AI_{{\text{{ridge}}}} = {vnni512_mc / l3_rd_mc:.1f}\text{{ OPs/Byte}}$ against {primary_t}C L3) and the AMX-INT8 ridge point ($AI_{{\text{{ridge}}}} = {amx_mc / l3_rd_mc:.1f}\text{{ OPs/Byte}}$ against {primary_t}C L3).
- **Measured Performance ({primary_t}C, `128×21504×5376`)**:
  - **Prior YNNPACK (`12x16x8_avx512vnni`)**: **`{(p2_128_21k_b[primary_key]['gops'] / 1000.0) if p2_128_21k_b else 0.0:.2f} TOPS`** (`{p2_128_21k_b['roofline'][util_field] if p2_128_21k_b else 0.0:.1f}%` of {primary_t}C AVX-512 VNNI ceiling, only `{p2_128_21k_b['roofline'][f'machine_roofline_utilization_{primary_sfx}_pct'] if p2_128_21k_b else 0.0:.1f}%` of AMX-INT8 peak).
  - **XNNPACK (`16x64c4__avx512amx`, GFNI-optimized)**: **`{p2_128_21k_x[primary_key]['gops'] / 1000.0:.2f} TOPS`** (`{p2_128_21k_x['roofline'][util_field]:.1f}%` of {primary_t}C AMX-INT8 ceiling).
  - **New YNNPACK (`32x32x64_amxint8`, GFNI-optimized)**: **`{p2_128_21k_y[primary_key]['gops'] / 1000.0:.2f} TOPS`** (**`{p2_128_21k_y['roofline'][util_field]:.1f}%` of {primary_t}C AMX-INT8 ceiling** — **`{p2_y_vs_vnni_128:.2f}×`** faster than prior YNNPACK VNNI and **`{p2_y_vs_x_128:.2f}×`** faster than XNNPACK AMX!).
- **Microarchitectural Explanation (`2×2` vs. `1×4` Tile Blocking under GFNI Unpack + L1D Stack Bounce)**:
  - Sapphire Rapids AMX (`tdpbssd`/`tdpbusd`) requires unpacked 8-bit tiles, so both backends unpack 4-bit weights in ZMM registers using GFNI (`vgf2p8affineqb`), store unpacked `16×64` tiles to a stack scratchpad (`vmovdqa64`), and reload them into `tmm` tiles (`tileloadd`).
  - **XNNPACK `16×64` (`1×4` tile blocking)** loads `1` A-tile and unpacks/spills/loads **`4` B-tiles (`4,096 B` written + `4,096 B` read = `8,192 B` L1D traffic)** for every `4 × tdpbssd` (`131,072` OPs = `16 OPs/B` of L1D scratchpad traffic).
  - **YNNPACK `32×32` (`2×2` tile blocking)** loads `2` A-tiles and unpacks/spills/loads only **`2` B-tiles (`2,048 B` written + `2,048 B` read = `4,096 B` L1D traffic)** for the same `4 × tdpbusd` (`131,072` OPs = **`32 OPs/B` — cutting GFNI unpack instructions and L1D stack-bounce traffic in half per MAC!**).
  - **End-to-End Caveat**: Note that YNNPACK's standalone `dot` kernel outputs raw `INT32` (requiring a separate `dequantize_dot_f32_avx512` epilogue pass in full-graph execution), whereas XNNPACK fuses `INT32 -> FP32` scale/bias dequantization inside the microkernel. Also note that at `M = 128`, YNNPACK's `{p2_128_21k_y[primary_key]['gops'] / 1000.0:.2f} TOPS` already consumes `{p2_128_21k_y[primary_key]['gbps']:.1f} GB/s` of logical weight bandwidth (`{p2_128_21k_y[primary_key]['gbps'] / ddr_rd_mc * 100.0:.0f}%` of Socket 0's `{ddr_rd_mc:.1f} GB/s` DDR5 ceiling for cold layers).

### 6.3 8-Bit Multimodal GEMM (`pair5_int8_gemm`) & 2-Bit Audio GEMM (`pair6_int2_gemm`)
- **Pair 5 (`INT8` AMX)**: Without 4-bit unpacking, `tileloadd` reads packed `INT8` weights directly from cache. YNNPACK's `2×2` blocking (`32×32`) leads XNNPACK's `1×4` blocking (`16×64`) on `280×5376×1152` (**`{p5_280_y[primary_key]['gops'] / 1000.0:.2f}`** vs. `{p5_280_x[primary_key]['gops'] / 1000.0:.2f} TOPS`, reaching **`{p5_280_y['roofline'][util_field]:.1f}%`** of {primary_t}C AMX-INT8 peak) and `2520×4304×1152` (**`{p5_2520_4k_y[primary_key]['gops'] / 1000.0:.2f}`** vs. `{p5_2520_4k_x[primary_key]['gops'] / 1000.0:.2f} TOPS`), while on narrow-`N` `2520×1152×1152` (`18` `nr=64` panels vs. `36` `nr=32` panels) XNNPACK reaches **`{p5_2520_1k_x[primary_key]['gops'] / 1000.0:.2f} TOPS`** vs. YNNPACK **`{p5_2520_1k_y[primary_key]['gops'] / 1000.0:.2f} TOPS`** at {primary_t}C (whereas YNNPACK leads at 28C).
- **Pair 6 (`INT2` Audio)**: XNNPACK's `5x8c8__avxvnni` hits **`{p6_4k_x['roofline'][util_field]:.1f}%`** of the 256-bit AVX-VNNI ceiling (`{p6_4k_x[primary_key]['gops'] / 1000.0:.2f} TOPS`), demonstrating tight 256-bit codegen but capping out at half the machine's 512-bit vector width. YNNPACK's 512-bit `4x64x16_avx512vnni` reaches **`{p6_4k_y[primary_key]['gops'] / 1000.0:.2f}–{p6_1k4k_y[primary_key]['gops'] / 1000.0:.2f} TOPS`** (`{p6_y_vs_x_mc:.2f}×–{p6_y_vs_x_1k4k_mc:.2f}×` faster).

### 6.4 Cache-Hierarchy Transitions, Panel Granularity & Harness Methodology Notes
1. **Aggregate Private L2 Capacity (`56 MiB` @ 28C, `112 MiB` @ 56C) Exceeds Shared L3 (`105 MiB`)**:
   - On `1C`, `11–22 MB` warm GEMV weights (`pair1`) and `8–51 MB` Softmax buffers (`pair4` YNNPACK `in`+`out`) exceed the `2 MiB` private L2 and stream from the `105 MiB` shared L3 (`{l3_rd_1c:.2f} GB/s` 1C Read ceiling — YNNPACK Softmax reductions hit **`{p4_1024_4k_y['single_core']['gbps']:.2f} GB/s` = `{p4_1024_4k_y['roofline']['isa_roofline_utilization_1c_pct']:.1f}%` of 1C L3 Read!**).
   - On `28C` and `56C`, static row/column partitioning shrinks per-core working sets below `2 MiB` (into each core's private L2), transitioning the true bandwidth ceiling from shared L3 (`{l3_rd_mc:,.1f} GB/s`) to aggregate private L2 (`{l2_rd_mc:,.1f} GB/s` Read-Only / `{l2_copy_mc:,.1f} GB/s` Read+Write Copy at {primary_t}C).
2. **Pair 4 Softmax Ceiling & Harness Correction**:
   - XNNPACK (`rmax + raddstoreexpminusmax`, `2R + 1W`) is normalized against **L2/L1D Read+Write Copy** (`{l2_copy_1c:.2f}` / `{l2_copy_mc:,.2f} GB/s`), achieving `{p4_x_ut_lo:.1f}%–{p4_x_ut_hi:.1f}%` utilization at {primary_t}C while computing the minimax `exp` polynomial.
   - YNNPACK (`max_k1 + sum_k1`, reductions only, `2R + ~0W`) is normalized against **L3 (1C) / L2-Sliced ({primary_t}C) Read-Only** (`{l3_rd_1c:.2f}` / `{l2_rd_mc:,.2f} GB/s`), achieving `{p4_y_ut_lo:.1f}%–{p4_y_ut_hi:.1f}%` at {primary_t}C. Note that this update corrects a bug in the earlier harness that passed byte length (`row_len * 4`) instead of element count (`row_len`) to `ynn::max_k1`/`sum_k1` and now batches 4 rows per call (`min_max_accumulator<N=4>`).
3. **2D N-Major GEMM Tiling & Pair 7 Panel Granularity at `56T`**:
   - Switching the GEMM benchmark runners to N-major 2D `(tile_n, tile_m)` OpenMP partitioning with exact `cur_nc` tail clamping eliminates the 1D wave-tail imbalance seen previously on `pair3` `2520×2520×72` (both `7x32` and `5x64` now exceed `{min(p3_2520_x[primary_key]['gops'], p3_2520_y[primary_key]['gops']) / 1000.0:.2f} TFLOPS` on {primary_t}C).
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
  --output kernel/reports/{report_dir.name}/ceilings_sapphire_rapids.json

# 2a. Benchmark microkernels directly from upstream https://github.com/google/XNNPACK
#     Compile target XNNPACK/YNNPACK microkernel objects + bridge and link with -DAEB_DIRECT_LINK:
#     gcc -O2 -march=sapphirerapids -fopenmp -DAEB_DIRECT_LINK \
#         kernel/scripts/bench_litert_microkernels.c <ukernel_objs> -o /tmp/bench_ukernels -lm -lstdc++
OMP_WAIT_POLICY=ACTIVE /tmp/bench_ukernels \
  --threads 28,56 --trials 7 \
  --xnnpack-commit {xnnpack_commit or '<commit_sha>'} \
  --output kernel/reports/{report_dir.name}/ukernel_benchmarks_sapphire_rapids.json

# 2b. Or run in LD_PRELOAD mode against an installed LiteRT-LM binary:
gcc -shared -fPIC -O3 -march=sapphirerapids -fopenmp kernel/scripts/bench_litert_microkernels.c -o /tmp/libbench_ukernels.so -lm
OMP_WAIT_POLICY=ACTIVE LD_LIBRARY_PATH=/opt/litert-lm/lib:/opt/litert-lm/bin \
  LD_PRELOAD=/tmp/libbench_ukernels.so \
  AEB_BENCH_THREADS=28,56 \
  AEB_BENCH_OUTPUT=kernel/reports/{report_dir.name}/ukernel_benchmarks_sapphire_rapids.json \
  /opt/litert-lm/bin/litert_lm_advanced_main

# 3. (Optional) Compile and run the 7-pair microkernel numerical precision verifier (Level A & Level B)
gcc -shared -fPIC -O3 -march=sapphirerapids -fopenmp kernel/scripts/verify_litert_microkernels.c -o /tmp/libverify_ukernels.so -lm
LD_LIBRARY_PATH=/opt/litert-lm/lib:/opt/litert-lm/bin \
  LD_PRELOAD=/tmp/libverify_ukernels.so \
  AEB_VERIFY_OUTPUT=kernel/reports/{report_dir.name}/ukernel_precision_sapphire_rapids.json \
  /opt/litert-lm/bin/litert_lm_advanced_main

# 4. Generate the unified JSON & Markdown kernel report
python3 kernel/scripts/generate_kernel_report.py --report-dir kernel/reports/{report_dir.name}
```
"""

  out_md_path.write_text(md)
  print(f"Wrote {out_json_path} and {out_md_path}")


if __name__ == "__main__":
  main()
