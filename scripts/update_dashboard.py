#!/usr/bin/env python3
"""Generate and update the AI Edge Bench HTML dashboard (dashboards/index.html).

Aggregates benchmark suite results over time from:
  1. reports/history/<suite>/<machine>.jsonl
  2. reports/<report_id>/data.json
  3. results/<machine>/perf/<framework>/<label>/*/meta.json + summary.json
     (for any completed suite runs not yet captured in reports/)

Usage:
  python3 scripts/update_dashboard.py
  python3 dashboards/update_dashboard.py
  python3 scripts/update_dashboard.py --out dashboards/index.html
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import html
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]

COLORS = {
    "llama.cpp": "#d9822b",
    "litert-lm": "#2b6cd9",
    "onnxruntime": "#9c4d9f",
    "xla": "#2e8b57",
    "openvino": "#0095a8",
}

HIST_STYLE = {  # config -> (color, marker, linestyle)
    "llama.cpp/matched": ("#d9822b", "circle", "solid"),
    "llama.cpp/fastest": ("#a85a10", "square", "dashed"),
    "llama.cpp/stock-q4_0": ("#e69138", "triangle", "dotted"),
    "llama.cpp/q8_0": ("#c46d1f", "diamond", "dashdot"),
    "llama.cpp/int8": ("#c46d1f", "diamond", "dashdot"),
    "llama.cpp/bf16": ("#8c4600", "inverted_triangle", "dotted"),
    "litert-lm/matched": ("#2b6cd9", "circle", "solid"),
    "litert-lm/fastest": ("#13408f", "square", "dashed"),
    "litert-lm/ynnpack": ("#5c9aff", "triangle", "dashdot"),
    "xla/matched-ynnpack": ("#2e8b57", "circle", "solid"),
    "xla/matched-onednn": ("#1b5e20", "square", "dashed"),
    "xla/bf16-ynnpack": ("#43a047", "triangle", "solid"),
    "xla/bf16-onednn": ("#2e7d32", "inverted_triangle", "dashed"),
    "xla/int8-ynnpack": ("#66bb6a", "diamond", "dashdot"),
    "xla/int8-onednn": ("#388e3c", "square", "dotted"),
    "xla/fp32-ynnpack": ("#81c784", "cross", "dotted"),
    "xla/fp32-onednn": ("#4caf50", "cross", "dashed"),
    "openvino/int4": ("#0095a8", "square", "dashed"),
    "openvino/int8": ("#006064", "circle", "solid"),
    "openvino/bf16": ("#00acc1", "inverted_triangle", "dashdot"),
    "openvino/fp16": ("#4dd0e1", "triangle", "dotted"),
    "openvino/fp32": ("#26c6da", "diamond", "dotted"),
    "onnxruntime/q4_k_m": ("#9c4d9f", "diamond", "solid"),
    "onnxruntime/fp16": ("#ba68c8", "inverted_triangle", "dotted"),
}

DTYPE_ORDER = {"int4": 0, "int8": 1, "fp16": 2, "bf16": 3, "fp32": 4}
DTYPE_LABELS = {
    "int4": "INT4 / Mixed",
    "int8": "INT8",
    "fp16": "FP16",
    "bf16": "BF16",
    "fp32": "FP32",
}
TRACK_ORDER = {"fastest": 0, "matched": 1, "reference": 2}


def _read_json(path: Path):
  try:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
  except (OSError, json.JSONDecodeError):
    return None


def _describe(vals: list[float]) -> dict | None:
  clean = [float(x) for x in vals if x is not None]
  if not clean:
    return None
  med = statistics.median(clean)
  mean = statistics.fmean(clean)
  sd = statistics.stdev(clean) if len(clean) > 1 else 0.0
  return {
      "n": len(clean),
      "median": med,
      "min": min(clean),
      "max": max(clean),
      "cv": (sd / mean) if mean else 0.0,
  }


def _infer_dtype_info(suite_cfg: dict | None, cid: str, meta: dict | None) -> dict:
  suite_cfg = suite_cfg or {}
  meta = meta or {}
  cfg_dict = next((c for c in suite_cfg.get("configs", []) if c.get("id") == cid), {})
  raw_dtype = (cfg_dict.get("dtype") or "").lower()
  cid_lo = cid.lower()
  mp = meta.get("model_path") or ""
  cmd = [str(x) for x in (meta.get("driver_cmd") or cfg_dict.get("args") or [])]
  info = (meta.get("driver_ready") or {}).get("info") or {}
  fw = meta.get("framework") or cfg_dict.get("framework") or cid.split("/")[0]

  if raw_dtype == "fp32" or "fp32" in cid_lo:
    group = "fp32"
  elif raw_dtype == "bf16" or "bf16" in cid_lo:
    group = "bf16"
  elif raw_dtype == "fp16" or "fp16" in cid_lo:
    group = "fp16"
  elif raw_dtype in ("int8", "q8_0") or "int8" in cid_lo or "q8_0" in cid_lo:
    group = "int8"
  else:
    group = "int4"

  threads = info.get("threads")
  if threads is None and "--threads" in cmd:
    idx = cmd.index("--threads")
    if idx + 1 < len(cmd):
      t_val = cmd[idx + 1]
      if not str(t_val).startswith("{"):
        threads = t_val
  threads_str = str(threads) if threads is not None else "—"

  if "qat-mobile" in mp or "litert-lm" in mp or fw == "litert-lm":
    checkpoint = "qat-mobile"
  elif fw == "llama.cpp" or mp.endswith(".gguf"):
    checkpoint = "it (GGUF)"
  elif fw == "openvino" or "-ov" in mp:
    checkpoint = "it (OV)"
  elif fw == "onnxruntime" or "onnx" in mp:
    checkpoint = "it (ONNX)"
  else:
    checkpoint = "it (HF)"

  if fw == "xla":
    if group == "fp32":
      wak = "fp32 / fp32 / fp32"
    elif group == "bf16":
      wak = "bf16 / bf16 / bf16"
    elif group == "int8":
      wak = "int8 / int8 / bf16"
    else:
      wak = "int2_int4_int8 / int8 / bf16"
  elif fw == "openvino":
    if group == "fp32":
      wak = "fp32 / fp32 / fp16"
    elif group == "bf16":
      wak = "bf16 / bf16 / bf16"
    elif group == "fp16":
      wak = "fp16 / fp16 / fp16"
    elif group == "int8":
      wak = "int8_asym / bf16 / int8"
    else:
      wak = "int4_asym / bf16 / int8"
  elif fw == "litert-lm":
    wak = "int2_int4_int8 / int8 / int8"
  elif fw == "onnxruntime":
    wak = "fp16 / fp16 / fp16" if group == "fp16" else "q4_k_m / fp32 / fp32"
  elif fw == "llama.cpp":
    kv = str(info.get("cache_type_k") or "")
    if not kv and "--cache-type-k" in cmd:
      idx = cmd.index("--cache-type-k")
      if idx + 1 < len(cmd):
        kv = cmd[idx + 1]
    kv = "fp16" if kv in ("", "f16", "fp16") else kv
    if group == "fp32":
      wak = f"fp32 / fp32 / {kv}"
    elif group == "bf16":
      wak = f"bf16 / fp32 / {kv}"
    elif group == "int8":
      wak = f"q8_0 / q8_0 / {kv}"
    elif "matched-q4" in mp or cid_lo.endswith("/fastest"):
      wak = f"q4_0_q8_0 / q8_0 / {kv}"
    elif "matched" in mp or cid_lo.endswith("/matched"):
      wak = f"q2_k_q4_0_q8_0 / q8_0 / {kv}"
    else:
      wak = f"q4_0 / q8_0 / {kv}"
  else:
    wak = group

  track = cfg_dict.get("track") or meta.get("track") or "reference"
  return {
      "dtype": group,
      "dtype_label": DTYPE_LABELS.get(group, group.upper()),
      "checkpoint": checkpoint,
      "weights_acts_kv": wak,
      "threads": threads_str,
      "track": track,
      "framework": fw,
  }


def _find_report_link(reports_dir: Path, report_id: str) -> str | None:
  rep_dir = reports_dir / report_id
  if not rep_dir.is_dir():
    return None
  for candidate in ("index.html", "README.html", "README.md"):
    if (rep_dir / candidate).is_file():
      return f"../reports/{report_id}/{candidate}"
  return None


def collect_dashboard_data(
    suites_dir: Path,
    reports_dir: Path,
    results_dir: Path,
) -> dict:
  """Collect all benchmark suite historical series from suites, reports, and results."""
  suite_defs: dict[str, dict] = {}
  if suites_dir.is_dir():
    for sp in sorted(suites_dir.glob("*.json")):
      s_obj = _read_json(sp)
      if s_obj and s_obj.get("name"):
        suite_defs[s_obj["name"]] = s_obj

  # Map: (suite_name, machine) -> {(report_id, config_id): entry_dict}
  records_by_suite: dict[tuple[str, str], dict[tuple[str, str], dict]] = collections.defaultdict(dict)
  config_meta_by_suite: dict[tuple[str, str], dict[str, dict]] = collections.defaultdict(dict)
  machine_notes_by_suite: dict[tuple[str, str], str] = {}

  # 1. Load reports/history/<suite>/<machine>.jsonl
  hist_root = reports_dir / "history"
  if hist_root.is_dir():
    for hp in sorted(hist_root.glob("*/*.jsonl")):
      for line in hp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
          continue
        try:
          row = json.loads(line)
        except json.JSONDecodeError:
          continue
        suite_name = row.get("suite") or hp.parent.name
        machine = row.get("machine") or hp.stem
        report_id = row.get("report_id") or f"{row.get('date')}-{suite_name}-{machine}"
        cid = row.get("config")
        if not cid or not row.get("date"):
          continue
        key = (suite_name, machine)
        records_by_suite[key][(report_id, cid)] = {
            "report_id": report_id,
            "date": row["date"],
            "suite": suite_name,
            "machine": machine,
            "config": cid,
            "track": row.get("track"),
            "framework_ref": row.get("framework_ref"),
            "framework_commit": row.get("framework_commit"),
            "harness_sha256": row.get("harness_sha256"),
            "repo_rev": row.get("repo_rev"),
            "model_sha256": row.get("model_sha256"),
            "n_requests": row.get("n_requests", 0),
            "metrics": row.get("metrics") or {},
            "accuracy": row.get("accuracy") or {},
            "reproducibility": row.get("reproducibility"),
            "report_url": _find_report_link(reports_dir, report_id),
        }

  # 2. Load reports/<report_id>/data.json to enrich metadata and pick up any reports not in history
  if reports_dir.is_dir():
    for dp in sorted(reports_dir.glob("*/data.json")):
      rep_data = _read_json(dp)
      if not rep_data:
        continue
      rep_dir = dp.parent
      report_id = rep_data.get("report_id") or rep_dir.name
      suite_obj = rep_data.get("suite") or {}
      suite_name = suite_obj.get("name")
      machine = rep_data.get("machine")
      if not suite_name or not machine:
        continue
      if suite_name not in suite_defs and suite_obj:
        suite_defs[suite_name] = suite_obj

      m_Date = report_id[:10] if len(report_id) >= 10 and report_id[4] == "-" else ""
      key = (suite_name, machine)
      configs_map = rep_data.get("configs") or {}
      acc_rows = rep_data.get("accuracy") or []

      # Build accuracy lookup (including shared framework + model_sha256)
      acc_by_id: dict[tuple[str, str], dict] = {}
      acc_by_model: dict[tuple[str, str, str], dict] = {}
      for a in acc_rows:
        aid = a.get("id")
        task = a.get("task")
        if not aid or not task:
          continue
        acc_by_id[(aid, task)] = {"acc": a.get("accuracy"), "n": a.get("n")}
        c_runs = (configs_map.get(aid) or {}).get("runs") or []
        if c_runs:
          m0 = c_runs[0].get("meta") or {}
          fw = m0.get("framework") or aid.split("/")[0]
          sha = m0.get("model_sha256")
          if fw and sha:
            acc_by_model[(fw, sha, task)] = {"acc": a.get("accuracy"), "n": a.get("n")}

      for cid, c_entry in configs_map.items():
        c_runs = c_entry.get("runs") or []
        agg = c_entry.get("aggregate") or {}
        m0 = (c_runs[0].get("meta") if c_runs else {}) or {}
        if m0.get("machine_notes") and key not in machine_notes_by_suite:
          machine_notes_by_suite[key] = re.sub(
              r"Platinum [0-9]{4}C|[0-9]{4}[A-Z]-C",
              "Sapphire Rapids",
              str(m0["machine_notes"]),
          )

        dtype_info = _infer_dtype_info(suite_defs.get(suite_name), cid, m0)
        config_meta_by_suite[key][cid] = dtype_info

        date_str = m_Date or (m0.get("started_at") or "")[:10]
        if not date_str:
          continue

        # Resolve accuracy for cid
        fw = m0.get("framework") or cid.split("/")[0]
        sha = m0.get("model_sha256")
        resolved_acc = {}
        for task in ("mmlu", "gsm8k"):
          if (cid, task) in acc_by_id:
            resolved_acc[task] = acc_by_id[(cid, task)]
          elif fw and sha and (fw, sha, task) in acc_by_model:
            resolved_acc[task] = acc_by_model[(fw, sha, task)]

        rec_key = (report_id, cid)
        if rec_key in records_by_suite[key]:
          rec = records_by_suite[key][rec_key]
          if not rec.get("accuracy") and resolved_acc:
            rec["accuracy"] = resolved_acc
          else:
            for t_k, t_v in resolved_acc.items():
              rec["accuracy"].setdefault(t_k, t_v)
          rec["threads"] = dtype_info["threads"]
          rec["dtype"] = dtype_info["dtype"]
          rec["checkpoint"] = dtype_info["checkpoint"]
          rec["weights_acts_kv"] = dtype_info["weights_acts_kv"]
        else:
          build = m0.get("build") or {}
          e2e_d = agg.get("e2e_ms") or {}
          records_by_suite[key][rec_key] = {
              "report_id": report_id,
              "date": date_str,
              "suite": suite_name,
              "machine": machine,
              "config": cid,
              "track": dtype_info["track"],
              "framework_ref": build.get("ref"),
              "framework_commit": build.get("commit"),
              "harness_sha256": build.get("harness_sha256"),
              "repo_rev": build.get("repo_rev"),
              "model_sha256": m0.get("model_sha256"),
              "n_requests": e2e_d.get("n", 0),
              "metrics": {
                  k: agg.get(k)
                  for k in (
                      "e2e_ms",
                      "ttft_ms",
                      "prefill_tps",
                      "decode_tps",
                      "itl_p99_ms",
                      "cpu_cores_prefill",
                      "cpu_cores_decode",
                      "cpu_seconds",
                      "rss_max_mb",
                      "pss_max_mb",
                  )
                  if agg.get(k)
              },
              "accuracy": resolved_acc,
              "reproducibility": None,
              "report_url": _find_report_link(reports_dir, report_id),
              "threads": dtype_info["threads"],
              "dtype": dtype_info["dtype"],
              "checkpoint": dtype_info["checkpoint"],
              "weights_acts_kv": dtype_info["weights_acts_kv"],
          }

  # 3. Scan results/<machine>/perf/*/*/*/meta.json for any new suite runs on dates
  #    not yet represented in reports_dir / history.
  if results_dir.is_dir():
    existing_dates_by_suite: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for key, recs in records_by_suite.items():
      for r in recs.values():
        existing_dates_by_suite[key].add(r["date"])

    raw_grouped: dict[tuple[str, str, str, str], list[tuple[dict, dict]]] = collections.defaultdict(list)
    for meta_p in sorted(results_dir.glob("*/perf/*/*/*/meta.json")):
      if ".bak" in str(meta_p) or meta_p.parts[-5] == "_cache":
        continue
      summ_p = meta_p.parent / "summary.json"
      if not summ_p.is_file():
        continue
      meta = _read_json(meta_p)
      summ = _read_json(summ_p)
      if not meta or not summ or summ.get("status") != "ok":
        continue
      if meta.get("track") in ("prime", "sweep-threads", "sweep-sequence", "prefix-cache"):
        continue
      notes = str(meta.get("notes") or "")
      if not notes.startswith("suite="):
        continue
      suite_name = notes.split("=", 1)[1].strip()
      machine = meta.get("machine") or meta_p.parts[-5]
      date_str = str(meta.get("started_at") or "")[:10]
      if not suite_name or not machine or not date_str:
        continue
      if date_str in existing_dates_by_suite[(suite_name, machine)]:
        continue
      cid = f"{meta.get('framework')}/{meta.get('label')}"
      raw_grouped[(suite_name, machine, date_str, cid)].append((meta, summ))

    for (suite_name, machine, date_str, cid), run_pairs in sorted(raw_grouped.items()):
      key = (suite_name, machine)
      suite_obj = suite_defs.get(suite_name) or {}
      rounds = int(suite_obj.get("rounds", 3))
      run_pairs.sort(key=lambda pair: pair[0].get("started_at", ""))
      selected = run_pairs[-rounds:]
      m0 = selected[0][0]
      if m0.get("machine_notes") and key not in machine_notes_by_suite:
        machine_notes_by_suite[key] = re.sub(
            r"Platinum [0-9]{4}C|[0-9]{4}[A-Z]-C",
            "Sapphire Rapids",
            str(m0["machine_notes"]),
        )

      dtype_info = _infer_dtype_info(suite_obj, cid, m0)
      config_meta_by_suite[key].setdefault(cid, dtype_info)

      metrics_agg = {}
      for m_key in ("e2e_ms", "ttft_ms", "prefill_tps", "decode_tps", "itl_p99_ms", "rss_max_mb", "pss_max_mb"):
        pooled_vals = []
        for _, s in selected:
          for req in s.get("per_request") or []:
            if req.get("measured") and req.get(m_key) is not None:
              pooled_vals.append(float(req[m_key]))
        desc = _describe(pooled_vals)
        if desc:
          metrics_agg[m_key] = desc

      if not metrics_agg.get("e2e_ms"):
        continue

      report_id = f"{date_str}-{suite_name}-{machine}"
      build = m0.get("build") or {}
      records_by_suite[key][(report_id, cid)] = {
          "report_id": report_id,
          "date": date_str,
          "suite": suite_name,
          "machine": machine,
          "config": cid,
          "track": dtype_info["track"],
          "framework_ref": build.get("ref"),
          "framework_commit": build.get("commit"),
          "harness_sha256": build.get("harness_sha256"),
          "repo_rev": build.get("repo_rev"),
          "model_sha256": m0.get("model_sha256"),
          "n_requests": metrics_agg["e2e_ms"]["n"],
          "metrics": metrics_agg,
          "accuracy": {},
          "reproducibility": None,
          "report_url": _find_report_link(reports_dir, report_id),
          "threads": dtype_info["threads"],
          "dtype": dtype_info["dtype"],
          "checkpoint": dtype_info["checkpoint"],
          "weights_acts_kv": dtype_info["weights_acts_kv"],
      }

  # 4. Assemble final structured suite list
  suites_out = []
  def _suite_sort_key(k: tuple[str, str]):
    vals = records_by_suite[k].values()
    n_reps = len({r["report_id"] for r in vals})
    max_d = max((r["date"] for r in vals), default="1970-01-01")
    try:
      d_ord = dt.date.fromisoformat(max_d).toordinal()
    except ValueError:
      d_ord = 0
    return (-n_reps, -d_ord, k[0], k[1])

  sorted_keys = sorted(records_by_suite.keys(), key=_suite_sort_key)

  for suite_name, machine in sorted_keys:
    key = (suite_name, machine)
    suite_obj = suite_defs.get(suite_name) or {}
    recs = list(records_by_suite[key].values())
    if not recs:
      continue

    # Propagate accuracy across configs sharing (framework, model_sha256) within the same report_id
    acc_by_rep_model: dict[tuple[str, str, str], dict] = {}
    for r in recs:
      fw = r["config"].split("/")[0]
      sha = r.get("model_sha256")
      if sha and r.get("accuracy"):
        for t_k, t_v in r["accuracy"].items():
          acc_by_rep_model[(r["report_id"], fw, sha, t_k)] = t_v
    for r in recs:
      fw = r["config"].split("/")[0]
      sha = r.get("model_sha256")
      if sha:
        for t_k in ("mmlu", "gsm8k"):
          if t_k not in r["accuracy"] and (r["report_id"], fw, sha, t_k) in acc_by_rep_model:
            r["accuracy"][t_k] = acc_by_rep_model[(r["report_id"], fw, sha, t_k)]

    reports_meta = {}
    for r in recs:
      rid = r["report_id"]
      if rid not in reports_meta:
        reports_meta[rid] = {
            "report_id": rid,
            "date": r["date"],
            "report_url": r.get("report_url"),
        }
    reports_list = sorted(reports_meta.values(), key=lambda x: (x["date"], x["report_id"]))

    by_cfg: dict[str, list[dict]] = collections.defaultdict(list)
    for r in recs:
      by_cfg[r["config"]].append(r)

    configs_out = []
    for cid, c_pts in by_cfg.items():
      c_pts.sort(key=lambda x: (x["date"], x["report_id"]))
      c_meta = config_meta_by_suite[key].get(cid) or _infer_dtype_info(suite_obj, cid, None)
      fw = c_meta["framework"]
      col, marker, ls = HIST_STYLE.get(cid, (COLORS.get(fw, "#59636e"), "circle", "solid"))

      def _stat_obj(m_dict: dict | None, n_fallback: int, ndigits: int = 2) -> dict | None:
        if not m_dict or m_dict.get("median") is None:
          return None
        med = float(m_dict["median"])
        return {
            "median": round(med, ndigits),
            "min": round(float(m_dict.get("min", med)), ndigits),
            "max": round(float(m_dict.get("max", med)), ndigits),
            "cv": round(float(m_dict.get("cv") or 0.0), 5),
            "n": int(m_dict.get("n") or n_fallback or 0),
        }

      points_out = []
      for pt in c_pts:
        metrics = pt.get("metrics") or {}
        n_req = int(pt.get("n_requests") or 0)
        prefill_obj = _stat_obj(metrics.get("prefill_tps"), n_req, 2)
        decode_obj = _stat_obj(metrics.get("decode_tps"), n_req, 2)
        e2e_obj = _stat_obj(metrics.get("e2e_ms"), n_req, 2)
        if not prefill_obj and not decode_obj and not e2e_obj:
          continue
        ttft = metrics.get("ttft_ms") or {}
        rss = metrics.get("rss_max_mb") or {}
        itl = metrics.get("itl_p99_ms") or {}
        acc = pt.get("accuracy") or {}
        points_out.append({
            "report_id": pt["report_id"],
            "date": pt["date"],
            "prefill_tps": prefill_obj,
            "decode_tps": decode_obj,
            "e2e_ms": e2e_obj,
            "ttft_ms": round(float(ttft["median"]), 1) if ttft.get("median") is not None else None,
            "itl_p99_ms": round(float(itl["median"]), 1) if itl.get("median") is not None else None,
            "rss_max_mb": round(float(rss["median"]), 0) if rss.get("median") is not None else None,
            "mmlu": round(float(acc["mmlu"]["acc"]), 4) if acc.get("mmlu") and acc["mmlu"].get("acc") is not None else None,
            "gsm8k": round(float(acc["gsm8k"]["acc"]), 4) if acc.get("gsm8k") and acc["gsm8k"].get("acc") is not None else None,
            "framework_ref": pt.get("framework_ref"),
            "framework_commit": pt.get("framework_commit"),
            "report_url": pt.get("report_url"),
        })

      if not points_out:
        continue

      def _latest_and_delta(metric_key: str):
        valid = [p[metric_key]["median"] for p in points_out if p.get(metric_key) and p[metric_key].get("median") is not None]
        if not valid:
          return None, None, None
        latest_v = valid[-1]
        prev_v = valid[-2] if len(valid) >= 2 else None
        d_abs = round(latest_v - prev_v, 2) if prev_v is not None else None
        d_pct = round((latest_v - prev_v) / prev_v * 100.0, 2) if prev_v else None
        return latest_v, d_abs, d_pct

      latest_prefill, delta_prefill_tps, delta_prefill_pct = _latest_and_delta("prefill_tps")
      latest_decode, delta_decode_tps, delta_decode_pct = _latest_and_delta("decode_tps")
      latest_e2e, delta_ms, delta_pct = _latest_and_delta("e2e_ms")

      configs_out.append({
          "id": cid,
          "framework": fw,
          "track": c_meta["track"],
          "dtype": c_meta["dtype"],
          "dtype_label": c_meta["dtype_label"],
          "checkpoint": c_meta["checkpoint"],
          "weights_acts_kv": c_meta["weights_acts_kv"],
          "threads": c_meta["threads"],
          "color": col,
          "marker": marker,
          "linestyle": ls,
          "latest_prefill_tps": latest_prefill,
          "delta_prefill_tps": delta_prefill_tps,
          "delta_prefill_pct": delta_prefill_pct,
          "latest_decode_tps": latest_decode,
          "delta_decode_tps": delta_decode_tps,
          "delta_decode_pct": delta_decode_pct,
          "latest_e2e_ms": latest_e2e,
          "delta_ms": delta_ms,
          "delta_pct": delta_pct,
          "points": points_out,
      })

    configs_out.sort(
        key=lambda c: (
            DTYPE_ORDER.get(c["dtype"], 9),
            -(c["latest_decode_tps"] if c["latest_decode_tps"] is not None else -1.0),
            c["id"],
        )
    )

    workload = suite_obj.get("workload") or {}
    suites_out.append({
        "id": f"{suite_name}--{machine}",
        "suite": suite_name,
        "machine": machine,
        "machine_notes": machine_notes_by_suite.get(key, ""),
        "description": suite_obj.get("description", ""),
        "workload": {
            "prompt_tokens": workload.get("prompt_tokens", 1024),
            "gen_tokens": workload.get("gen_tokens", 128),
            "ctx": workload.get("ctx", 4096),
            "rounds": suite_obj.get("rounds", 3),
            "repetitions": workload.get("repetitions", 3),
        },
        "reports": reports_list,
        "configs": configs_out,
    })

  return {
      "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
      "suites": suites_out,
  }


def render_dashboard_html(dashboard_data: dict) -> str:
  data_json = json.dumps(dashboard_data, indent=2)
  generated_at = html.escape(dashboard_data.get("generated_at", ""))
  n_suites = len(dashboard_data.get("suites", []))
  n_configs = sum(len(s.get("configs", [])) for s in dashboard_data.get("suites", []))
  n_reports = sum(len(s.get("reports", [])) for s in dashboard_data.get("suites", []))

  return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Edge Bench — Benchmark Suites Over Time (Prefill &amp; Decode tokens/s)</title>
  <style>
    :root {{
      --bg: #f6f8fa;
      --surface: #ffffff;
      --surface-alt: #f8fafc;
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
      --code-bg: #eff1f3;
      --shadow: 0 1px 3px rgba(31, 35, 40, 0.06), 0 1px 2px rgba(31, 35, 40, 0.04);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      padding: 0 0 4rem;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
      font-size: 14.5px;
      line-height: 1.55;
      color: var(--fg);
      background: var(--bg);
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 0.86em;
      padding: 0.15em 0.38em;
      background: var(--code-bg);
      border-radius: 4px;
    }}
    header.top-bar {{
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 1.5rem 2rem 1.25rem;
      position: sticky;
      top: 0;
      z-index: 30;
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
    .brand-title {{
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }}
    .brand-title h1 {{
      margin: 0;
      font-size: 1.5rem;
      font-weight: 700;
      letter-spacing: -0.01em;
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }}
    .brand-subtitle {{
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .header-meta {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.75rem;
      font-size: 0.84rem;
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
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
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
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
      font-weight: 600;
    }}
    .kpi-value {{
      font-size: 1.55rem;
      font-weight: 700;
      margin-top: 0.2rem;
      color: var(--fg);
    }}
    .kpi-sub {{
      font-size: 0.82rem;
      color: var(--muted);
      margin-top: 0.15rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .suite-nav {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.75rem 1rem;
      margin-bottom: 1.5rem;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      box-shadow: var(--shadow);
    }}
    .suite-tabs {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.45rem;
    }}
    .suite-tab {{
      appearance: none;
      border: 1px solid var(--border);
      background: var(--surface-alt);
      color: var(--fg);
      font-size: 0.86rem;
      font-weight: 600;
      padding: 0.42rem 0.85rem;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.12s ease;
    }}
    .suite-tab:hover {{
      border-color: var(--accent);
      color: var(--accent);
    }}
    .suite-tab.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
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
      font-size: 1.3rem;
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
    .badge-workload {{
      background: #f0f4f8;
      border-color: #d0d7de;
      color: #32383f;
    }}
    .suite-meta-notes {{
      margin-top: 0.4rem;
      font-size: 0.85rem;
      color: var(--muted);
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
      gap: 1rem 1.5rem;
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
    .ctrl-btn:hover {{
      border-color: var(--accent);
    }}
    .ctrl-btn.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
      font-weight: 600;
    }}
    .chart-section {{
      padding: 1.25rem 1.5rem 0.75rem;
    }}
    .charts-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 1.25rem;
    }}
    .charts-grid.side-by-side {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    @media (max-width: 1180px) {{
      .charts-grid.side-by-side {{
        grid-template-columns: 1fr;
      }}
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
      font-size: 1rem;
      font-weight: 700;
      color: var(--fg);
      display: flex;
      align-items: center;
      gap: 0.45rem;
    }}
    .chart-panel-subtitle {{
      font-size: 0.79rem;
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
      gap: 0.4rem 0.65rem;
      margin-top: 0.9rem;
      padding: 0.65rem 0.85rem;
      background: var(--surface-alt);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      font-size: 0.8rem;
      padding: 0.2rem 0.5rem;
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
    .table-section {{
      padding: 0.5rem 1.5rem 1.5rem;
    }}
    .table-section h3 {{
      font-size: 1.02rem;
      margin: 0.75rem 0 0.6rem;
      color: var(--fg);
      display: flex;
      align-items: center;
      justify-content: space-between;
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
      font-size: 13px;
      white-space: nowrap;
    }}
    th, td {{
      padding: 0.48rem 0.7rem;
      border-bottom: 1px solid var(--border-subtle);
      border-right: 1px solid var(--border-subtle);
      text-align: left;
      vertical-align: middle;
    }}
    th:last-child, td:last-child {{
      border-right: none;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    thead th {{
      background: var(--surface-alt);
      font-weight: 600;
      color: var(--fg);
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    tbody tr:hover {{
      background: #f0f6fc;
    }}
    tbody tr.group-row td {{
      background: #eef2f7;
      font-weight: 700;
      font-size: 12.5px;
      color: var(--fg);
      letter-spacing: 0.02em;
      border-top: 1.5px solid var(--border);
      border-bottom: 1px solid var(--border);
      padding: 0.4rem 0.7rem;
    }}
    tbody tr.best-row {{
      background: #f0fdf4;
    }}
    tbody tr.best-row:hover {{
      background: #dcfce7;
    }}
    tbody tr.best-row td:first-child {{
      border-left: 3.5px solid #16a34a;
    }}
    .num {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    .sub-range {{
      display: block;
      font-size: 11px;
      color: var(--muted);
      font-weight: 400;
    }}
    .delta-pill {{
      display: inline-block;
      padding: 0.1rem 0.42rem;
      border-radius: 4px;
      font-size: 11.5px;
      font-weight: 600;
    }}
    .delta-good {{
      background: var(--success-bg);
      color: var(--success);
    }}
    .delta-bad {{
      background: var(--danger-bg);
      color: var(--danger);
    }}
    .delta-neutral {{
      background: var(--code-bg);
      color: var(--muted);
    }}
    .chart-tooltip {{
      position: fixed;
      pointer-events: none;
      z-index: 100;
      background: rgba(24, 28, 34, 0.96);
      color: #f0f6fc;
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 8px;
      padding: 0.65rem 0.85rem;
      font-size: 12.5px;
      line-height: 1.45;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
      max-width: 360px;
      display: none;
    }}
    .tt-title {{
      font-weight: 700;
      font-size: 13.5px;
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
    .tt-key {{
      color: #9198a1;
    }}
    .tt-val {{
      text-align: right;
      font-weight: 600;
      font-variant-numeric: tabular-nums;
    }}
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
    @media (max-width: 768px) {{
      header.top-bar, main {{
        padding-left: 1rem;
        padding-right: 1rem;
      }}
    }}
  </style>
</head>
<body>
  <header class="top-bar">
    <div class="top-bar-inner">
      <div class="brand-title">
        <h1>
          <span>AI Edge Bench — Suite Throughput Over Time (tokens/s)</span>
        </h1>
        <div class="brand-subtitle">
          Prefill and Decode throughput (tokens/s on y-axis, higher is better ↑) over time (benchmark date on x-axis) across all benchmark suites
        </div>
      </div>
      <div class="header-meta">
        <span class="meta-pill">Updated: {generated_at}</span>
        <span>Update command: <code>python3 scripts/update_dashboard.py</code></span>
      </div>
    </div>
  </header>

  <main>
    <div class="kpi-grid" id="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Benchmark Suites</div>
        <div class="kpi-value">{n_suites}</div>
        <div class="kpi-sub">Tracked across x86 AMX &amp; ARM CPUs ({n_reports} snapshots)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Configurations Tracked</div>
        <div class="kpi-value">{n_configs}</div>
        <div class="kpi-sub">Across FP32, BF16, INT8, INT4/Mixed</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Fastest Prefill Throughput</div>
        <div class="kpi-value" id="kpi-prefill-val">—</div>
        <div class="kpi-sub" id="kpi-prefill-sub">—</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Fastest Decode Throughput</div>
        <div class="kpi-value" id="kpi-decode-val">—</div>
        <div class="kpi-sub" id="kpi-decode-sub">—</div>
      </div>
    </div>

    <div class="suite-nav">
      <div class="suite-tabs" id="suite-tabs"></div>
      <div style="font-size: 0.82rem; color: var(--muted);">
        Tip: Hover points for full breakdown · Click legend items to toggle series · Double-click legend to solo
      </div>
    </div>

    <div id="suites-root"></div>

    <div class="update-banner">
      <div>
        <strong>Updating this dashboard:</strong> As new benchmark runs or reports become available, run
        <code>python3 scripts/update_dashboard.py</code> (or <code>./dashboards/update_dashboard.py</code> / <code>make dashboard</code>)
        to refresh <code>dashboards/index.html</code> and <code>dashboards/data.json</code>.
      </div>
      <div>
        <a href="data.json">Raw Dashboard JSON (data.json)</a>
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
    const suites = rawData.suites || [];
    const tooltip = document.getElementById("chart-tooltip");

    const PREFILL_SPEC = {{
      key: "prefill_tps",
      title: "Prefill Throughput Over Time",
      yAxisLabel: "Prefill Throughput (tokens/s)",
      unit: "tok/s",
      decimals: 1,
    }};

    const DECODE_SPEC = {{
      key: "decode_tps",
      title: "Decode Throughput Over Time",
      yAxisLabel: "Decode Throughput (tokens/s)",
      unit: "tok/s",
      decimals: 2,
    }};

    // Populate Fastest Prefill & Fastest Decode KPI cards
    let bestPrefill = null;
    let bestDecode = null;
    for (const s of suites) {{
      for (const c of s.configs || []) {{
        if (c.latest_prefill_tps != null && (!bestPrefill || c.latest_prefill_tps > bestPrefill.val)) {{
          bestPrefill = {{ val: c.latest_prefill_tps, cid: c.id, suite: s.suite, machine: s.machine }};
        }}
        if (c.latest_decode_tps != null && (!bestDecode || c.latest_decode_tps > bestDecode.val)) {{
          bestDecode = {{ val: c.latest_decode_tps, cid: c.id, suite: s.suite, machine: s.machine }};
        }}
      }}
    }}
    if (bestPrefill) {{
      document.getElementById("kpi-prefill-val").textContent = `${{bestPrefill.val.toLocaleString(undefined, {{ minimumFractionDigits: 1, maximumFractionDigits: 1 }})}} tok/s`;
      document.getElementById("kpi-prefill-sub").textContent = `${{bestPrefill.cid}} on ${{bestPrefill.machine}}`;
    }}
    if (bestDecode) {{
      document.getElementById("kpi-decode-val").textContent = `${{bestDecode.val.toFixed(2)}} tok/s`;
      document.getElementById("kpi-decode-sub").textContent = `${{bestDecode.cid}} on ${{bestDecode.machine}}`;
    }}

    // Per-suite interactive state
    const suiteStates = {{}};
    for (const s of suites) {{
      suiteStates[s.id] = {{
        yScale: "linear",       // "linear" | "log"
        xSpacing: "time",       // "time" | "ordinal"
        layout: "stacked",      // "stacked" | "side-by-side"
        dtypeFilter: "all",     // "all" | "int4" | "int8" | "bf16" | "fp32"
        trackFilter: "all",     // "all" | "fastest" | "matched" | "reference"
        frameworkFilter: "all", // "all" | framework name
        showWhiskers: true,
        hiddenConfigs: new Set(),
        hoveredConfig: null,
      }};
    }}

    let activeSuiteFilter = "all";

    function renderSuiteTabs() {{
      const tabsEl = document.getElementById("suite-tabs");
      tabsEl.innerHTML = "";
      const allBtn = document.createElement("button");
      allBtn.className = "suite-tab" + (activeSuiteFilter === "all" ? " active" : "");
      allBtn.textContent = `All Suites (${{suites.length}})`;
      allBtn.onclick = () => {{
        activeSuiteFilter = "all";
        renderSuiteTabs();
        renderAllSuites();
      }};
      tabsEl.appendChild(allBtn);

      for (const s of suites) {{
        const btn = document.createElement("button");
        btn.className = "suite-tab" + (activeSuiteFilter === s.id ? " active" : "");
        btn.textContent = `${{s.suite}} (${{s.machine}})`;
        btn.onclick = () => {{
          activeSuiteFilter = s.id;
          renderSuiteTabs();
          renderAllSuites();
        }};
        tabsEl.appendChild(btn);
      }}
    }}

    function dashArrayFor(ls) {{
      if (ls === "dashed") return "7,4";
      if (ls === "dotted") return "2.5,3.5";
      if (ls === "dashdot") return "7,3.5,2,3.5";
      return "";
    }}

    function formatTps(val, decimals) {{
      if (val == null || !isFinite(val)) return "—";
      return Number(val).toLocaleString(undefined, {{
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      }});
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
      if (markerType === "inverted_triangle") {{
        const el = document.createElementNS(ns, "polygon");
        const h = r * 2.0;
        const pts = `${{cx}},${{cy + h * 0.62}} ${{cx - h * 0.58}},${{cy - h * 0.45}} ${{cx + h * 0.58}},${{cy - h * 0.45}}`;
        el.setAttribute("points", pts);
        el.setAttribute("fill", color);
        el.setAttribute("stroke", "#ffffff");
        el.setAttribute("stroke-width", strokeW);
        return el;
      }}
      if (markerType === "diamond") {{
        const el = document.createElementNS(ns, "polygon");
        const d = r * 1.35;
        const pts = `${{cx}},${{cy - d}} ${{cx + d}},${{cy}} ${{cx}},${{cy + d}} ${{cx - d}},${{cy}}`;
        el.setAttribute("points", pts);
        el.setAttribute("fill", color);
        el.setAttribute("stroke", "#ffffff");
        el.setAttribute("stroke-width", strokeW);
        return el;
      }}
      if (markerType === "cross") {{
        const g = document.createElementNS(ns, "g");
        const d = r * 1.1;
        for (const [x1, y1, x2, y2] of [[cx - d, cy - d, cx + d, cy + d], [cx - d, cy + d, cx + d, cy - d]]) {{
          const line = document.createElementNS(ns, "line");
          line.setAttribute("x1", x1);
          line.setAttribute("y1", y1);
          line.setAttribute("x2", x2);
          line.setAttribute("y2", y2);
          line.setAttribute("stroke", color);
          line.setAttribute("stroke-width", isHighlighted ? "3.2" : "2.4");
          line.setAttribute("stroke-linecap", "round");
          g.appendChild(line);
        }}
        return g;
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

    function visibleConfigsFor(suite, st) {{
      return (suite.configs || []).filter(c => {{
        if (st.dtypeFilter !== "all" && c.dtype !== st.dtypeFilter) return false;
        if (st.trackFilter !== "all" && c.track !== st.trackFilter) return false;
        if (st.frameworkFilter !== "all" && c.framework !== st.frameworkFilter) return false;
        return true;
      }});
    }}

    function showTooltip(evt, suite, cfg, pt, prevPt, metricSpec) {{
      const mKey = metricSpec.key;
      const mObj = pt[mKey];
      if (!mObj) return;
      const dec = metricSpec.decimals;
      let deltaHtml = "";
      if (prevPt && prevPt[mKey] && prevPt[mKey].median) {{
        const diff = mObj.median - prevPt[mKey].median;
        const pct = (diff / prevPt[mKey].median) * 100;
        const sign = diff > 0 ? "+" : "";
        const col = pct > 1.0 ? "#4ade80" : (pct < -1.0 ? "#f87171" : "#9ca3af");
        deltaHtml = `<div class="tt-key">Δ vs ${{prevPt.date}}</div><div class="tt-val" style="color:${{col}}">${{sign}}${{diff.toFixed(dec)}} tok/s (${{sign}}${{pct.toFixed(1)}}%)</div>`;
      }}
      const otherSpec = mKey === "prefill_tps" ? DECODE_SPEC : PREFILL_SPEC;
      const otherObj = pt[otherSpec.key];
      const otherStr = otherObj && otherObj.median != null ? `${{formatTps(otherObj.median, otherSpec.decimals)}} tok/s` : "—";
      const e2eStr = pt.e2e_ms && pt.e2e_ms.median != null ? `${{Math.round(pt.e2e_ms.median).toLocaleString()}} ms` : "—";
      const ttftStr = pt.ttft_ms != null ? `${{Math.round(pt.ttft_ms).toLocaleString()}} ms` : "—";
      const mmluStr = pt.mmlu != null ? `${{(pt.mmlu * 100).toFixed(2)}}%` : "—";
      const gsm8kStr = pt.gsm8k != null ? `${{(pt.gsm8k * 100).toFixed(2)}}%` : "—";

      tooltip.innerHTML = `
        <div class="tt-title">
          <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${{cfg.color}}"></span>
          <span>${{cfg.id}}</span>
          <span style="margin-left:auto;font-size:11px;color:#9198a1;font-weight:500">${{cfg.dtype_label}} · ${{cfg.track}}</span>
        </div>
        <div class="tt-grid">
          <div class="tt-key">Date</div><div class="tt-val">${{pt.date}}</div>
          <div class="tt-key">${{metricSpec.title.replace(" Over Time", "")}} (median)</div><div class="tt-val">${{formatTps(mObj.median, dec)}} tok/s</div>
          <div class="tt-key">Min – Max (n=${{mObj.n}})</div><div class="tt-val">${{formatTps(mObj.min, dec)}} – ${{formatTps(mObj.max, dec)}} tok/s</div>
          <div class="tt-key">CV</div><div class="tt-val">${{(mObj.cv * 100).toFixed(2)}}%</div>
          ${{deltaHtml}}
          <div class="tt-key">${{otherSpec.title.replace(" Over Time", "")}}</div><div class="tt-val">${{otherStr}}</div>
          <div class="tt-key">E2E / TTFT</div><div class="tt-val">${{e2eStr}} / ${{ttftStr}}</div>
          <div class="tt-key">Peak RSS</div><div class="tt-val">${{pt.rss_max_mb != null ? Math.round(pt.rss_max_mb).toLocaleString() + " MB" : "—"}}</div>
          <div class="tt-key">Accuracy (MMLU / GSM8K)</div><div class="tt-val">${{mmluStr}} / ${{gsm8kStr}}</div>
          <div class="tt-key">Framework Ref</div><div class="tt-val">${{pt.framework_ref || "—"}}</div>
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
      if (left + rect.width > window.innerWidth - 12) {{
        left = evt.clientX - rect.width - pad;
      }}
      if (top + rect.height > window.innerHeight - 12) {{
        top = evt.clientY - rect.height - pad;
      }}
      tooltip.style.left = `${{Math.max(8, left)}}px`;
      tooltip.style.top = `${{Math.max(8, top)}}px`;
    }}

    function hideTooltip() {{
      tooltip.style.display = "none";
    }}

    function renderMetricChart(suite, st, svgEl, metricSpec, onHoverChange) {{
      const ns = "http://www.w3.org/2000/svg";
      svgEl.innerHTML = "";
      const mKey = metricSpec.key;
      const dec = metricSpec.decimals;

      const W = 1220;
      const H = 450;
      const padL = 82;
      const padR = 245;
      const padT = 26;
      const padB = 60;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;

      svgEl.setAttribute("viewBox", `0 0 ${{W}} ${{H}}`);

      const reports = suite.reports || [];
      const filteredConfigs = visibleConfigsFor(suite, st);
      const activeConfigs = filteredConfigs.filter(c => !st.hiddenConfigs.has(c.id));

      // Compute X positions for each report_id (by time or ordinal)
      const xByReport = {{}};
      const parsedTimestamps = reports.map(r => Date.parse(r.date + "T00:00:00Z"));
      const minTs = Math.min(...parsedTimestamps);
      const maxTs = Math.max(...parsedTimestamps);

      if (reports.length === 1) {{
        xByReport[reports[0].report_id] = padL + plotW / 2;
      }} else if (st.xSpacing === "time" && maxTs > minTs) {{
        const innerPad = 36;
        reports.forEach((r, idx) => {{
          const frac = (parsedTimestamps[idx] - minTs) / (maxTs - minTs);
          xByReport[r.report_id] = padL + innerPad + frac * (plotW - 2 * innerPad);
        }});
      }} else {{
        const innerPad = 36;
        reports.forEach((r, idx) => {{
          const frac = idx / (reports.length - 1);
          xByReport[r.report_id] = padL + innerPad + frac * (plotW - 2 * innerPad);
        }});
      }}

      // Gather Y range across active configs for metricSpec.key
      let minVal = Infinity;
      let maxVal = -Infinity;
      for (const c of activeConfigs) {{
        for (const p of c.points || []) {{
          const mObj = p[mKey];
          if (!mObj || mObj.median == null) continue;
          const lo = st.showWhiskers ? mObj.min : mObj.median;
          const hi = st.showWhiskers ? mObj.max : mObj.median;
          if (lo < minVal) minVal = lo;
          if (hi > maxVal) maxVal = hi;
        }}
      }}
      if (!isFinite(minVal) || !isFinite(maxVal)) {{
        minVal = 10;
        maxVal = 100;
      }}

      const isLog = st.yScale === "log";
      let yMin, yMax;
      if (isLog) {{
        yMin = Math.max(1, minVal * 0.85);
        yMax = maxVal * 1.18;
      }} else {{
        yMin = 0;
        yMax = maxVal * 1.12;
      }}

      function yPos(val) {{
        const clamped = Math.max(isLog ? yMin : 0, val);
        if (isLog) {{
          const frac = (Math.log10(clamped) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin));
          return padT + plotH * (1 - frac);
        }}
        const frac = (clamped - yMin) / (yMax - yMin || 1);
        return padT + plotH * (1 - frac);
      }}

      // Generate Y ticks
      let yTicks = [];
      if (isLog) {{
        const candidates = [1, 2, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000];
        yTicks = candidates.filter(v => v >= yMin * 0.95 && v <= yMax * 1.02);
        if (yTicks.length < 3) {{
          yTicks = [minVal, Math.sqrt(minVal * maxVal), maxVal];
        }}
      }} else {{
        const nTicks = 6;
        const rawStep = (yMax - yMin) / nTicks;
        const mag = Math.pow(10, Math.floor(Math.log10(rawStep || 1)));
        const residual = rawStep / mag;
        const niceStep = residual > 5 ? 10 * mag : residual > 2 ? 5 * mag : residual > 1 ? 2 * mag : mag;
        for (let v = 0; v <= yMax; v += niceStep) {{
          yTicks.push(v);
        }}
      }}

      // Draw Y grid lines and labels
      for (const tickVal of yTicks) {{
        const y = yPos(tickVal);
        if (y < padT - 2 || y > padT + plotH + 2) continue;
        const line = document.createElementNS(ns, "line");
        line.setAttribute("x1", padL);
        line.setAttribute("y1", y);
        line.setAttribute("x2", padL + plotW);
        line.setAttribute("y2", y);
        line.setAttribute("stroke", "#e6ebf1");
        line.setAttribute("stroke-width", "1");
        svgEl.appendChild(line);

        const txt = document.createElementNS(ns, "text");
        txt.setAttribute("x", padL - 10);
        txt.setAttribute("y", y + 4);
        txt.setAttribute("text-anchor", "end");
        txt.setAttribute("font-size", "11.5");
        txt.setAttribute("fill", "#59636e");
        txt.textContent = `${{Math.round(tickVal).toLocaleString()}}`;
        svgEl.appendChild(txt);
      }}

      // Draw X vertical guide lines and date labels
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

        const tick = document.createElementNS(ns, "line");
        tick.setAttribute("x1", x);
        tick.setAttribute("y1", padT + plotH);
        tick.setAttribute("x2", x);
        tick.setAttribute("y2", padT + plotH + 6);
        tick.setAttribute("stroke", "#8c959f");
        svgEl.appendChild(tick);

        const dTxt = document.createElementNS(ns, "text");
        dTxt.setAttribute("x", x);
        dTxt.setAttribute("y", padT + plotH + 22);
        dTxt.setAttribute("text-anchor", "middle");
        dTxt.setAttribute("font-size", "12");
        dTxt.setAttribute("font-weight", "600");
        dTxt.setAttribute("fill", "#1f2328");
        dTxt.textContent = r.date;
        svgEl.appendChild(dTxt);
      }}

      // Axes border lines
      const xAxisLine = document.createElementNS(ns, "line");
      xAxisLine.setAttribute("x1", padL);
      xAxisLine.setAttribute("y1", padT + plotH);
      xAxisLine.setAttribute("x2", padL + plotW);
      xAxisLine.setAttribute("y2", padT + plotH);
      xAxisLine.setAttribute("stroke", "#8c959f");
      xAxisLine.setAttribute("stroke-width", "1.2");
      svgEl.appendChild(xAxisLine);

      const yAxisLine = document.createElementNS(ns, "line");
      yAxisLine.setAttribute("x1", padL);
      yAxisLine.setAttribute("y1", padT);
      yAxisLine.setAttribute("x2", padL);
      yAxisLine.setAttribute("y2", padT + plotH);
      yAxisLine.setAttribute("stroke", "#8c959f");
      yAxisLine.setAttribute("stroke-width", "1.2");
      svgEl.appendChild(yAxisLine);

      // Axis titles
      const yTitle = document.createElementNS(ns, "text");
      yTitle.setAttribute("transform", `translate(20, ${{padT + plotH / 2}}) rotate(-90)`);
      yTitle.setAttribute("text-anchor", "middle");
      yTitle.setAttribute("font-size", "12.5");
      yTitle.setAttribute("font-weight", "600");
      yTitle.setAttribute("fill", "#1f2328");
      yTitle.textContent = `${{metricSpec.yAxisLabel}} (${{isLog ? "log scale" : "linear scale"}}) — higher is better ↑`;
      svgEl.appendChild(yTitle);

      const xTitle = document.createElementNS(ns, "text");
      xTitle.setAttribute("x", padL + plotW / 2);
      xTitle.setAttribute("y", H - 10);
      xTitle.setAttribute("text-anchor", "middle");
      xTitle.setAttribute("font-size", "12.5");
      xTitle.setAttribute("font-weight", "600");
      xTitle.setAttribute("fill", "#1f2328");
      xTitle.textContent = "Time (Benchmark Report Date)";
      svgEl.appendChild(xTitle);

      if (activeConfigs.length === 0) {{
        const emptyTxt = document.createElementNS(ns, "text");
        emptyTxt.setAttribute("x", padL + plotW / 2);
        emptyTxt.setAttribute("y", padT + plotH / 2);
        emptyTxt.setAttribute("text-anchor", "middle");
        emptyTxt.setAttribute("font-size", "14");
        emptyTxt.setAttribute("fill", "#59636e");
        emptyTxt.textContent = "No configurations match the current filter selection.";
        svgEl.appendChild(emptyTxt);
        return;
      }}

      // Collect right-edge labels for de-overlapping
      const endLabels = [];

      // Draw series lines, whiskers, and markers
      for (const cfg of activeConfigs) {{
        const isHovered = st.hoveredConfig === cfg.id;
        const isDimmed = st.hoveredConfig && st.hoveredConfig !== cfg.id;
        const opacity = isDimmed ? "0.22" : "0.95";
        const strokeW = isHovered ? "3.2" : "2.1";

        const pts = (cfg.points || []).filter(p => p[mKey] && p[mKey].median != null);
        if (pts.length === 0) continue;

        const g = document.createElementNS(ns, "g");
        g.setAttribute("opacity", opacity);

        // Polyline connecting dates
        if (pts.length > 1) {{
          const pathCoords = pts.map(p => `${{xByReport[p.report_id]}},${{yPos(p[mKey].median)}}`).join(" ");
          const poly = document.createElementNS(ns, "polyline");
          poly.setAttribute("fill", "none");
          poly.setAttribute("stroke", cfg.color);
          poly.setAttribute("stroke-width", strokeW);
          const dash = dashArrayFor(cfg.linestyle);
          if (dash) poly.setAttribute("stroke-dasharray", dash);
          poly.setAttribute("points", pathCoords);
          poly.style.cursor = "pointer";
          poly.addEventListener("mouseenter", () => {{
            st.hoveredConfig = cfg.id;
            onHoverChange();
          }});
          poly.addEventListener("mouseleave", () => {{
            st.hoveredConfig = null;
            onHoverChange();
          }});
          g.appendChild(poly);
        }}

        // Whiskers & points
        pts.forEach((pt, idx) => {{
          const mObj = pt[mKey];
          const cx = xByReport[pt.report_id];
          const cy = yPos(mObj.median);

          if (st.showWhiskers && mObj.max > mObj.min) {{
            const yLo = yPos(mObj.min);
            const yHi = yPos(mObj.max);
            const wLine = document.createElementNS(ns, "line");
            wLine.setAttribute("x1", cx);
            wLine.setAttribute("y1", yHi);
            wLine.setAttribute("x2", cx);
            wLine.setAttribute("y2", yLo);
            wLine.setAttribute("stroke", cfg.color);
            wLine.setAttribute("stroke-width", "1.3");
            wLine.setAttribute("opacity", "0.65");
            g.appendChild(wLine);

            for (const capY of [yHi, yLo]) {{
              const cap = document.createElementNS(ns, "line");
              cap.setAttribute("x1", cx - 4);
              cap.setAttribute("y1", capY);
              cap.setAttribute("x2", cx + 4);
              cap.setAttribute("y2", capY);
              cap.setAttribute("stroke", cfg.color);
              cap.setAttribute("stroke-width", "1.3");
              cap.setAttribute("opacity", "0.65");
              g.appendChild(cap);
            }}
          }}

          const markerEl = createMarkerSVG(cfg.marker, cx, cy, isHovered ? 5.8 : 4.5, cfg.color, isHovered);
          g.appendChild(markerEl);

          // Invisible larger hit target for smooth hover
          const hit = document.createElementNS(ns, "circle");
          hit.setAttribute("cx", cx);
          hit.setAttribute("cy", cy);
          hit.setAttribute("r", 11);
          hit.setAttribute("fill", "transparent");
          hit.style.cursor = "pointer";
          const prevPt = idx > 0 ? pts[idx - 1] : null;
          hit.addEventListener("mouseenter", (e) => {{
            st.hoveredConfig = cfg.id;
            onHoverChange();
            showTooltip(e, suite, cfg, pt, prevPt, metricSpec);
          }});
          hit.addEventListener("mousemove", (e) => positionTooltip(e));
          hit.addEventListener("mouseleave", () => {{
            st.hoveredConfig = null;
            onHoverChange();
            hideTooltip();
          }});
          if (pt.report_url) {{
            hit.addEventListener("click", () => {{
              window.location.href = pt.report_url;
            }});
          }}
          g.appendChild(hit);
        }});

        const lastPt = pts[pts.length - 1];
        const lastVal = lastPt[mKey].median;
        endLabels.push({{
          cfg,
          reportId: lastPt.report_id,
          x: xByReport[lastPt.report_id],
          rawY: yPos(lastVal),
          y: yPos(lastVal),
          val: lastVal,
          isHovered,
          isDimmed,
        }});

        svgEl.appendChild(g);
      }}

      // Group labels by their final x column so each column de-overlaps independently
      const labelsByCol = {{}};
      for (const item of endLabels) {{
        (labelsByCol[item.reportId] = labelsByCol[item.reportId] || []).push(item);
      }}
      const lastReportId = reports.length > 0 ? reports[reports.length - 1].report_id : null;
      const minGap = 14.5;
      const maxLabelY = padT + plotH + 8;

      for (const [repId, colItems] of Object.entries(labelsByCol)) {{
        colItems.sort((a, b) => a.rawY - b.rawY);
        for (let i = 1; i < colItems.length; i++) {{
          if (colItems[i].y - colItems[i - 1].y < minGap) {{
            colItems[i].y = colItems[i - 1].y + minGap;
          }}
        }}
        if (colItems.length > 0 && colItems[colItems.length - 1].y > maxLabelY) {{
          colItems[colItems.length - 1].y = maxLabelY;
          for (let i = colItems.length - 2; i >= 0; i--) {{
            if (colItems[i + 1].y - colItems[i].y < minGap) {{
              colItems[i].y = Math.max(padT - 4, colItems[i + 1].y - minGap);
            }}
          }}
        }}

        const isRightmostMulti = reports.length > 1 && repId === lastReportId;
        for (const item of colItems) {{
          const lg = document.createElementNS(ns, "g");
          lg.setAttribute("opacity", item.isDimmed ? "0.25" : "1.0");
          lg.style.cursor = "pointer";

          const labelStartX = isRightmostMulti ? (padL + plotW + 12) : (item.x + 14);
          const leader = document.createElementNS(ns, "path");
          const midX = (item.x + labelStartX - 3) / 2;
          leader.setAttribute("d", `M ${{item.x + 5}} ${{item.rawY}} C ${{midX}} ${{item.rawY}}, ${{midX}} ${{item.y}}, ${{labelStartX - 3}} ${{item.y}}`);
          leader.setAttribute("fill", "none");
          leader.setAttribute("stroke", item.cfg.color);
          leader.setAttribute("stroke-width", "0.9");
          leader.setAttribute("opacity", "0.55");
          lg.appendChild(leader);

          const lbl = document.createElementNS(ns, "text");
          lbl.setAttribute("x", labelStartX);
          lbl.setAttribute("y", item.y + 3.5);
          lbl.setAttribute("font-size", "11");
          lbl.setAttribute("font-weight", item.isHovered ? "700" : "500");
          lbl.setAttribute("fill", item.cfg.color);
          lbl.textContent = `${{item.cfg.id}} (${{formatTps(item.val, dec)}} tok/s)`;
          lg.appendChild(lbl);

          lg.addEventListener("mouseenter", () => {{
            st.hoveredConfig = item.cfg.id;
            onHoverChange();
          }});
          lg.addEventListener("mouseleave", () => {{
            st.hoveredConfig = null;
            onHoverChange();
          }});

          svgEl.appendChild(lg);
        }}
      }}
    }}

    function renderLegendForSuite(suite, st, legendEl, redrawCharts) {{
      legendEl.innerHTML = "";
      const visibleConfigs = visibleConfigsFor(suite, st);
      for (const cfg of visibleConfigs) {{
        const item = document.createElement("div");
        const isHidden = st.hiddenConfigs.has(cfg.id);
        item.className = "legend-item" + (isHidden ? " muted" : "");
        item.title = "Click to toggle visibility · Double-click to solo this configuration";

        const dash = dashArrayFor(cfg.linestyle);
        const dashAttr = dash ? `stroke-dasharray="${{dash}}"` : "";
        item.innerHTML = `
          <svg class="legend-swatch" viewBox="0 0 24 12">
            <line x1="1" y1="6" x2="23" y2="6" stroke="${{cfg.color}}" stroke-width="2.2" ${{dashAttr}} />
            <circle cx="12" cy="6" r="3.2" fill="${{cfg.color}}" stroke="#fff" stroke-width="1" />
          </svg>
          <span><code>${{cfg.id}}</code> <span style="color:var(--muted);font-size:11px">(${{cfg.dtype_label}})</span></span>
        `;

        item.addEventListener("mouseenter", () => {{
          if (!isHidden) {{
            st.hoveredConfig = cfg.id;
            redrawCharts();
          }}
        }});
        item.addEventListener("mouseleave", () => {{
          st.hoveredConfig = null;
          redrawCharts();
        }});
        item.addEventListener("click", () => {{
          if (st.hiddenConfigs.has(cfg.id)) {{
            st.hiddenConfigs.delete(cfg.id);
          }} else {{
            st.hiddenConfigs.add(cfg.id);
          }}
          renderLegendForSuite(suite, st, legendEl, redrawCharts);
          redrawCharts();
        }});
        item.addEventListener("dblclick", (e) => {{
          e.stopPropagation();
          const allIds = visibleConfigs.map(c => c.id);
          const onlyThisVisible = st.hiddenConfigs.size === allIds.length - 1 && !st.hiddenConfigs.has(cfg.id);
          st.hiddenConfigs.clear();
          if (!onlyThisVisible) {{
            for (const id of allIds) {{
              if (id !== cfg.id) st.hiddenConfigs.add(id);
            }}
          }}
          renderLegendForSuite(suite, st, legendEl, redrawCharts);
          redrawCharts();
        }});

        legendEl.appendChild(item);
      }}
    }}

    function renderTableForSuite(suite, st, tableContainer) {{
      const reports = suite.reports || [];
      const configs = visibleConfigsFor(suite, st);

      // Group configs by dtype
      const dtypeOrder = [
        ["fp32", "FP32 Track"],
        ["bf16", "BF16 Track"],
        ["fp16", "FP16 Track"],
        ["int8", "INT8 Track"],
        ["int4", "INT4 / Mixed Track"],
      ];
      const byDtype = {{}};
      for (const c of configs) {{
        (byDtype[c.dtype] = byDtype[c.dtype] || []).push(c);
      }}

      const prefillDateHeaders = reports.map(r => {{
        const lbl = r.report_url ? `<a href="${{r.report_url}}">${{r.date}} ↗</a>` : r.date;
        return `<th class="num">Prefill ${{lbl}} (tok/s) ↑</th>`;
      }}).join("");

      const decodeDateHeaders = reports.map(r => {{
        const lbl = r.report_url ? `<a href="${{r.report_url}}">${{r.date}} ↗</a>` : r.date;
        return `<th class="num">Decode ${{lbl}} (tok/s) ↑</th>`;
      }}).join("");

      const hasHistory = reports.length >= 2;
      const prefillDeltaHeader = hasHistory ? `<th class="num">Δ Prefill</th>` : "";
      const decodeDeltaHeader = hasHistory ? `<th class="num">Δ Decode</th>` : "";

      let rowsHtml = "";
      for (const [dtypeKey, dtypeTitle] of dtypeOrder) {{
        const groupCfgs = byDtype[dtypeKey];
        if (!groupCfgs || groupCfgs.length === 0) continue;

        groupCfgs.sort((a, b) => (b.latest_decode_tps ?? -1) - (a.latest_decode_tps ?? -1));
        const winnerId = groupCfgs[0].id;

        const totalCols = 4 + (reports.length * 2) + (hasHistory ? 2 : 0) + 4;
        rowsHtml += `<tr class="group-row"><td colspan="${{totalCols}}">${{dtypeTitle}}</td></tr>`;

        for (const cfg of groupCfgs) {{
          const isWinner = cfg.id === winnerId;
          const ptByRep = {{}};
          for (const p of cfg.points || []) {{
            ptByRep[p.report_id] = p;
          }}
          const lastPt = (cfg.points || [])[cfg.points.length - 1] || {{}};

          const prefillCells = reports.map(r => {{
            const p = ptByRep[r.report_id];
            if (!p || !p.prefill_tps) return `<td class="num" style="color:var(--muted)">—</td>`;
            const med = formatTps(p.prefill_tps.median, 1);
            const lo = formatTps(p.prefill_tps.min, 0);
            const hi = formatTps(p.prefill_tps.max, 0);
            const refBadge = p.framework_ref ? ` · ${{p.framework_ref}}` : "";
            return `<td class="num"><strong>${{med}}</strong><span class="sub-range">[${{lo}}–${{hi}}]${{refBadge}}</span></td>`;
          }}).join("");

          const decodeCells = reports.map(r => {{
            const p = ptByRep[r.report_id];
            if (!p || !p.decode_tps) return `<td class="num" style="color:var(--muted)">—</td>`;
            const med = formatTps(p.decode_tps.median, 2);
            const lo = formatTps(p.decode_tps.min, 1);
            const hi = formatTps(p.decode_tps.max, 1);
            const refBadge = p.framework_ref ? ` · ${{p.framework_ref}}` : "";
            return `<td class="num"><strong>${{med}}</strong><span class="sub-range">[${{lo}}–${{hi}}]${{refBadge}}</span></td>`;
          }}).join("");

          function renderThroughputDelta(dAbs, dPct, decimals) {{
            if (!hasHistory) return "";
            if (dPct == null || dAbs == null) {{
              return `<td class="num"><span class="delta-pill delta-neutral">n/a</span></td>`;
            }}
            const sign = dAbs > 0 ? "+" : "";
            const cls = dPct > 1.0 ? "delta-good" : (dPct < -1.0 ? "delta-bad" : "delta-neutral");
            return `<td class="num"><span class="delta-pill ${{cls}}">${{sign}}${{dAbs.toFixed(decimals)}} (${{sign}}${{dPct.toFixed(1)}}%)</span></td>`;
          }}

          const prefillDeltaCell = renderThroughputDelta(cfg.delta_prefill_tps, cfg.delta_prefill_pct, 1);
          const decodeDeltaCell = renderThroughputDelta(cfg.delta_decode_tps, cfg.delta_decode_pct, 2);

          const e2eStr = lastPt.e2e_ms && lastPt.e2e_ms.median != null ? Math.round(lastPt.e2e_ms.median).toLocaleString() : "—";
          const mmluStr = lastPt.mmlu != null ? `${{(lastPt.mmlu * 100).toFixed(2)}}%` : "—";
          const gsm8kStr = lastPt.gsm8k != null ? `${{(lastPt.gsm8k * 100).toFixed(2)}}%` : "—";

          rowsHtml += `
            <tr class="${{isWinner ? "best-row" : ""}}">
              <td>
                <span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${{cfg.color}};margin-right:5px;"></span>
                ${{isWinner ? "🏆 " : ""}}<code><strong>${{cfg.id}}</strong></code>
              </td>
              <td><code>${{cfg.track}}</code></td>
              <td><code>${{cfg.weights_acts_kv}}</code></td>
              <td class="num">${{cfg.threads}}</td>
              ${{prefillCells}}
              ${{prefillDeltaCell}}
              ${{decodeCells}}
              ${{decodeDeltaCell}}
              <td class="num">${{lastPt.ttft_ms != null ? Math.round(lastPt.ttft_ms).toLocaleString() : "—"}}</td>
              <td class="num">${{e2eStr}}</td>
              <td class="num">${{lastPt.rss_max_mb != null ? Math.round(lastPt.rss_max_mb).toLocaleString() : "—"}}</td>
              <td class="num">${{mmluStr}} / ${{gsm8kStr}}</td>
            </tr>
          `;
        }}
      }}

      tableContainer.innerHTML = `
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Framework / Config</th>
                <th>Track</th>
                <th>Weights / Acts / KV</th>
                <th class="num">Threads</th>
                ${{prefillDateHeaders}}
                ${{prefillDeltaHeader}}
                ${{decodeDateHeaders}}
                ${{decodeDeltaHeader}}
                <th class="num">Latest TTFT (ms) ↓</th>
                <th class="num">Latest E2E (ms) ↓</th>
                <th class="num">Peak RSS (MB) ↓</th>
                <th class="num">MMLU / GSM8K ↑</th>
              </tr>
            </thead>
            <tbody>
              ${{rowsHtml}}
            </tbody>
          </table>
        </div>
      `;
    }}

    function renderAllSuites() {{
      const root = document.getElementById("suites-root");
      root.innerHTML = "";

      const visibleSuites = activeSuiteFilter === "all"
        ? suites
        : suites.filter(s => s.id === activeSuiteFilter);

      for (const suite of visibleSuites) {{
        const st = suiteStates[suite.id];
        const card = document.createElement("section");
        card.className = "suite-card";
        card.id = `suite-${{suite.id}}`;

        const w = suite.workload || {{}};
        const reportLinksHtml = (suite.reports || []).map(r => {{
          if (r.report_url) {{
            return `<a class="report-link-pill" href="${{r.report_url}}">📄 ${{r.date}} Report ↗</a>`;
          }}
          return `<span class="report-link-pill">📅 ${{r.date}}</span>`;
        }}).join("");

        card.innerHTML = `
          <div class="suite-header">
            <div class="suite-title-row">
              <h2>
                <span>${{suite.suite}}</span>
                <span class="badge badge-machine">Machine: ${{suite.machine}}</span>
                <span class="badge badge-workload">${{w.prompt_tokens || 1024}} prompt → ${{w.gen_tokens || 128}} decode tokens (ctx ${{w.ctx || 4096}})</span>
              </h2>
              <span class="badge">${{(suite.configs || []).length}} configs · ${{(suite.reports || []).length}} report date(s)</span>
            </div>
            ${{suite.machine_notes ? `<div class="suite-meta-notes">${{suite.machine_notes}}</div>` : ""}}
            <div class="report-links-bar">
              <strong style="color:var(--muted)">Full Reports:</strong>
              ${{reportLinksHtml}}
            </div>
          </div>
          <div class="controls-bar" data-controls></div>
          <div class="chart-section">
            <div class="charts-grid" data-charts-grid>
              <div class="chart-panel">
                <div class="chart-panel-header">
                  <div class="chart-panel-title">⚡ Prefill Throughput Over Time (tokens/s ↑)</div>
                  <div class="chart-panel-subtitle">${{w.prompt_tokens || 1024}}-token prompt processing speed vs. report date</div>
                </div>
                <svg class="chart-svg" data-chart-prefill></svg>
              </div>
              <div class="chart-panel">
                <div class="chart-panel-header">
                  <div class="chart-panel-title">🚀 Decode Throughput Over Time (tokens/s ↑)</div>
                  <div class="chart-panel-subtitle">${{w.gen_tokens || 128}}-token autoregressive generation speed vs. report date</div>
                </div>
                <svg class="chart-svg" data-chart-decode></svg>
              </div>
            </div>
            <div class="legend-grid"></div>
          </div>
          <div class="table-section">
            <h3>
              <span>Prefill &amp; Decode Throughput Over Time (tokens/s) &amp; Latest Breakdown</span>
              <span style="font-size:0.8rem;font-weight:400;color:var(--muted)">Ordered by fastest latest Decode tok/s per data type track (🏆 = fastest in track)</span>
            </h3>
            <div data-table></div>
          </div>
        `;

        const controlsEl = card.querySelector("[data-controls]");
        const chartsGridEl = card.querySelector("[data-charts-grid]");
        const svgPrefillEl = card.querySelector("[data-chart-prefill]");
        const svgDecodeEl = card.querySelector("[data-chart-decode]");
        const legendEl = card.querySelector(".legend-grid");
        const tableEl = card.querySelector("[data-table]");

        function redrawBothCharts() {{
          chartsGridEl.className = "charts-grid" + (st.layout === "side-by-side" ? " side-by-side" : "");
          renderMetricChart(suite, st, svgPrefillEl, PREFILL_SPEC, redrawBothCharts);
          renderMetricChart(suite, st, svgDecodeEl, DECODE_SPEC, redrawBothCharts);
        }}

        function refreshSuiteView() {{
          renderControls();
          redrawBothCharts();
          renderLegendForSuite(suite, st, legendEl, redrawBothCharts);
          renderTableForSuite(suite, st, tableEl);
        }}

        function renderControls() {{
          controlsEl.innerHTML = "";

          // Y-Axis Scale group
          const yGrp = document.createElement("div");
          yGrp.className = "control-group";
          yGrp.innerHTML = `<span class="control-label">Y-Axis (tokens/s):</span>`;
          for (const [val, label] of [["linear", "Linear (tok/s)"], ["log", "Log Scale (tok/s)"]]) {{
            const b = document.createElement("button");
            b.className = "ctrl-btn" + (st.yScale === val ? " active" : "");
            b.textContent = label;
            b.onclick = () => {{ st.yScale = val; refreshSuiteView(); }};
            yGrp.appendChild(b);
          }}
          controlsEl.appendChild(yGrp);

          // Layout group
          const lGrp = document.createElement("div");
          lGrp.className = "control-group";
          lGrp.innerHTML = `<span class="control-label">Charts Layout:</span>`;
          for (const [val, label] of [["stacked", "Stacked (Full Width)"], ["side-by-side", "Side-by-Side (2 Columns)"]]) {{
            const b = document.createElement("button");
            b.className = "ctrl-btn" + (st.layout === val ? " active" : "");
            b.textContent = label;
            b.onclick = () => {{ st.layout = val; refreshSuiteView(); }};
            lGrp.appendChild(b);
          }}
          controlsEl.appendChild(lGrp);

          // X-Axis Spacing group
          if ((suite.reports || []).length > 2) {{
            const xGrp = document.createElement("div");
            xGrp.className = "control-group";
            xGrp.innerHTML = `<span class="control-label">X-Axis (Time):</span>`;
            for (const [val, label] of [["time", "Proportional Date"], ["ordinal", "Equal Report Spacing"]]) {{
              const b = document.createElement("button");
              b.className = "ctrl-btn" + (st.xSpacing === val ? " active" : "");
              b.textContent = label;
              b.onclick = () => {{ st.xSpacing = val; refreshSuiteView(); }};
              xGrp.appendChild(b);
            }}
            controlsEl.appendChild(xGrp);
          }}

          // Data Type Filter group
          const dtypesPresent = Array.from(new Set((suite.configs || []).map(c => c.dtype)));
          if (dtypesPresent.length > 1) {{
            const dtGrp = document.createElement("div");
            dtGrp.className = "control-group";
            dtGrp.innerHTML = `<span class="control-label">Data Type:</span>`;
            for (const [val, label] of [["all", "All"], ["int4", "INT4 / Mixed"], ["int8", "INT8"], ["bf16", "BF16"], ["fp32", "FP32"]]) {{
              if (val !== "all" && !dtypesPresent.includes(val)) continue;
              const b = document.createElement("button");
              b.className = "ctrl-btn" + (st.dtypeFilter === val ? " active" : "");
              b.textContent = label;
              b.onclick = () => {{ st.dtypeFilter = val; refreshSuiteView(); }};
              dtGrp.appendChild(b);
            }}
            controlsEl.appendChild(dtGrp);
          }}

          // Track Filter group
          const tracksPresent = Array.from(new Set((suite.configs || []).map(c => c.track)));
          if (tracksPresent.length > 1) {{
            const trGrp = document.createElement("div");
            trGrp.className = "control-group";
            trGrp.innerHTML = `<span class="control-label">Track:</span>`;
            for (const [val, label] of [["all", "All"], ["fastest", "Fastest"], ["matched", "Matched"], ["reference", "Reference"]]) {{
              if (val !== "all" && !tracksPresent.includes(val)) continue;
              const b = document.createElement("button");
              b.className = "ctrl-btn" + (st.trackFilter === val ? " active" : "");
              b.textContent = label;
              b.onclick = () => {{ st.trackFilter = val; refreshSuiteView(); }};
              trGrp.appendChild(b);
            }}
            controlsEl.appendChild(trGrp);
          }}

          // Framework Filter group
          const fwsPresent = Array.from(new Set((suite.configs || []).map(c => c.framework)));
          if (fwsPresent.length > 1) {{
            const fwGrp = document.createElement("div");
            fwGrp.className = "control-group";
            fwGrp.innerHTML = `<span class="control-label">Framework:</span>`;
            for (const val of ["all", ...fwsPresent]) {{
              const b = document.createElement("button");
              b.className = "ctrl-btn" + (st.frameworkFilter === val ? " active" : "");
              b.textContent = val === "all" ? "All" : val;
              b.onclick = () => {{ st.frameworkFilter = val; refreshSuiteView(); }};
              fwGrp.appendChild(b);
            }}
            controlsEl.appendChild(fwGrp);
          }}

          // Whiskers toggle
          const wGrp = document.createElement("div");
          wGrp.className = "control-group";
          const wBtn = document.createElement("button");
          wBtn.className = "ctrl-btn" + (st.showWhiskers ? " active" : "");
          wBtn.textContent = "Min–Max Whiskers";
          wBtn.onclick = () => {{ st.showWhiskers = !st.showWhiskers; refreshSuiteView(); }};
          wGrp.appendChild(wBtn);

          if (st.hiddenConfigs.size > 0) {{
            const resetBtn = document.createElement("button");
            resetBtn.className = "ctrl-btn";
            resetBtn.textContent = `Show All Series (${{st.hiddenConfigs.size}} hidden)`;
            resetBtn.onclick = () => {{ st.hiddenConfigs.clear(); refreshSuiteView(); }};
            wGrp.appendChild(resetBtn);
          }}
          controlsEl.appendChild(wGrp);
        }}

        refreshSuiteView();
        root.appendChild(card);
      }}
    }}

    renderSuiteTabs();
    renderAllSuites();
  }})();
  </script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--suites", type=Path, default=ROOT / "suites", help="Directory containing suite JSON files")
  parser.add_argument("--reports", type=Path, default=ROOT / "reports", help="Directory containing reports and history/")
  parser.add_argument("--results", type=Path, default=ROOT / "results", help="Directory containing raw benchmark results")
  parser.add_argument("-o", "--out", type=Path, default=ROOT / "dashboards" / "index.html", help="Output HTML file path")
  args = parser.parse_args(argv)

  out_html = args.out.resolve()
  out_html.parent.mkdir(parents=True, exist_ok=True)
  out_json = out_html.parent / "data.json"

  dashboard_data = collect_dashboard_data(
      suites_dir=args.suites.resolve(),
      reports_dir=args.reports.resolve(),
      results_dir=args.results.resolve(),
  )
  out_json.write_text(json.dumps(dashboard_data, indent=2) + "\n", encoding="utf-8")

  html_doc = render_dashboard_html(dashboard_data)
  out_html.write_text(html_doc, encoding="utf-8")

  n_suites = len(dashboard_data.get("suites", []))
  n_configs = sum(len(s.get("configs", [])) for s in dashboard_data.get("suites", []))
  print(f"Updated dashboard: {out_html} ({n_suites} suites, {n_configs} tracked configs)")
  print(f"Updated dashboard data: {out_json}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
