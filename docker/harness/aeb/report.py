"""Aggregate a suite's run directories into a standardized report + charts,
and maintain the performance-over-time history.

  python3 -m aeb.report --machine apple-m5-max --suite /suites/gemma4-e2b-cpu.json \\
      --results /results --out /reports [--since 2026-10-01T09:00:00Z] [--date 2026-10-01]

Writes reports/<date>-<suite>-<machine>/{README.md, README.html, index.html, data.json, *.png} and
appends one line per configuration to reports/history/<suite>/<machine>.jsonl
(idempotent per report id), then re-renders the history charts from that file.
The history file is the long-term record: it is small, committed, and keyed by
framework commits and harness digest so regressions can be attributed.
"""

from __future__ import annotations

import argparse
import base64
import collections
import csv
import datetime as dt
import hashlib
import html
import json
import mimetypes
from pathlib import Path
import re
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import accuracy as acc_mod  # noqa: E402
from . import stats  # noqa: E402

COLORS = {
    "llama.cpp": "#d9822b",
    "litert-lm": "#2b6cd9",
    "onnxruntime": "#9c4d9f",
    "xla": "#2e8b57",
    "openvino": "#0095a8",
}
TRACK_ORDER = {"matched": 0, "fastest": 1, "reference": 2}
HEADLINE = [  # metric, label, unit, higher_is_better, fmt
    ("ttft_ms", "TTFT", "ms", False, "{:.0f}"),
    ("prefill_tps", "Prefill", "tok/s", True, "{:.1f}"),
    ("decode_tps", "Decode", "tok/s", True, "{:.2f}"),
    ("e2e_ms", "End-to-end", "ms", False, "{:.0f}"),
    ("itl_p99_ms", "Inter-token p99", "ms", False, "{:.1f}"),
    ("cpu_cores_prefill", "CPU cores (prefill)", "cores", False, "{:.2f}"),
    ("cpu_cores_decode", "CPU cores (decode)", "cores", False, "{:.2f}"),
    ("cpu_seconds", "CPU time / request", "s", False, "{:.1f}"),
    ("rss_max_mb", "Peak RSS (request)", "MB", False, "{:.0f}"),
    ("rss_anon_max_mb", "Peak RSS anon", "MB", False, "{:.0f}"),
    ("rss_file_max_mb", "Peak RSS file-backed", "MB", False, "{:.0f}"),
    ("pss_max_mb", "Peak PSS", "MB", False, "{:.0f}"),
]
CHANGE_THRESHOLD = 0.03  # flag history changes beyond max(3%, 3 x pooled CV)


# --------------------------------------------------------------------- load


def _read_json(p: Path):
  return json.loads(p.read_text()) if p.is_file() else None


def load_perf(results: Path, machine: str) -> list[dict]:
  runs = []
  for meta_p in sorted((results / machine / "perf").glob("*/*/*/meta.json")):
    d = meta_p.parent
    summ = _read_json(d / "summary.json")
    if summ is None:
      continue
    runs.append({"dir": d, "meta": _read_json(meta_p), "summary": summ})
  return runs


def load_accuracy(results: Path, machine: str) -> list[dict]:
  out = []
  for summ_p in sorted(results.glob(f"{machine}/accuracy-*/*/*/*/summary.json")):
    meta = _read_json(summ_p.parent / "meta.json")
    out.append({"dir": summ_p.parent, "meta": meta, "summary": _read_json(summ_p)})
  return out


def read_timeseries(path: Path) -> dict[str, list]:
  cols = collections.defaultdict(list)
  with open(path) as f:
    for row in csv.DictReader(f):
      for k, v in row.items():
        cols[k].append(float(v) if v not in ("", None) else None)
  return cols


# ------------------------------------------------------------------ helpers


def merge_accuracy(acc_runs, agg):
  """Combine shards (k/n) of the same (config, task, limit) into one row,
  using the most recent run of every shard and re-scoring from predictions."""
  groups = collections.defaultdict(dict)
  for a in acc_runs:
    s, m = a["summary"], a["meta"] or {}
    cid = f"{s['framework']}/{s['label']}"
    if cid not in agg or not s.get("complete"):
      continue
    k, n = (int(x) for x in str(m.get("shard", "0/1")).split("/"))
    key = (cid, s.get("task", "mmlu"), m.get("limit") or 0, n)
    prev = groups[key].get(k)
    if prev is None or m.get("started_at", "") > prev["meta"].get("started_at", ""):
      groups[key][k] = a
  rows, preds_by = [], {}
  for (cid, task, limit, n), shards in groups.items():
    if len(shards) != n:
      continue
    preds = []
    for k in range(n):
      preds += [json.loads(l) for l in open(shards[k]["dir"] / "predictions.jsonl")]
    seen = {}
    for p in preds:
      seen[p["id"]] = p
    preds = sorted(seen.values(), key=lambda p: p["id"])
    s = acc_mod.summarize(preds)
    gens = [p.get("n_gen") for p in preds if p.get("n_gen") is not None]
    max_tok = shards[0]["meta"].get("max_tokens")
    rows.append({"id": cid, "task": task, "n": s["n"], "accuracy": s["accuracy"],
                 "ci95": s["accuracy_ci95"], "macro": s["macro_accuracy"] if task == "mmlu" else None,
                 "parse_failures": s["parse_failures"], "complete": True, "limit": limit,
                 "mean_gen": (sum(gens) / len(gens)) if gens else None,
                 # MMLU deliberately stops after a few tokens (only the letter
                 # matters), so the cap is only meaningful for long-form tasks.
                 "truncated": (sum(1 for g in gens if max_tok and g >= max_tok)
                               if task != "mmlu" else None),
                 "max_tokens": max_tok,
                 "shards": n, "dirs": [str(shards[k]["dir"]) for k in range(n)],
                 "dataset_sha256": ((shards[0]["meta"].get("dataset_manifest") or {}).get("jsonl_sha256"))})
    preds_by[(cid, task, limit)] = {p["id"]: p for p in preds}
  # Prefer the largest run per (config, task) for the table, keep the rest.
  rows.sort(key=lambda a: (a["task"] != "mmlu", -a["n"], a["id"]))
  return rows, preds_by


def paired_agreement(preds_by) -> list[dict]:
  """Per task, compare every pair of configs on the items both answered."""
  out = []
  keys = sorted(preds_by)
  for i, a in enumerate(keys):
    for b in keys[i + 1:]:
      if a[1] != b[1]:
        continue
      pa, pb = preds_by[a], preds_by[b]
      common = sorted(set(pa) & set(pb))
      if len(common) < 50:
        continue
      same_fw = a[0].split("/")[0] == b[0].split("/")[0]
      same_track = a[0].split("/")[1] == b[0].split("/")[1]
      if not (same_fw or same_track):
        continue
      same = sum(pa[k]["pred"] == pb[k]["pred"] for k in common)
      a_only = sum(pa[k]["correct"] and not pb[k]["correct"] for k in common)
      b_only = sum(pb[k]["correct"] and not pa[k]["correct"] for k in common)
      # McNemar's exact test (two-sided) on the discordant pairs.
      nd = a_only + b_only
      p = 1.0
      if nd:
        from math import comb
        tail = sum(comb(nd, x) for x in range(0, min(a_only, b_only) + 1)) / 2 ** nd
        p = min(1.0, 2 * tail)
      out.append({"task": a[1], "a": a[0], "b": b[0], "n": len(common),
                  "same_prediction": same / len(common),
                  "acc_a": sum(pa[k]["correct"] for k in common) / len(common),
                  "acc_b": sum(pb[k]["correct"] for k in common) / len(common),
                  "a_only_correct": a_only, "b_only_correct": b_only, "mcnemar_p": p})
  return out


def load_repro(results: Path, machine: str, cfg_ids) -> dict:
  """Latest repro process per (config, mode, seed, process) -> outputs."""
  latest = {}
  for meta_p in sorted((results / machine / "repro").glob("*/*/*/meta.json")):
    m = _read_json(meta_p)
    cid = f"{m['framework']}/{m['label']}"
    if cid not in cfg_ids or not (meta_p.parent / "summary.json").is_file():
      continue
    key = (cid, m.get("mode"), m.get("seed"), m.get("process"))
    if key not in latest or m["started_at"] > latest[key]["meta"]["started_at"]:
      outs = {}
      for line in open(meta_p.parent / "outputs.jsonl"):
        r = json.loads(line)
        if r.get("ok"):
          outs[(r["prompt"], r["repeat"])] = r
      latest[key] = {"meta": m, "summary": _read_json(meta_p.parent / "summary.json"), "outs": outs}
  return latest


def _cmp(a: dict, b: dict):
  """Compare two processes' outputs on shared (prompt, repeat) keys."""
  from .repro import first_divergence
  keys = sorted(set(a["outs"]) & set(b["outs"]))
  same, div = 0, []
  for k in keys:
    x, y = a["outs"][k], b["outs"][k]
    if x["text"] == y["text"]:
      same += 1
    else:
      d = first_divergence(x["ids"], y["ids"])
      div.append(d if d is not None else 0)
  return same, len(keys), div


def repro_rows(rep: dict, cfg_ids) -> tuple[list[dict], list[dict]]:
  rows, cross = [], []
  for cid in cfg_ids:
    row = {"id": cid}
    for mode in ("greedy", "seeded"):
      procs = {k[3]: v for k, v in rep.items() if k[0] == cid and k[1] == mode
               and (mode == "greedy" or k[2] == min(kk[2] for kk in rep if kk[0] == cid and kk[1] == "seeded"))}
      if not procs:
        continue
      w_same = sum(p["summary"]["within_process"]["identical"] for p in procs.values())
      w_tot = sum(p["summary"]["within_process"]["compared"] for p in procs.values())
      w_div = [d for p in procs.values() for d in p["summary"]["within_process"]["divergence_tokens"]
               if d is not None]
      row[f"{mode}_within"] = (w_same, w_tot, w_div)
      if 0 in procs and 1 in procs:
        row[f"{mode}_process"] = _cmp(procs[0], procs[1])
      if 0 in procs and 2 in procs:
        row[f"{mode}_threads"] = _cmp(procs[0], procs[2]) + (
            procs[0]["meta"].get("threads"), procs[2]["meta"].get("threads"))
      if mode == "greedy" and 0 in procs:
        h = hashlib.sha256()
        for k in sorted(procs[0]["outs"]):
          h.update(json.dumps([k, procs[0]["outs"][k]["text"]]).encode())
        row["greedy_fingerprint"] = h.hexdigest()
    seeded = sorted({k[2] for k in rep if k[0] == cid and k[1] == "seeded"})
    if len(seeded) >= 2:
      base = rep.get((cid, "seeded", seeded[0], 0))
      alt = next((v for k, v in rep.items() if k[0] == cid and k[1] == "seeded" and k[2] == seeded[1]), None)
      if base and alt:
        same, n, _ = _cmp(base, alt)
        row["seed_control"] = (n - same, n)
    rows.append(row)
  a = rep.get(("llama.cpp/matched", "greedy", None, 0))
  b = rep.get(("litert-lm/matched", "greedy", None, 0))
  if a and b:
    same, n, div = _cmp(a, b)
    cross.append({"a": "llama.cpp/matched", "b": "litert-lm/matched", "same": same, "n": n, "div": div})
  return rows, cross


def repro_verdict(row, mode) -> str:
  w, p = row.get(f"{mode}_within"), row.get(f"{mode}_process")
  if not w or not p:
    return "incomplete"
  within_ok = w[0] == w[1]
  proc_ok = p[0] == p[1]
  if within_ok and proc_ok:
    return "**reproducible per request**"
  if proc_ok:
    return "**reproducible per process only** (state carries across requests)"
  return "**not reproducible**"


def config_runs(runs, suite, since: str | None) -> dict[str, list[dict]]:
  rounds = suite.get("rounds", 3)
  tag = f"suite={suite['name']}"
  out = {}
  for c in suite["configs"]:
    fw, label = c["framework"], c["id"].split("/", 1)[1]
    mine = [r for r in runs if r["meta"].get("framework") == fw and r["meta"].get("label") == label
            and r["meta"].get("track") != "prefix-cache"
            and r["meta"].get("notes", "").startswith(tag)
            and (since is None or r["meta"].get("started_at", "") >= since)]
    mine.sort(key=lambda r: r["meta"]["started_at"])
    if since is None:
      mine = mine[-rounds:]
    out[c["id"]] = mine
  return out


def pooled(runs: list[dict], metric: str) -> list[float]:
  vals = []
  for r in runs:
    for m in r["summary"].get("per_request", []):
      if m.get("measured") and m.get(metric) is not None:
        val = m[metric]
        if (
            metric == "rss_file_max_mb"
            and val == 0
            and m.get("rss_max_mb") is not None
            and m.get("rss_anon_max_mb") is not None
        ):
          val = max(0.0, float(m["rss_max_mb"]) - float(m["rss_anon_max_mb"]))
        vals.append(val)
  return vals


def sequence_data(runs, suite, since=None) -> dict:
  sweep = suite.get("sequence_sweep")
  if not sweep:
    return {}
  tag = f"suite={suite['name']}"
  candidates = [r for r in runs if r["meta"].get("track") == "sweep-sequence"
                and r["meta"].get("notes") == tag
                and (since is None or r["meta"].get("started_at", "") >= since)]
  if not candidates:
    return {}

  def _has_sequence_run(c):
    base = c["id"].split("/", 1)[1]
    expected = {f"{base}-n{n}-d{sweep['gen_tokens']}" for n in sweep["prompt_tokens"]}
    return any(r["meta"].get("framework") == c["framework"]
               and r["meta"].get("label") in expected for r in candidates)

  sweep_configs = [c for c in suite["configs"] if _has_sequence_run(c)] or list(suite["configs"])
  data = dict(sweep, rounds=sweep.get("rounds", suite.get("rounds", 3)),
              warmup=sweep.get("warmup", suite["workload"]["warmup"]),
              repetitions=sweep.get("repetitions", suite["workload"]["repetitions"]), points={})
  for n in sweep["prompt_tokens"]:
    point = {"configs": {}, "prompt_identity": "unverified"}
    hashes = set()
    identity_configs = set()
    for c in sweep_configs:
      label = f"{c['id'].split('/', 1)[1]}-n{n}-d{sweep['gen_tokens']}"
      latest = {}
      for r in sorted(candidates, key=lambda r: r["meta"]["started_at"]):
        m = r["meta"]
        cmd = m.get("driver_cmd", [])
        if (m["framework"] == c["framework"] and m["label"] == label
            and m["params"]["prompt_tokens"] == n
            and m["params"]["gen_tokens"] == sweep["gen_tokens"]
            and "--ctx" in cmd and cmd[cmd.index("--ctx") + 1] == str(sweep["ctx"])
            and m["round"] in range(data["rounds"])):
          latest[m["round"]] = r
      selected_runs = list(latest.values())
      valid, errors = [], []
      for r in selected_runs:
        m, s = r["meta"], r["summary"]
        workload = m.get("workload", {})
        if workload.get("prompt_tokens") == n and workload.get("prompt_ids_sha256"):
          hashes.add(workload["prompt_ids_sha256"])
          identity_configs.add(c["id"])
        requests = [q for q in s.get("per_request", []) if q.get("measured")]
        if (s.get("status") == "ok" and len(requests) >= data["repetitions"]
            and m.get("workload", {}).get("prompt_tokens") == n
            and all(q.get("n_prompt") == n and q.get("n_gen") == sweep["gen_tokens"]
                    and q.get("prefill_tps", 0) and q.get("decode_tps", 0)
                    for q in requests)):
          valid.append(r)
        else:
          errors.append(m.get("error") or
                        f"{r['dir']}: {s.get('status')}; incomplete or wrong-length requests")
      complete = len(valid) == data["rounds"]
      point["configs"][c["id"]] = {
          "status": "complete" if complete else "partial" if valid else "unavailable",
          "errors": errors,
          "n_requests": len(pooled(valid, "prefill_tps")),
          "prefill_tps": stats.describe(pooled(valid, "prefill_tps")),
          "decode_tps": stats.describe(pooled(valid, "decode_tps")),
          "runs": [{"run_dir": str(r["dir"]), "meta": _meta_slim(r["meta"]),
                    "summary": r["summary"]} for r in selected_runs],
      }
    if len(identity_configs) == len(sweep_configs):
      point["prompt_identity"] = "identical" if len(hashes) == 1 else "DIFFERENT"
    data["points"][str(n)] = point
  return data


def prefix_cache_data(runs, suite, cfg_runs, agg, since=None) -> dict:
  pc = suite.get("prefix_cache")
  if not pc:
    return {}
  tag = f"suite={suite['name']}"
  candidates = [r for r in runs if r["meta"].get("track") == "prefix-cache"
                and r["meta"].get("notes", "").startswith(tag)
                and (since is None or r["meta"].get("started_at", "") >= since)]
  if not candidates:
    return {}

  w = suite["workload"]
  prompt_tokens = int(pc.get("prompt_tokens", w["prompt_tokens"]))
  prefix_tokens = int(pc["prefix_tokens"])
  suffix_tokens = prompt_tokens - prefix_tokens
  gen_tokens = int(pc.get("gen_tokens", w["gen_tokens"]))
  rounds = int(pc.get("rounds", suite.get("rounds", 3)))
  warmup = int(pc.get("warmup", w["warmup"]))
  repetitions = int(pc.get("repetitions", w["repetitions"]))

  configs = {}
  for c in suite["configs"]:
    cid = c["id"]
    base_label = cid.split("/", 1)[1]
    expected_labels = {
        f"{base_label}-prefix-p{prefix_tokens}-s{suffix_tokens}-d{gen_tokens}",
        f"{base_label}-p{prefix_tokens}",
    }
    latest = {}
    for r in sorted(candidates, key=lambda r: r["meta"]["started_at"]):
      m = r["meta"]
      params = m.get("params", {})
      if (m.get("framework") == c["framework"] and m.get("label") in expected_labels
          and params.get("prompt_tokens") == prompt_tokens
          and params.get("prefix_tokens") == prefix_tokens
          and params.get("gen_tokens") == gen_tokens
          and m.get("round") in range(rounds)):
        latest[m["round"]] = r
    selected_runs = list(latest.values())
    if not selected_runs:
      continue
    valid, errors = [], []
    for r in selected_runs:
      m, s = r["meta"], r["summary"]
      requests = [q for q in s.get("per_request", []) if q.get("measured")]
      if (s.get("status") == "ok" and len(requests) >= repetitions
          and all(q.get("n_prompt") == prompt_tokens and q.get("n_gen") == gen_tokens
                  and q.get("n_cached") == prefix_tokens
                  and q.get("n_prefilled") == suffix_tokens
                  for q in requests)):
        valid.append(r)
      else:
        errors.append(m.get("error") or
                      f"{r['dir']}: {s.get('status')}; incomplete or wrong-length requests")
    if not valid and not errors:
      continue
    cold_runs = cfg_runs.get(cid, [])
    cold_ttft_vals = pooled(cold_runs, "ttft_ms")
    warm_ttft_vals = pooled(valid, "ttft_ms")
    cold_e2e_vals = pooled(cold_runs, "e2e_ms")
    warm_e2e_vals = pooled(valid, "e2e_ms")
    configs[cid] = {
        "status": "complete" if len(valid) == rounds else "partial" if valid else "unavailable",
        "errors": errors,
        "n_requests": len(warm_ttft_vals),
        "cold_ttft_ms": (agg.get(cid) or {}).get("ttft_ms"),
        "warm_ttft_ms": stats.describe(warm_ttft_vals),
        "ttft_speedup": (stats.bootstrap_ratio_ci(cold_ttft_vals, warm_ttft_vals)
                         if cold_ttft_vals and warm_ttft_vals else None),
        "cold_prefill_tps": (agg.get(cid) or {}).get("prefill_tps"),
        "warm_effective_prefill_tps": stats.describe(pooled(valid, "prefill_tps")),
        "warm_suffix_prefill_tps": stats.describe(pooled(valid, "suffix_prefill_tps")),
        "warm_decode_tps": stats.describe(pooled(valid, "decode_tps")),
        "cold_e2e_ms": (agg.get(cid) or {}).get("e2e_ms"),
        "warm_e2e_ms": stats.describe(warm_e2e_vals),
        "e2e_speedup": (stats.bootstrap_ratio_ci(cold_e2e_vals, warm_e2e_vals)
                        if cold_e2e_vals and warm_e2e_vals else None),
        "runs": [{"run_dir": str(r["dir"]), "meta": _meta_slim(r["meta"]),
                  "summary": r["summary"]} for r in selected_runs],
    }
  if not configs:
    return {}
  return {
      "prompt_tokens": prompt_tokens,
      "prefix_tokens": prefix_tokens,
      "suffix_tokens": suffix_tokens,
      "gen_tokens": gen_tokens,
      "rounds": rounds,
      "warmup": warmup,
      "repetitions": repetitions,
      "configs": configs,
  }


def fmt(v, f="{:.2f}"):
  return "n/a" if v is None else f.format(v)


def describe_cfg(runs):
  return {m[0]: stats.describe(pooled(runs, m[0])) for m in HEADLINE} | {
      "load_ms": stats.describe([r["summary"].get("load_ms") for r in runs]),
      "peak_rss_mb_lifetime": stats.describe([r["summary"].get("peak_rss_mb_lifetime") for r in runs]),
      "idle_rss_mb": stats.describe([(r["summary"].get("idle_after_load") or {}).get("rss_mb") for r in runs]),
      "idle_rss_anon_mb": stats.describe([(r["summary"].get("idle_after_load") or {}).get("rss_anon_mb") for r in runs]),
      "cgroup_peak_mb": stats.describe([
          v for r in runs
          if (v := r["summary"].get("cgroup_memory_peak_mb")) is not None and v > 0
      ]),
      "system_cpu_busy_frac": stats.describe(pooled(runs, "system_cpu_busy_frac")),
      "round_medians_decode": [((r["summary"]["metrics"].get("decode_tps") or {}).get("median")) for r in runs],
      "round_medians_ttft": [((r["summary"]["metrics"].get("ttft_ms") or {}).get("median")) for r in runs],
  }


# ------------------------------------------------------------------- charts


def _label(cid: str) -> str:
  return cid.replace("/", "\n")


def bar_chart(path: Path, cfg_ids, agg, metrics, title):
  panel_w = max(10.5, 0.72 * len(cfg_ids))
  fig, axes = plt.subplots(len(metrics), 1, figsize=(panel_w, 4.2 * len(metrics)), squeeze=False)
  axes_flat = [axes[i][0] for i in range(len(metrics))]
  rot = 25 if len(cfg_ids) > 6 else 0
  for ax, (m, lab, unit, hib, _) in zip(axes_flat, metrics):
    xs, meds, lo, hi, cols = [], [], [], [], []
    for cid in cfg_ids:
      d = agg[cid].get(m)
      if not d:
        continue
      xs.append(_label(cid))
      meds.append(d["median"])
      lo.append(d["median"] - d["min"])
      hi.append(d["max"] - d["median"])
      cols.append(HIST_STYLE.get(cid, (COLORS.get(cid.split("/")[0], "gray"),))[0])
    ax.bar(xs, meds, yerr=[lo, hi], color=cols, capsize=4, alpha=0.9)
    if meds:
      ax.set_ylim(0, max(m_val + h_val for m_val, h_val in zip(meds, hi)) * 1.14)
    for x, v in zip(xs, meds):
      ax.annotate(f"{v:.1f}" if v < 100 else f"{v:.0f}", (x, v), ha="center", va="bottom",
                  fontsize=8, xytext=(0, 3), textcoords="offset points")
    ax.set_title(f"{lab} ({unit}) {'↑' if hib else '↓'}", fontsize=10.5, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8, rotation=rot)
    ax.grid(axis="y", alpha=0.3)
  fig.suptitle(title + "  (median; whiskers = min-max over all measured requests)", fontsize=11, fontweight="bold")
  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)


def sequence_charts(out: Path, data: dict):
  for metric, phase in (("prefill_tps", "Prefill"), ("decode_tps", "Decode")):
    fig, ax = plt.subplots(figsize=(10, 5))
    lengths = data["prompt_tokens"]
    cfg_ids = data["points"][str(lengths[0])]["configs"]
    for cid in cfg_ids:
      points = [data["points"][str(n)]["configs"][cid].get(metric) for n in lengths]
      meds = [p["median"] if p else float("nan") for p in points]
      lo = [p["median"] - p["min"] if p else 0 for p in points]
      hi = [p["max"] - p["median"] if p else 0 for p in points]
      col, mk, ls = HIST_STYLE.get(cid, (COLORS.get(cid.split("/")[0], "gray"), "o", "-"))
      ax.errorbar(lengths, meds, yerr=[lo, hi], marker=mk, ls=ls, capsize=3,
                  color=col, label=cid)
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths)
    ax.set_xticklabels([f"{n:,}" for n in lengths], rotation=30)
    ax.set_xlabel("Prefill sequence length N (tokens, including BOS)")
    ax.set_ylabel(f"{phase} throughput (tokens/s)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{phase} vs input length; fixed {data['gen_tokens']}-token decode\n"
                 "Median; whiskers = min-max; unavailable points are gaps")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8 if len(cfg_ids) > 6 else 9)
    fig.tight_layout()
    fig.savefig(out / f"sequence_{phase.lower()}.png", dpi=130)
    plt.close(fig)


def sequence_section(data: dict) -> str:
  if not data:
    return ""
  cfg_ids = list(next(iter(data["points"].values()))["configs"])
  dates = sorted({r["meta"]["started_at"][:10] for p in data["points"].values()
                  for c in p["configs"].values() for r in c["runs"]})
  lines = ["#### Input-length throughput sweep\n",
           "Sweep measurement dates (UTC): " + ", ".join(dates) + ".\n",
           f"Fixed **{data['gen_tokens']} generated tokens**, greedy with stop tokens ignored; "
           f"common context capacity **{data['ctx']:,}**. Model artifacts and thread counts "
           "are unchanged from the configurations above. These are reference configurations, "
           "not matched weights. The original headline workload remains unchanged.\n"]
  if data.get("select_fastest_by_dtype") or data.get("min_accuracy"):
    min_acc = data.get("min_accuracy")
    if isinstance(min_acc, dict):
      gate_desc = ", ".join(f"{k} ≥ {v:.0%}" for k, v in sorted(min_acc.items()))
    elif isinstance(min_acc, (int, float)):
      gate_desc = f"accuracy ≥ {min_acc:.0%}"
    else:
      gate_desc = "the accuracy gate"
    lines.append(f"Swept configurations are the empirically fastest configuration for each "
                 f"`(framework, dtype)` pair that cleared the minimum accuracy gate ({gate_desc}).\n")
  lines += [f"Each point uses {data['rounds']} interleaved processes per configuration, "
            f"{data['warmup']} warm-ups and {data['repetitions']} measured requests per process. "
            "Long inputs repeat the same public-domain passage deterministically. "
            "N includes BOS and chat-template tokens. Each request starts with an empty KV cache.\n",
            "Prefill tokens/s = N / time to first token (includes tokenization and first-token "
            f"generation). Decode tokens/s = {data['gen_tokens'] - 1} / time from the first to the last generated "
            "token. Rates are medians; brackets show min-max. Failed, truncated or wrong-length "
            "processes are excluded, never replaced by estimated rates.\n"]
  for metric, phase in (("prefill_tps", "Prefill"), ("decode_tps", "Decode")):
    lines += [f"##### {phase} (tokens/s)\n",
              "| N | " + " | ".join(f"`{cid}`" for cid in cfg_ids) + " |",
              "| --- " * (len(cfg_ids) + 1) + "|"]
    for n in data["prompt_tokens"]:
      cells = []
      for cid in cfg_ids:
        point = data["points"][str(n)]["configs"][cid]
        d = point[metric]
        value = (f"{d['median']:.2f} [{d['min']:.2f}-{d['max']:.2f}]" if d else "n/a")
        if point["status"] != "complete":
          value += f" ({point['status']}, n={point['n_requests']})"
        cells.append(value)
      lines.append(f"| {n:,} | " + " | ".join(cells) + " |")
    lines.append(f"\n![{phase} by input length](sequence_{phase.lower()}.png)\n")
  return "\n".join(lines) + "\n"


def prefix_cache_chart(path: Path, data: dict):
  if not data or not data.get("configs"):
    return
  cfg_ids = [cid for cid, c in data["configs"].items() if c.get("warm_ttft_ms")]
  if not cfg_ids:
    return
  panel_w = max(10.5, 0.75 * len(cfg_ids))
  fig, axes = plt.subplots(3, 1, figsize=(panel_w, 4.2 * 3))
  rot = 25 if len(cfg_ids) > 6 else 0
  xs = [_label(cid) for cid in cfg_ids]
  x_pos = list(range(len(cfg_ids)))
  w = 0.36
  panels = [
      ("ttft_ms", "TTFT (ms) ↓", False),
      ("prefill_tps", "Effective Prefill (tok/s) ↑", True),
      ("e2e_ms", "End-to-end (ms) ↓", False),
  ]
  for ax, (metric, title, _) in zip(axes, panels):
    cold_key = f"cold_{metric}"
    warm_key = "warm_effective_prefill_tps" if metric == "prefill_tps" else f"warm_{metric}"
    c_meds = [(data["configs"][cid].get(cold_key) or {}).get("median") or 0.0 for cid in cfg_ids]
    w_meds = [(data["configs"][cid].get(warm_key) or {}).get("median") or 0.0 for cid in cfg_ids]
    cols = [HIST_STYLE.get(cid, (COLORS.get(cid.split("/")[0], "gray"),))[0] for cid in cfg_ids]
    ax.bar([x - w / 2 for x in x_pos], c_meds, width=w, color=cols, alpha=0.45,
           edgecolor="black", linewidth=0.5, label="Cold (0 cached)")
    ax.bar([x + w / 2 for x in x_pos], w_meds, width=w, color=cols, alpha=0.95,
           edgecolor="black", linewidth=0.5, label=f"Warm ({data['prefix_tokens']} cached)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xs, fontsize=8, rotation=rot)
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    if metric == "ttft_ms":
      ax.legend(fontsize=8.5)
  fig.suptitle(
      f"Prefix caching: {data['prompt_tokens']}-token prompt "
      f"({data['prefix_tokens']} cached prefix + {data['suffix_tokens']} new suffix) -> "
      f"{data['gen_tokens']} generated tokens",
      fontsize=11,
      fontweight="bold",
  )
  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)


def prefix_cache_section(data: dict) -> str:
  if not data or not data.get("configs"):
    return ""
  p_tok = data["prompt_tokens"]
  c_tok = data["prefix_tokens"]
  s_tok = data["suffix_tokens"]
  g_tok = data["gen_tokens"]
  lines = [
      "### Prefix caching (warm vs. cold KV cache)\n",
      f"Measures multi-turn / shared-system-prompt inference where **{c_tok:,} prefix tokens** "
      f"of a **{p_tok:,}-token prompt** are already present in the KV cache, and each request "
      f"appends a distinct **{s_tok:,}-token suffix** before generating **{g_tok:,} tokens** "
      f"(greedy, stop tokens ignored). The headline suite above remains strictly stateless "
      f"(fresh KV cache per request).\n",
      f"- **Cold TTFT / Cold prefill tok/s**: full {p_tok:,}-token prompt evaluated from an empty KV cache.\n"
      f"- **Warm TTFT**: latency to first generated token when reusing the {c_tok:,}-token prefix KV cache and prefilling only the {s_tok:,}-token suffix.\n"
      f"- **Effective prefill tok/s**: `{p_tok} / warm_ttft_s` (effective prompt processing rate seen by the caller).\n"
      f"- **Incremental suffix tok/s**: `{s_tok} / warm_ttft_s` (actual compute throughput on the {s_tok} uncached suffix tokens at KV position {c_tok}..{p_tok}).\n",
      "| Config | Cold TTFT (ms) ↓ | Warm TTFT (ms) ↓ | TTFT speedup ↑ | Cold prefill (tok/s) ↑ | Warm effective prefill (tok/s) ↑ | Warm suffix prefill (tok/s) ↑ | Cold E2E (ms) ↓ | Warm E2E (ms) ↓ | E2E speedup ↑ |",
      "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
  ]
  for cid, c in data["configs"].items():
    def _cell(d, f="{:.0f}"):
      if not d:
        return "n/a"
      return f"{f.format(d['median'])} [{f.format(d['min'])}–{f.format(d['max'])}]"

    def _ratio(r):
      if not r:
        return "n/a"
      return f"**{r[0]:.2f}×** [{r[1]:.2f}, {r[2]:.2f}]"

    lines.append(
        f"| `{cid}` | {_cell(c.get('cold_ttft_ms'), '{:.0f}')} | "
        f"{_cell(c.get('warm_ttft_ms'), '{:.0f}')} | {_ratio(c.get('ttft_speedup'))} | "
        f"{_cell(c.get('cold_prefill_tps'), '{:.1f}')} | "
        f"{_cell(c.get('warm_effective_prefill_tps'), '{:.1f}')} | "
        f"{_cell(c.get('warm_suffix_prefill_tps'), '{:.1f}')} | "
        f"{_cell(c.get('cold_e2e_ms'), '{:.0f}')} | "
        f"{_cell(c.get('warm_e2e_ms'), '{:.0f}')} | {_ratio(c.get('e2e_speedup'))} |"
    )
  lines.append("\n![Prefix caching comparison](prefix_cache.png)\n")
  for cid, c in data["configs"].items():
    for err in c.get("errors", []):
      lines.append(f"\n- `{cid}`: {err.replace(chr(10), ' ')}")
  return "\n".join(lines) + "\n"


def timeline_chart(path: Path, cfg_ids, cfg_runs):
  rows = [cid for cid in cfg_ids if cfg_runs.get(cid)]
  fig, (ax_mem, ax_cpu) = plt.subplots(2, 1, figsize=(11.5, 9.0))
  for cid in rows:
    run = cfg_runs[cid][0]
    ts = read_timeseries(run["dir"] / "timeseries.csv")
    if not ts.get("t_ns"):
      continue
    t0 = ts["t_ns"][0]
    t = [(x - t0) / 1e9 for x in ts["t_ns"]]
    rss_mb = [
        ((a or 0) + (f or 0)) / 1024.0
        for a, f in zip(ts["rss_anon_kb"], ts["rss_file_kb"])
    ]
    col, _, ls = HIST_STYLE.get(cid, (COLORS.get(cid.split("/")[0], "gray"), "o", "-"))
    ax_mem.plot(t, rss_mb, color=col, ls=ls, lw=1.6, alpha=0.9, label=cid)

    ticks = ts["cpu_ticks"]
    cores_t, cores = [], []
    for j in range(1, len(t)):
      dt_ = t[j] - t[j - 1]
      if dt_ > 0:
        cores_t.append(t[j])
        cores.append((ticks[j] - ticks[j - 1]) / 100.0 / dt_)
    # Smooth over 5 samples for readability.
    sm = [statistics.fmean(cores[max(0, k - 2):k + 3]) for k in range(len(cores))]
    ax_cpu.plot(cores_t, sm, color=col, ls=ls, lw=1.4, alpha=0.88, label=cid)

  ax_mem.set_ylabel("Total RSS (MB)")
  ax_mem.set_xlabel("Seconds since process start")
  ax_mem.set_title("Memory (Total RSS) over time across configurations (round 0 process)",
                   fontsize=10.5, fontweight="bold")
  ax_mem.grid(alpha=0.3)
  ax_mem.legend(fontsize=8, ncol=3, loc="upper right", framealpha=0.9)

  ax_cpu.set_ylabel("CPU cores busy")
  ax_cpu.set_xlabel("Seconds since process start")
  ax_cpu.set_title("CPU utilization (cores busy) over time across configurations (round 0 process)",
                   fontsize=10.5, fontweight="bold")
  ax_cpu.grid(alpha=0.3)
  ax_cpu.legend(fontsize=8, ncol=3, loc="upper right", framealpha=0.9)

  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)


def _requests(d: Path):
  out = []
  p = d / "requests.jsonl"
  if p.is_file():
    for line in open(p):
      r = json.loads(line)
      if r.get("ok") is not False and "t_req_ns" in r:
        out.append(r)
  return out


THREAD_SWEEP_STYLES = {
    "llama.cpp matched": ("#d9822b", "-o"),
    "llama.cpp q4_0": ("#a85a10", "--s"),
    "litert-lm matched": ("#2b6cd9", "-o"),
    "litert-lm ynnpack": ("#13408f", "--s"),
    "xla ynnpack-bf16": ("#2e8b57", "-o"),
    "xla onednn-bf16": ("#1b5e20", "--s"),
    "xla ynnpack-int8": ("#66bb6a", ":^"),
    "openvino int4": ("#0095a8", "-o"),
    "openvino int8": ("#006064", "--s"),
    "onnxruntime q4_k_m": ("#9c4d9f", "-d"),
}


def scaling_chart(path: Path, runs, machine):
  series = collections.defaultdict(dict)
  for r in sorted(runs, key=lambda r: r["meta"].get("started_at", "")):
    m = r["meta"]
    if m.get("track") != "sweep-threads":
      continue
    label = m["label"]
    name, sep, t = label.rpartition("-t")
    if not sep or not t.isdigit():
      continue
    key = f"{m['framework']} {name}"
    s = r["summary"]["metrics"]
    if not s.get("decode_tps"):
      continue
    series[key][int(t)] = (s["prefill_tps"]["median"], s["decode_tps"]["median"],
                           (s.get("cpu_cores_decode") or {}).get("median"))
  if not series:
    return False
  fig, axes = plt.subplots(3, 1, figsize=(10.5, 4.0 * 3))
  for key, pts in sorted(series.items()):
    ts = sorted(pts)
    fw = key.split()[0]
    col, st = THREAD_SWEEP_STYLES.get(key, (COLORS.get(fw, "gray"), "-o"))
    for k, ax in enumerate(axes):
      ax.plot(ts, [pts[t][k] for t in ts], st, color=col, label=key)
  for ax, title in zip(axes, ["Prefill tok/s (1024-token prompt)", "Decode tok/s (64 tokens)",
                               "CPU cores busy during decode"]):
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("threads")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
  fig.suptitle(f"Thread scaling on {machine} (sweep: 1 warm-up + 2 measured requests per point)",
               fontsize=11, fontweight="bold")
  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)
  return True


def accuracy_chart(path: Path, acc_rows, refs=None):
  tasks = sorted({a["task"] for a in acc_rows})
  max_rows = max((sum(1 for a in acc_rows if a["task"] == t) for t in tasks), default=4)
  panel_w = max(10.5, 0.75 * max_rows)
  fig, axes = plt.subplots(len(tasks), 1, figsize=(panel_w, 4.4 * len(tasks)), squeeze=False)
  axes_flat = [axes[i][0] for i in range(len(tasks))]
  for ax, task in zip(axes_flat, tasks):
    rows = [a for a in acc_rows if a["task"] == task]
    xs = [f"{a['id']}\n(n={a['n']})" for a in rows]
    ys = [a["accuracy"] * 100 for a in rows]
    err = [a["ci95"] * 100 for a in rows]
    ax.bar(xs, ys, yerr=err, capsize=4, color=[COLORS.get(a["id"].split("/")[0], "gray") for a in rows])
    if ys:
      ax.set_ylim(0, max(y + e for y, e in zip(ys, err)) * 1.15)
    for x, y, e in zip(xs, ys, err):
      ax.annotate(f"{y:.1f}%", (x, y + e), ha="center", va="bottom",
                  fontsize=8, xytext=(0, 3), textcoords="offset points")
    # Published non-thinking bf16 references for the same benchmark.
    for r in (refs or {}).get("results", []):
      if r["benchmark"].lower() == task and "thinking on" not in r["setup"]:
        ax.axhline(r["value"], ls="--", lw=1, color="gray")
        ax.annotate(f"published {r['value']:.1f}% ({r['source_name'].split(',')[0]}; {r['setup'][:38]}…)",
                    (0.01, r["value"]), xycoords=("axes fraction", "data"), fontsize=7,
                    color="dimgray", va="bottom")
    ax.set_title(f"{task.upper()} accuracy (95% CI)", fontsize=10.5, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8, rotation=25 if len(rows) > 6 else 0)
    ax.grid(axis="y", alpha=0.3)
  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)


HIST_STYLE = {  # config -> (color, marker, linestyle)
    "llama.cpp/matched": ("#d9822b", "o", "-"),
    "llama.cpp/fastest": ("#a85a10", "s", "--"),
    "llama.cpp/stock-q4_0": ("#f0b070", "^", ":"),
    "llama.cpp/q8_0": ("#c46d1f", "v", "-."),
    "llama.cpp/int8": ("#c46d1f", "v", "-."),
    "llama.cpp/bf16": ("#8c4600", "d", ":"),
    "litert-lm/matched": ("#2b6cd9", "o", "-"),
    "litert-lm/fastest": ("#13408f", "s", "--"),
    "litert-lm/ynnpack": ("#5c9aff", "^", "-."),
    "xla/matched-ynnpack": ("#2e8b57", "o", "-"),
    "xla/matched-onednn": ("#1b5e20", "s", "--"),
    "xla/bf16-ynnpack": ("#43a047", "^", "-"),
    "xla/bf16-onednn": ("#2e7d32", "v", "--"),
    "xla/int8-ynnpack": ("#66bb6a", "d", "-."),
    "xla/int8-onednn": ("#388e3c", "p", ":"),
    "xla/fp32-ynnpack": ("#81c784", "x", ":"),
    "xla/fp32-onednn": ("#4caf50", "+", ":"),
    "openvino/int4": ("#0095a8", "s", "--"),
    "openvino/int8": ("#006064", "o", "-"),
    "openvino/bf16": ("#00acc1", "v", "-."),
    "openvino/fp16": ("#4dd0e1", "^", ":"),
    "openvino/fp32": ("#26c6da", "P", ":"),
    "onnxruntime/q4_k_m": ("#9c4d9f", "d", "-"),
    "onnxruntime/fp16": ("#ba68c8", "v", ":"),
}


def history_chart(path: Path, history: list[dict]):
  """One x position per report (ordered by date), so irregular cadence and a
  single first report still render legibly."""
  metrics = [("decode_tps", "Decode tok/s ↑"), ("prefill_tps", "Prefill tok/s ↑"),
             ("ttft_ms", "TTFT ms ↓"), ("rss_max_mb", "Peak RSS MB ↓")]
  reports = sorted({(h["date"], h["report_id"]) for h in history})
  xpos = {rid: i for i, (_, rid) in enumerate(reports)}
  fig, axes = plt.subplots(len(metrics), 1, figsize=(10.5, 3.8 * len(metrics)))
  by_cfg = collections.defaultdict(list)
  for h in history:
    by_cfg[h["config"]].append(h)
  for ax, (m, title) in zip(axes, metrics):
    for cid, pts in sorted(by_cfg.items()):
      pts = sorted((h for h in pts if h["metrics"].get(m)), key=lambda h: xpos[h["report_id"]])
      if not pts:
        continue
      col, mk, ls = HIST_STYLE.get(cid, (COLORS.get(cid.split("/")[0], "gray"), "o", "-"))
      xs = [xpos[h["report_id"]] for h in pts]
      ys = [h["metrics"][m]["median"] for h in pts]
      lo = [h["metrics"][m]["median"] - h["metrics"][m]["min"] for h in pts]
      hi = [h["metrics"][m]["max"] - h["metrics"][m]["median"] for h in pts]
      ax.errorbar(xs, ys, yerr=[lo, hi], marker=mk, ls=ls, color=col, label=cid, capsize=3, ms=6)
    ax.set_xticks(range(len(reports)))
    ax.set_xticklabels([d for d, _ in reports], rotation=20, fontsize=8)
    ax.set_xlim(-0.5, max(len(reports) - 0.5, 0.5))
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.3)
  axes[0].legend(fontsize=8, ncol=3)
  fig.suptitle("Performance over time (one column per report; whiskers = min-max over requests)",
               fontsize=11, fontweight="bold")
  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)


# ------------------------------------------------------------------ history


def update_history(hist_path: Path, entries: list[dict]) -> list[dict]:
  hist_path.parent.mkdir(parents=True, exist_ok=True)
  existing = []
  if hist_path.is_file():
    existing = [json.loads(l) for l in hist_path.read_text().splitlines() if l.strip()]
  keys = {(e["report_id"], e["config"]) for e in entries}
  existing = [e for e in existing if (e["report_id"], e["config"]) not in keys] + entries
  existing.sort(key=lambda e: (e["date"], e["config"]))
  hist_path.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in existing))
  return existing


def history_changes(history: list[dict], report_id: str) -> list[str]:
  notes = []
  by_cfg = collections.defaultdict(list)
  for h in history:
    by_cfg[h["config"]].append(h)
  for cid, pts in sorted(by_cfg.items()):
    pts.sort(key=lambda h: h["date"])
    cur = [p for p in pts if p["report_id"] == report_id]
    prev = [p for p in pts if p["date"] < (cur[0]["date"] if cur else "")]
    if not cur or not prev:
      continue
    a, b = prev[-1], cur[0]
    fa = (a.get("reproducibility") or {}).get("greedy_output_fingerprint")
    fb = (b.get("reproducibility") or {}).get("greedy_output_fingerprint")
    if fa and fb and fa != fb:
      why = ("same model file and framework commit — investigate"
             if a.get("model_sha256") == b.get("model_sha256")
             and a.get("framework_commit") == b.get("framework_commit")
             else "model or framework changed")
      notes.append(f"{cid}: greedy outputs on the reproducibility prompts changed since "
                   f"{a['date']} ({why}).")
    for m in ("decode_tps", "prefill_tps", "ttft_ms", "rss_max_mb"):
      ma, mb = a["metrics"].get(m), b["metrics"].get(m)
      if not ma or not mb:
        continue
      rel = (mb["median"] - ma["median"]) / ma["median"]
      noise = 3 * max(ma.get("cv") or 0, mb.get("cv") or 0)
      if abs(rel) > max(CHANGE_THRESHOLD, noise):
        notes.append(f"{cid} {m}: {ma['median']:.4g} -> {mb['median']:.4g} ({rel:+.1%}) "
                     f"since {a['date']} [threshold {max(CHANGE_THRESHOLD, noise):.1%}]")
  return notes


# ------------------------------------------------------------------- report


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--machine", required=True)
  ap.add_argument("--suite", type=Path, required=True)
  ap.add_argument("--results", type=Path, default=Path("/results"))
  ap.add_argument("--out", type=Path, default=Path("/reports"))
  ap.add_argument("--since", help="only use suite runs started at/after this UTC timestamp")
  ap.add_argument("--date", default=dt.date.today().isoformat())
  ap.add_argument("--extra-notes", type=Path, help="markdown appended to the caveats section")
  ap.add_argument("--models", type=Path, default=Path("/models"),
                  help="models dir holding aeb/*.report.json and aeb/litert-parity.json")
  args = ap.parse_args(argv)

  suite = json.loads(args.suite.read_text())
  report_id = f"{args.date}-{suite['name']}-{args.machine}"
  out = args.out / report_id
  out.mkdir(parents=True, exist_ok=True)
  runs = load_perf(args.results, args.machine)
  cfg_runs = config_runs(runs, suite, args.since)
  cfg_ids = sorted([c["id"] for c in suite["configs"] if cfg_runs.get(c["id"])],
                   key=lambda cid: (TRACK_ORDER.get(_track(suite, cid), 9), cid))
  agg = {cid: describe_cfg(cfg_runs[cid]) for cid in cfg_ids}
  sequence = sequence_data(runs, suite, args.since)
  prefix_cache = prefix_cache_data(runs, suite, cfg_runs, agg, args.since)

  # ---- accuracy: latest complete set of shards per (config, task, limit)
  acc_rows, acc_preds = merge_accuracy(load_accuracy(args.results, args.machine), agg)
  agreement = paired_agreement(acc_preds)

  repro = load_repro(args.results, args.machine, set(cfg_ids))
  repro_tbl, repro_cross = repro_rows(repro, cfg_ids) if repro else ([], [])

  # ---- charts
  bar_chart(out / "throughput.png", cfg_ids, agg, [HEADLINE[1], HEADLINE[2]], "Throughput")
  bar_chart(out / "latency.png", cfg_ids, agg, [HEADLINE[0], HEADLINE[3], HEADLINE[4]], "Latency")
  bar_chart(out / "cpu.png", cfg_ids, agg, [HEADLINE[5], HEADLINE[6], HEADLINE[7]], "CPU use")
  bar_chart(out / "memory.png", cfg_ids, agg, [HEADLINE[9], HEADLINE[10], HEADLINE[11]], "Memory")
  timeline_chart(out / "timeline.png", cfg_ids, cfg_runs)
  if sequence:
    sequence_charts(out, sequence)
  if prefix_cache:
    prefix_cache_chart(out / "prefix_cache.png", prefix_cache)
  sweep_thread_runs = [
      r for r in runs
      if r["meta"].get("track") == "sweep-threads"
      and (not r["meta"].get("notes") or r["meta"].get("notes", "").startswith(f"suite={suite['name']}"))
      and (args.since is None or r["meta"].get("started_at", "") >= args.since)
  ]
  has_scaling = scaling_chart(
      out / "thread_scaling.png", sweep_thread_runs, args.machine)
  if acc_rows:
    refs_path = args.suite.parent / suite["references"] if suite.get("references") else None
    accuracy_chart(out / "accuracy.png", acc_rows,
                   json.loads(refs_path.read_text()) if refs_path and refs_path.is_file() else None)
  has_pareto = pareto_chart(out / "pareto.png", suite, cfg_ids, cfg_runs, agg, acc_rows)

  # ---- history
  def first_meta(cid):
    return cfg_runs[cid][0]["meta"]

  entries = []
  for cid in cfg_ids:
    meta = first_meta(cid)
    entries.append({
        "report_id": report_id, "date": args.date, "suite": suite["name"],
        "machine": args.machine, "config": cid, "track": _track(suite, cid),
        "framework_ref": meta["build"].get("ref"), "framework_commit": meta["build"].get("commit"),
        "harness_sha256": meta["build"].get("harness_sha256"),
        "repo_rev": meta["build"].get("repo_rev"), "model_sha256": meta.get("model_sha256"),
        "n_requests": agg[cid]["e2e_ms"]["n"] if agg[cid].get("e2e_ms") else 0,
        "metrics": {k: _slim(agg[cid].get(k)) for k, *_ in HEADLINE},
        "accuracy": {a["task"]: {"acc": a["accuracy"], "n": a["n"]}
                     for a in reversed(acc_rows) if a["id"] == cid},
        "reproducibility": next(({"greedy": repro_verdict(r, "greedy").strip("*").split(" (")[0],
                                  "seeded": repro_verdict(r, "seeded").strip("*").split(" (")[0],
                                  "greedy_output_fingerprint": r.get("greedy_fingerprint")}
                                 for r in repro_tbl if r["id"] == cid), None),
    })
  hist_path = args.out / "history" / suite["name"] / f"{args.machine}.jsonl"
  history = update_history(hist_path, entries)
  history_chart(out / "history.png", [h for h in history if h["machine"] == args.machine])
  changes = history_changes(history, report_id)

  # ---- data.json (committed alongside the report: raw per-request metrics)
  data = {"report_id": report_id, "suite": suite, "machine": args.machine,
          "configs": {cid: {"runs": [{"run_dir": str(r["dir"]), "meta": _meta_slim(r["meta"]),
                                      "summary": r["summary"]} for r in cfg_runs[cid]],
                            "aggregate": agg[cid]} for cid in cfg_ids},
          "sequence_sweep": sequence,
          "prefix_cache": prefix_cache,
          "accuracy": acc_rows, "accuracy_agreement": agreement,
          "reproducibility": {"rows": repro_tbl, "cross_framework": repro_cross}}
  (out / "data.json").write_text(json.dumps(data, indent=1, default=str) + "\n")

  extra = args.extra_notes.read_text() if (args.extra_notes and args.extra_notes.is_file()) else ""
  args._quant_md = (quant_section(args.models / "aeb")
                    if any(c["track"] == "matched" for c in suite["configs"]) else "")
  refs_path = args.suite.parent / suite["references"] if suite.get("references") else None
  args._refs = json.loads(refs_path.read_text()) if refs_path and refs_path.is_file() else None
  md_text = render_markdown(
      suite, args, report_id, cfg_ids, cfg_runs, agg, acc_rows, has_scaling, history, changes,
      hist_path, extra, agreement, (repro_tbl, repro_cross), sequence, prefix_cache,
      has_pareto=has_pareto)
  (out / "README.md").write_text(md_text, encoding="utf-8")
  html_doc = markdown_to_html(md_text, base_dir=out, embed_images=True)
  (out / "README.html").write_text(html_doc, encoding="utf-8")
  (out / "index.html").write_text(html_doc, encoding="utf-8")
  print(f"wrote {out} (README.md, README.html, index.html)")
  return 0


def _track(suite, cid):
  return next(c["track"] for c in suite["configs"] if c["id"] == cid)


def _slim(d):
  if not d:
    return None
  return {k: d[k] for k in ("n", "median", "min", "max", "cv")}


def _meta_slim(m):
  m = dict(m)
  ready = dict(m.get("driver_ready") or {})
  if "info" in ready:
    ready["info"] = {k: v for k, v in ready["info"].items() if k != "system_info"}
  m["driver_ready"] = ready
  return m


def reference_section(refs: dict, acc_rows) -> str:
  """Publicly reported accuracy for the model, next to what this harness measured."""
  L = ["### Publicly reported accuracy\n", refs.get("summary", "") + "\n"]
  measured = {}
  for a in acc_rows:
    if a["limit"] == 0 or a["task"] == "gsm8k":
      measured.setdefault(a["task"], []).append(f"`{a['id']}` {a['accuracy']:.1%} (n={a['n']})")
  L.append("| Benchmark | Published | Model variant | Setup | Source | Measured here (non-thinking, greedy) |")
  L.append("| --- | --- | --- | --- | --- | --- |")
  for r in refs["results"]:
    key = r.get("task_key")
    here = "; ".join(measured.get(key, [])) if key else "—"
    val = r["value"] if isinstance(r["value"], str) else f"{r['value']:.1f}%"
    L.append(f"| {r['benchmark']} | {val} | {r['variant']} | {r['setup']} | "
             f"[{r['source_name']}]({r['source']}) | {here or '—'} |")
  L.append("")
  if refs.get("notes"):
    L.append("\n".join(f"- {n}" for n in refs["notes"]) + "\n")
  return "\n".join(L)


def quant_section(d: Path) -> str:
  """Summarize the weight-equivalence evidence produced by aeb-prep-matched-gguf."""
  parity = _read_json(d / "litert-parity.json")
  convs = {p.name.replace(".report.json", ""): _read_json(p) for p in sorted(d.glob("*.report.json"))}
  if not parity and not convs:
    return ""
  L = ["## Quantization equivalence\n"]
  if parity:
    L.append(f"- `.litertlm` vs `google/gemma-4-E2B-it-qat-mobile-transformers`: "
             f"**{parity['tensors_compared']} quantized tensors compared element by element, "
             f"{'all integers and per-channel scales identical' if parity['all_identical'] else str(len(parity['mismatches'])) + ' mismatches'}** "
             f"(attention, MLP, per-layer gate/projection for every layer, lm_head, token and "
             f"per-layer embedding tables).")
  for name, rep in convs.items():
    t = rep["tensors"]
    q = [v for v in t.values() if "bits" in v]
    types = collections.Counter(f"int{v['bits']}→{v['type']}" for v in q)
    exact = all(v.get("ints_roundtrip_exact", False) for v in q)
    above = [k for k, v in t.items() if (v.get("scale_fp16_max_rel_err") or 0) > 4.9e-4]
    L.append(f"- `{name}.gguf`: {len(q)} quantized tensors "
             f"({', '.join(f'{n}× {k}' for k, n in sorted(types.items()))}); integer round-trip "
             f"{'exact for every tensor' if exact else '**NOT exact**'}; per-row scales stored as "
             f"fp16 (LiteRT: fp32), worst relative scale error {rep['max_scale_fp16_rel_err']:.2%}"
             f"{'' if not above else f' (only {len(above)} tensors exceed fp16 rounding, from subnormal scales: ' + ', '.join(above) + ')'}.")
  L.append("")
  L.append("| Tensor group | Checkpoint / LiteRT-LM | Matched GGUF | `-q4` GGUF | Stock ggml-org Q4_0 |")
  L.append("| --- | --- | --- | --- | --- |")
  L.append("| Token embedding, lm_head | INT2 per-row (separate lm_head) | Q2_K, d = dmin = s | Q4_0, d = s | Q8_0 (tied) |")
  L.append("| Attention q/k/v/o | INT4 per-channel | Q4_0, d = s | Q4_0, d = s | Q4_0 (block-32 scales) |")
  L.append("| MLP layers 0-14 | INT4 per-channel | Q4_0, d = s | Q4_0, d = s | Q4_0 |")
  L.append("| MLP layers 15-34 (double-wide) | INT2 per-channel | Q2_K | Q4_0 | Q4_0 |")
  L.append("| Per-layer embeddings | INT4 per (token, layer) | Q4_0 | Q4_0 | Q4_0 |")
  L.append("| Per-layer gate / projection | INT8 per-channel | Q8_0 | Q8_0 | Q4_0 |")
  L.append("| Per-layer model projection | INT8 per-channel (LiteRT PTQ) | Q8_0 (same ints) | Q8_0 | BF16 |")
  L.append("| Norms, layer scalars | BF16 values (fp32 tensors in LiteRT) | F32, same values | F32, same values | F32 (different QAT checkpoint) |")
  L.append("| Activations | static per-tensor INT8 (QAT scales) | dynamic Q8_0/Q8_K per block | same | same |")
  L.append("| KV cache | INT8, static per-tensor scale | Q8_0 (matched track) | F16 (fastest) | F16 |")
  L.append("")
  return "\n".join(L)


def ratio_line(agg, cfg_runs, a, b, metric, higher_better):
  va, vb = pooled(cfg_runs[a], metric), pooled(cfg_runs[b], metric)
  if not va or not vb:
    return "n/a"
  r, lo, hi = stats.bootstrap_ratio_ci(va, vb)
  return f"{r:.2f}× [{lo:.2f}, {hi:.2f}]"


def _config_dtype_info(suite: dict, cid: str, meta: dict) -> tuple[str, str, str, str]:
  """Return (dtype_group, checkpoint_label, weights_acts_kv, threads_str) for cid."""
  cfg_dict = next((c for c in suite.get("configs", []) if c["id"] == cid), {})
  raw_dtype = (cfg_dict.get("dtype") or "").lower()
  cid_lo = cid.lower()
  mp = meta.get("model_path") or ""
  cmd = [str(x) for x in (meta.get("driver_cmd") or [])]
  info = (meta.get("driver_ready") or {}).get("info") or {}
  fw = meta.get("framework") or cid.split("/")[0]

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
      threads = cmd[idx + 1]
  threads_str = str(threads) if threads is not None else "—"

  if "qat-mobile" in mp or "litert-lm" in mp or fw == "litert-lm":
    checkpoint = "`qat-mobile`"
  elif fw == "llama.cpp" or mp.endswith(".gguf"):
    checkpoint = "`it` (GGUF)"
  elif fw == "openvino" or "-ov" in mp:
    checkpoint = "`it` (OV)"
  elif fw == "onnxruntime" or "onnx" in mp:
    checkpoint = "`it` (ONNX)"
  else:
    checkpoint = "`it` (HF)"

  if fw == "xla":
    if group == "fp32":
      wak = "`fp32 / fp32 / fp32`"
    elif group == "bf16":
      wak = "`bf16 / bf16 / bf16`"
    elif group == "int8":
      wak = "`int8 / int8 / bf16`"
    else:
      wak = "`int2_int4_int8 / int8 / bf16`"
  elif fw == "openvino":
    if group == "fp32":
      wak = "`fp32 / fp32 / fp16`"
    elif group == "bf16":
      wak = "`bf16 / bf16 / bf16`"
    elif group == "fp16":
      wak = "`fp16 / fp16 / fp16`"
    elif group == "int8":
      wak = "`int8_asym / bf16 / int8`"
    else:
      wak = "`int4_asym / bf16 / int8`"
  elif fw == "litert-lm":
    wak = "`int2_int4_int8 / int8 / int8`"
  elif fw == "onnxruntime":
    wak = "`fp16 / fp16 / fp16`" if group == "fp16" else "`q4_k_m / fp32 / fp32`"
  elif fw == "llama.cpp":
    kv = str(info.get("cache_type_k") or "")
    if not kv and "--cache-type-k" in cmd:
      idx = cmd.index("--cache-type-k")
      if idx + 1 < len(cmd):
        kv = cmd[idx + 1]
    kv = "fp16" if kv in ("", "f16", "fp16") else kv
    if group == "fp32":
      wak = f"`fp32 / fp32 / {kv}`"
    elif group == "bf16":
      wak = f"`bf16 / fp32 / {kv}`"
    elif group == "int8":
      wak = f"`q8_0 / q8_0 / {kv}`"
    elif "matched-q4" in mp or cid_lo.endswith("/fastest"):
      wak = f"`q4_0_q8_0 / q8_0 / {kv}`"
    elif "matched" in mp or cid_lo.endswith("/matched"):
      wak = f"`q2_k_q4_0_q8_0 / q8_0 / {kv}`"
    else:
      wak = f"`q4_0 / q8_0 / {kv}`"
  else:
    wak = f"`{group}`"

  return group, checkpoint, wak, threads_str


def _resolve_acc_lookup(cfg_runs: dict, acc_rows: list[dict]):
  """Return a helper _get_acc(cid, task) that resolves accuracy by cid or shared (framework, model_sha256)."""
  acc_by_id: dict[tuple[str, str], dict] = {}
  acc_by_model: dict[tuple[str, str, str], dict] = {}
  for a in acc_rows:
    acc_by_id[(a["id"], a["task"])] = a
    runs_for_a = cfg_runs.get(a["id"])
    if runs_for_a:
      m0 = runs_for_a[0]["meta"]
      fw = m0.get("framework") or a["id"].split("/")[0]
      sha = m0.get("model_sha256")
      if fw and sha:
        acc_by_model[(fw, sha, a["task"])] = a

  def _get_acc(cid: str, task: str) -> float | None:
    if (cid, task) in acc_by_id:
      return acc_by_id[(cid, task)]["accuracy"]
    runs_for_c = cfg_runs.get(cid)
    if not runs_for_c:
      return None
    m0 = runs_for_c[0]["meta"]
    fw = m0.get("framework") or cid.split("/")[0]
    sha = m0.get("model_sha256")
    if fw and sha and (fw, sha, task) in acc_by_model:
      return acc_by_model[(fw, sha, task)]["accuracy"]
    return None

  return _get_acc


def pareto_chart(path: Path, suite: dict, cfg_ids: list[str], cfg_runs: dict,
                 agg: dict, acc_rows: list[dict]) -> bool:
  """Scatter plot of MMLU accuracy (y-axis) vs. End-to-End latency (x-axis) with Pareto frontier."""
  _get_acc = _resolve_acc_lookup(cfg_runs, acc_rows)
  pts = []
  for cid in cfg_ids:
    e2e_d = agg.get(cid, {}).get("e2e_ms")
    e2e = e2e_d.get("median") if e2e_d else None
    mmlu = _get_acc(cid, "mmlu")
    if e2e is None or mmlu is None:
      continue
    m0 = cfg_runs[cid][0]["meta"]
    grp, _, _, _ = _config_dtype_info(suite, cid, m0)
    fw = m0.get("framework") or cid.split("/")[0]
    pts.append({
        "cid": cid,
        "fw": fw,
        "dtype": grp.upper() if grp != "int4" else "INT4/Mixed",
        "grp": grp,
        "e2e_ms": float(e2e),
        "mmlu_pct": float(mmlu) * 100.0,
    })

  if not pts:
    return False

  # Identify Pareto-optimal points (minimize e2e_ms, maximize mmlu_pct)
  pts_sorted = sorted(pts, key=lambda p: (p["e2e_ms"], -p["mmlu_pct"]))
  pareto = []
  best_y = -1.0
  for p in pts_sorted:
    if p["mmlu_pct"] > best_y + 1e-9:
      pareto.append(p)
      best_y = p["mmlu_pct"]
  pareto_cids = {p["cid"] for p in pareto}

  dtype_markers = {
      "fp32": ("D", "FP32"),
      "bf16": ("^", "BF16"),
      "fp16": ("v", "FP16"),
      "int8": ("s", "INT8"),
      "int4": ("o", "INT4 / Mixed"),
  }

  fig, ax = plt.subplots(figsize=(10.5, 6.2))

  # Plot each point colored by framework and shaped by data type
  for p in pts:
    col = COLORS.get(p["fw"], "#555555")
    mk, _ = dtype_markers.get(p["grp"], ("o", p["dtype"]))
    is_par = p["cid"] in pareto_cids
    if is_par:
      ax.scatter(
          [p["e2e_ms"]], [p["mmlu_pct"]],
          s=190, facecolors="none", edgecolors="#2e8b57", linewidths=2.2, zorder=3,
      )
    ax.scatter(
        [p["e2e_ms"]], [p["mmlu_pct"]],
        s=95 if is_par else 70, color=col, marker=mk,
        edgecolors="black" if is_par else "white", linewidths=0.9, zorder=4,
    )

  # De-overlap point annotations with vertical spreading for close clusters
  xs = [p["e2e_ms"] for p in pts]
  ys = [p["mmlu_pct"] for p in pts]
  x_span = max(max(xs) - min(xs), 500.0)
  y_span = max(max(ys) - min(ys), 2.0)
  x_min = max(0.0, min(xs) - 0.10 * x_span)
  x_max = max(xs) + 0.22 * x_span
  y_min = min(ys) - 0.16 * y_span
  y_max = max(ys) + 0.36 * y_span
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(y_min, y_max)

  ax_w_pts = 620.0
  ax_h_pts = 340.0
  placed_boxes: list[tuple[float, float, float, float]] = [
      # Reserve top-left box region for the Pareto corner callout
      (0.0, 0.44, 0.82, 1.0),
  ]
  for p in pts:
    px_n = (p["e2e_ms"] - x_min) / (x_max - x_min)
    py_n = (p["mmlu_pct"] - y_min) / (y_max - y_min)
    placed_boxes.append((px_n - 8.0 / ax_w_pts, px_n + 8.0 / ax_w_pts,
                         py_n - 8.0 / ax_h_pts, py_n + 8.0 / ax_h_pts))

  candidate_offsets = [
      (9, 0), (9, 12), (9, -12), (9, 24), (9, -24), (9, 36), (9, -36),
      (9, 48), (9, -48), (9, 60), (9, -60), (-9, 0), (-9, 14), (-9, -14),
      (-9, 28), (-9, -28), (28, 18), (28, -18), (28, 32), (28, -32),
  ]
  for p in sorted(pts, key=lambda item: (item["e2e_ms"], -item["mmlu_pct"])):
    nx = (p["e2e_ms"] - x_min) / (x_max - x_min)
    ny = (p["mmlu_pct"] - y_min) / (y_max - y_min)
    lbl = f"{p['cid']} ({p['dtype']})"
    w_box = (len(lbl) * 4.6 + 4.0) / ax_w_pts
    h_box = 11.0 / ax_h_pts
    best_cand = candidate_offsets[0]
    best_rect = (nx, nx + w_box, ny - h_box / 2.0, ny + h_box / 2.0)
    best_score = float("inf")
    for dx, dy in candidate_offsets:
      cx = nx + dx / ax_w_pts
      cy = ny + dy / ax_h_pts
      x0 = cx if dx >= 0 else cx - w_box
      x1 = cx + w_box if dx >= 0 else cx
      y0 = cy - h_box / 2.0
      y1 = cy + h_box / 2.0
      if x0 < 0.01 or x1 > 0.99 or y0 < 0.02 or y1 > 0.98:
        continue
      ov_area = 0.0
      for bx0, bx1, by0, by1 in placed_boxes:
        ix = max(0.0, min(x1, bx1) - max(x0, bx0))
        iy = max(0.0, min(y1, by1) - max(y0, by0))
        ov_area += ix * iy
      dist_pen = 1e-6 * (abs(dx) + abs(dy))
      score = ov_area + dist_pen
      if score < best_score:
        best_score = score
        best_cand = (dx, dy)
        best_rect = (x0, x1, y0, y1)
      if ov_area == 0.0:
        break
    chosen = best_cand
    placed_boxes.append(best_rect)

    ha = "left" if chosen[0] >= 0 else "right"
    use_leader = abs(chosen[1]) >= 11 or abs(chosen[0]) >= 15
    relpos = (0.0, 0.5) if chosen[0] >= 0 else (1.0, 0.5)
    ax.annotate(
        lbl,
        (p["e2e_ms"], p["mmlu_pct"]),
        xytext=chosen,
        textcoords="offset points",
        ha=ha,
        va="center",
        fontsize=8,
        fontweight="bold" if p["cid"] in pareto_cids else "normal",
        color="#1b5e20" if p["cid"] in pareto_cids else "#24292f",
        arrowprops=({"arrowstyle": "-", "color": "#999999", "lw": 0.7, "relpos": relpos}
                    if use_leader else None),
        zorder=5,
    )

  # Highlight the Pareto corner (Top-Left: lower E2E latency + higher MMLU accuracy)
  ax.annotate(
      "↖ Pareto Optimal Corner\n(Lower E2E Latency, Higher MMLU Accuracy)",
      xy=(0.012, 0.98),
      xycoords="axes fraction",
      xytext=(0.055, 0.955),
      textcoords="axes fraction",
      ha="left",
      va="top",
      fontsize=9,
      fontweight="bold",
      color="#1b5e20",
      bbox={"boxstyle": "round,pad=0.45", "facecolor": "#e8f5e9", "edgecolor": "#2e8b57", "lw": 1.5},
      arrowprops={"arrowstyle": "->", "color": "#2e8b57", "lw": 2.0},
      zorder=6,
  )

  # Build combined legend for Frameworks (color) and Data Types (marker shape)
  from matplotlib.lines import Line2D
  seen_fws = sorted({p["fw"] for p in pts})
  seen_grps = [g for g, _ in [("fp32", ""), ("bf16", ""), ("fp16", ""), ("int8", ""), ("int4", "")]
               if any(p["grp"] == g for p in pts)]
  handles = []
  for fw in seen_fws:
    handles.append(Line2D([0], [0], marker="o", color="w", label=f"{fw}",
                          markerfacecolor=COLORS.get(fw, "#555555"), markersize=8))
  for grp in seen_grps:
    mk, grp_lbl = dtype_markers[grp]
    handles.append(Line2D([0], [0], marker=mk, color="w", label=f"dtype: {grp_lbl}",
                          markerfacecolor="#666666", markeredgecolor="black", markersize=7))
  handles.append(Line2D([0], [0], ls="none", color="#2e8b57", marker="o",
                        markerfacecolor="none", markeredgecolor="#2e8b57", markeredgewidth=2.0,
                        markersize=9, label="Pareto optimal"))
  ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.92, ncol=2)

  w = suite.get("workload") or {}
  mmlu_ns = {a["n"] for a in acc_rows if a["task"] == "mmlu"}
  mmlu_n_str = f" (n={next(iter(mmlu_ns))})" if len(mmlu_ns) == 1 else ""
  ax.set_xlabel(
      f"End-to-End Latency (ms, {w.get('prompt_tokens', 1024)} prompt → "
      f"{w.get('gen_tokens', 128)} decode tokens) — lower is better ←",
      fontsize=9.5,
  )
  ax.set_ylabel(f"MMLU Accuracy (%{mmlu_n_str}) — higher is better ↑", fontsize=9.5)
  ax.set_title("MMLU Accuracy vs. End-to-End Latency by Framework & Data Type", fontsize=11, fontweight="bold")
  ax.grid(True, alpha=0.3)
  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)
  return True


def summary_by_dtype_section(suite: dict, cfg_ids: list[str], cfg_runs: dict, agg: dict,
                             acc_rows: list[dict], has_pareto: bool = False) -> str:
  """Render a single top-level summary table grouped by data type, highlighting the best row per data type."""
  w = suite.get("workload") or {}
  n_prompt = w.get("prompt_tokens", 1024)
  n_gen = w.get("gen_tokens", 128)
  _get_acc = _resolve_acc_lookup(cfg_runs, acc_rows)

  mmlu_ns = {a["n"] for a in acc_rows if a["task"] == "mmlu"}
  gsm8k_ns = {a["n"] for a in acc_rows if a["task"] == "gsm8k"}
  mmlu_hdr = f"MMLU (n={next(iter(mmlu_ns))})" if len(mmlu_ns) == 1 else "MMLU"
  gsm8k_hdr = f"GSM8K (n={next(iter(gsm8k_ns))})" if len(gsm8k_ns) == 1 else "GSM8K"

  dtype_tracks = [
      ("fp32", "FP32 Track"),
      ("bf16", "BF16 Track"),
      ("fp16", "FP16 Track"),
      ("int8", "INT8 Track"),
      ("int4", "INT4 / Mixed Track"),
  ]
  by_group: dict[str, list[tuple[str, str, str, str]]] = collections.defaultdict(list)
  for cid in cfg_ids:
    m0 = cfg_runs[cid][0]["meta"]
    grp, ckpt, wak, thr = _config_dtype_info(suite, cid, m0)
    by_group[grp].append((cid, ckpt, wak, thr))

  L = [
      "## Summary\n",
      f"Headline medians (`{n_prompt}` prompt → `{n_gen}` generated tokens, stateless cold KV cache) "
      "and accuracy grouped by data type. Within each data type, rows are ordered from fastest to "
      "slowest end-to-end latency, with the best-performing row highlighted (**🏆**) and best "
      "per-column metrics bolded.\n",
  ]
  if has_pareto:
    L.append("![MMLU Accuracy vs. End-to-End Latency (Pareto Frontier)](pareto.png)\n")
  L.extend([
      f"| Framework / Config | Checkpoint | Weights / Acts / KV | Threads | TTFT (ms) ↓ | "
      f"Prefill (tok/s) ↑ | Decode (tok/s) ↑ | E2E (ms) ↓ | Peak RSS (MB) ↓ | {mmlu_hdr} ↑ | {gsm8k_hdr} ↑ |",
      "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ])

  def _med(cid: str, key: str) -> float | None:
    d = agg.get(cid, {}).get(key)
    return d["median"] if d and d.get("median") is not None else None

  for grp_key, grp_label in dtype_tracks:
    items = by_group.get(grp_key)
    if not items:
      continue
    items.sort(key=lambda item: (
        _med(item[0], "e2e_ms") if _med(item[0], "e2e_ms") is not None else float("inf"),
        item[0],
    ))
    best_row_cid = items[0][0]

    def _best(vals: list[float | None], higher: bool) -> float | None:
      clean = [v for v in vals if v is not None]
      if not clean:
        return None
      return max(clean) if higher else min(clean)

    cids_in_grp = [it[0] for it in items]
    best_ttft = _best([_med(c, "ttft_ms") for c in cids_in_grp], False)
    best_prefill = _best([_med(c, "prefill_tps") for c in cids_in_grp], True)
    best_decode = _best([_med(c, "decode_tps") for c in cids_in_grp], True)
    best_e2e = _best([_med(c, "e2e_ms") for c in cids_in_grp], False)
    best_rss = _best([_med(c, "rss_max_mb") for c in cids_in_grp], False)
    best_mmlu = _best([_get_acc(c, "mmlu") for c in cids_in_grp], True)
    best_gsm8k = _best([_get_acc(c, "gsm8k") for c in cids_in_grp], True)

    L.append(f"| **{grp_label}** | | | | | | | | | | |")

    for cid, ckpt, wak, thr in items:
      is_winner = (cid == best_row_cid)
      v_ttft = _med(cid, "ttft_ms")
      v_prefill = _med(cid, "prefill_tps")
      v_decode = _med(cid, "decode_tps")
      v_e2e = _med(cid, "e2e_ms")
      v_rss = _med(cid, "rss_max_mb")
      v_mmlu = _get_acc(cid, "mmlu")
      v_gsm8k = _get_acc(cid, "gsm8k")

      def _fmt_cell(val: float | None, best_val: float | None, fmt_str: str) -> str:
        if val is None:
          return "—"
        s = fmt_str.format(val)
        if is_winner or (best_val is not None and abs(val - best_val) <= 1e-9):
          return f"**{s}**"
        return s

      cfg_cell = f"🏆 **`{cid}`**" if is_winner else f"`{cid}`"
      ckpt_cell = f"**{ckpt}**" if is_winner else ckpt
      wak_cell = f"**{wak}**" if is_winner else wak
      thr_cell = f"**{thr}**" if is_winner else thr

      L.append(
          f"| {cfg_cell} | {ckpt_cell} | {wak_cell} | {thr_cell} | "
          f"{_fmt_cell(v_ttft, best_ttft, '{:,.0f}')} | "
          f"{_fmt_cell(v_prefill, best_prefill, '{:,.1f}')} | "
          f"{_fmt_cell(v_decode, best_decode, '{:.2f}')} | "
          f"{_fmt_cell(v_e2e, best_e2e, '{:,.0f}')} | "
          f"{_fmt_cell(v_rss, best_rss, '{:,.0f}')} | "
          f"{_fmt_cell(v_mmlu, best_mmlu, '{:.2%}')} | "
          f"{_fmt_cell(v_gsm8k, best_gsm8k, '{:.2%}')} |"
      )

  L.append("")
  L.append(
      "**Legend:**\n"
      "- **`matched`**: Apples-to-apples comparison using the **exact same pre-quantized integer weights** "
      "from the mobile QAT checkpoint (`qat-mobile`: mixed per-channel `INT2` / `INT4` / `INT8` tensors), "
      "matching thread counts, context length, and closest KV-cache precision (`Q2_K`/`Q4_0`/`Q8_0` in "
      "`llama.cpp/matched`, `.litertlm` in `litert-lm/matched`, and `QArray` in `xla/matched-ynnpack`).\n"
      "- **`fastest`**: Each framework's empirically fastest configuration on the mobile QAT checkpoint "
      "selected by the thread-scaling and kernel sweep (e.g., `llama.cpp/fastest` stores the mobile QAT "
      "`INT2` tensors in `Q4_0` blocks so `llama.cpp` can dispatch repacked AVX-512/AMX kernels instead "
      "of scalar `Q2_K`, and `litert-lm/fastest` uses the sweep-selected thread count).\n"
  )
  return "\n".join(L)


def render_markdown(suite, args, report_id, cfg_ids, cfg_runs, agg, acc_rows, has_scaling,
                    history, changes, hist_path, extra, agreement=(), repro=([], []), sequence=None,
                    prefix_cache=None, has_pareto=False):
  L = []
  w = suite["workload"]
  first = cfg_runs[cfg_ids[0]][0]["meta"]
  sysinfo = _read_json(cfg_runs[cfg_ids[0]][0]["dir"] / "sysinfo.json") or {}
  L.append(f"# {suite['name']} on {args.machine} — {args.date}\n")
  sweep_count = sum(len(c["runs"]) for p in (sequence or {}).get("points", {}).values()
                    for c in p["configs"].values())
  pc_count = sum(len(c["runs"]) for c in (prefix_cache or {}).get("configs", {}).values())
  extra_procs = []
  if sweep_count:
    extra_procs.append(f"{sweep_count} input-length sweep processes")
  if pc_count:
    extra_procs.append(f"{pc_count} prefix-cache processes")
  process_description = ("benchmark processes" if not extra_procs else
                         "headline benchmark processes plus " + " and ".join(extra_procs))
  L.append(f"> Generated by `python3 -m aeb.report` from {sum(len(v) for v in cfg_runs.values())} "
           f"{process_description}. Raw per-request data: [data.json](data.json). "
           f"Methodology: [benchmark_methodology.md](../../docs/benchmark_methodology.md).\n")
  L.append(suite["description"] + "\n")
  L.append(summary_by_dtype_section(suite, cfg_ids, cfg_runs, agg, acc_rows, has_pareto=has_pareto))

  # headline table (Results right after Summary)
  L.append("## Results\n")
  L.append("### Throughput\n")
  L.append("![Throughput](throughput.png)\n")
  if sequence:
    L.append(sequence_section(sequence))
  L.append("### Latency\n")
  L.append("![Latency](latency.png)\n")
  if prefix_cache:
    L.append(prefix_cache_section(prefix_cache))
  L.append("### CPU\n")
  L.append("![CPU](cpu.png)\n")
  L.append("### Memory\n")
  L.append("![Memory](memory.png)\n")
  L.append("### Memory and CPU over time\n")
  L.append("Combined timeline (round 0 process per configuration: model load, warm-up, and "
           "measured requests) showing Total RSS (MB) and active CPU cores over time.\n")
  L.append("![Timeline](timeline.png)\n")

  if has_scaling:
    L.append("### Thread scaling\n")
    L.append("![Thread scaling](thread_scaling.png)\n")

  # generated text on the timing prompt
  L.append("### Generated text on the timing prompt\n")
  L.append("Greedy decoding should be deterministic within a configuration. Different "
           "configurations may legitimately diverge (different activation/KV quantization and "
           "kernels change near-tie logits); the accuracy gate below is the validity check.\n")
  L.append("| Config | Identical across all measured requests | First 140 characters |")
  L.append("| --- | --- | --- |")
  for cid in cfg_ids:
    texts = [r.get("text", "") for run in cfg_runs[cid] for r in _requests(run["dir"])
             if r.get("measured")]
    sample = (texts[0] if texts else "")[:140].replace("\n", " ⏎ ").replace("|", "\\|")
    L.append(f"| `{cid}` | {'yes' if texts and len(set(texts)) == 1 else f'no ({len(set(texts))} variants)'} | {sample} |")
  L.append("")

  # accuracy
  if acc_rows:
    L.append("## Accuracy (validity gate)\n")
    L.append("Same prompts, greedy decoding, through the same in-process drivers as the timing "
             "runs. Use these to confirm each configuration produces correct tokens, and to "
             "compare frameworks on identical items; they are not leaderboard numbers.\n")
    L.append("| Config | Task | Items | Accuracy | 95% CI | Macro (MMLU) | Parse failures | "
             "Mean generated tokens | Hit token limit |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for a in acc_rows:
      scope = "full test set" if not a["limit"] else (
          "stratified subset" if a["task"] == "mmlu" else "evenly spaced subset")
      trunc = "n/a (letter only)" if a["truncated"] is None else f"{a['truncated']} (cap {a['max_tokens']})"
      L.append(f"| `{a['id']}` | {a['task']} | {a['n']} ({scope}) | {a['accuracy']:.2%} | "
               f"±{a['ci95']:.2%} | {fmt(a['macro'], '{:.2%}')} | {a['parse_failures']} | "
               f"{fmt(a['mean_gen'], '{:.1f}')} | {trunc} |")
    L.append("")
    if agreement:
      L.append("**Paired comparison on identical items.** Same prediction = both configs chose "
               "the same answer; McNemar's exact test on items only one config got right "
               "(p < 0.05 means a real accuracy difference).\n")
      L.append("| Task | A | B | Items | Same prediction | Acc A | Acc B | Only A right | "
               "Only B right | McNemar p |")
      L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
      for g in sorted(agreement, key=lambda g: (g["task"] != "mmlu", -g["n"])):
        L.append(f"| {g['task']} | `{g['a']}` | `{g['b']}` | {g['n']} | {g['same_prediction']:.1%} | "
                 f"{g['acc_a']:.2%} | {g['acc_b']:.2%} | {g['a_only_correct']} | "
                 f"{g['b_only_correct']} | {g['mcnemar_p']:.3g} |")
      L.append("")
    L.append("![Accuracy](accuracy.png)\n")
    diag = (_read_json(args.results / args.machine / "diagnostics" / "hf-reference-mmlu.json")
            if {"llama.cpp/matched", "litert-lm/matched"} <= set(cfg_ids) else None)
    if diag and diag.get("summary"):
      L.append("### Diagnostic: why do identical weights score differently?\n")
      L.append("The upstream PyTorch reference of the same checkpoint (transformers `gemma` "
               "quantizer, weights dequantized to fp32) re-answered MMLU items with its static "
               "int8 activation rounding (SRQ) **on** — as trained, and as LiteRT-LM runs it — and "
               f"**off** (full-precision activations, what llama.cpp approximates). A = `{Path(diag['a']).parts[-2]}/{Path(diag['a']).name}`, "
               f"B = `{Path(diag['b']).parts[-2]}/{Path(diag['b']).name}`. "
               "Produced by `python3 -m aeb.prep.hf_reference`.\n")
      L.append("| Items | n | Reference SRQ on: accuracy / agrees with A / agrees with B | "
               "Reference SRQ off: accuracy / agrees with A / agrees with B | A accuracy | B accuracy |")
      L.append("| --- | --- | --- | --- | --- | --- |")
      for kind, sm in diag["summary"].items():
        on, off = sm["srq-on"], sm["srq-off"]
        L.append(f"| {kind} (A and B {'disagree on correctness' if kind == 'discordant' else 'both right or both wrong'}) "
                 f"| {sm['n']} | {on['acc']:.0%} / {on['agrees_with_a']:.0%} / {on['agrees_with_b']:.0%} | "
                 f"{off['acc']:.0%} / {off['agrees_with_a']:.0%} / {off['agrees_with_b']:.0%} | "
                 f"{sm['a_acc']:.0%} | {sm['b_acc']:.0%} |")
      L.append("")
    refs = getattr(args, "_refs", None)
    if refs:
      L.append(reference_section(refs, acc_rows))

  # reproducibility
  rtbl, rcross = repro
  if rtbl:
    L.append("## Output reproducibility\n")
    L.append("Same model file, build, prompt and settings, repeated: 12 prompts × 3 repeats per "
             "process, fresh KV cache per request, stop tokens honoured, ≤128 tokens. "
             "*Within process* = repeats vs. the first repeat; *across processes* = a second fresh "
             "process with the same request sequence; *across threads* = a process with half the "
             "threads. Seeded = temperature 1.0, top-k 64, top-p 0.95, fixed seed; the *seed "
             "control* reruns with seed+1 and must change outputs, otherwise seeded "
             "reproducibility would be vacuous. Method: "
             "[benchmark_methodology.md](../../docs/benchmark_methodology.md#output-reproducibility).\n")
    def frac(t):
      if not t:
        return "n/a"
      s_, n_ = t[0], t[1]
      div = t[2] if len(t) > 2 else []
      med = f"; first divergence at token {statistics.median(div):.0f} (median)" if div else ""
      return f"{s_}/{n_} identical{med}"
    L.append("| Config | Mode | Verdict | Within process | Across processes | Across threads | Seed control |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rtbl:
      for mode in ("greedy", "seeded"):
        if f"{mode}_within" not in r:
          continue
        thr = r.get(f"{mode}_threads")
        thr_s = (frac(thr[:3]) + f" ({thr[3]}→{thr[4]} threads)") if thr else "n/a"
        sc = r.get("seed_control") if mode == "seeded" else None
        sc_s = f"{sc[0]}/{sc[1]} outputs changed" if sc else "—"
        L.append(f"| `{r['id']}` | {mode} | {repro_verdict(r, mode)} | {frac(r.get(f'{mode}_within'))} | "
                 f"{frac(r.get(f'{mode}_process'))} | {thr_s} | {sc_s} |")
    L.append("")
    for c in rcross:
      med = f", median first divergence at token {statistics.median(c['div']):.0f}" if c["div"] else ""
      L.append(f"Cross-framework (informational, greedy): `{c['a']}` and `{c['b']}` produce "
               f"identical text for {c['same']}/{c['n']} prompt-repeats{med}. Identical weights do "
               f"not imply identical outputs when activation/KV quantization and kernels differ.\n")

  # Hand-written notes: anything before "## Caveats" (e.g. "## Key findings")
  # goes here, the caveats section to the end.
  top, _, caveats = extra.partition("## Caveats")
  if top.strip():
    L.append(top.strip() + "\n")

  # environment
  L.append("## Environment\n")
  builds = {}
  for cid in cfg_ids:
    m = cfg_runs[cid][0]["meta"]
    builds[m["framework"]] = m["build"]
  L.append("| Item | Value |\n| --- | --- |")
  L.append(f"| Machine | {args.machine} — {first.get('machine_notes', '')} |")
  cpu = sysinfo.get("cpu", {})
  kernel_str = re.sub(r"-smp-[0-9.]+", "", str(sysinfo.get("container", {}).get("kernel", "?")))
  L.append(f"| Container view | {sysinfo.get('container', {}).get('os', '?')}, kernel "
           f"{kernel_str}, {cpu.get('nproc', '?')} vCPUs, "
           f"{(sysinfo.get('memory', {}).get('total_kb', 0) / 2**20):.1f} GiB RAM |")
  for fw, b in sorted(builds.items()):
    commit_val = str(b.get("commit") or "")
    if commit_val and "==" in commit_val:
      revision = f", `{commit_val}`"
    elif commit_val:
      revision = f", commit `{commit_val[:12]}`"
    else:
      revision = ""
    L.append(f"| {fw} | `{b.get('ref')}`{revision}, built {b.get('built_at')} |")
  L.append(f"| Harness | digest `{str(first['build'].get('harness_sha256'))[:12]}`, repo rev "
           f"`{first['build'].get('repo_rev')}` |")
  L.append(f"| Workload | {w['prompt_tokens']}-token real-text chat prompt → {w['gen_tokens']} "
           f"greedy tokens (fixed length), context {w['ctx']}, {w['warmup']} warm-up + "
           f"{w['repetitions']} measured requests per process, {suite.get('rounds', 3)} processes per "
           f"config interleaved A-B-…/…-B-A |")
  sha = {cfg_runs[cid][0]["meta"].get("workload", {}).get("prompt_ids_sha256") for cid in cfg_ids}
  L.append(f"| Prompt identity | {'identical token ids in every config' if len(sha) == 1 else 'DIFFERENT token ids: ' + ', '.join(map(str, sha))} "
           f"(sha256 `{str(next(iter(sha)))[:16]}…`) |\n")

  # configs
  L.append("## Configurations\n")
  L.append("| Config | Track | Model file | Driver arguments |\n| --- | --- | --- | --- |")
  for cid in cfg_ids:
    m = cfg_runs[cid][0]["meta"]
    args_ = " ".join(m["driver_cmd"][1:] if m["framework"] == "llama.cpp" else m["driver_cmd"][3:])
    args_ = args_.replace(m.get("model_path") or "\0", "<model>")
    L.append(f"| `{cid}` | {_track(suite, cid)} | `{Path(m.get('model_path') or '').name}` "
             f"(sha256 `{str(m.get('model_sha256'))[:12]}`) | `{args_}` |")
  L.append("")

  if getattr(args, "_quant_md", ""):
    L.append(args._quant_md)

  # automatic data-quality notes
  L.append("## Data-quality checks\n")
  checks = []
  for cid in cfg_ids:
    d = agg[cid]
    cv = (d.get("decode_tps") or {}).get("cv")
    if cv is not None and cv > 0.05:
      checks.append(f"`{cid}`: decode throughput CV {cv:.1%} > 5% — treat differences below that as noise.")
    busy = (d.get("system_cpu_busy_frac") or {}).get("median")
    for r in cfg_runs[cid]:
      if r["summary"].get("status") != "ok":
        checks.append(f"`{cid}` run `{r['dir'].name}` status **{r['summary'].get('status')}**.")
      if r["summary"].get("outliers_e2e"):
        checks.append(f"`{cid}` run `{r['dir'].name}`: end-to-end outlier request indices "
                      f"{r['summary']['outliers_e2e']} (MAD rule) — kept, not discarded.")
    rm = [x for x in d.get("round_medians_decode", []) if x]
    if len(rm) > 1 and (max(rm) - min(rm)) / statistics.median(rm) > 0.05:
      checks.append(f"`{cid}`: per-process decode medians spread {min(rm):.1f}–{max(rm):.1f} tok/s "
                    f"(>5%); between-process variance dominates.")
  L.append("\n".join(f"- {c}" for c in checks) if checks else "- All runs completed; no outliers; "
           "decode CV ≤ 5% everywhere.")
  L.append("")

  L.append("## Caveats\n")
  L.append(caveats.strip() or "_none recorded_")
  L.append("")
  L.append(reproduction_section(suite, args.machine, args.suite.name, cfg_ids, cfg_runs, builds, first, w))
  return "\n".join(L)


def reproduction_section(suite: dict, machine: str, suite_filename: str, cfg_ids: list[str],
                         cfg_runs: dict, builds: dict, first_meta: dict, w: dict) -> str:
  suite_rel = f"suites/{suite_filename}"
  service_map = {
      "llama.cpp": "harness-llama-cpp",
      "litert-lm": "harness-litert-lm",
      "onnxruntime": "harness-onnxruntime",
      "xla": "harness-xla",
      "openvino": "harness-openvino",
  }
  repo_urls = {
      "llama.cpp": "[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)",
      "litert-lm": "[`google-ai-edge/LiteRT-LM`](https://github.com/google-ai-edge/LiteRT-LM)",
      "xla": "[`openxla/xla`](https://github.com/openxla/xla) / [`jax-ml/jax`](https://github.com/jax-ml/jax) / [`google/tunix`](https://github.com/google/tunix) / [`google/qwix`](https://github.com/google/qwix)",
      "openvino": "[`openvinotoolkit/openvino.genai`](https://github.com/openvinotoolkit/openvino.genai)",
      "onnxruntime": "[`microsoft/onnxruntime-genai`](https://github.com/microsoft/onnxruntime-genai)",
  }
  L = [
      "## Reproducing these results\n",
      "### 1. Exact framework & repository versions used in this report\n",
      "| Framework | Upstream Repository | Ref / Tag | Exact Commit / Package Versions | Built At |",
      "| --- | --- | --- | --- | --- |",
  ]
  for fw, b in sorted(builds.items()):
    repo = repo_urls.get(fw, f"`{fw}`")
    ref = f"`{b.get('ref', '—')}`"
    if fw == "xla":
      pkgs = [b.get(k) for k in ("runtime", "tunix", "qwix", "flax", "tokenizers") if b.get(k)]
      exact_ver = ", ".join(f"`{p}`" for p in pkgs) if pkgs else f"`{b.get('commit', '—')}`"
    elif fw in ("openvino", "onnxruntime"):
      pkgs = [b.get(k) for k in ("runtime", "genai") if b.get(k)]
      exact_ver = ", ".join(f"`{p}`" for p in pkgs) if pkgs else f"`{b.get('commit', '—')}`"
    else:
      commit = b.get("commit", "—")
      extra_flags = b.get("bazel_extra_flags") or b.get("cmake_extra_flags")
      exact_ver = f"`{commit}`" + (f" (`{extra_flags}`)" if extra_flags else "")
    L.append(f"| `{fw}` | {repo} | {ref} | {exact_ver} | `{b.get('built_at', '—')}` |")
  h_info = first_meta.get("build", {})
  L.append(
      f"| `ai-edge-bench` (harness) | [`mariecwhite/ai-edge-bench`](https://github.com/mariecwhite/ai-edge-bench) | "
      f"`{h_info.get('repo_rev', 'unknown')}` | harness sha256 `{h_info.get('harness_sha256', '—')}` | — |\n"
  )

  # Build commands (exact pinned vs. latest upstream)
  llama_ref = (builds.get("llama.cpp") or {}).get("ref", "b11259")
  litert_ref = (builds.get("litert-lm") or {}).get("ref", "v0.17.1")
  xla_b = builds.get("xla") or {}
  jax_ver = str(xla_b.get("ref", "jax==0.11.2")).split("==")[-1]
  tunix_ver = str(xla_b.get("tunix", "google-tunix==0.1.7")).split("==")[-1]
  qwix_ver = str(xla_b.get("qwix", "qwix==0.1.8")).split("==")[-1]
  flax_ver = str(xla_b.get("flax", "flax==0.12.10")).split("==")[-1]
  ov_b = builds.get("openvino") or {}
  ov_ver = str(ov_b.get("runtime", "openvino==2026.4.1.0")).split("==")[-1]
  ov_genai_ver = str(ov_b.get("genai", "openvino-genai==2026.4.1.0")).split("==")[-1]
  ort_b = builds.get("onnxruntime") or {}
  ort_ver = str(ort_b.get("runtime", "onnxruntime==1.30.0")).split("==")[-1]
  ort_genai_ver = str(ort_b.get("genai", "onnxruntime-genai==0.17.1")).split("==")[-1]

  L.append("### 2. Building framework container images (pinned vs. latest upstream)\n")
  L.append("**Option A — Rebuild with the exact pinned versions used in this report:**\n")
  L.append("```bash")
  L.append(f"COMPOSE=\"docker compose -f compose.yaml -f compose/machines/{machine}.yaml\"")
  L.append(f"export AEB_LLAMA_CPP_REF=\"{llama_ref}\"")
  L.append(f"export AEB_LITERTLM_REF=\"{litert_ref}\"")
  L.append("${COMPOSE} --profile base build base")
  L.append("${COMPOSE} --profile tools build tools")
  L.append("${COMPOSE} --profile cpu build llama-cpp litert-lm-xnnpack")
  if "xla" in builds:
    L.append(f"${{COMPOSE}} --profile x86-sota build \\\n"
             f"  --build-arg JAX_VERSION=\"{jax_ver}\" --build-arg FLAX_VERSION=\"{flax_ver}\" \\\n"
             f"  --build-arg QWIX_VERSION=\"{qwix_ver}\" --build-arg TUNIX_VERSION=\"{tunix_ver}\" xla")
  if "openvino" in builds:
    L.append(f"${{COMPOSE}} --profile x86-sota build \\\n"
             f"  --build-arg OPENVINO_VERSION=\"{ov_ver}\" --build-arg OPENVINO_GENAI_VERSION=\"{ov_genai_ver}\" openvino")
  if "onnxruntime" in builds:
    prof = "x86-sota" if "xla" in builds or "openvino" in builds else "onnxruntime"
    L.append(f"${{COMPOSE}} --profile {prof} build \\\n"
             f"  --build-arg ONNXRUNTIME_VERSION=\"{ort_ver}\" --build-arg ONNXRUNTIME_GENAI_VERSION=\"{ort_genai_ver}\" onnxruntime")
  L.append("```\n")

  L.append("**Option B — Pull & build with the latest upstream versions of every framework:**\n")
  L.append("```bash")
  L.append("# Query latest upstream git tags for llama.cpp and LiteRT-LM:")
  L.append("export AEB_LLAMA_CPP_REF=\"$(git ls-remote --tags --sort='-v:refname' https://github.com/ggml-org/llama.cpp.git 'b*' | head -n1 | sed 's|.*refs/tags/||')\"")
  L.append("export AEB_LITERTLM_REF=\"$(git ls-remote --tags --sort='-v:refname' https://github.com/google-ai-edge/LiteRT-LM.git 'v*' | head -n1 | sed 's|.*refs/tags/||; s/\\^{}//')\"")
  L.append("")
  L.append("# Query latest PyPI versions for OpenXLA (jax/flax/qwix/tunix), OpenVINO, and ONNX Runtime:")
  L.append("read -r JAX_VER FLAX_VER QWIX_VER TUNIX_VER OV_VER OV_GENAI_VER ORT_VER ORT_GENAI_VER < <(python3 - << 'PY'")
  L.append("import json, urllib.request")
  L.append("def latest(pkg):")
  L.append("  with urllib.request.urlopen(f\"https://pypi.org/pypi/{pkg}/json\") as r:")
  L.append("    return json.load(r)[\"info\"][\"version\"]")
  L.append("print(")
  L.append("    latest(\"jax\"), latest(\"flax\"), latest(\"qwix\"), latest(\"google-tunix\"),")
  L.append("    latest(\"openvino\"), latest(\"openvino-genai\"),")
  L.append("    latest(\"onnxruntime\"), latest(\"onnxruntime-genai\")")
  L.append(")")
  L.append("PY")
  L.append(")")
  L.append("")
  L.append(f"COMPOSE=\"docker compose -f compose.yaml -f compose/machines/{machine}.yaml\"")
  L.append("${COMPOSE} --profile base build --pull base")
  L.append("${COMPOSE} --profile tools build --no-cache tools")
  L.append("${COMPOSE} --profile cpu build --no-cache llama-cpp litert-lm-xnnpack")
  if "xla" in builds or "openvino" in builds:
    L.append("${COMPOSE} --profile x86-sota build --no-cache \\")
    L.append("  --build-arg JAX_VERSION=\"${JAX_VER}\" --build-arg FLAX_VERSION=\"${FLAX_VER}\" \\")
    L.append("  --build-arg QWIX_VERSION=\"${QWIX_VER}\" --build-arg TUNIX_VERSION=\"${TUNIX_VER}\" xla")
    L.append("${COMPOSE} --profile x86-sota build --no-cache \\")
    L.append("  --build-arg OPENVINO_VERSION=\"${OV_VER}\" --build-arg OPENVINO_GENAI_VERSION=\"${OV_GENAI_VER}\" openvino")
    L.append("${COMPOSE} --profile x86-sota build --no-cache \\")
    L.append("  --build-arg ONNXRUNTIME_VERSION=\"${ORT_VER}\" --build-arg ONNXRUNTIME_GENAI_VERSION=\"${ORT_GENAI_VER}\" onnxruntime")
  elif "onnxruntime" in builds:
    L.append("${COMPOSE} --profile onnxruntime build --no-cache \\")
    L.append("  --build-arg ONNXRUNTIME_VERSION=\"${ORT_VER}\" --build-arg ONNXRUNTIME_GENAI_VERSION=\"${ORT_GENAI_VER}\" onnxruntime")
  L.append("```\n")

  L.append("### 3. Machine setup & running the full benchmark suite\n")
  if machine == "sapphire-rapids":
    L.append(
        "```bash\n"
        "# Step 1 (from workstation): bootstrap /data/aeb and sync repo, container rootfs bundles,\n"
        "# datasets, and SHA-256 verified models to the target x86 AMX machine:\n"
        "./scripts/setup_amx_machine.sh <user@hostname> --bundles\n\n"
        "# Step 2 (on the target machine): archive any previous results/caches (so --skip-existing\n"
        "# does not skip runs) and launch all benchmark phases:\n"
        "ssh <user@hostname> '\n"
        "  set -e\n"
        "  if [[ -d /data/aeb/results/sapphire-rapids ]]; then\n"
        "    mv /data/aeb/results/sapphire-rapids \"/data/aeb/results/sapphire-rapids.bak-$(date -u +%Y%m%dT%H%M%SZ)\"\n"
        "  fi\n"
        "  rm -rf /data/aeb/results/_cache\n"
        "  nohup /data/aeb/repo/scripts/run_amx_benchmarks.sh > /data/aeb/bench.log 2>&1 &\n"
        "  echo \"Started PID $! — tail -f /data/aeb/bench.log\"\n"
        "'\n\n"
        "# Step 3 (from workstation): fetch results and generated report back to workstation:\n"
        "./scripts/fetch_results.sh <user@hostname>\n"
        "```\n"
    )
  L.append(
      f"```bash\n"
      f"# Step-by-step suite phases on {machine}:\n"
      f"python3 scripts/suite.py perf         --machine {machine} --suite {suite_rel}\n"
      f"python3 scripts/suite.py prefix-cache --machine {machine} --suite {suite_rel}\n"
      f"python3 scripts/suite.py accuracy     --machine {machine} --suite {suite_rel} --parallel 2 --shards 2 --threads half\n"
      f"python3 scripts/suite.py sequence     --machine {machine} --suite {suite_rel}\n"
      f"python3 scripts/suite.py repro        --machine {machine} --suite {suite_rel}\n"
      f"make MACHINE={machine} SUITE={suite_rel} report\n"
      f"```\n"
  )
  L.append("### 4. Single-configuration performance commands\n")
  L.append("```bash")
  for cid in cfg_ids:
    m = cfg_runs[cid][0]["meta"]
    fw = m["framework"]
    svc = service_map.get(fw, f"harness-{fw}")
    drv_args = " ".join(m["driver_cmd"][1:] if fw == "llama.cpp" else m["driver_cmd"][3:])
    L.append(
        f"# {cid}\n"
        f"docker compose -f compose.yaml -f compose/machines/{machine}.yaml --profile harness run --rm -T \\\n"
        f"  {svc} aeb.perf --framework {fw} --label {m['label']} --track {_track(suite, cid)} \\\n"
        f"  --prompt-tokens {w['prompt_tokens']} --gen-tokens {w['gen_tokens']} "
        f"--warmup {w['warmup']} --repetitions {w['repetitions']} -- \\\n"
        f"  {drv_args}"
    )
  L.append("```\n")
  return "\n".join(L)


def _slugify(text: str) -> str:
  plain = re.sub(r"`([^`]+)`", r"\1", text)
  plain = re.sub(r"[^\w\s-]", "", plain.lower())
  return re.sub(r"[\s_]+", "-", plain.strip())


def _render_inline_text_only(text: str, tokens: dict[str, str]) -> str:
  escaped = html.escape(text)
  escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
  escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)
  escaped = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"<em>\1</em>", escaped)
  for k, v in tokens.items():
    escaped = escaped.replace(k, v)
  return escaped


def _render_inline(text: str, base_dir: Path, embed_images: bool = True) -> str:
  tokens: dict[str, str] = {}

  def _stash(html_fragment: str) -> str:
    key = f"\x00TOK{len(tokens)}\x00"
    tokens[key] = html_fragment
    return key

  def _repl_code(m: re.Match[str]) -> str:
    return _stash(f"<code>{html.escape(m.group(1))}</code>")

  text = re.sub(r"`([^`]+)`", _repl_code, text)

  def _repl_img(m: re.Match[str]) -> str:
    alt = html.escape(m.group(1))
    src = m.group(2).strip()
    img_path = (base_dir / src).resolve()
    if embed_images and not src.startswith(("http://", "https://", "data:")) and img_path.is_file():
      mime, _ = mimetypes.guess_type(str(img_path))
      mime = mime or "image/png"
      b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
      src_attr = f"data:{mime};base64,{b64}"
    else:
      src_attr = html.escape(src)
    return _stash(
        f'<figure class="report-figure">'
        f'<img src="{src_attr}" alt="{alt}" loading="lazy" />'
        f'<figcaption>{alt}</figcaption>'
        f'</figure>'
    )

  text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _repl_img, text)

  def _repl_link(m: re.Match[str]) -> str:
    label_raw = m.group(1)
    href = html.escape(m.group(2).strip())
    label_html = _render_inline_text_only(label_raw, tokens)
    return _stash(f'<a href="{href}">{label_html}</a>')

  text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _repl_link, text)
  text = _render_inline_text_only(text, tokens)
  for k, v in tokens.items():
    text = text.replace(k, v)
  return text


def _parse_list(lines: list[str], start: int, base_dir: Path, embed_images: bool) -> tuple[str, int]:
  list_item_re = re.compile(r"^(\s*)(?:([-*+])|(\d+)\.)\s+(.*)$")

  def _collect_items(idx: int, base_indent: int, ordered: bool) -> tuple[str, int]:
    tag = "ol" if ordered else "ul"
    out = [f"<{tag}>"]
    n = len(lines)
    while idx < n:
      line = lines[idx]
      if not line.strip():
        look = idx + 1
        while look < n and not lines[look].strip():
          look += 1
        if look < n:
          m_next = list_item_re.match(lines[look])
          if m_next and len(m_next.group(1)) >= base_indent:
            idx = look
            continue
        break

      m = list_item_re.match(line)
      if not m:
        break
      indent = len(m.group(1))
      is_ord = m.group(3) is not None
      if indent < base_indent:
        break
      if indent > base_indent:
        sub_html, idx = _collect_items(idx, indent, is_ord)
        if out[-1].endswith("</li>"):
          out[-1] = out[-1][:-5] + "\n" + sub_html + "</li>"
        else:
          out.append(sub_html)
        continue
      if is_ord != ordered:
        break

      item_lines = [m.group(4)]
      idx += 1
      sub_chunks: list[str] = []
      while idx < n:
        nxt = lines[idx]
        if not nxt.strip():
          break
        m_sub = list_item_re.match(nxt)
        if m_sub:
          sub_indent = len(m_sub.group(1))
          if sub_indent > base_indent:
            sub_html, idx = _collect_items(idx, sub_indent, m_sub.group(3) is not None)
            sub_chunks.append(sub_html)
            continue
          break
        nxt_indent = len(nxt) - len(nxt.lstrip(" "))
        if nxt_indent > base_indent:
          item_lines.append(nxt.strip())
          idx += 1
        else:
          break

      item_html = _render_inline(" ".join(item_lines), base_dir, embed_images)
      if sub_chunks:
        out.append(f"<li>{item_html}\n" + "\n".join(sub_chunks) + "</li>")
      else:
        out.append(f"<li>{item_html}</li>")

    out.append(f"</{tag}>")
    return "\n".join(out), idx

  m0 = list_item_re.match(lines[start])
  assert m0 is not None
  return _collect_items(start, len(m0.group(1)), m0.group(3) is not None)


def _split_table_row(line: str) -> list[str]:
  s = line.strip()
  if s.startswith("|"):
    s = s[1:]
  if s.endswith("|"):
    s = s[:-1]
  return [cell.strip() for cell in s.split("|")]


def markdown_to_html(md_text: str, base_dir: Path, embed_images: bool = True) -> str:
  lines = md_text.splitlines()
  n = len(lines)
  i = 0
  body_parts: list[str] = []
  title = "AI Edge Bench Report"
  list_item_re = re.compile(r"^(\s*)(?:([-*+])|(\d+)\.)\s+(.*)$")

  while i < n:
    line = lines[i]
    stripped = line.strip()

    if not stripped:
      i += 1
      continue

    if stripped.startswith("```"):
      lang = stripped[3:].strip()
      code_lines: list[str] = []
      i += 1
      while i < n and not lines[i].strip().startswith("```"):
        code_lines.append(lines[i])
        i += 1
      if i < n:
        i += 1
      cls = f' class="language-{html.escape(lang)}"' if lang else ""
      body_parts.append(f"<pre><code{cls}>{html.escape(chr(10).join(code_lines))}</code></pre>")
      continue

    m_head = re.match(r"^(#{1,6})\s+(.*)$", stripped)
    if m_head:
      level = len(m_head.group(1))
      raw_heading = m_head.group(2).strip()
      if level == 1 and title == "AI Edge Bench Report":
        title = re.sub(r"`([^`]+)`", r"\1", raw_heading)
      slug = _slugify(raw_heading)
      heading_html = _render_inline(raw_heading, base_dir, embed_images)
      body_parts.append(f'<h{level} id="{slug}">{heading_html}</h{level}>')
      i += 1
      continue

    if stripped.startswith(">"):
      bq_lines: list[str] = []
      while i < n and lines[i].strip().startswith(">"):
        bq_lines.append(re.sub(r"^>\s?", "", lines[i].strip()))
        i += 1
      bq_html = _render_inline(" ".join(bq_lines), base_dir, embed_images)
      body_parts.append(f"<blockquote><p>{bq_html}</p></blockquote>")
      continue

    if "|" in stripped and i + 1 < n and re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", lines[i + 1].strip()):
      headers = _split_table_row(lines[i])
      align_specs = _split_table_row(lines[i + 1])
      aligns = []
      for idx_col in range(len(headers)):
        sp = align_specs[idx_col] if idx_col < len(align_specs) else ""
        if sp.startswith(":") and sp.endswith(":"):
          aligns.append(' style="text-align: center;"')
        elif sp.endswith(":"):
          aligns.append(' style="text-align: right;"')
        else:
          aligns.append("")
      i += 2
      rows: list[list[str]] = []
      while i < n and "|" in lines[i] and lines[i].strip():
        rows.append(_split_table_row(lines[i]))
        i += 1
      th_html = "".join(
          f"<th{aligns[idx_col]}>{_render_inline(h, base_dir, embed_images)}</th>"
          for idx_col, h in enumerate(headers)
      )
      tr_list: list[str] = []
      for r in rows:
        if r and r[0] and len(r) > 1 and all(not c for c in r[1:]):
          tr_list.append(
              f'<tr class="group-row"><td colspan="{len(headers)}">'
              f"{_render_inline(r[0], base_dir, embed_images)}</td></tr>"
          )
          continue
        tr_cls = ' class="best-row"' if (r and "🏆" in r[0]) else ""
        cells = [
            f"<td{aligns[idx_col] if idx_col < len(aligns) else ''}>"
            f"{_render_inline(c, base_dir, embed_images)}</td>"
            for idx_col, c in enumerate(r)
        ]
        tr_list.append(f"<tr{tr_cls}>" + "".join(cells) + "</tr>")
      body_parts.append(
          '<div class="table-wrap"><table>'
          f"<thead><tr>{th_html}</tr></thead>"
          f"<tbody>{''.join(tr_list)}</tbody>"
          "</table></div>"
      )
      continue

    if list_item_re.match(line):
      list_html, i = _parse_list(lines, i, base_dir, embed_images)
      body_parts.append(list_html)
      continue

    if re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped):
      body_parts.append(_render_inline(stripped, base_dir, embed_images))
      i += 1
      continue

    para_lines: list[str] = []
    while i < n:
      cur = lines[i].strip()
      if (
          not cur
          or cur.startswith("#")
          or cur.startswith("```")
          or cur.startswith(">")
          or list_item_re.match(lines[i])
          or ("|" in cur and i + 1 < n and re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", lines[i + 1].strip()))
      ):
        break
      para_lines.append(cur)
      i += 1
    para_html = _render_inline(" ".join(para_lines), base_dir, embed_images)
    body_parts.append(f"<p>{para_html}</p>")

  body_joined = "\n".join(body_parts)
  return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #ffffff;
      --fg: #1f2328;
      --muted: #59636e;
      --border: #d1d9e0;
      --surface: #f6f8fa;
      --accent: #0969da;
      --code-bg: #eff1f3;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      padding: 2rem 1.5rem 4rem;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
      font-size: 15px;
      line-height: 1.6;
      color: var(--fg);
      background: var(--bg);
    }}
    main {{
      max-width: 1440px;
      margin: 0 auto;
    }}
    h1, h2, h3, h4, h5 {{
      margin-top: 1.8rem;
      margin-bottom: 0.75rem;
      font-weight: 600;
      line-height: 1.25;
    }}
    h1 {{
      font-size: 1.85rem;
      padding-bottom: 0.4rem;
      border-bottom: 1px solid var(--border);
    }}
    h2 {{
      font-size: 1.45rem;
      padding-bottom: 0.3rem;
      border-bottom: 1px solid var(--border);
    }}
    h3 {{
      font-size: 1.25rem;
    }}
    h4 {{
      font-size: 1.08rem;
      margin-top: 1.4rem;
    }}
    h5 {{
      font-size: 0.98rem;
      margin-top: 1.1rem;
    }}
    p, ul, ol, blockquote {{
      margin-top: 0;
      margin-bottom: 1rem;
    }}
    ul, ol {{
      padding-left: 1.75rem;
    }}
    li {{
      margin-bottom: 0.25rem;
    }}
    li > ul, li > ol {{
      margin-top: 0.25rem;
      margin-bottom: 0.25rem;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    blockquote {{
      margin-left: 0;
      padding: 0.5rem 1rem;
      color: var(--muted);
      border-left: 4px solid var(--border);
      background: var(--surface);
      border-radius: 0 6px 6px 0;
    }}
    blockquote p:last-child {{
      margin-bottom: 0;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 0.88em;
      padding: 0.15em 0.35em;
      background: var(--code-bg);
      border-radius: 4px;
    }}
    pre {{
      padding: 1rem;
      overflow-x: auto;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
    }}
    pre code {{
      padding: 0;
      background: transparent;
    }}
    .table-wrap {{
      width: 100%;
      overflow-x: auto;
      margin: 1rem 0 1.5rem;
      border: 1px solid var(--border);
      border-radius: 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13.5px;
      white-space: nowrap;
    }}
    th, td {{
      padding: 0.5rem 0.75rem;
      border-bottom: 1px solid var(--border);
      border-right: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
    }}
    th:last-child, td:last-child {{
      border-right: none;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    thead th {{
      background: var(--surface);
      font-weight: 600;
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    tbody tr:nth-child(even) {{
      background: #fafbfc;
    }}
    tbody tr:hover {{
      background: #f0f4f8;
    }}
    tbody tr.group-row td {{
      background: #eef2f7;
      font-weight: 700;
      color: var(--fg);
      letter-spacing: 0.02em;
      border-top: 2px solid var(--border);
      border-bottom: 1px solid var(--border);
    }}
    tbody tr.best-row {{
      background: #e8f5e9 !important;
    }}
    tbody tr.best-row:hover {{
      background: #dcf0de !important;
    }}
    tbody tr.best-row td:first-child {{
      border-left: 4px solid #2e8b57;
    }}
    .report-figure {{
      margin: 1.5rem 0;
      text-align: center;
    }}
    .report-figure img {{
      max-width: 100%;
      height: auto;
      border: 1px solid var(--border);
      border-radius: 6px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }}
    .report-figure figcaption {{
      margin-top: 0.4rem;
      font-size: 0.85rem;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <main>
{body_joined}
  </main>
</body>
</html>
"""


if __name__ == "__main__":
  sys.exit(main())
