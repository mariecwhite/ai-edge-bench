"""Aggregate a suite's run directories into a standardized report + charts,
and maintain the performance-over-time history.

  python3 -m aeb.report --machine apple-m5-max --suite /suites/gemma4-e2b-cpu.json \\
      --results /results --out /reports [--since 2026-10-01T09:00:00Z] [--date 2026-10-01]

Writes reports/<date>-<suite>-<machine>/{README.md, data.json, *.png} and
appends one line per configuration to reports/history/<suite>/<machine>.jsonl
(idempotent per report id), then re-renders the history charts from that file.
The history file is the long-term record: it is small, committed, and keyed by
framework commits and harness digest so regressions can be attributed.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import accuracy as acc_mod  # noqa: E402
from . import stats  # noqa: E402

COLORS = {"llama.cpp": "#d9822b", "litert-lm": "#2b6cd9",
          "onnxruntime": "#9c4d9f"}
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
        vals.append(m[metric])
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
  data = dict(sweep, rounds=suite.get("rounds", 3),
              warmup=suite["workload"]["warmup"],
              repetitions=suite["workload"]["repetitions"], points={})
  for n in sweep["prompt_tokens"]:
    point = {"configs": {}, "prompt_identity": "unverified"}
    hashes = set()
    identity_configs = set()
    for c in suite["configs"]:
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
        if (s.get("status") == "ok" and len(requests) == data["repetitions"]
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
    if len(identity_configs) == len(suite["configs"]):
      point["prompt_identity"] = "identical" if len(hashes) == 1 else "DIFFERENT"
    data["points"][str(n)] = point
  return data


def fmt(v, f="{:.2f}"):
  return "n/a" if v is None else f.format(v)


def describe_cfg(runs):
  return {m[0]: stats.describe(pooled(runs, m[0])) for m in HEADLINE} | {
      "load_ms": stats.describe([r["summary"].get("load_ms") for r in runs]),
      "peak_rss_mb_lifetime": stats.describe([r["summary"].get("peak_rss_mb_lifetime") for r in runs]),
      "idle_rss_mb": stats.describe([(r["summary"].get("idle_after_load") or {}).get("rss_mb") for r in runs]),
      "idle_rss_anon_mb": stats.describe([(r["summary"].get("idle_after_load") or {}).get("rss_anon_mb") for r in runs]),
      "cgroup_peak_mb": stats.describe([r["summary"].get("cgroup_memory_peak_mb") for r in runs]),
      "system_cpu_busy_frac": stats.describe(pooled(runs, "system_cpu_busy_frac")),
      "round_medians_decode": [((r["summary"]["metrics"].get("decode_tps") or {}).get("median")) for r in runs],
      "round_medians_ttft": [((r["summary"]["metrics"].get("ttft_ms") or {}).get("median")) for r in runs],
  }


# ------------------------------------------------------------------- charts


def _label(cid: str) -> str:
  return cid.replace("/", "\n")


def bar_chart(path: Path, cfg_ids, agg, metrics, title):
  fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.2))
  axes = axes if len(metrics) > 1 else [axes]
  for ax, (m, lab, unit, hib, _) in zip(axes, metrics):
    xs, meds, lo, hi, cols = [], [], [], [], []
    for cid in cfg_ids:
      d = agg[cid].get(m)
      if not d:
        continue
      xs.append(_label(cid))
      meds.append(d["median"])
      lo.append(d["median"] - d["min"])
      hi.append(d["max"] - d["median"])
      cols.append(HIST_STYLE.get(cid, (COLORS[cid.split("/")[0]],))[0])
    ax.bar(xs, meds, yerr=[lo, hi], color=cols, capsize=4, alpha=0.9)
    for x, v in zip(xs, meds):
      ax.annotate(f"{v:.1f}" if v < 100 else f"{v:.0f}", (x, v), ha="center", va="bottom",
                  fontsize=8, xytext=(0, 3), textcoords="offset points")
    ax.set_title(f"{lab} ({unit}) {'↑' if hib else '↓'}")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", alpha=0.3)
  fig.suptitle(title + "  (median; whiskers = min-max over all measured requests)", fontsize=10)
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
      ax.errorbar(lengths, meds, yerr=[lo, hi], marker="o", capsize=3,
                  color=COLORS[cid.split("/")[0]], label=cid)
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths)
    ax.set_xticklabels([f"{n:,}" for n in lengths], rotation=30)
    ax.set_xlabel("Prefill sequence length N (tokens, including BOS)")
    ax.set_ylabel(f"{phase} throughput (tokens/s)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{phase} vs input length; fixed {data['gen_tokens']}-token decode\n"
                 "Median; whiskers = min-max; unavailable points are gaps")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"sequence_{phase.lower()}.png", dpi=130)
    plt.close(fig)


def sequence_section(data: dict) -> str:
  if not data:
    return ""
  cfg_ids = list(next(iter(data["points"].values()))["configs"])
  dates = sorted({r["meta"]["started_at"][:10] for p in data["points"].values()
                  for c in p["configs"].values() for r in c["runs"]})
  lines = ["## Input-length throughput sweep\n",
           "Sweep measurement dates (UTC): " + ", ".join(dates) + ".\n",
           f"Fixed **{data['gen_tokens']} generated tokens**, greedy with stop tokens ignored; "
           f"common context capacity **{data['ctx']:,}**. Model artifacts and thread counts "
           "are unchanged from the configurations above. These are reference configurations, "
           "not matched weights. The original headline workload remains unchanged.\n",
           f"Each point uses {data['rounds']} interleaved processes per configuration, "
           f"{data['warmup']} warm-ups and {data['repetitions']} measured requests per process. "
           "Long inputs repeat the same public-domain passage deterministically. "
           "N includes BOS and chat-template tokens. Each request starts with an empty KV cache.\n",
           "Prefill tokens/s = N / time to first token (includes tokenization and first-token "
           f"generation). Decode tokens/s = {data['gen_tokens'] - 1} / time from the first to the last generated "
           "token. Rates are medians; brackets show min-max. Failed, truncated or wrong-length "
           "processes are excluded, never replaced by estimated rates.\n"]
  for metric, phase in (("prefill_tps", "Prefill"), ("decode_tps", "Decode")):
    lines += [f"### {phase} (tokens/s)\n",
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
  lines += ["### Sweep validation\n",
            "| N | Token-ID identity | Measured requests by configuration |",
            "| --- | --- | --- |"]
  for n in data["prompt_tokens"]:
    point = data["points"][str(n)]
    counts = "; ".join(f"`{cid}`: {c['n_requests']}" for cid, c in point["configs"].items())
    lines.append(f"| {n:,} | {point['prompt_identity']} | {counts} |")
  for n, point in data["points"].items():
    for cid, c in point["configs"].items():
      for error in c["errors"]:
        lines.append(f"\n- N={n}, `{cid}`: {error.replace(chr(10), ' ')}")
  return "\n".join(lines) + "\n"


def timeline_chart(path: Path, cfg_ids, cfg_runs):
  rows = [cid for cid in cfg_ids if cfg_runs.get(cid)]
  fig, axes = plt.subplots(len(rows), 2, figsize=(13, 2.6 * len(rows)), squeeze=False)
  for i, cid in enumerate(rows):
    run = cfg_runs[cid][0]
    ts = read_timeseries(run["dir"] / "timeseries.csv")
    if not ts.get("t_ns"):
      continue
    t0 = ts["t_ns"][0]
    t = [(x - t0) / 1e9 for x in ts["t_ns"]]
    ax = axes[i][0]
    anon = [(v or 0) / 1024 for v in ts["rss_anon_kb"]]
    file_ = [(v or 0) / 1024 for v in ts["rss_file_kb"]]
    ax.stackplot(t, anon, file_, labels=["RSS anon", "RSS file-backed (mmap)"],
                 colors=["#555555", "#bbbbbb"], alpha=0.85)
    pss = [(tt, v / 1024) for tt, v in zip(t, ts["pss_kb"]) if v is not None]
    if pss:
      ax.plot(*zip(*pss), "k.", ms=3, label="PSS")
    ax.set_ylabel("MB")
    ax.set_title(f"{cid}: memory over time (round 0 process)", fontsize=9)
    ax2 = axes[i][1]
    ticks = ts["cpu_ticks"]
    cores_t, cores = [], []
    for j in range(1, len(t)):
      dt_ = t[j] - t[j - 1]
      if dt_ > 0:
        cores_t.append(t[j])
        cores.append((ticks[j] - ticks[j - 1]) / 100.0 / dt_)
    # 10 ms tick granularity: smooth over 5 samples for readability.
    sm = [statistics.fmean(cores[max(0, k - 2):k + 3]) for k in range(len(cores))]
    ax2.plot(cores_t, sm, color=COLORS[cid.split("/")[0]], lw=1)
    ax2.set_ylabel("cores busy")
    ax2.set_title(f"{cid}: CPU over time", fontsize=9)
    for r in _requests(run["dir"]):
      a, b, c = ((r["t_req_ns"] - t0) / 1e9, (r["t_first_ns"] - t0) / 1e9, (r["t_end_ns"] - t0) / 1e9)
      for axx in (ax, ax2):
        axx.axvspan(a, b, color="#f4c542", alpha=0.25, lw=0)
        axx.axvspan(b, c, color="#7ec97e", alpha=0.18, lw=0)
    if i == 0:
      ax.legend(fontsize=7, loc="lower right")
    for axx in (ax, ax2):
      axx.grid(alpha=0.3)
      axx.set_xlabel("seconds since process start (yellow = prefill, green = decode)", fontsize=7)
  fig.tight_layout()
  fig.savefig(path, dpi=110)
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


def scaling_chart(path: Path, runs, machine):
  series = collections.defaultdict(dict)
  for r in runs:
    m = r["meta"]
    if m.get("track") != "sweep-threads":
      continue
    label = m["label"]
    name, _, t = label.rpartition("-t")
    key = f"{m['framework']} {name}"
    s = r["summary"]["metrics"]
    if not s.get("decode_tps"):
      continue
    series[key][int(t)] = (s["prefill_tps"]["median"], s["decode_tps"]["median"],
                           (s.get("cpu_cores_decode") or {}).get("median"))
  if not series:
    return False
  fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
  styles = {"llama.cpp matched": ("#d9822b", "-o"), "litert-lm matched": ("#2b6cd9", "-o"),
            "litert-lm ynnpack": ("#2b6cd9", "--s")}
  for key, pts in sorted(series.items()):
    ts = sorted(pts)
    col, st = styles.get(key, ("gray", "-x"))
    for k, ax in enumerate(axes):
      ax.plot(ts, [pts[t][k] for t in ts], st, color=col, label=key)
  for ax, title in zip(axes, ["Prefill tok/s (1024-token prompt)", "Decode tok/s (64 tokens)",
                               "CPU cores busy during decode"]):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("threads")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
  fig.suptitle(f"Thread scaling on {machine} (sweep: 1 warm-up + 2 measured requests per point)",
               fontsize=10)
  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)
  return True


def accuracy_chart(path: Path, acc_rows, refs=None):
  tasks = sorted({a["task"] for a in acc_rows})
  fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4), squeeze=False)
  for ax, task in zip(axes[0], tasks):
    rows = [a for a in acc_rows if a["task"] == task]
    xs = [f"{a['id']}\n(n={a['n']})" for a in rows]
    ys = [a["accuracy"] * 100 for a in rows]
    err = [a["ci95"] * 100 for a in rows]
    ax.bar(xs, ys, yerr=err, capsize=4, color=[COLORS[a["id"].split("/")[0]] for a in rows])
    for x, y, e in zip(xs, ys, err):
      ax.annotate(f"{y:.1f}%", (x, y + e), ha="center", va="bottom", fontsize=8,
                  xytext=(0, 3), textcoords="offset points")
    # Published non-thinking bf16 references for the same benchmark.
    for r in (refs or {}).get("results", []):
      if r["benchmark"].lower() == task and "thinking on" not in r["setup"]:
        ax.axhline(r["value"], ls="--", lw=1, color="gray")
        ax.annotate(f"published {r['value']:.1f}% ({r['source_name'].split(',')[0]}; {r['setup'][:38]}…)",
                    (0.01, r["value"]), xycoords=("axes fraction", "data"), fontsize=6,
                    color="dimgray", va="bottom")
    ax.set_title(f"{task.upper()} accuracy (95% CI)", fontsize=10)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", alpha=0.3)
  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)


HIST_STYLE = {  # config -> (color, marker, linestyle)
    "llama.cpp/matched": ("#d9822b", "o", "-"), "llama.cpp/fastest": ("#a85a10", "s", "--"),
    "llama.cpp/stock-q4_0": ("#f0b070", "^", ":"), "litert-lm/matched": ("#2b6cd9", "o", "-"),
    "litert-lm/fastest": ("#13408f", "s", "--"),
}


def history_chart(path: Path, history: list[dict]):
  """One x position per report (ordered by date), so irregular cadence and a
  single first report still render legibly."""
  metrics = [("decode_tps", "Decode tok/s ↑"), ("prefill_tps", "Prefill tok/s ↑"),
             ("ttft_ms", "TTFT ms ↓"), ("rss_max_mb", "Peak RSS MB ↓")]
  reports = sorted({(h["date"], h["report_id"]) for h in history})
  xpos = {rid: i for i, (_, rid) in enumerate(reports)}
  fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 4.2))
  by_cfg = collections.defaultdict(list)
  for h in history:
    by_cfg[h["config"]].append(h)
  for ax, (m, title) in zip(axes, metrics):
    for cid, pts in sorted(by_cfg.items()):
      pts = sorted((h for h in pts if h["metrics"].get(m)), key=lambda h: xpos[h["report_id"]])
      if not pts:
        continue
      col, mk, ls = HIST_STYLE.get(cid, ("gray", "x", "-"))
      xs = [xpos[h["report_id"]] for h in pts]
      ys = [h["metrics"][m]["median"] for h in pts]
      lo = [h["metrics"][m]["median"] - h["metrics"][m]["min"] for h in pts]
      hi = [h["metrics"][m]["max"] - h["metrics"][m]["median"] for h in pts]
      ax.errorbar(xs, ys, yerr=[lo, hi], marker=mk, ls=ls, color=col, label=cid, capsize=3, ms=6)
    ax.set_xticks(range(len(reports)))
    ax.set_xticklabels([d for d, _ in reports], rotation=30, fontsize=7)
    ax.set_xlim(-0.5, max(len(reports) - 0.5, 0.5))
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
  axes[0].legend(fontsize=7)
  fig.suptitle("Performance over time (one column per report; whiskers = min-max over requests)",
               fontsize=10)
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
  has_scaling = scaling_chart(
      out / "thread_scaling.png",
      [r for r in runs if r["meta"].get("notes", "").startswith(f"suite={suite['name']}")],
      args.machine)
  if acc_rows:
    refs_path = args.suite.parent / suite["references"] if suite.get("references") else None
    accuracy_chart(out / "accuracy.png", acc_rows,
                   json.loads(refs_path.read_text()) if refs_path and refs_path.is_file() else None)

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
          "accuracy": acc_rows, "accuracy_agreement": agreement,
          "reproducibility": {"rows": repro_tbl, "cross_framework": repro_cross}}
  (out / "data.json").write_text(json.dumps(data, indent=1, default=str) + "\n")

  extra = args.extra_notes.read_text() if args.extra_notes else ""
  args._quant_md = (quant_section(args.models / "aeb")
                    if any(c["track"] == "matched" for c in suite["configs"]) else "")
  refs_path = args.suite.parent / suite["references"] if suite.get("references") else None
  args._refs = json.loads(refs_path.read_text()) if refs_path and refs_path.is_file() else None
  (out / "README.md").write_text(render_markdown(
      suite, args, report_id, cfg_ids, cfg_runs, agg, acc_rows, has_scaling, history, changes,
      hist_path, extra, agreement, (repro_tbl, repro_cross), sequence))
  print(f"wrote {out}")
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


def render_markdown(suite, args, report_id, cfg_ids, cfg_runs, agg, acc_rows, has_scaling,
                    history, changes, hist_path, extra, agreement=(), repro=([], []), sequence=None):
  L = []
  w = suite["workload"]
  first = cfg_runs[cfg_ids[0]][0]["meta"]
  sysinfo = _read_json(cfg_runs[cfg_ids[0]][0]["dir"] / "sysinfo.json") or {}
  L.append(f"# {suite['name']} on {args.machine} — {args.date}\n")
  sweep_count = sum(len(c["runs"]) for p in (sequence or {}).get("points", {}).values()
                    for c in p["configs"].values())
  process_description = ("benchmark processes" if not sweep_count else
                         f"headline benchmark processes plus {sweep_count} input-length sweep processes")
  L.append(f"> Generated by `python3 -m aeb.report` from {sum(len(v) for v in cfg_runs.values())} "
           f"{process_description}. Raw per-request data: [data.json](data.json). "
           f"Methodology: [benchmark_methodology.md](../../docs/benchmark_methodology.md).\n")
  L.append(suite["description"] + "\n")
  # Hand-written notes: anything before "## Caveats" (e.g. "## Key findings")
  # goes to the top, the caveats section to the end.
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
  L.append(f"| Container view | {sysinfo.get('container', {}).get('os', '?')}, kernel "
           f"{sysinfo.get('container', {}).get('kernel', '?')}, {cpu.get('nproc', '?')} vCPUs, "
           f"{(sysinfo.get('memory', {}).get('total_kb', 0) / 2**20):.1f} GiB RAM |")
  for fw, b in sorted(builds.items()):
    revision = f", commit `{str(b['commit'])[:12]}`" if b.get("commit") else ""
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

  # headline table
  L.append("## Results\n")
  L.append("Medians over all measured requests (n per cell in the last row); "
           "[min – max] in brackets.\n")
  hdr = "| Metric | " + " | ".join(f"`{c}`" for c in cfg_ids) + " |"
  L.append(hdr)
  L.append("| --- " * (len(cfg_ids) + 1) + "|")
  for m, lab, unit, hib, f in HEADLINE:
    cells = []
    for cid in cfg_ids:
      d = agg[cid].get(m)
      cells.append("n/a" if not d else f"{f.format(d['median'])} [{f.format(d['min'])}–{f.format(d['max'])}]")
    L.append(f"| {lab} ({unit}) {'↑' if hib else '↓'} | " + " | ".join(cells) + " |")
  for key, lab, f in (("load_ms", "Model load, warm cache (ms)", "{:.0f}"),
                      ("peak_rss_mb_lifetime", "Peak RSS, process lifetime (VmHWM, MB)", "{:.0f}"),
                      ("idle_rss_mb", "RSS after load, before 1st request (MB)", "{:.0f}"),
                      ("idle_rss_anon_mb", "  …of which anonymous (MB)", "{:.0f}"),
                      ("cgroup_peak_mb", "Container memory.peak incl. page cache (MB)", "{:.0f}")):
    cells = [fmt((agg[cid].get(key) or {}).get("median"), f) for cid in cfg_ids]
    L.append(f"| {lab} | " + " | ".join(cells) + " |")
  L.append("| Measured requests (n) | " + " | ".join(
      str((agg[cid].get("e2e_ms") or {}).get("n", 0)) for cid in cfg_ids) + " |")
  L.append("| Decode CV across requests | " + " | ".join(
      fmt((agg[cid].get("decode_tps") or {}).get("cv"), "{:.1%}") for cid in cfg_ids) + " |\n")

  # ratios
  pairs = []
  if "llama.cpp/matched" in agg and "litert-lm/matched" in agg:
    pairs.append(("matched", "llama.cpp/matched", "litert-lm/matched"))
  if "llama.cpp/fastest" in agg and "litert-lm/fastest" in agg:
    pairs.append(("fastest", "llama.cpp/fastest", "litert-lm/fastest"))
  if pairs:
    L.append("### llama.cpp ÷ LiteRT-LM\n")
    L.append("Ratio of medians with a 95% bootstrap interval over requests. >1 means llama.cpp has "
             "the larger value (better for throughput, worse for latency/CPU/memory).\n")
    L.append("| Track | Prefill tok/s | Decode tok/s | TTFT | End-to-end | CPU s / request | Peak RSS |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, a, b in pairs:
      L.append(f"| {name} | " + " | ".join(ratio_line(agg, cfg_runs, a, b, m, True) for m in
                                           ("prefill_tps", "decode_tps", "ttft_ms", "e2e_ms",
                                            "cpu_seconds", "rss_max_mb")) + " |")
    L.append("")

  L.append("![Throughput](throughput.png)\n")
  L.append("![Latency](latency.png)\n")
  L.append("![CPU](cpu.png)\n")
  L.append("![Memory](memory.png)\n")
  L.append("### Memory and CPU over time\n")
  L.append("One process per configuration (round 0): load, warm-up and measured requests. "
           "Yellow = prefill, green = decode. File-backed RSS is mmap'd model pages (shared, "
           "reclaimable); anonymous RSS is private memory (KV cache, activations, repacked or "
           "converted weights).\n")
  L.append("![Timeline](timeline.png)\n")
  if has_scaling:
    L.append("### Thread scaling\n")
    L.append("![Thread scaling](thread_scaling.png)\n")

  if sequence:
    L.append(sequence_section(sequence))

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

  # history
  L.append("## Performance over time\n")
  L.append(f"Every report appends one line per configuration to "
           f"[`{hist_path.relative_to(args.out)}`](../{hist_path.relative_to(args.out)}) "
           f"(medians, ranges, framework commits, harness digest, model hashes). A change is "
           f"flagged when the median moves by more than max({CHANGE_THRESHOLD:.0%}, 3 × the "
           f"larger coefficient of variation) between consecutive reports of the same config.\n")
  L.append("![History](history.png)\n")
  rows = [h for h in history if h["machine"] == args.machine]
  L.append("| Date | Config | Framework commit | Decode tok/s | Prefill tok/s | TTFT ms | Peak RSS MB | MMLU |")
  L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
  for h in sorted(rows, key=lambda h: (h["date"], h["config"]))[-30:]:
    mt = h["metrics"]
    mm = h.get("accuracy", {}).get("mmlu")
    L.append(f"| {h['date']} | `{h['config']}` | `{str(h['framework_commit'])[:10]}` | "
             f"{fmt((mt.get('decode_tps') or {}).get('median'))} | "
             f"{fmt((mt.get('prefill_tps') or {}).get('median'), '{:.1f}')} | "
             f"{fmt((mt.get('ttft_ms') or {}).get('median'), '{:.0f}')} | "
             f"{fmt((mt.get('rss_max_mb') or {}).get('median'), '{:.0f}')} | "
             f"{fmt(mm['acc'], '{:.2%}') if mm else 'n/a'} |")
  L.append("")
  L.append("**Flagged changes vs. previous report:** " +
           ("; ".join(changes) if changes else "none (first report, or all within noise).") + "\n")

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
  return "\n".join(L)


if __name__ == "__main__":
  sys.exit(main())
