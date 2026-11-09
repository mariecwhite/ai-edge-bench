#!/usr/bin/env python3
"""Run a benchmark suite one container at a time (host side, stdlib only).

  python3 scripts/suite.py perf     --machine apple-m5-max [--suite suites/gemma4-e2b-cpu.json]
  python3 scripts/suite.py accuracy --machine apple-m5-max [--only llama.cpp/matched,...]
  python3 scripts/suite.py repro    --machine apple-m5-max   # output reproducibility
  python3 scripts/suite.py sequence --machine apple-m5-max --suite suites/gemma4-e2b-cpu-arm-runtimes.json
  python3 scripts/suite.py plan     --machine apple-m5-max   # print commands only
  python3 scripts/suite.py perf     --machine apple-m5-max --suite suites/gemma4-26b-a4b-gpu.json
                                    # "execution": "native": runs on the macOS host (Metal)

perf: first primes every config once (unmeasured; populates framework caches),
then every config runs in its own container/process for `rounds` rounds. The
order is reversed on every other round (A B C, C B A, A B C, ...) so slow
drift (thermal, background load) is spread across configs instead of landing
on whichever runs last. Never run two benchmark containers at once.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE = {
    "llama.cpp": "harness-llama-cpp",
    "litert-lm": "harness-litert-lm",
    "onnxruntime": "harness-onnxruntime",
    "xla": "harness-xla",
    "openvino": "harness-openvino",
}
# Native (host) execution: suites with "execution": "native" run the harness
# directly on the host (e.g. Metal GPU on macOS, unreachable from the Docker VM)
# with the stack built by scripts/native-apple-gpu.sh.
NATIVE = ROOT / "build" / "native"
NATIVE_BUILD_INFO = {"llama.cpp": "llama.cpp", "litert-lm": "litert-lm", "mlx": "mlx",
                     "basert": "basert"}
EXECUTION = {"mode": "container", "machine_notes": "", "chat_template": "gemma4"}


def resolve(value: str, ctx: dict) -> str:
  def sub(m):
    key = m.group(1)
    cur = ctx
    for part in key.split("."):
      if part not in cur:
        # allow dotted framework names like "llama.cpp"
        return _lookup_dotted(ctx, key)
      cur = cur[part]
    return str(cur)
  return re.sub(r"\{([^{}]+)\}", sub, value)


def _lookup_dotted(ctx: dict, key: str) -> str:
  # fastest.llama.cpp.threads -> ctx["fastest"]["llama.cpp"]["threads"]
  head, rest = key.split(".", 1)
  node = ctx[head]
  for name in sorted(node, key=len, reverse=True):
    if rest.startswith(name + "."):
      return str(node[name][rest[len(name) + 1:]])
  raise KeyError(key)


def git_rev() -> str:
  try:
    rev = subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT,
                                  text=True, stderr=subprocess.DEVNULL).strip()
    dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=ROOT,
                            stderr=subprocess.DEVNULL) != 0
    return rev + ("-dirty" if dirty else "")
  except (OSError, subprocess.CalledProcessError):
    return "unknown"


def compose_base(machine: str, dev: bool) -> list[str]:
  cmd = ["docker", "compose", "-f", str(ROOT / "compose.yaml"),
         "-f", str(ROOT / "compose" / "machines" / f"{machine}.yaml"), "--profile", "harness"]
  return cmd


def host_path(arg: str) -> str:
  """Map the container mount points used in suite files to the repo directories."""
  for mount in ("/models", "/results", "/datasets"):
    if arg == mount or arg.startswith(mount + "/"):
      return str(ROOT / mount.lstrip("/")) + arg[len(mount):]
  return arg


def native_cmd(machine: str, framework: str, module_args: list[str]) -> list[str]:
  env = {
      "AEB_MACHINE": machine,
      "AEB_MACHINE_NOTES": EXECUTION["machine_notes"],
      "AEB_RESULTS_DIR": str(ROOT / "results"),
      "AEB_DATASETS_DIR": str(ROOT / "datasets"),
      "AEB_GIT_REV": git_rev(),
      "AEB_BUILD_INFO": str(NATIVE / NATIVE_BUILD_INFO[framework] / "build-info.json"),
      "PYTHONPATH": str(ROOT / "docker" / "harness"),
      "PATH": f"{NATIVE / 'llama.cpp' / 'bin'}:{NATIVE / 'venv' / 'bin'}:/usr/bin:/bin:/usr/sbin:/sbin",
      "BASERT_LIB_PATH": str(NATIVE / "basert" / "engine" / "libbaseRT.dylib"),
      "PYTHONUNBUFFERED": "1",
      "AEB_CHAT_TEMPLATE": EXECUTION["chat_template"],
  }
  return (["env", "-i", *(f"{k}={v}" for k, v in env.items()),
           str(NATIVE / "venv" / "bin" / "python"), "-m"]
          + [host_path(a) for a in module_args])


def run_cmd(machine: str, framework: str, module_args: list[str], dev: bool) -> list[str]:
  if EXECUTION["mode"] == "native":
    return native_cmd(machine, framework, module_args)
  cmd = compose_base(machine, dev) + ["run", "--rm", "-T", "-e", f"AEB_GIT_REV={git_rev()}"]
  if EXECUTION["chat_template"] != "gemma4":
    cmd += ["-e", f"AEB_CHAT_TEMPLATE={EXECUTION['chat_template']}"]
  if dev:
    cmd += ["-v", f"{ROOT / 'docker' / 'harness'}:/opt/aeb/harness:ro"]
  return cmd + [SERVICE[framework], *module_args]


def load(suite_path: Path, machine: str) -> tuple[dict, dict]:
  suite = json.loads(suite_path.read_text())
  mcfg = suite.get("machines", {}).get(machine)
  if mcfg is None:
    raise SystemExit(f"suite has no 'machines.{machine}' section (threads / fastest config)")
  ctx = dict(suite["models"])
  ctx.update(threads=mcfg["threads"], ctx=suite["workload"]["ctx"], fastest=mcfg.get("fastest", {}))
  EXECUTION.update(mode=suite.get("execution", "container"),
                   machine_notes=mcfg.get("machine_notes", ""),
                   chat_template=suite["workload"].get("chat_template", "gemma4"))
  return suite, ctx


def selected(suite: dict, only: str | None) -> list[dict]:
  cfgs = [c for c in suite["configs"] if not c.get("disabled")]
  if only:
    want = set(only.split(","))
    missing = want - {c["id"] for c in suite["configs"]}
    if missing:
      raise ValueError(f"unknown suite config(s): {', '.join(sorted(missing))}")
    cfgs = [c for c in cfgs if c["id"] in want]
  return cfgs


def _has_complete_prime(results_dir: Path | None, machine: str, fw: str, label: str) -> bool:
  if results_dir is None:
    return False
  for lbl in (f"prime-{label}", label):
    cfg_dir = results_dir / machine / "perf" / fw / lbl
    if not cfg_dir.is_dir():
      continue
    for summ_p in sorted(cfg_dir.glob("*/summary.json")):
      try:
        summ = json.loads(summ_p.read_text())
      except (OSError, json.JSONDecodeError):
        continue
      if summ.get("status") in ("ok", "warning") and len(summ.get("per_request") or []) >= 1:
        return True
  return False


def _has_complete_perf(results_dir: Path | None, machine: str, fw: str, label: str,
                       track: str, round_idx: int, expected_reps: int) -> bool:
  if results_dir is None:
    return False
  cfg_dir = results_dir / machine / "perf" / fw / label
  if not cfg_dir.is_dir():
    return False
  for summ_p in sorted(cfg_dir.glob("*/summary.json")):
    meta_p = summ_p.parent / "meta.json"
    if not meta_p.is_file():
      continue
    try:
      summ = json.loads(summ_p.read_text())
      meta = json.loads(meta_p.read_text())
    except (OSError, json.JSONDecodeError):
      continue
    if summ.get("status") not in ("ok", "warning"):
      continue
    if int(summ.get("n_measured", 0)) < int(expected_reps):
      continue
    if meta.get("track") == track and int(meta.get("round", 0)) == int(round_idx):
      return True
  return False


def _has_complete_accuracy(results_dir: Path | None, machine: str, fw: str, label: str,
                           task: str, limit: int, shard: str) -> bool:
  if results_dir is None:
    return False
  task_dir = results_dir / machine / f"accuracy-{task}" / fw / label
  if not task_dir.is_dir():
    return False
  target_limit = int(limit or TASK_ITEMS.get(task, 0))
  for summ_p in sorted(task_dir.glob("*/summary.json")):
    meta_p = summ_p.parent / "meta.json"
    if not meta_p.is_file():
      continue
    try:
      summ = json.loads(summ_p.read_text())
      meta = json.loads(meta_p.read_text())
    except (OSError, json.JSONDecodeError):
      continue
    if not summ.get("complete") or not summ.get("n"):
      continue
    meta_limit = int(meta.get("limit") or TASK_ITEMS.get(task, 0))
    if meta_limit >= target_limit and str(meta.get("shard", "0/1")) == shard:
      return True
  return False


def _find_resumable_accuracy(results_dir: Path | None, machine: str, fw: str, label: str,
                             task: str, limit: int, shard: str) -> str | None:
  if results_dir is None:
    return None
  task_dir = results_dir / machine / f"accuracy-{task}" / fw / label
  if not task_dir.is_dir():
    return None
  target_limit = int(limit or TASK_ITEMS.get(task, 0))
  candidates: list[tuple[str, str]] = []
  for meta_p in sorted(task_dir.glob("*/meta.json")):
    pred_p = meta_p.parent / "predictions.jsonl"
    if not pred_p.is_file() or pred_p.stat().st_size == 0:
      continue
    try:
      meta = json.loads(meta_p.read_text())
    except (OSError, json.JSONDecodeError):
      continue
    meta_limit = int(meta.get("limit") or TASK_ITEMS.get(task, 0))
    if meta_limit >= target_limit and str(meta.get("shard", "0/1")) == shard:
      candidates.append((str(meta.get("started_at", "")), meta_p.parent.name))
  if not candidates:
    return None
  candidates.sort()
  return f"/results/{machine}/accuracy-{task}/{fw}/{label}/{candidates[-1][1]}"


def _has_complete_repro(results_dir: Path | None, machine: str, fw: str, label: str,
                        mode: str, seed: int | None, proc: int, threads: int) -> bool:
  if results_dir is None:
    return False
  cfg_dir = results_dir / machine / "repro" / fw / label
  if not cfg_dir.is_dir():
    return False
  expected_seed = int(seed) if (mode == "seeded" and seed is not None) else None
  for summ_p in sorted(cfg_dir.glob("*/summary.json")):
    meta_p = summ_p.parent / "meta.json"
    if not meta_p.is_file():
      continue
    try:
      summ = json.loads(summ_p.read_text())
      meta = json.loads(meta_p.read_text())
    except (OSError, json.JSONDecodeError):
      continue
    if int(summ.get("errors", 1)) != 0:
      continue
    if int((summ.get("within_process") or {}).get("compared", 0)) <= 0:
      continue
    if (meta.get("mode") == mode and int(meta.get("process", -1)) == int(proc)
        and meta.get("seed") == expected_seed and int(meta.get("threads") or 0) == int(threads)):
      return True
  return False


def perf_commands(suite, ctx, machine, rounds, only, dev,
                  results_dir: Path | None = None, skip_existing: bool = False):
  w = suite["workload"]
  cfgs = selected(suite, only)
  out = []
  for r in range(rounds):
    order = cfgs if r % 2 == 0 else list(reversed(cfgs))
    for c in order:
      fw, label = c["framework"], c["id"].split("/", 1)[1]
      if skip_existing and _has_complete_perf(
          results_dir, machine, fw, label, c["track"], r, int(w["repetitions"])):
        print(f"[suite] skip existing perf {c['id']} (round {r})", flush=True)
        continue
      args = ["aeb.perf", "--framework", fw, "--label", label, "--track", c["track"],
              "--round", str(r), "--prompt-tokens", str(w["prompt_tokens"])]
      if int(w.get("prefix_tokens") or 0) > 0:
        args += ["--prefix-tokens", str(w["prefix_tokens"])]
      args += ["--gen-tokens", str(w["gen_tokens"]), "--warmup", str(w["warmup"]),
               "--repetitions", str(w["repetitions"]), "--cooldown-s", str(w["cooldown_s"]),
               "--notes", f"suite={suite['name']}", "--"]
      args += [resolve(a, ctx) for a in c["args"]]
      out.append((c["id"], run_cmd(machine, fw, args, dev)))
  return out


def prime_commands(suite, ctx, machine, only, dev,
                   results_dir: Path | None = None, skip_existing: bool = False):
  """One unmeasured request per config so framework caches (e.g. the XNNPACK
  weight cache) exist before any measured process starts."""
  out = []
  for c in selected(suite, only):
    fw, label = c["framework"], c["id"].split("/", 1)[1]
    if skip_existing and _has_complete_prime(results_dir, machine, fw, label):
      print(f"[suite] skip existing prime {c['id']}", flush=True)
      continue
    args = ["aeb.perf", "--framework", fw, "--label", f"prime-{label}", "--track", "prime",
            "--prompt-tokens", "128", "--gen-tokens", "8", "--warmup", "1", "--repetitions", "0",
            "--cooldown-s", "0", "--notes", f"prime:{suite['name']}", "--"]
    args += [resolve(a, ctx) for a in c["args"]]
    out.append((f"prime {c['id']}", run_cmd(machine, fw, args, dev)))
  return out


def config_dtype(c: dict) -> str:
  if c.get("dtype"):
    return str(c["dtype"])
  args = c.get("args", [])
  if "--dtype" in args:
    dt = args[args.index("--dtype") + 1]
    return "int4" if dt == "qat-mobile" else dt
  cid = c["id"].lower()
  for dt in ("fp32", "bf16", "fp16", "int8"):
    if dt in cid:
      return dt
  if "q8_0" in cid:
    return "int8"
  return "int4"


def _config_accuracy(results_dir: Path, machine: str, fw: str, label: str, task: str) -> float | None:
  import collections
  task_dir = results_dir / machine / f"accuracy-{task}" / fw / label
  if not task_dir.is_dir():
    return None
  groups: dict[tuple[int, int], dict[int, tuple[str, dict]]] = collections.defaultdict(dict)
  for summ_p in sorted(task_dir.glob("*/summary.json")):
    meta_p = summ_p.parent / "meta.json"
    if not meta_p.is_file():
      continue
    try:
      summ = json.loads(summ_p.read_text())
      meta = json.loads(meta_p.read_text())
    except (OSError, json.JSONDecodeError):
      continue
    if not summ.get("complete") or not summ.get("n"):
      continue
    k, n_shards = (int(x) for x in str(meta.get("shard", "0/1")).split("/"))
    key = (int(meta.get("limit") or 0), n_shards)
    started = str(meta.get("started_at", ""))
    prev = groups[key].get(k)
    if prev is None or started >= prev[0]:
      groups[key][k] = (started, summ)
  best_n = -1
  best_acc = None
  for (_, n_shards), shards in groups.items():
    if len(shards) != n_shards:
      continue
    tot_n = sum(int(s["n"]) for _, s in shards.values())
    tot_c = sum(int(s.get("correct", round(s["accuracy"] * s["n"]))) for _, s in shards.values())
    if tot_n > 0 and tot_n > best_n:
      best_n = tot_n
      best_acc = tot_c / tot_n
  return best_acc


def _config_median_e2e_ms(results_dir: Path, machine: str, suite_name: str, rounds: int,
                          fw: str, label: str) -> float | None:
  import statistics
  cfg_dir = results_dir / machine / "perf" / fw / label
  if not cfg_dir.is_dir():
    return None
  tag = f"suite={suite_name}"
  runs = []
  for summ_p in sorted(cfg_dir.glob("*/summary.json")):
    meta_p = summ_p.parent / "meta.json"
    if not meta_p.is_file():
      continue
    try:
      summ = json.loads(summ_p.read_text())
      meta = json.loads(meta_p.read_text())
    except (OSError, json.JSONDecodeError):
      continue
    if summ.get("status") != "ok":
      continue
    if meta.get("track") in ("sweep-sequence", "prime", "sweep-threads", "prefix-cache"):
      continue
    runs.append((meta.get("notes", "").startswith(tag), str(meta.get("started_at", "")), summ))
  if not runs:
    return None
  suite_runs = [r for r in runs if r[0]] or runs
  suite_runs.sort(key=lambda r: r[1])
  selected_runs = suite_runs[-max(1, rounds):]
  vals = []
  for _, _, summ in selected_runs:
    for req in summ.get("per_request", []):
      if req.get("measured") and req.get("e2e_ms") is not None:
        vals.append(float(req["e2e_ms"]))
    if not vals:
      med = (summ.get("metrics", {}).get("e2e_ms") or {}).get("median")
      if med is not None:
        vals.append(float(med))
  return statistics.median(vals) if vals else None


def _normalize_min_accuracy(raw) -> dict[str, float]:
  if raw is None:
    return {}
  if isinstance(raw, (int, float)):
    return {"mmlu": float(raw), "gsm8k": float(raw)}
  if isinstance(raw, dict):
    return {str(k): float(v) for k, v in raw.items()}
  raise ValueError(f"invalid min_accuracy: {raw!r}")


def select_sequence_configs(suite: dict, machine: str, results_dir: Path | None = None,
                            min_accuracy: float | dict | None = None,
                            allow_unmeasured_fallback: bool = True) -> list[dict]:
  """Select the fastest configuration per (framework, dtype) pair that clears
  the minimum accuracy threshold."""
  sweep = suite.get("sequence_sweep") or {}
  results_dir = results_dir or (ROOT / "results")
  thresholds = _normalize_min_accuracy(sweep.get("min_accuracy"))
  if min_accuracy is not None:
    override = _normalize_min_accuracy(min_accuracy)
    if thresholds and isinstance(min_accuracy, (int, float)):
      thresholds = {k: float(min_accuracy) for k in thresholds}
    else:
      thresholds.update(override)
  by_dtype = bool(sweep.get("select_fastest_by_dtype") or thresholds)
  cfgs = [c for c in suite.get("configs", []) if not c.get("disabled")]
  if not by_dtype:
    return cfgs

  machine_dir = results_dir / machine
  has_acc = machine_dir.is_dir() and any(machine_dir.glob("accuracy-*/*/*/*/summary.json"))
  has_perf = (machine_dir / "perf").is_dir() and any((machine_dir / "perf").glob("*/*/*/summary.json"))

  track_rank = {"fastest": 0, "matched": 1, "reference": 2}
  if not has_acc and not has_perf and allow_unmeasured_fallback:
    groups: dict[tuple[str, str], list[tuple[int, int, dict]]] = {}
    for idx, c in enumerate(cfgs):
      key = (c["framework"], config_dtype(c))
      groups.setdefault(key, []).append((track_rank.get(c.get("track", ""), 9), idx, c))
    chosen_ids = {min(items)[2]["id"] for items in groups.values()}
    return [c for c in cfgs if c["id"] in chosen_ids]

  rounds = suite.get("rounds", 3)
  candidates_by_pair: dict[tuple[str, str], list[tuple[float, int, int, dict, dict[str, float]]]] = {}
  for idx, c in enumerate(cfgs):
    fw, label = c["framework"], c["id"].split("/", 1)[1]
    dtype = config_dtype(c)
    acc_cfg = c.get("accuracy") or {}
    measured_acc: dict[str, float] = {}
    failed_reasons = []
    for task, min_val in thresholds.items():
      acc_val = _config_accuracy(results_dir, machine, fw, label, task)
      if acc_val is not None:
        measured_acc[task] = acc_val
      required = (f"{task}_limit" in acc_cfg) or (not acc_cfg and task == "mmlu")
      if required or acc_val is not None:
        if acc_val is None:
          failed_reasons.append(f"{task}=missing (need >={min_val:.1%})")
        elif acc_val < min_val:
          failed_reasons.append(f"{task}={acc_val:.1%} < {min_val:.1%}")
    if thresholds and not measured_acc and not failed_reasons:
      failed_reasons.append("no accuracy tasks evaluated")
    if failed_reasons:
      print(f"[suite] sequence-gate skip {c['id']} ({fw}/{dtype}): {', '.join(failed_reasons)}",
            flush=True)
      continue
    e2e = _config_median_e2e_ms(results_dir, machine, suite.get("name", ""), rounds, fw, label)
    sort_e2e = e2e if e2e is not None else float("inf")
    candidates_by_pair.setdefault((fw, dtype), []).append(
        (sort_e2e, track_rank.get(c.get("track", ""), 9), idx, c, measured_acc)
    )

  chosen_ids = set()
  for (fw, dtype), items in candidates_by_pair.items():
    items.sort(key=lambda x: (x[0], x[1], x[2]))
    best_e2e, _, _, winner, winner_acc = items[0]
    chosen_ids.add(winner["id"])
    e2e_str = f"{best_e2e:.1f} ms" if best_e2e != float("inf") else "unmeasured"
    acc_str = ", ".join(f"{t}={v:.1%}" for t, v in sorted(winner_acc.items())) or "n/a"
    print(f"[suite] sequence-gate select {winner['id']} for ({fw}, {dtype}): "
          f"median e2e={e2e_str}, accuracy=({acc_str})", flush=True)

  return [c for c in cfgs if c["id"] in chosen_ids]


def select_fastest_by_dtype_configs(suite: dict, machine: str,
                                    results_dir: Path | None = None) -> list[dict]:
  """Select the empirically fastest configuration per (framework, dtype) pair
  based on measured Phase 3 perf latency (falling back to track rank if unmeasured)."""
  results_dir = results_dir or (ROOT / "results")
  cfgs = [c for c in suite.get("configs", []) if not c.get("disabled")]
  rounds = suite.get("rounds", 3)
  track_rank = {"fastest": 0, "matched": 1, "reference": 2}
  candidates_by_pair: dict[tuple[str, str], list[tuple[float, int, int, dict]]] = {}
  for idx, c in enumerate(cfgs):
    fw, label = c["framework"], c["id"].split("/", 1)[1]
    dtype = config_dtype(c)
    e2e = _config_median_e2e_ms(results_dir, machine, suite.get("name", ""), rounds, fw, label)
    sort_e2e = e2e if e2e is not None else float("inf")
    candidates_by_pair.setdefault((fw, dtype), []).append(
        (sort_e2e, track_rank.get(c.get("track", ""), 9), idx, c)
    )
  chosen_ids = set()
  for (fw, dtype), items in candidates_by_pair.items():
    items.sort(key=lambda x: (x[0], x[1], x[2]))
    best_e2e, _, _, winner = items[0]
    chosen_ids.add(winner["id"])
    e2e_str = f"{best_e2e:.1f} ms" if best_e2e != float("inf") else "unmeasured (track rank)"
    print(f"[suite] accuracy-select {winner['id']} for ({fw}, {dtype}): median e2e={e2e_str}",
          flush=True)
  return [c for c in cfgs if c["id"] in chosen_ids]


def sequence_commands(suite, ctx, machine, rounds, only, dev, results_dir: Path | None = None,
                      min_accuracy: float | dict | None = None, skip_existing: bool = False):
  sweep = suite.get("sequence_sweep")
  if not sweep:
    raise ValueError("suite has no sequence_sweep section")
  lengths = sweep["prompt_tokens"]
  gen = sweep["gen_tokens"]
  if (not lengths or any(type(n) is not int or n < 1 for n in lengths)
      or len(set(lengths)) != len(lengths) or type(gen) is not int or gen < 2
      or type(sweep["ctx"]) is not int or sweep["ctx"] < max(lengths) + gen):
    raise ValueError("invalid sequence_sweep lengths, decode length or context capacity")
  if only:
    cfgs = selected(suite, only)
  else:
    cfgs = select_sequence_configs(suite, machine, results_dir=results_dir,
                                   min_accuracy=min_accuracy)
  out = []
  sweep_ctx = dict(ctx, ctx=sweep["ctx"])
  sweep_rounds = sweep.get("rounds", rounds)
  for n in lengths:
    point = dict(suite)
    point["workload"] = dict(
        suite["workload"],
        prompt_tokens=n,
        gen_tokens=gen,
        warmup=sweep.get("warmup", suite["workload"]["warmup"]),
        repetitions=sweep.get("repetitions", suite["workload"]["repetitions"]),
        cooldown_s=sweep.get("cooldown_s", suite["workload"]["cooldown_s"]),
    )
    point["configs"] = [dict(c, id=f"{c['id']}-n{n}-d{gen}", track="sweep-sequence")
                        for c in cfgs]
    out += perf_commands(point, sweep_ctx, machine, sweep_rounds, None, dev,
                         results_dir=results_dir, skip_existing=skip_existing)
  return out


def prefix_cache_commands(suite, ctx, machine, rounds, only, dev,
                          results_dir: Path | None = None, skip_existing: bool = False):
  """Run cached-prefix benchmarks where `prefix_tokens` of the prompt are pre-cached
  in the KV cache and each request appends a distinct suffix of
  `prompt_tokens - prefix_tokens` tokens."""
  pc = suite.get("prefix_cache") or {}
  w = suite["workload"]
  prompt_tokens = int(pc.get("prompt_tokens", w["prompt_tokens"]))
  prefix_tokens = int(pc.get("prefix_tokens", prompt_tokens * 7 // 8))
  gen_tokens = int(pc.get("gen_tokens", w["gen_tokens"]))
  pc_ctx_len = int(pc.get("ctx", w["ctx"]))
  if prefix_tokens < 8 or prefix_tokens >= prompt_tokens or pc_ctx_len < prompt_tokens + gen_tokens:
    raise ValueError("invalid prefix_cache lengths or context capacity")
  if only:
    cfgs = selected(suite, only)
  elif pc.get("only"):
    present_ids = {c["id"] for c in suite.get("configs", [])}
    active_only = [cid for cid in pc["only"] if cid in present_ids]
    cfgs = selected(suite, ",".join(active_only)) if active_only else []
  else:
    cfgs = [c for c in selected(suite, None) if c.get("track") in ("matched", "fastest")]
  suffix_tokens = prompt_tokens - prefix_tokens
  point = dict(suite)
  point["workload"] = dict(
      w,
      prompt_tokens=prompt_tokens,
      prefix_tokens=prefix_tokens,
      gen_tokens=gen_tokens,
      warmup=int(pc.get("warmup", w["warmup"])),
      repetitions=int(pc.get("repetitions", w["repetitions"])),
      cooldown_s=pc.get("cooldown_s", w["cooldown_s"]),
  )
  point["configs"] = [
      dict(c, id=f"{c['id']}-prefix-p{prefix_tokens}-s{suffix_tokens}-d{gen_tokens}",
           track="prefix-cache")
      for c in cfgs
  ]
  pc_ctx = dict(ctx, ctx=pc_ctx_len)
  pc_rounds = int(pc.get("rounds", rounds))
  return perf_commands(point, pc_ctx, machine, pc_rounds, None, dev,
                       results_dir=results_dir, skip_existing=skip_existing)


TASK_ITEMS = {"mmlu": 14042, "gsm8k": 1319}
TASK_COST = {"mmlu": 1.0, "gsm8k": 8.0}  # rough relative seconds per item


def accuracy_commands(suite, ctx, machine, only, dev, tasks, threads=None, shards=1,
                      results_dir: Path | None = None, skip_existing: bool = False):
  """Returns (id, cmd, est_cost). Full-set runs are split into `shards`."""
  if not only and suite.get("accuracy_select_fastest_by_dtype"):
    cfgs = select_fastest_by_dtype_configs(suite, machine, results_dir=results_dir)
  else:
    cfgs = selected(suite, only)
  out = []
  for c in cfgs:
    fw, label = c["framework"], c["id"].split("/", 1)[1]
    acc = c.get("accuracy", {})
    driver = [resolve(a, ctx) for a in c["args"]]
    if threads and fw in ("llama.cpp", "litert-lm") and EXECUTION["mode"] != "native":
      # Thread invariance was validated for these two drivers only.
      driver[driver.index("--threads") + 1] = str(threads)
    for task in tasks:
      limit = acc.get(f"{task}_limit")
      if limit is None:
        continue
      n_items = limit or TASK_ITEMS[task]
      k = shards if not limit else 1
      for shard in range(k):
        shard_str = f"{shard}/{k}"
        if skip_existing and _has_complete_accuracy(
            results_dir, machine, fw, label, task, int(limit or 0), shard_str):
          print(f"[suite] skip existing accuracy {c['id']}:{task}:{shard_str}", flush=True)
          continue
        args = ["aeb.accuracy", "--framework", fw, "--label", label, "--task", task,
                "--limit", str(limit), "--shard", shard_str]
        if skip_existing:
          resume_dir = _find_resumable_accuracy(
              results_dir, machine, fw, label, task, int(limit or 0), shard_str)
          if resume_dir:
            print(f"[suite] resuming partial accuracy {c['id']}:{task}:{shard_str} from {resume_dir}",
                  flush=True)
            args += ["--resume", resume_dir]
        args += ["--"] + driver
        cost = n_items / k * TASK_COST[task] * (2.5 if fw == "llama.cpp" else 1.0)
        out.append((f"{c['id']}:{task}:{shard_str}", run_cmd(machine, fw, args, dev), cost))
  return out


def repro_commands(suite, ctx, machine, only, dev, seed=1234,
                   results_dir: Path | None = None, skip_existing: bool = False):
  """Per config: greedy x3 and seeded x3 processes (two at the configured
  thread count, one at half) plus a seed+1 control. Interleaved across
  configs by process slot. See docs/benchmark_methodology.md#reproducibility."""
  repro_cfg = suite.get("repro") or {}
  if "slots" in repro_cfg:
    slots = [tuple(s) for s in repro_cfg["slots"]]
  else:
    slots = [("greedy", None, 1.0, 0), ("seeded", seed, 1.0, 0), ("greedy", None, 1.0, 1),
             ("seeded", seed, 1.0, 1), ("greedy", None, 0.5, 2), ("seeded", seed, 0.5, 2),
             ("seeded", seed + 1, 1.0, 3)]
  eff_only = only
  if not eff_only and repro_cfg.get("only"):
    eff_only = ",".join(repro_cfg["only"])
  out = []
  for mode, sd, tfrac, proc in slots:
    for c in selected(suite, eff_only):
      fw, label = c["framework"], c["id"].split("/", 1)[1]
      driver = [resolve(a, ctx) for a in c["args"]]
      i = driver.index("--threads") + 1
      driver[i] = str(max(1, int(int(driver[i]) * tfrac)))
      if skip_existing and _has_complete_repro(
          results_dir, machine, fw, label, mode, sd, proc, int(driver[i])):
        print(f"[suite] skip existing repro {c['id']}:{mode}:seed={sd}:p{proc}:t{driver[i]}",
              flush=True)
        continue
      args = ["aeb.repro", "--framework", fw, "--label", label, "--mode", mode,
              "--process", str(proc), "--notes", f"suite={suite['name']}"]
      if repro_cfg.get("repeats") is not None:
        args += ["--repeats", str(repro_cfg["repeats"])]
      if repro_cfg.get("max_tokens") is not None:
        args += ["--max-tokens", str(repro_cfg["max_tokens"])]
      if sd is not None:
        args += ["--seed", str(sd)]
      out.append((f"{c['id']}:{mode}:seed={sd}:p{proc}:t{driver[i]}",
                  run_cmd(machine, fw, args + ["--"] + driver, dev)))
  return out


def execute_parallel(cmds, dry: bool, n_queues: int) -> int:
  """Balance commands over n concurrent sequential queues (longest first)."""
  import threading
  queues = [[] for _ in range(n_queues)]
  load = [0.0] * n_queues
  for cid, cmd, cost in sorted(cmds, key=lambda c: -c[2]):
    i = load.index(min(load))
    queues[i].append((cid, cmd))
    load[i] += cost
  results = []
  threads = [threading.Thread(target=lambda q=q: results.append(execute(q, dry, 0.0)))
             for q in queues if q]
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  return sum(results)


def execute(cmds, dry: bool, cooldown: float) -> int:
  failures = 0
  for i, (cid, cmd) in enumerate(cmds):
    print(f"\n[suite] ({i + 1}/{len(cmds)}) {cid}\n  {shlex.join(cmd)}", flush=True)
    if dry:
      continue
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
      failures += 1
      print(f"[suite] {cid} exited {rc}", flush=True)
    time.sleep(cooldown)
  return failures


def keep_awake() -> None:
  """On macOS, idle sleep pauses the Docker VM mid-run (observed: an hour-long
  stall and a perturbed request). Hold a no-idle-sleep assertion for as long
  as this process lives. Native GPU runs also keep the display awake, so the
  window server's GPU load does not change part-way through a run."""
  if sys.platform == "darwin":
    flags = ["-i", "-s"] + (["-d"] if EXECUTION["mode"] == "native" else [])
    try:
      subprocess.Popen(["caffeinate", *flags, "-w", str(os.getpid())])
    except OSError:
      print("[suite] warning: caffeinate unavailable; disable sleep manually", flush=True)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("action", choices=["perf", "sequence", "prefix-cache", "accuracy", "repro", "plan"])
  ap.add_argument("--machine", required=True)
  ap.add_argument("--suite", type=Path, default=ROOT / "suites" / "gemma4-e2b-cpu.json")
  ap.add_argument("--rounds", type=int)
  ap.add_argument("--only", help="comma-separated config ids")
  ap.add_argument("--tasks", default="mmlu,gsm8k")
  ap.add_argument("--cooldown-s", type=float, default=10.0,
                  help="idle time between containers (thermal recovery)")
  ap.add_argument("--dev", action="store_true",
                  help="bind-mount docker/harness over the baked-in copy (development only)")
  ap.add_argument("--dry-run", action="store_true")
  ap.add_argument("--threads",
                  help="accuracy only: override the driver thread count "
                       "('half' = half of the machine's suite thread count)")
  ap.add_argument("--parallel", type=int, default=0,
                  help="accuracy only: number of concurrent queues (never use for perf)")
  ap.add_argument("--shards", type=int, default=1,
                  help="accuracy only: split full-dataset runs into N shards")
  ap.add_argument("--results", type=Path, default=ROOT / "results",
                  help="results directory for accuracy/perf gating and --skip-existing")
  ap.add_argument("--min-accuracy", type=float,
                  help="sequence only: override minimum accuracy threshold (0.0-1.0)")
  ap.add_argument("--skip-existing", action="store_true",
                  help="skip any measurement that already has complete results on disk")
  args = ap.parse_args(argv)
  suite, ctx = load(args.suite, args.machine)
  dry = args.dry_run or args.action == "plan"
  if not dry:
    if EXECUTION["mode"] == "native" and not (NATIVE / "venv" / "bin" / "python").exists():
      raise SystemExit("native suite: run scripts/native-apple-gpu.sh (make native) first")
    keep_awake()
  rounds = suite.get("rounds", 3) if args.rounds is None else args.rounds
  if args.action in ("perf", "plan"):
    cmds = prime_commands(suite, ctx, args.machine, args.only, args.dev,
                          results_dir=args.results, skip_existing=args.skip_existing)
    cmds += perf_commands(suite, ctx, args.machine, rounds, args.only, args.dev,
                          results_dir=args.results, skip_existing=args.skip_existing)
    if args.action == "plan" and suite.get("prefix_cache"):
      cmds += prefix_cache_commands(suite, ctx, args.machine, rounds, args.only, args.dev,
                                    results_dir=args.results, skip_existing=args.skip_existing)
  elif args.action == "prefix-cache":
    cmds = prefix_cache_commands(suite, ctx, args.machine, rounds, args.only, args.dev,
                                 results_dir=args.results, skip_existing=args.skip_existing)
  elif args.action == "sequence":
    seq_only = args.only
    if not seq_only:
      chosen = select_sequence_configs(suite, args.machine, results_dir=args.results,
                                       min_accuracy=args.min_accuracy,
                                       allow_unmeasured_fallback=dry)
      if not chosen:
        print("[suite] error: no configurations passed the sequence accuracy gate", flush=True)
        return 1
      seq_only = ",".join(c["id"] for c in chosen)
    cmds = prime_commands(suite, ctx, args.machine, seq_only, args.dev,
                          results_dir=args.results, skip_existing=args.skip_existing)
    cmds += sequence_commands(suite, ctx, args.machine, rounds, seq_only, args.dev,
                              results_dir=args.results, min_accuracy=args.min_accuracy,
                              skip_existing=args.skip_existing)
  else:
    cmds = []
  if args.action in ("accuracy", "plan"):
    threads = args.threads
    if threads == "half":
      threads = max(1, int(ctx["threads"]) // 2)
    acc = accuracy_commands(suite, ctx, args.machine, args.only, args.dev,
                            args.tasks.split(","), threads, args.shards,
                            results_dir=args.results, skip_existing=args.skip_existing)
  else:
    acc = []
  if args.action in ("repro", "plan"):
    cmds += repro_commands(suite, ctx, args.machine, args.only, args.dev,
                           results_dir=args.results, skip_existing=args.skip_existing)
  if args.parallel > 1 and EXECUTION["mode"] == "native":
    # One GPU: concurrent processes would share it and change each other's numerics timing.
    print("[suite] native suite: ignoring --parallel; running one process at a time", flush=True)
    args.parallel = 0
  if args.action == "accuracy" and args.parallel:
    return 1 if execute_parallel(acc, dry, args.parallel) else 0
  cmds += [(cid, cmd) for cid, cmd, _ in acc]
  return 1 if execute(cmds, dry, args.cooldown_s) else 0


if __name__ == "__main__":
  sys.exit(main())
