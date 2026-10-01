#!/usr/bin/env python3
"""Run a benchmark suite one container at a time (host side, stdlib only).

  python3 scripts/suite.py perf     --machine apple-m5-max [--suite suites/gemma4-e2b-cpu.json]
  python3 scripts/suite.py accuracy --machine apple-m5-max [--only llama.cpp/matched,...]
  python3 scripts/suite.py repro    --machine apple-m5-max   # output reproducibility
  python3 scripts/suite.py plan     --machine apple-m5-max   # print commands only

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
SERVICE = {"llama.cpp": "harness-llama-cpp", "litert-lm": "harness-litert-lm"}


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


def run_cmd(machine: str, framework: str, module_args: list[str], dev: bool) -> list[str]:
  cmd = compose_base(machine, dev) + ["run", "--rm", "-T", "-e", f"AEB_GIT_REV={git_rev()}"]
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
  return suite, ctx


def selected(suite: dict, only: str | None) -> list[dict]:
  cfgs = suite["configs"]
  if only:
    want = set(only.split(","))
    cfgs = [c for c in cfgs if c["id"] in want]
  return cfgs


def perf_commands(suite, ctx, machine, rounds, only, dev):
  w = suite["workload"]
  cfgs = selected(suite, only)
  out = []
  for r in range(rounds):
    order = cfgs if r % 2 == 0 else list(reversed(cfgs))
    for c in order:
      fw, label = c["framework"], c["id"].split("/", 1)[1]
      args = ["aeb.perf", "--framework", fw, "--label", label, "--track", c["track"],
              "--round", str(r), "--prompt-tokens", str(w["prompt_tokens"]),
              "--gen-tokens", str(w["gen_tokens"]), "--warmup", str(w["warmup"]),
              "--repetitions", str(w["repetitions"]), "--cooldown-s", str(w["cooldown_s"]),
              "--notes", f"suite={suite['name']}", "--"]
      args += [resolve(a, ctx) for a in c["args"]]
      out.append((c["id"], run_cmd(machine, fw, args, dev)))
  return out


def prime_commands(suite, ctx, machine, only, dev):
  """One unmeasured request per config so framework caches (e.g. the XNNPACK
  weight cache) exist before any measured process starts."""
  out = []
  for c in selected(suite, only):
    fw, label = c["framework"], c["id"].split("/", 1)[1]
    args = ["aeb.perf", "--framework", fw, "--label", f"prime-{label}", "--track", "prime",
            "--prompt-tokens", "128", "--gen-tokens", "8", "--warmup", "1", "--repetitions", "0",
            "--cooldown-s", "0", "--notes", f"prime:{suite['name']}", "--"]
    args += [resolve(a, ctx) for a in c["args"]]
    out.append((f"prime {c['id']}", run_cmd(machine, fw, args, dev)))
  return out


TASK_ITEMS = {"mmlu": 14042, "gsm8k": 1319}
TASK_COST = {"mmlu": 1.0, "gsm8k": 8.0}  # rough relative seconds per item


def accuracy_commands(suite, ctx, machine, only, dev, tasks, threads=None, shards=1):
  """Returns (id, cmd, est_cost). Full-set runs are split into `shards`."""
  out = []
  for c in selected(suite, only):
    fw, label = c["framework"], c["id"].split("/", 1)[1]
    acc = c.get("accuracy", {})
    driver = [resolve(a, ctx) for a in c["args"]]
    if threads:
      # Thread count does not change results for either framework's CPU
      # kernels (work is split by output rows), so accuracy runs may use fewer
      # threads and run side by side.
      driver[driver.index("--threads") + 1] = str(threads)
    for task in tasks:
      limit = acc.get(f"{task}_limit")
      if limit is None:
        continue
      n_items = limit or TASK_ITEMS[task]
      k = shards if not limit else 1
      for shard in range(k):
        args = ["aeb.accuracy", "--framework", fw, "--label", label, "--task", task,
                "--limit", str(limit), "--shard", f"{shard}/{k}", "--"] + driver
        cost = n_items / k * TASK_COST[task] * (2.5 if fw == "llama.cpp" else 1.0)
        out.append((f"{c['id']}:{task}:{shard}/{k}", run_cmd(machine, fw, args, dev), cost))
  return out


def repro_commands(suite, ctx, machine, only, dev, seed=1234):
  """Per config: greedy x3 and seeded x3 processes (two at the configured
  thread count, one at half) plus a seed+1 control. Interleaved across
  configs by process slot. See docs/benchmark_methodology.md#reproducibility."""
  slots = [("greedy", None, 1.0, 0), ("seeded", seed, 1.0, 0), ("greedy", None, 1.0, 1),
           ("seeded", seed, 1.0, 1), ("greedy", None, 0.5, 2), ("seeded", seed, 0.5, 2),
           ("seeded", seed + 1, 1.0, 3)]
  out = []
  for mode, sd, tfrac, proc in slots:
    for c in selected(suite, only):
      fw, label = c["framework"], c["id"].split("/", 1)[1]
      driver = [resolve(a, ctx) for a in c["args"]]
      i = driver.index("--threads") + 1
      driver[i] = str(max(1, int(int(driver[i]) * tfrac)))
      args = ["aeb.repro", "--framework", fw, "--label", label, "--mode", mode,
              "--process", str(proc), "--notes", f"suite={suite['name']}"]
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
  as this process lives."""
  if sys.platform == "darwin":
    try:
      subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])
    except OSError:
      print("[suite] warning: caffeinate unavailable; disable sleep manually", flush=True)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("action", choices=["perf", "accuracy", "repro", "plan"])
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
  args = ap.parse_args(argv)
  suite, ctx = load(args.suite, args.machine)
  if not (args.dry_run or args.action == "plan"):
    keep_awake()
  rounds = args.rounds or suite.get("rounds", 3)
  if args.action in ("perf", "plan"):
    cmds = prime_commands(suite, ctx, args.machine, args.only, args.dev)
    cmds += perf_commands(suite, ctx, args.machine, rounds, args.only, args.dev)
  else:
    cmds = []
  if args.action in ("accuracy", "plan"):
    threads = args.threads
    if threads == "half":
      threads = max(1, int(ctx["threads"]) // 2)
    acc = accuracy_commands(suite, ctx, args.machine, args.only, args.dev,
                            args.tasks.split(","), threads, args.shards)
  else:
    acc = []
  if args.action in ("repro", "plan"):
    cmds += repro_commands(suite, ctx, args.machine, args.only, args.dev)
  dry = args.dry_run or args.action == "plan"
  if args.action == "accuracy" and args.parallel:
    return 1 if execute_parallel(acc, dry, args.parallel) else 0
  cmds += [(cid, cmd) for cid, cmd, _ in acc]
  return 1 if execute(cmds, dry, args.cooldown_s) else 0


if __name__ == "__main__":
  sys.exit(main())
