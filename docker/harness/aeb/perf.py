"""Framework-neutral latency / throughput / CPU / memory benchmark.

  python3 -m aeb.perf --framework llama.cpp --label matched \\
      --prompt-tokens 1024 --gen-tokens 256 --warmup 2 --repetitions 5 \\
      -- --model /models/... --threads 8 ...

Everything after `--` is passed to the framework driver. One invocation is one
driver *process*; statistically independent repetitions across processes (and
interleaving between frameworks) are the caller's job, see Makefile `perf`.

Per request the driver reports CLOCK_MONOTONIC timestamps for request start,
each generated token and completion. The monitor samples the driver process
every --sample-ms. The run directory receives:

  meta.json        configuration, environment, workload identity, load time
  requests.jsonl   one line per request (warm-up and measured, flagged)
  timeseries.csv   resource samples (see aeb/monitor.py for columns)
  summary.json     per-metric median/min/max/CV over measured requests
  driver.log       driver stderr
  sysinfo.json     hardware / OS snapshot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

from . import common, monitor, stats, workload
from .driver import FRAMEWORKS, Driver, driver_command


def request_metrics(r: dict, samples: list[tuple]) -> dict:
  n_gen = r["n_gen"]
  t_req, t_first, t_end = r["t_req_ns"], r["t_first_ns"], r["t_end_ns"]
  tok = r.get("token_ns") or []
  # Use the per-token timestamp of the last token when every token was
  # streamed; otherwise (LiteRT-LM filters some special tokens from the
  # stream) fall back to completion time.
  t_last = tok[-1] if len(tok) == n_gen and tok else t_end
  m = {
      "n_prompt": r.get("n_prompt"),
      "n_gen": n_gen,
      "e2e_ms": (t_end - t_req) / 1e6,
      "ttft_ms": (t_first - t_req) / 1e6,
      "prefill_tps": r["n_prompt"] / ((t_first - t_req) / 1e9) if r.get("n_prompt") else None,
      "decode_tps": (n_gen - 1) / ((t_last - t_first) / 1e9) if n_gen >= 2 and t_last > t_first else None,
      "tpot_ms": ((t_last - t_first) / 1e6) / (n_gen - 1) if n_gen >= 2 else None,
  }
  if len(tok) >= 3:
    gaps = sorted((b - a) / 1e6 for a, b in zip(tok, tok[1:]))
    m["itl_p50_ms"] = stats.quantile(gaps, 0.5)
    m["itl_p99_ms"] = stats.quantile(gaps, 0.99)
  if samples:
    ts = [s[0] for s in samples]
    ticks = [s[1] for s in samples]

    def cores(a, b):
      if b <= a:
        return None
      return ((stats.interp(ts, ticks, b) - stats.interp(ts, ticks, a)) / monitor.CLK_TCK) / ((b - a) / 1e9)

    m["cpu_cores_prefill"] = cores(t_req, t_first)
    m["cpu_cores_decode"] = cores(t_first, t_end)
    m["cpu_cores_request"] = cores(t_req, t_end)
    window = [s for s in samples if t_req <= s[0] <= t_end]
    if window:
      m["rss_max_mb"] = max(s[2] for s in window) / 1024
      m["rss_anon_max_mb"] = max(s[3] for s in window) / 1024
      m["rss_file_max_mb"] = max(s[4] for s in window) / 1024
      pss = [s[8] for s in window if s[8] is not None]
      m["pss_max_mb"] = max(pss) / 1024 if pss else None
      sys_busy = (window[-1][10] - window[0][10])
      sys_total = (window[-1][11] - window[0][11])
      m["system_cpu_busy_frac"] = sys_busy / sys_total if sys_total else None
  if r.get("ticks_before") is not None and r.get("ticks_after") is not None:
    wall = (r["orch_after_ns"] - r["orch_before_ns"]) / 1e9
    m["cpu_seconds"] = (r["ticks_after"] - r["ticks_before"]) / monitor.CLK_TCK
    m["cpu_cores_exact"] = m["cpu_seconds"] / wall if wall > 0 else None
  return m


def cache_state(driver_args: list[str]) -> dict | None:
  """Record whether a framework weight cache already existed (warm vs cold)."""
  if "--cache-dir" not in driver_args:
    return None
  d = driver_args[driver_args.index("--cache-dir") + 1]
  try:
    files = {f: os.path.getsize(os.path.join(d, f)) for f in sorted(os.listdir(d))}
  except OSError:
    files = {}
  return {"dir": d, "files": files, "warm": bool(files)}


METRICS = ("e2e_ms", "ttft_ms", "prefill_tps", "decode_tps", "tpot_ms", "itl_p50_ms",
           "itl_p99_ms", "cpu_cores_prefill", "cpu_cores_decode", "cpu_cores_request",
           "cpu_seconds", "rss_max_mb", "rss_anon_max_mb", "rss_file_max_mb", "pss_max_mb",
           "system_cpu_busy_frac")


def main(argv=None) -> int:
  argv = list(sys.argv[1:] if argv is None else argv)
  driver_args: list[str] = []
  if "--" in argv:
    i = argv.index("--")
    argv, driver_args = argv[:i], argv[i + 1:]
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--framework", required=True, choices=FRAMEWORKS)
  ap.add_argument("--label", required=True, help="configuration label, e.g. matched")
  ap.add_argument("--track", default="", help="report track: matched | fastest | sweep | ...")
  ap.add_argument("--prompt-tokens", type=int, default=1024)
  ap.add_argument("--gen-tokens", type=int, default=256)
  ap.add_argument("--warmup", type=int, default=2)
  ap.add_argument("--repetitions", type=int, default=5)
  ap.add_argument("--cooldown-s", type=float, default=2.0)
  ap.add_argument("--sample-ms", type=float, default=50.0)
  ap.add_argument("--round", type=int, default=0, help="interleaving round (for ordering analysis)")
  ap.add_argument("--notes", default="")
  args = ap.parse_args(argv)

  if args.framework == "litert-lm" and "--fixed-decode-tokens" not in driver_args:
    driver_args += ["--fixed-decode-tokens", str(args.gen_tokens)]
  ctx_needed = args.prompt_tokens + args.gen_tokens
  run_dir = common.new_run_dir("perf", args.framework, args.label)
  cmd = driver_command(args.framework, driver_args)
  model = common.model_arg(driver_args)
  meta = {
      "schema": "aeb.perf/1",
      "framework": args.framework,
      "label": args.label,
      "track": args.track,
      "round": args.round,
      "notes": args.notes,
      "machine": os.environ.get("AEB_MACHINE", "unknown-machine"),
      "machine_notes": os.environ.get("AEB_MACHINE_NOTES", ""),
      "started_at": common.utc_now(),
      "driver_cmd": cmd,
      "model_path": model,
      "model_sha256": common.model_sha256(model) if model else None,
      "build": common.build_info(),
      "params": {k: getattr(args, k) for k in ("prompt_tokens", "gen_tokens", "warmup",
                                                "repetitions", "cooldown_s", "sample_ms")},
  }
  meta["cache_state_before"] = cache_state(driver_args)
  common.write_sysinfo(run_dir / "sysinfo.json")
  print(f"[aeb] perf {args.framework}/{args.label} -> {run_dir}", file=sys.stderr, flush=True)

  t_spawn = time.monotonic_ns()
  drv = Driver(cmd, run_dir / "driver.log")
  mon = monitor.Monitor(drv.pid, interval_s=args.sample_ms / 1000).start()
  status = "ok"
  requests = []
  idle_after_load = None
  try:
    ready = drv.wait_ready()
    t_ready = time.monotonic_ns()
    meta["driver_ready"] = ready
    meta["load_ms_driver"] = ready.get("load_ns", 0) / 1e6
    meta["load_ms_spawn_to_ready"] = (t_ready - t_spawn) / 1e6
    idle_after_load = mon.sample(with_pss=True)

    tok = lambda text: drv.request("tokenize", text=text, add_bos=True)["ids"]
    prompt, ids = workload.build_perf_prompt(tok, args.prompt_tokens)
    meta["workload"] = {
        "kind": "chat-template real-text prompt (aeb.workload.build_perf_prompt)",
        "prompt_tokens": len(ids),
        "gen_tokens": args.gen_tokens,
        "prompt_ids_sha256": workload.ids_sha256(ids),
        "prompt_text_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "decode": "greedy, fixed length (stop tokens ignored)",
        "context_tokens_needed": ctx_needed,
    }
    time.sleep(args.cooldown_s)

    total = args.warmup + args.repetitions
    for i in range(total):
      measured = i >= args.warmup
      before_ticks = monitor.read_cpu_ticks(drv.pid)
      before = time.monotonic_ns()
      r = drv.request("generate", prompt=prompt, max_tokens=args.gen_tokens,
                      ignore_eos=True, return_text=True, stop_ids=workload.STOP_IDS)
      after = time.monotonic_ns()
      after_ticks = monitor.read_cpu_ticks(drv.pid)
      if not r.get("ok"):
        r = {"ok": False, "error": r.get("error"), "measured": measured, "index": i}
        requests.append(r)
        status = "error"
        print(f"[aeb] request {i} failed: {r['error']}", file=sys.stderr)
        break
      r.update(index=i, measured=measured, ticks_before=before_ticks, ticks_after=after_ticks,
               orch_before_ns=before, orch_after_ns=after)
      if r.get("n_gen") != args.gen_tokens:
        r["warning"] = f"generated {r.get('n_gen')} tokens, expected {args.gen_tokens}"
        status = "warning"
      requests.append(r)
      print(f"[aeb] {'run' if measured else 'warmup'} {i}: e2e {(r['t_end_ns'] - r['t_req_ns']) / 1e6:.0f} ms, "
            f"ttft {(r['t_first_ns'] - r['t_req_ns']) / 1e6:.0f} ms, n_gen {r['n_gen']}",
            file=sys.stderr, flush=True)
      time.sleep(args.cooldown_s)
  except Exception as e:  # keep partial data; never discard silently
    status = "error"
    meta["error"] = str(e)
    print(f"[aeb] error: {e}", file=sys.stderr)
  finally:
    samples = mon.stop()
    end_sample = monitor.Monitor(drv.pid).sample(with_pss=True)
    meta["exit_code"] = drv.close()
    meta["cgroup_memory_peak_bytes"] = monitor.read_cgroup_peak()

  mon.write_csv(run_dir / "timeseries.csv")
  with open(run_dir / "requests.jsonl", "w") as f:
    for r in requests:
      f.write(json.dumps(r) + "\n")

  per_req = []
  for r in requests:
    if r.get("ok") is False:
      continue
    m = request_metrics(r, samples)
    m.update(index=r["index"], measured=r["measured"])
    per_req.append(m)
  measured = [m for m in per_req if m["measured"]]
  summary = {
      "schema": "aeb.perf.summary/1",
      "framework": args.framework, "label": args.label, "track": args.track,
      "status": status,
      "n_measured": len(measured),
      "metrics": {k: stats.describe([m.get(k) for m in measured]) for k in METRICS},
      "load_ms": meta.get("load_ms_driver"),
      "peak_rss_mb_lifetime": (end_sample[6] / 1024) if end_sample else
                              (max(s[6] for s in samples) / 1024 if samples else None),
      "idle_after_load": {"rss_mb": idle_after_load[2] / 1024, "rss_anon_mb": idle_after_load[3] / 1024,
                          "rss_file_mb": idle_after_load[4] / 1024,
                          "pss_mb": (idle_after_load[8] or 0) / 1024} if idle_after_load else None,
      "cgroup_memory_peak_mb": (meta["cgroup_memory_peak_bytes"] or 0) / 2**20,
      "outliers_e2e": stats.mad_outliers([m["e2e_ms"] for m in measured]),
      "per_request": per_req,
  }
  meta["ended_at"] = common.utc_now()
  meta["status"] = status
  common.dump(run_dir / "meta.json", meta)
  common.dump(run_dir / "summary.json", summary)
  d = summary["metrics"]
  def med(k):
    return d[k]["median"] if d.get(k) else float("nan")
  print(f"[aeb] {args.framework}/{args.label}: ttft {med('ttft_ms'):.0f} ms, "
        f"prefill {med('prefill_tps'):.1f} tok/s, decode {med('decode_tps'):.2f} tok/s, "
        f"cores(decode) {med('cpu_cores_decode'):.2f}, rss max {med('rss_max_mb'):.0f} MB "
        f"[{status}] -> {run_dir}", file=sys.stderr)
  return 0 if status != "error" else 1


if __name__ == "__main__":
  sys.exit(main())
