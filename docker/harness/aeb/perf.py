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
import datetime as dt
import hashlib
import json
import os
import sys
import time

from . import common, gpucheck, monitor, stats, workload
from .driver import FRAMEWORKS, Driver, driver_command


def request_metrics(r: dict, samples: list[tuple]) -> dict:
  n_gen = r["n_gen"]
  t_req, t_first, t_end = r["t_req_ns"], r["t_first_ns"], r["t_end_ns"]
  tok = r.get("token_ns") or []
  # Use the per-token timestamp of the last token when every token was
  # streamed; otherwise (LiteRT-LM filters some special tokens from the
  # stream) fall back to completion time.
  t_last = tok[-1] if len(tok) == n_gen and tok else t_end
  n_prompt = r.get("n_prompt")
  n_cached = r.get("n_cached", 0) or 0
  n_prefilled = r.get("n_prefilled") if r.get("n_prefilled") is not None else (
      (n_prompt - n_cached) if n_prompt is not None else None
  )
  ttft_s = (t_first - t_req) / 1e9
  m = {
      "n_prompt": n_prompt,
      "n_cached": n_cached,
      "n_prefilled": n_prefilled,
      "n_gen": n_gen,
      "e2e_ms": (t_end - t_req) / 1e6,
      "ttft_ms": (t_first - t_req) / 1e6,
      "prefill_tps": n_prompt / ttft_s if n_prompt and ttft_s > 0 else None,
      "suffix_prefill_tps": (
          n_prefilled / ttft_s if n_prefilled and ttft_s > 0 else None
      ),
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
    # System-wide GPU utilization (IOAccelerator "Device Utilization %"), mean
    # over each phase. Coarse: it tracks sustained load but overstates idle
    # gaps shorter than ~1 s (docs/harness.md, "Native Apple GPU suite").
    util = [(s[0], s[15]) for s in samples if len(s) > 15 and s[15] is not None]

    def util_mean(a, b):
      v = [u for t, u in util if a <= t <= b]
      return sum(v) / len(v) / 100 if v else None

    if util:
      m["gpu_util_prefill"] = util_mean(t_req, t_first)
      m["gpu_util_decode"] = util_mean(t_first, t_end)
    window = [s for s in samples if t_req <= s[0] <= t_end]
    if window:
      m["rss_max_mb"] = max(s[2] for s in window) / 1024
      for key, col in (("rss_anon_max_mb", 3), ("rss_file_max_mb", 4), ("pss_max_mb", 8),
                       ("footprint_max_mb", 12), ("gpu_mem_max_mb", 16)):
        vals = [s[col] for s in window if len(s) > col and s[col] is not None]
        m[key] = max(vals) / 1024 if vals else None
      if not monitor.DARWIN:
        m["rss_file_max_mb"] = max((s[4] or 0) + (s[5] or 0) for s in window) / 1024
      # rusage ri_energy_nj: the kernel's CPU energy estimate; it does not
      # include GPU work (verified: ~0 W for a GPU-saturating process).
      energy = [s[17] for s in window if len(s) > 17 and s[17] is not None]
      if len(energy) >= 2 and energy[-1] >= energy[0]:
        m["cpu_energy_j"] = (energy[-1] - energy[0]) / 1e9
      sys_busy = (window[-1][10] - window[0][10])
      sys_total = (window[-1][11] - window[0][11])
      m["system_cpu_busy_frac"] = sys_busy / sys_total if sys_total else None
  if r.get("ticks_before") is not None and r.get("ticks_after") is not None:
    wall = (r["orch_after_ns"] - r["orch_before_ns"]) / 1e9
    m["cpu_seconds"] = (r["ticks_after"] - r["ticks_before"]) / monitor.CLK_TCK
    m["cpu_cores_exact"] = m["cpu_seconds"] / wall if wall > 0 else None
  return m


def _mb(kb):
  return kb / 1024 if kb is not None else None


def _peak(end_sample, samples, col):
  """Lifetime high-water mark column (VmHWM on Linux, max footprint on macOS)."""
  if end_sample and end_sample[col] is not None:
    return end_sample[col] / 1024
  vals = [s[col] for s in samples if s[col] is not None]
  return max(vals) / 1024 if vals else None


def _idle_gpu_util(samples, requests, t_ready) -> float | None:
  """Mean system GPU utilization after load, away from any request (cool-down
  gaps): the background load other GPU clients (e.g. the window server) add."""
  if t_ready is None:
    return None
  spans = [(r["t_req_ns"] - 300_000_000, r["t_end_ns"] + 1_000_000_000)
           for r in requests if r.get("ok") is not False and "t_req_ns" in r]
  vals = [s[15] for s in samples if len(s) > 15 and s[15] is not None and s[0] > t_ready
          and not any(a <= s[0] <= b for a, b in spans)]
  return sum(vals) / len(vals) / 100 if vals else None


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


METRICS = ("e2e_ms", "ttft_ms", "prefill_tps", "suffix_prefill_tps", "decode_tps",
           "tpot_ms", "itl_p50_ms", "itl_p99_ms", "cpu_cores_prefill",
           "cpu_cores_decode", "cpu_cores_request", "cpu_seconds", "rss_max_mb",
           "rss_anon_max_mb", "rss_file_max_mb", "pss_max_mb",
           "system_cpu_busy_frac", "gpu_util_prefill", "gpu_util_decode", "footprint_max_mb",
           "gpu_mem_max_mb", "cpu_energy_j")


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
  ap.add_argument("--prefix-tokens", type=int, default=0,
                  help="cached shared prefix tokens (0 = stateless cold KV cache per request)")
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
  if args.prefix_tokens > 0 and "--cache-prefix" not in driver_args:
    driver_args += ["--cache-prefix"]
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
      "params": {k: getattr(args, k) for k in ("prompt_tokens", "prefix_tokens", "gen_tokens",
                                                "warmup", "repetitions", "cooldown_s", "sample_ms")},
      "platform": sys.platform,
      "clk_tck": monitor.CLK_TCK,
  }
  meta["cache_state_before"] = cache_state(driver_args)
  common.write_sysinfo(run_dir / "sysinfo.json")
  print(f"[aeb] perf {args.framework}/{args.label} -> {run_dir}", file=sys.stderr, flush=True)

  t_spawn = time.monotonic_ns()
  wall_start = dt.datetime.now(dt.timezone.utc)
  drv = Driver(cmd, run_dir / "driver.log")
  meta["driver_pid"] = drv.pid
  mon = monitor.Monitor(drv.pid, interval_s=args.sample_ms / 1000).start()
  status = "ok"
  requests = []
  idle_after_load = None
  t_ready = None
  try:
    ready = drv.wait_ready()
    t_ready = time.monotonic_ns()
    meta["driver_ready"] = ready
    meta["load_ms_driver"] = ready.get("load_ns", 0) / 1e6
    meta["load_ms_spawn_to_ready"] = (t_ready - t_spawn) / 1e6
    idle_after_load = mon.sample(with_pss=True)

    tok = lambda text: drv.request("tokenize", text=text, add_bos=True)["ids"]
    total = args.warmup + args.repetitions
    if args.prefix_tokens > 0:
      prompt_variants = [
          workload.build_prefix_cache_prompt(
              tok, args.prompt_tokens, args.prefix_tokens, variant=i
          )
          for i in range(total)
      ]
      prompt0, _, _, ids0, prefix_ids0 = prompt_variants[0]
      meta["workload"] = {
          "kind": "chat-template real-text prompt with shared prefix (aeb.workload.build_prefix_cache_prompt)",
          "prompt_tokens": len(ids0),
          "prefix_tokens": len(prefix_ids0),
          "suffix_tokens": len(ids0) - len(prefix_ids0),
          "gen_tokens": args.gen_tokens,
          "prompt_ids_sha256": workload.ids_sha256(ids0),
          "prefix_ids_sha256": workload.ids_sha256(prefix_ids0),
          "prompt_text_sha256": hashlib.sha256(prompt0.encode()).hexdigest(),
          "decode": "greedy, fixed length (stop tokens ignored)",
          "context_tokens_needed": ctx_needed,
      }
    else:
      prompt, ids = workload.build_perf_prompt(tok, args.prompt_tokens)
      meta["workload"] = {
          "kind": "chat-template real-text prompt (aeb.workload.build_perf_prompt)",
          "prompt_tokens": len(ids),
          "prefix_tokens": 0,
          "suffix_tokens": len(ids),
          "gen_tokens": args.gen_tokens,
          "prompt_ids_sha256": workload.ids_sha256(ids),
          "prompt_text_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
          "decode": "greedy, fixed length (stop tokens ignored)",
          "context_tokens_needed": ctx_needed,
      }
    meta["workload"]["chat_template"] = workload.CHAT_TEMPLATE
    time.sleep(args.cooldown_s)

    for i in range(total):
      measured = i >= args.warmup
      before_ticks = monitor.read_cpu_ticks(drv.pid)
      before = time.monotonic_ns()
      if args.prefix_tokens > 0:
        req_prompt, req_prefix, req_suffix, _, _ = prompt_variants[i]
        r = drv.request(
            "generate",
            prompt=req_prompt,
            prefix_text=req_prefix,
            suffix_text=req_suffix,
            prefix_tokens=args.prefix_tokens,
            cache_prefix=True,
            max_tokens=args.gen_tokens,
            ignore_eos=True,
            return_text=True,
            stop_ids=workload.STOP_IDS,
        )
      else:
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
      if not before <= r["t_req_ns"] <= r["t_end_ns"] <= after:
        # Driver and monitor timestamps must share one clock, or per-phase
        # CPU/GPU/memory attribution is wrong (latencies stay valid).
        r["warning"] = "driver timestamps outside the orchestrator's request window (clock domain)"
        meta["clock_domain_mismatch"] = True
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

  if monitor.DARWIN:
    # An engine that ignores Metal command-buffer errors keeps "running" after
    # a GPU hang without doing GPU work; such a run is invalid (aeb/gpucheck.py).
    time.sleep(2)  # let the unified log persist the last entries
    errors = gpucheck.metal_errors(wall_start - dt.timedelta(seconds=1),
                                   dt.datetime.now(dt.timezone.utc))
    if errors is not None:
      check = gpucheck.summarize(errors, wall_start - dt.timedelta(seconds=1),
                                 dt.datetime.now(dt.timezone.utc), drv.pid)
      common.dump(run_dir / "gpu_check.json", check)
      if check["errors"]:
        status = "error"
        meta["error"] = gpucheck.error_text(check)
        print(f"[aeb] {meta['error']}", file=sys.stderr)

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
      "peak_rss_mb_lifetime": _peak(end_sample, samples, 6),
      "peak_footprint_mb_lifetime": _peak(end_sample, samples, 13),
      "idle_after_load": {"rss_mb": idle_after_load[2] / 1024,
                          "rss_anon_mb": _mb(idle_after_load[3]),
                          "rss_file_mb": (_mb(idle_after_load[4]) if monitor.DARWIN else
                                          ((idle_after_load[4] or 0) + (idle_after_load[5] or 0)) / 1024),
                          "pss_mb": (idle_after_load[8] or 0) / 1024,
                          "footprint_mb": _mb(idle_after_load[12])} if idle_after_load else None,
      "cgroup_memory_peak_mb": (meta["cgroup_memory_peak_bytes"] / 2**20) if meta["cgroup_memory_peak_bytes"] else None,
      "gpu_util_idle": _idle_gpu_util(samples, requests, t_ready),
      "outliers_e2e": stats.mad_outliers([m["e2e_ms"] for m in measured]),
      "per_request": per_req,
  }
  meta["ended_at"] = common.utc_now()
  meta["status"] = status
  common.dump(run_dir / "meta.json", meta)
  common.dump(run_dir / "summary.json", summary)
  d = summary["metrics"]
  fallback = {k: stats.describe([m.get(k) for m in per_req]) for k in METRICS} if not measured else d
  def med(k):
    src = d if d.get(k) else fallback
    return src[k]["median"] if src.get(k) else float("nan")
  print(f"[aeb] {args.framework}/{args.label}: ttft {med('ttft_ms'):.0f} ms, "
        f"prefill {med('prefill_tps'):.1f} tok/s, decode {med('decode_tps'):.2f} tok/s, "
        f"cores(decode) {med('cpu_cores_decode'):.2f}, rss max {med('rss_max_mb'):.0f} MB "
        f"[{status}] -> {run_dir}", file=sys.stderr)
  return 0 if status != "error" else 1


if __name__ == "__main__":
  sys.exit(main())
