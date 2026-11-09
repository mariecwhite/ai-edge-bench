"""Detect Metal GPU errors (hangs, faults, aborted command buffers) during
native macOS runs, from the unified log.

  python3 -m aeb.gpucheck --results results --machine apple-m5-max [--since 2026-10-22T14:00:00Z]

A GPU hang makes macOS restart the GPU and then ignore the offending process's
command buffers. An engine that does not check command-buffer status keeps
"running" without doing GPU work: it returns stale output with impossible
timings and reports success (observed with LiteRT-LM 0.18.1). Such a run must
be excluded, not averaged.

Benchmark processes never overlap, so an error is attributed to the run whose
[started_at, ended_at] window contains it. Errors from other processes in the
window (e.g. the window server's "innocent victim" command buffers during a
GPU recovery) are recorded as `recoveries`. Writes `gpu_check.json` next to
each run's meta.json; aeb.perf runs the same check at the end of every native
process. The unified log is only retained for a few days.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

PREDICATE = 'eventMessage CONTAINS "command buffer was aborted"'
_KIND = re.compile(r"\((\d+):(kIOGPU\w+)\)")
DRIVER_PROCESSES = ("python", "aeb-llama-driver")


def _parse_utc(s: str) -> dt.datetime:
  return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def _local(t: dt.datetime) -> str:
  return t.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def metal_errors(start: dt.datetime, end: dt.datetime) -> list[dict] | None:
  """Aborted Metal command buffers logged between start and end (UTC-aware).
  None if the log could not be read (not macOS, or `log` unavailable)."""
  if sys.platform != "darwin":
    return None
  try:
    out = subprocess.run(["log", "show", "--style", "ndjson", "--start", _local(start),
                          "--end", _local(end), "--predicate", PREDICATE],
                         capture_output=True, text=True, timeout=600).stdout
  except (OSError, subprocess.SubprocessError):
    return None
  errors = []
  for line in out.splitlines():
    try:
      e = json.loads(line)
    except json.JSONDecodeError:
      continue
    if "eventMessage" not in e:
      continue
    ts = dt.datetime.strptime(e["timestamp"][:26], "%Y-%m-%d %H:%M:%S.%f")
    tz = e["timestamp"][26:]
    offset = dt.timedelta(hours=int(tz[1:3]), minutes=int(tz[3:5])) * (1 if tz[0] == "+" else -1)
    kind = _KIND.search(e["eventMessage"])
    errors.append({"t": (ts - offset).replace(tzinfo=dt.timezone.utc),
                   "process": Path(e.get("processImagePath", "?")).name,
                   "pid": e.get("processID"),
                   "kind": kind.group(2) if kind else "unknown",
                   "message": e["eventMessage"][:300]})
  return errors


def summarize(errors: list[dict], start: dt.datetime, end: dt.datetime,
              pid: int | None = None) -> dict:
  mine, others = [], []
  for e in errors:
    if not start <= e["t"] <= end:
      continue
    ours = (e["pid"] == pid) if pid else e["process"].startswith(DRIVER_PROCESSES)
    (mine if ours else others).append(e)
  first = min(mine, key=lambda e: e["t"]) if mine else None
  root = next((e for e in sorted(mine, key=lambda e: e["t"])
               if "InnocentVictim" not in e["kind"] and "Ignored" not in e["message"]), first)
  return {
      "schema": "aeb.gpucheck/1",
      "checked_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "window_utc": [start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")],
      "errors": len(mine),
      "kinds": dict(collections.Counter(e["kind"] for e in mine)),
      "first_at": first["t"].strftime("%Y-%m-%dT%H:%M:%S.%fZ") if first else None,
      "root_cause": f"{root['kind']}: {root['message']}" if root else None,
      "recoveries_other_processes": len(others),
  }


def error_text(check: dict | None) -> str | None:
  if not check or not check.get("errors"):
    return None
  return (f"GPU error during the run ({check['errors']} aborted Metal command buffers, "
          f"first at {check['first_at']}; {check['root_cause']})")


def _runs(results: Path, machine: str, since: str | None):
  base = results / machine
  for meta_p in sorted(list(base.glob("perf/*/*/*/meta.json")) +
                       list(base.glob("accuracy-*/*/*/*/meta.json")) +
                       list(base.glob("repro/*/*/*/meta.json"))):
    m = json.loads(meta_p.read_text())
    if not m.get("started_at") or (since and m["started_at"] < since):
      continue
    yield meta_p.parent, m


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--results", type=Path, default=Path("results"))
  ap.add_argument("--machine", required=True)
  ap.add_argument("--since", help="only runs started at/after this UTC timestamp")
  args = ap.parse_args(argv)
  runs = list(_runs(args.results, args.machine, args.since))
  if not runs:
    print("no runs found")
    return 0
  starts = [_parse_utc(m["started_at"]) for _, m in runs]
  ends = [_parse_utc(m.get("ended_at") or m["started_at"]) for _, m in runs]
  errors = metal_errors(min(starts) - dt.timedelta(seconds=1), max(ends) + dt.timedelta(seconds=5))
  if errors is None:
    print("could not read the unified log (macOS only)", file=sys.stderr)
    return 1
  bad = 0
  for (d, m), s, e in zip(runs, starts, ends):
    pid = (m.get("driver_ready") or {}).get("info", {}).get("pid") or m.get("driver_pid")
    check = summarize(errors, s - dt.timedelta(seconds=1), e + dt.timedelta(seconds=5), pid)
    (d / "gpu_check.json").write_text(json.dumps(check, indent=1) + "\n")
    if check["errors"]:
      bad += 1
      print(f"GPU ERRORS {d}: {error_text(check)}")
  print(f"checked {len(runs)} runs against {len(errors)} logged Metal errors; {bad} runs affected")
  return 0


if __name__ == "__main__":
  sys.exit(main())
