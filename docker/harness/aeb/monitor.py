"""Low-overhead /proc and cgroup sampler for a single process.

Samples, at a fixed interval, the driver process's
  * CPU time (utime + stime from /proc/<pid>/stat),
  * RSS split into anonymous / file-backed / shmem (/proc/<pid>/status),
  * kernel high-water mark VmHWM,
  * thread count,
plus the container cgroup's memory.current (anon + page cache charged to the
container) and whole-machine busy time from /proc/stat (to detect background
load). PSS from /proc/<pid>/smaps_rollup is sampled at a lower rate because it
walks page tables.

Every sample carries a CLOCK_MONOTONIC timestamp (time.monotonic_ns), the same
clock the drivers use for request/token timestamps, so samples can be split
into load / prefill / decode / idle phases afterwards.
"""

from __future__ import annotations

import os
import threading
import time

CLK_TCK = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")

FIELDS = ("t_ns", "cpu_ticks", "rss_kb", "rss_anon_kb", "rss_file_kb", "rss_shmem_kb",
          "hwm_kb", "threads", "pss_kb", "cg_mem_bytes", "sys_busy_ticks", "sys_total_ticks")


def read_cpu_ticks(pid: int) -> int | None:
  try:
    with open(f"/proc/{pid}/stat", "rb") as f:
      data = f.read().decode()
  except OSError:
    return None
  # comm may contain spaces; fields after the closing paren are positional.
  rest = data[data.rindex(")") + 2:].split()
  return int(rest[11]) + int(rest[12])  # utime, stime (fields 14, 15)


def read_status(pid: int) -> dict | None:
  out = {}
  try:
    with open(f"/proc/{pid}/status") as f:
      for line in f:
        k, _, v = line.partition(":")
        if k in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmHWM"):
          out[k] = int(v.split()[0])
        elif k == "Threads":
          out[k] = int(v)
  except OSError:
    return None
  return out


def read_pss_kb(pid: int) -> int | None:
  try:
    with open(f"/proc/{pid}/smaps_rollup") as f:
      for line in f:
        if line.startswith("Pss:"):
          return int(line.split()[1])
  except OSError:
    return None
  return None


def read_cgroup_mem() -> int | None:
  try:
    with open("/sys/fs/cgroup/memory.current") as f:
      return int(f.read())
  except (OSError, ValueError):
    return None


def read_cgroup_peak() -> int | None:
  try:
    with open("/sys/fs/cgroup/memory.peak") as f:
      return int(f.read())
  except (OSError, ValueError):
    return None


def read_sys_ticks() -> tuple[int, int]:
  with open("/proc/stat") as f:
    parts = f.readline().split()[1:]
  vals = [int(x) for x in parts]
  idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
  total = sum(vals[:8])
  return total - idle, total


class Monitor:
  """Background sampler. start() before the driver loads; stop() at the end."""

  def __init__(self, pid: int, interval_s: float = 0.05, pss_every: int = 20):
    self.pid = pid
    self.interval = interval_s
    self.pss_every = pss_every
    self.samples: list[tuple] = []
    self._stop = threading.Event()
    self._thread = threading.Thread(target=self._run, name="aeb-monitor", daemon=True)

  def sample(self, with_pss: bool = False) -> tuple | None:
    t = time.monotonic_ns()
    ticks = read_cpu_ticks(self.pid)
    st = read_status(self.pid)
    if ticks is None or st is None:
      return None
    busy, total = read_sys_ticks()
    pss = read_pss_kb(self.pid) if with_pss else None
    return (t, ticks, st.get("VmRSS", 0), st.get("RssAnon", 0), st.get("RssFile", 0),
            st.get("RssShmem", 0), st.get("VmHWM", 0), st.get("Threads", 0), pss,
            read_cgroup_mem(), busy, total)

  def _run(self) -> None:
    i = 0
    next_t = time.monotonic()
    while not self._stop.is_set():
      s = self.sample(with_pss=(i % self.pss_every == 0))
      if s is None:
        break
      self.samples.append(s)
      i += 1
      next_t += self.interval
      delay = next_t - time.monotonic()
      if delay > 0:
        self._stop.wait(delay)
      else:
        next_t = time.monotonic()

  def start(self) -> "Monitor":
    self._thread.start()
    return self

  def stop(self) -> list[tuple]:
    self._stop.set()
    self._thread.join()
    return self.samples

  def write_csv(self, path) -> None:
    with open(path, "w") as f:
      f.write(",".join(FIELDS) + "\n")
      for s in self.samples:
        f.write(",".join("" if v is None else str(v) for v in s) + "\n")
