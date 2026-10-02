"""Driver process management for the framework-neutral harness.

Every framework is driven through a long-lived child process that speaks the
JSON-lines protocol documented in harness/README.md. Keeping the framework in
its own process lets the monitor attribute CPU time and memory to the
framework alone (the orchestrator's own Python overhead is excluded).
"""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

FRAMEWORKS = ("llama.cpp", "litert-lm", "onnxruntime")


def driver_command(framework: str, driver_args: list[str]) -> list[str]:
  if framework == "llama.cpp":
    return ["aeb-llama-driver", *driver_args]
  if framework == "litert-lm":
    return ["python3", "-m", "aeb.drivers.litert_lm_driver", *driver_args]
  if framework == "onnxruntime":
    return ["python3", "-m", "aeb.drivers.onnxruntime_driver", *driver_args]
  raise ValueError(f"unknown framework {framework!r}; expected one of {FRAMEWORKS}")


class DriverError(RuntimeError):
  pass


class Driver:
  """A running driver. Use as a context manager."""

  def __init__(self, cmd: list[str], stderr_path: Path):
    self.cmd = cmd
    self._stderr = open(stderr_path, "ab")
    self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=self._stderr, bufsize=1, text=True)
    self.pid = self.proc.pid
    self.ready: dict | None = None
    self._lock = threading.Lock()
    self._next_id = 0

  def wait_ready(self) -> dict:
    msg = self._read()
    if msg.get("event") != "ready":
      raise DriverError(f"driver failed to start: {msg}")
    self.ready = msg
    return msg

  def _read(self) -> dict:
    while True:
      line = self.proc.stdout.readline()
      if not line:
        code = self.proc.wait()
        raise DriverError(f"driver exited with code {code}; see stderr log")
      line = line.strip()
      if not line:
        continue
      try:
        return json.loads(line)
      except json.JSONDecodeError:
        # Some native libraries print to stdout; never let that corrupt a run.
        self._stderr.write(("[stdout] " + line + "\n").encode())
        continue

  def request(self, op: str, **kwargs) -> dict:
    with self._lock:
      self._next_id += 1
      rid = self._next_id
      self.proc.stdin.write(json.dumps({"op": op, "id": rid, **kwargs}) + "\n")
      self.proc.stdin.flush()
      while True:
        msg = self._read()
        if msg.get("id") == rid:
          return msg

  def close(self) -> int:
    if self.proc.poll() is None:
      try:
        self.request("quit")
      except (DriverError, BrokenPipeError, OSError):
        pass
      try:
        self.proc.wait(timeout=60)
      except subprocess.TimeoutExpired:
        self.proc.kill()
        self.proc.wait()
    self._stderr.close()
    return self.proc.returncode

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    self.close()
