"""Run-directory and metadata helpers shared by the runners."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import secrets
import subprocess
from pathlib import Path

RESULTS = Path(os.environ.get("AEB_RESULTS_DIR", "/results"))


def utc_now() -> str:
  return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id() -> str:
  return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)


def new_run_dir(kind: str, framework: str, label: str) -> Path:
  machine = os.environ.get("AEB_MACHINE", "unknown-machine")
  d = RESULTS / machine / kind / framework / label / run_id()
  d.mkdir(parents=True, exist_ok=False)
  return d


def _file_sha256(path: Path) -> str:
  side = Path(str(path) + ".sha256")
  if side.is_file():
    return side.read_text().split()[0]
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
      h.update(chunk)
  digest = h.hexdigest()
  try:
    side.write_text(digest + "\n")
  except OSError:
    pass
  return digest


def model_sha256(path: str) -> str:
  if Path(path).is_dir():
    h = hashlib.sha256()
    files = sorted(f for f in Path(path).rglob("*")
                   if f.is_file() and ".cache" not in f.relative_to(path).parts
                   and not f.name.endswith((".sha256", ".manifest.json")))
    if not files:
      raise ValueError(f"model directory contains no files: {path}")
    for f in files:
      h.update(str(f.relative_to(path)).encode())
      h.update(b"\0")
      h.update(_file_sha256(f).encode())
      h.update(b"\n")
    return h.hexdigest()
  return _file_sha256(Path(path))


def build_info() -> dict:
  p = Path("/opt/aeb/build-info.json")
  info = json.loads(p.read_text()) if p.is_file() else {}
  info["harness_sha256"] = harness_digest()
  info["repo_rev"] = os.environ.get("AEB_GIT_REV", "unknown")
  return info


def harness_digest() -> str:
  """Content hash of the harness package, so results identify the exact code."""
  root = Path(__file__).resolve().parent
  h = hashlib.sha256()
  for f in sorted(root.rglob("*")):
    if f.is_file() and f.suffix in (".py", ".txt", ".json"):
      h.update(str(f.relative_to(root)).encode())
      h.update(f.read_bytes())
  return h.hexdigest()


def write_sysinfo(path: Path) -> None:
  try:
    subprocess.run(["aeb-sysinfo", str(path)], check=True, timeout=60)
  except (OSError, subprocess.SubprocessError) as e:
    path.write_text(json.dumps({"error": str(e)}))


def model_arg(driver_args: list[str]) -> str | None:
  for i, a in enumerate(driver_args):
    if a == "--model" and i + 1 < len(driver_args):
      return driver_args[i + 1]
  return None


def dump(path: Path, obj) -> None:
  path.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n")
