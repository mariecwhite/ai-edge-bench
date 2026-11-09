#!/usr/bin/env python3
"""Docker / Docker Compose CLI shim for bare-metal hosts without dockerd.

Translates `docker compose ... run ... <service> <args...>` and
`docker run ... <image> <args...>` into isolated rootfs container executions:
- Uses Linux `unshare --mount` + `chroot` into `/data/aeb/bundles/<bundle>/rootfs`
  with private bind mounts for `/proc`, `/sys`, `/dev`, `/models`, `/results`,
  `/datasets`, `/reports`, `/opt/aeb/harness`, and `/tmp`.
- Automatically pins execution to the physical cores of NUMA node 0 (or node 1
  for odd accuracy shards) via `numactl` (or `taskset -c` if `numactl` is absent).
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

AEB_ROOT = Path(os.environ.get("AEB_DATA_ROOT", "/data/aeb"))
BUNDLES_DIR = AEB_ROOT / "bundles"

SERVICE_MAP = {
    "base": ("base", [], ["aeb-sysinfo"], {}),
    "fetch-models": ("base", [], ["aeb-fetch-model"], {}),
    "tools": ("tools", ["python3", "-m"], ["aeb.report", "--help"], {}),
    "llama-cpp": ("llama-cpp", [], ["aeb-bench-llama-cpp"], {"AEB_VARIANT": "cpu"}),
    "litert-lm-xnnpack": (
        "litert-lm",
        [],
        ["aeb-bench-litert-lm"],
        {"AEB_YNNPACK": "false", "AEB_BACKEND": "cpu", "AEB_CTX": "4096"},
    ),
    "litert-lm-ynnpack": (
        "litert-lm",
        [],
        ["aeb-bench-litert-lm"],
        {"AEB_YNNPACK": "true", "AEB_BACKEND": "cpu", "AEB_CTX": "4096"},
    ),
    "onnxruntime": ("onnxruntime", [], ["python3", "-m", "aeb.perf", "--help"], {}),
    "xla": ("xla", [], ["python3", "-m", "aeb.perf", "--help"], {}),
    "openvino": ("openvino", [], ["python3", "-m", "aeb.perf", "--help"], {}),
    "harness-llama-cpp": ("llama-cpp", ["python3", "-m"], ["aeb.perf", "--help"], {}),
    "harness-litert-lm": ("litert-lm", ["python3", "-m"], ["aeb.perf", "--help"], {}),
    "harness-onnxruntime": ("onnxruntime", ["python3", "-m"], ["aeb.perf", "--help"], {}),
    "harness-xla": ("xla", ["python3", "-m"], ["aeb.perf", "--help"], {}),
    "harness-openvino": ("openvino", ["python3", "-m"], ["aeb.perf", "--help"], {}),
}

IMAGE_MAP = {
    "aeb/base:latest": "base",
    "aeb/tools:latest": "tools",
    "aeb/llama-cpp:cpu": "llama-cpp",
    "aeb/litert-lm:cpu": "litert-lm",
    "aeb/onnxruntime:cpu": "onnxruntime",
    "aeb/xla:cpu": "xla",
    "aeb/openvino:cpu": "openvino",
}

DEFAULT_ENV = {
    "PATH": (
        "/opt/aeb/venv/bin:/opt/aeb/bin:/opt/llama.cpp/bin:/opt/litert-lm/bin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    ),
    "PYTHONPATH": "/opt/aeb/harness:/opt/litert-lm/python",
    "LD_LIBRARY_PATH": "/opt/llama.cpp/bin:/opt/litert-lm/bin",
    "AEB_MODELS_DIR": "/models",
    "AEB_RESULTS_DIR": "/results",
    "TZ": "UTC",
    "HOME": "/root",
    "LANG": "C.UTF-8",
    "AEB_MACHINE": os.environ.get("AEB_MACHINE", "sapphire-rapids"),
    "AEB_MACHINE_NOTES": os.environ.get(
        "AEB_MACHINE_NOTES",
        "x86_64 server pinned to Socket 0 physical cores",
    ),
    "AEB_PROMPT_TOKENS": os.environ.get("AEB_PROMPT_TOKENS", "1024"),
    "AEB_GEN_TOKENS": os.environ.get("AEB_GEN_TOKENS", "256"),
    "AEB_REPETITIONS": os.environ.get("AEB_REPETITIONS", "5"),
    "AEB_THREADS": os.environ.get("AEB_THREADS", "16"),
}


def _socket_physical_cpulist(socket_idx: int) -> str:
  """Return comma/range list of physical core IDs on a CPU socket (package) from sysfs.

  Grouping by `physical_package_id` (rather than `node0` alone) ensures that
  machines with Sub-NUMA Clustering (SNC-2 / SNC-3) still expose all physical
  cores of the socket instead of only 1/2 or 1/3 of the socket.
  """
  cpu_root = Path("/sys/devices/system/cpu")
  by_pkg: dict[int, list[int]] = {}
  seen_cores: set[tuple[int, str]] = set()
  for cpu_dir in sorted(
      cpu_root.glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])
  ):
    c = int(cpu_dir.name[3:])
    online_file = cpu_dir / "online"
    if online_file.exists() and online_file.read_text().strip() == "0":
      continue
    pkg_file = cpu_dir / "topology" / "physical_package_id"
    pkg = int(pkg_file.read_text().strip()) if pkg_file.exists() else 0
    sib_file = cpu_dir / "topology" / "thread_siblings_list"
    sib_key = sib_file.read_text().strip() if sib_file.exists() else str(c)
    if (pkg, sib_key) not in seen_cores:
      seen_cores.add((pkg, sib_key))
      by_pkg.setdefault(pkg, []).append(c)

  if not by_pkg:
    return "0-39"
  pkgs = sorted(by_pkg)
  target_pkg = pkgs[socket_idx % len(pkgs)]
  phys = sorted(by_pkg[target_pkg])
  ranges: list[str] = []
  start = prev = phys[0]
  for c in phys[1:]:
    if c == prev + 1:
      prev = c
    else:
      ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
      start = prev = c
  ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
  return ",".join(ranges)


def _select_numa(cmd_args: list[str]) -> list[str]:
  """Pin container to Socket 0 physical cores by default, or Socket 1 for odd accuracy shards."""
  socket_idx = int(os.environ.get("AEB_NUMA_NODE", "0"))
  for i, arg in enumerate(cmd_args):
    if arg == "--shard" and i + 1 < len(cmd_args):
      shard_spec = cmd_args[i + 1]
      if "/" in shard_spec:
        try:
          shard_idx = int(shard_spec.split("/", 1)[0])
          socket_idx = shard_idx % 2
        except ValueError:
          pass
    elif arg.startswith("--shard="):
      shard_spec = arg.split("=", 1)[1]
      if "/" in shard_spec:
        try:
          shard_idx = int(shard_spec.split("/", 1)[0])
          socket_idx = shard_idx % 2
        except ValueError:
          pass
  cpulist = _socket_physical_cpulist(socket_idx)
  if shutil.which("numactl"):
    return ["numactl", f"--physcpubind={cpulist}", "--localalloc"]
  if shutil.which("taskset"):
    return ["taskset", "-c", cpulist]
  return []


def _normalize_mount(spec: str) -> tuple[str, str, str]:
  parts = spec.split(":")
  src = os.path.abspath(parts[0])
  dst = parts[1]
  opts = parts[2:]
  base_name = os.path.basename(src.rstrip("/"))
  if base_name in ("models", "results", "datasets", "reports") and not src.startswith(str(AEB_ROOT)):
    candidate = AEB_ROOT / base_name
    if candidate.exists() or base_name in ("results", "reports"):
      src = str(candidate)
  if not os.path.exists(src):
    os.makedirs(src, exist_ok=True)
  mount_str = f"{src}:{dst}" + (":" + ":".join(opts) if opts else "")
  return dst, src, mount_str


def _exec_unshare(
    rootfs: Path,
    numa_prefix: list[str],
    cmd_argv: list[str],
    env: dict[str, str],
    mounts_src_by_dst: dict[str, str],
) -> int:
  """Execute command inside rootfs using Linux unshare --mount + chroot."""
  script_lines = [
      "set -e",
      "mount --make-rslave /",
  ]
  for sys_dir in ("proc", "sys", "dev"):
    target = rootfs / sys_dir
    target.mkdir(parents=True, exist_ok=True)
    script_lines.append(f"mount --rbind /{sys_dir} {shlex.quote(str(target))}")
    script_lines.append(f"mount --make-rslave {shlex.quote(str(target))}")

  for dst, src in mounts_src_by_dst.items():
    rel_dst = dst.lstrip("/")
    target = rootfs / rel_dst
    target.mkdir(parents=True, exist_ok=True)
    script_lines.append(f"mount --bind {shlex.quote(src)} {shlex.quote(str(target))}")

  env_kv = [f"{k}={v}" for k, v in env.items()]
  chroot_cmd = [
      "exec",
      "chroot",
      shlex.quote(str(rootfs)),
      "/usr/bin/env",
      "-i",
      *(shlex.quote(kv) for kv in env_kv),
      *(shlex.quote(a) for a in cmd_argv),
  ]
  script_lines.append(" ".join(chroot_cmd))
  full_cmd = [*numa_prefix, "unshare", "--mount", "/bin/bash", "-c", "\n".join(script_lines)]
  proc = subprocess.run(full_cmd)
  return proc.returncode


def _exec_container(
    bundle: str,
    cmd_argv: list[str],
    extra_env: dict[str, str],
    extra_mounts: list[str],
) -> int:
  cid = f"aeb-{bundle}-{uuid.uuid4().hex[:8]}"
  tmp_host = Path(f"/tmp/aeb-tmp-{cid}")
  tmp_host.mkdir(parents=True, exist_ok=True)

  repo_notes = AEB_ROOT / "repo" / "reports" / "notes"
  dst_notes = AEB_ROOT / "reports" / "notes"
  if repo_notes.is_dir() and repo_notes.resolve() != dst_notes.resolve():
    dst_notes.mkdir(parents=True, exist_ok=True)
    for f in repo_notes.iterdir():
      if f.is_file():
        dst_f = dst_notes / f.name
        if not (dst_f.exists() and os.path.samefile(f, dst_f)):
          shutil.copy2(f, dst_f)

  mounts_src_by_dst: dict[str, str] = {}
  default_mounts = [
      f"{AEB_ROOT / 'models'}:/models",
      f"{AEB_ROOT / 'results'}:/results",
      f"{AEB_ROOT / 'datasets'}:/datasets:ro",
      f"{AEB_ROOT / 'reports'}:/reports",
      f"{AEB_ROOT / 'repo' / 'docker' / 'harness'}:/opt/aeb/harness:ro",
      f"{tmp_host}:/tmp",
  ]
  for m in default_mounts + extra_mounts:
    dst, src, _ = _normalize_mount(m)
    mounts_src_by_dst[dst] = src

  env = dict(DEFAULT_ENV)
  env.update(extra_env)

  numa_prefix = _select_numa(cmd_argv)
  bundle_dir = BUNDLES_DIR / bundle
  rootfs_dir = bundle_dir / "rootfs"

  try:
    return _exec_unshare(rootfs_dir, numa_prefix, cmd_argv, env, mounts_src_by_dst)
  finally:
    shutil.rmtree(tmp_host, ignore_errors=True)


def _handle_compose(argv: list[str]) -> int:
  i = 0
  while i < len(argv):
    a = argv[i]
    if a in ("-f", "--file", "--profile", "-p", "--project-name", "--env-file"):
      i += 2
      continue
    if a.startswith(("--file=", "--profile=", "--project-name=", "--env-file=")):
      i += 1
      continue
    break

  if i >= len(argv):
    return 0
  subcmd = argv[i]
  rest = argv[i + 1 :]

  if subcmd == "build":
    return 0
  if subcmd == "config":
    if "--services" in rest:
      for s in SERVICE_MAP:
        print(s)
    return 0
  if subcmd != "run":
    sys.stderr.write(f"dock_shim: unsupported docker compose subcommand: {subcmd}\n")
    return 1

  extra_env: dict[str, str] = {}
  extra_mounts: list[str] = []
  entrypoint_override: str | None = None

  j = 0
  while j < len(rest):
    a = rest[j]
    if a in ("--rm", "-T", "-i", "-t", "--init"):
      j += 1
    elif a in ("-e", "--env"):
      kv = rest[j + 1]
      if "=" in kv:
        k, v = kv.split("=", 1)
        extra_env[k] = v
      elif kv in os.environ:
        extra_env[kv] = os.environ[kv]
      j += 2
    elif a.startswith("--env="):
      kv = a.split("=", 1)[1]
      if "=" in kv:
        k, v = kv.split("=", 1)
        extra_env[k] = v
      j += 1
    elif a in ("-v", "--volume"):
      extra_mounts.append(rest[j + 1])
      j += 2
    elif a.startswith("--volume="):
      extra_mounts.append(a.split("=", 1)[1])
      j += 1
    elif a == "--entrypoint":
      entrypoint_override = rest[j + 1]
      j += 2
    elif a.startswith("--entrypoint="):
      entrypoint_override = a.split("=", 1)[1]
      j += 1
    else:
      break

  if j >= len(rest):
    sys.stderr.write("dock_shim: missing service name in docker compose run\n")
    return 1

  service = rest[j]
  service_args = rest[j + 1 :]
  if service not in SERVICE_MAP:
    sys.stderr.write(f"dock_shim: unknown compose service {service!r}\n")
    return 1

  bundle, default_ep, default_cmd, svc_env = SERVICE_MAP[service]
  merged_env = dict(svc_env)
  merged_env.update(extra_env)

  if entrypoint_override is not None:
    ep = [entrypoint_override] if entrypoint_override else []
  else:
    ep = list(default_ep)

  cmd_tail = service_args if service_args else list(default_cmd)
  full_cmd = ep + cmd_tail
  return _exec_container(bundle, full_cmd, merged_env, extra_mounts)


def _handle_run(argv: list[str]) -> int:
  extra_env: dict[str, str] = {}
  extra_mounts: list[str] = []
  entrypoint_override: str | None = None

  j = 0
  while j < len(argv):
    a = argv[j]
    if a in ("--rm", "-i", "-t", "-T", "--init"):
      j += 1
    elif a in ("-e", "--env"):
      kv = argv[j + 1]
      if "=" in kv:
        k, v = kv.split("=", 1)
        extra_env[k] = v
      elif kv in os.environ:
        extra_env[kv] = os.environ[kv]
      j += 2
    elif a in ("-v", "--volume"):
      extra_mounts.append(argv[j + 1])
      j += 2
    elif a == "--entrypoint":
      entrypoint_override = argv[j + 1]
      j += 2
    elif a.startswith("--entrypoint="):
      entrypoint_override = a.split("=", 1)[1]
      j += 1
    else:
      break

  if j >= len(argv):
    sys.stderr.write("dock_shim: missing image name in docker run\n")
    return 1

  image = argv[j]
  cmd_args = argv[j + 1 :]
  bundle = IMAGE_MAP.get(image, image.split("/")[-1].split(":")[0])
  ep = [entrypoint_override] if entrypoint_override else []
  full_cmd = ep + cmd_args
  return _exec_container(bundle, full_cmd, extra_env, extra_mounts)


def main() -> int:
  argv = sys.argv[1:]
  if not argv:
    return 0
  if argv[0] == "compose":
    return _handle_compose(argv[1:])
  if argv[0] == "run":
    return _handle_run(argv[1:])
  if argv[0] == "build":
    return 0
  sys.stderr.write(f"dock_shim: unsupported docker command {argv[0]!r}\n")
  return 1


if __name__ == "__main__":
  sys.exit(main())
