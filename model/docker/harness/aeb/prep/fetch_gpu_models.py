"""Stage the pinned Gemma 4 26B A4B artifacts for the native Apple GPU suite.

  python3 -m aeb.prep.fetch_gpu_models [--models models] [--only gguf,mlx,...]

Standard library only (runs on the macOS host before any framework is
installed). Every file is fetched from a pinned Hugging Face revision with a
resumable curl transfer into `<file>.part`, then verified against the LFS
SHA-256 that the Hub publishes for that revision (and, for BaseRT, the checksum
in the BaseRT v0.4.0 model catalog) before it is renamed into place. Each
model file gets the `.sha256` and `.manifest.json` sidecars used by the rest
of the harness (see docker/bin/aeb-fetch-model).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

# key -> (repo, revision, files (None = whole repo), extra expected sha256 by file)
ARTIFACTS = {
    "gguf": ("google/gemma-4-26B-A4B-it-qat-q4_0-gguf",
             "d1c082be9cf3c8a514acf63b8761f4b41935842e",
             ["gemma-4-26B_q4_0-it.gguf", "README.md"], {}),
    "mlx": ("lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit",
            "aa5e2e50376f54254d503276ca2222ea8569b95e", None, {}),
    "litertlm": ("litert-community/gemma-4-26B-A4B-it-litert-lm",
                 "a6268756268ac3eab06333eb28f279cff56e9ea6",
                 ["gemma-4-26B-A4B-it-gpu.litertlm", "README.md"], {}),
    # Pre-converted BaseRT catalog bundle (base-convert/crates/base-hub/catalog.json
    # at basecompute/baseRT v0.4.0: id basecompute/gemma-4-26B-A4B-it, quant default-q4).
    "basert": ("basecompute/gemma-4-26B-A4B-it",
               "1e3d21ea4a49f9acd0957326ec295eb90298ea89",
               ["gemma-4-26B-A4B-it-Q4.base", "config.json", "README.md"],
               {"gemma-4-26B-A4B-it-Q4.base":
                "efabbac27832f43d17eaac05c9a5c32545c9a02bea975836f038d32de8368782"}),
}
ORDER = ("gguf", "mlx", "litertlm", "basert")


def _auth() -> list[str]:
  tok = os.environ.get("HF_TOKEN")
  return ["-H", f"Authorization: Bearer {tok}"] if tok else []


def tree(repo: str, revision: str) -> dict[str, dict]:
  req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=1")
  if os.environ.get("HF_TOKEN"):
    req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
  with urllib.request.urlopen(req, timeout=60) as r:
    return {f["path"]: f for f in json.load(r) if f["type"] == "file"}


def sha256(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
      h.update(chunk)
  return h.hexdigest()


def fetch(repo: str, revision: str, name: str, info: dict, dest: Path, expect: str | None) -> None:
  dest.parent.mkdir(parents=True, exist_ok=True)
  lfs = (info.get("lfs") or {}).get("oid")
  want = [x for x in (lfs, expect) if x]
  if len(set(want)) > 1:
    raise SystemExit(f"{repo}/{name}: Hub sha256 {lfs} != pinned catalog sha256 {expect}")
  side = Path(str(dest) + ".sha256")
  if dest.is_file() and dest.stat().st_size == info["size"]:
    if not want or (side.is_file() and side.read_text().split()[0] == want[0]):
      print(f"present  {repo}/{name}", flush=True)
      return
  part = Path(str(dest) + ".part")
  url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
  print(f"fetching {repo}@{revision[:12]}/{name} ({info['size'] / 1e9:.2f} GB)", flush=True)
  for _ in range(20):  # resume across dropped connections
    rc = subprocess.call(["curl", "--fail", "--location", "--silent", "--show-error",
                          "--retry", "5", "--retry-delay", "10", "--continue-at", "-",
                          "--speed-limit", "10000", "--speed-time", "120",
                          *_auth(), "--output", str(part), url])
    if rc == 0 and part.stat().st_size == info["size"]:
      break
    print(f"  curl exited {rc}; resuming", flush=True)
  else:
    raise SystemExit(f"download failed: {url}")
  digest = sha256(part) if (want or name.endswith((".gguf", ".litertlm", ".base", ".safetensors"))) else None
  if want and digest != want[0]:
    raise SystemExit(f"{repo}/{name}: sha256 {digest} does not match expected {want[0]}")
  part.rename(dest)
  if digest:
    side.write_text(digest + "\n")
    (Path(str(dest) + ".manifest.json")).write_text(json.dumps(
        {"repo": repo, "revision": revision, "file": name, "path": str(dest),
         "sha256": digest, "bytes": info["size"], "verified_against": "hub-lfs-sha256"
         + ("+basert-catalog" if expect else "")}, indent=1) + "\n")
  print(f"ready    {dest} ({'sha256 ' + digest[:16] + '…' if digest else 'small file'})", flush=True)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--models", type=Path, default=ROOT / "models")
  ap.add_argument("--only", default=",".join(ORDER))
  args = ap.parse_args(argv)
  for key in args.only.split(","):
    repo, revision, files, expect = ARTIFACTS[key]
    listing = tree(repo, revision)
    names = files or sorted(n for n in listing if n != ".gitattributes")
    for name in names:
      fetch(repo, revision, name, listing[name], args.models / repo / name, expect.get(name))
    print(f"staged {key}: {repo}@{revision}", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
