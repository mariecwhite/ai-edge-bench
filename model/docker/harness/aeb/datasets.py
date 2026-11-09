"""Stage evaluation datasets into ./datasets (runs in the aeb/tools image).

  python3 -m aeb.datasets mmlu  --out /datasets
  python3 -m aeb.datasets gsm8k --out /datasets

Downloads a pinned revision of `cais/mmlu` (all subjects, test split: 14,042
questions) or `openai/gsm8k` (main config, test split: 1,319 problems),
converts it to JSON lines with a stable `id`, and writes a manifest with the
source URL, revision and SHA-256 of both files. The benchmark containers only
read the JSONL, so they need no pyarrow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

MMLU_REPO = "cais/mmlu"
# Pin the dataset revision so accuracy numbers stay comparable over time.
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
MMLU_FILE = "all/test-00000-of-00001.parquet"
GSM8K_REPO = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
GSM8K_FILE = "main/test-00000-of-00001.parquet"


def sha256(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def _download(repo: str, revision: str, file: str, d: Path) -> tuple[str, Path]:
  d.mkdir(parents=True, exist_ok=True)
  url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{file}"
  pq_path = d / "test.parquet"
  if not pq_path.exists():
    tmp = pq_path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(pq_path)
  return url, pq_path


def fetch_gsm8k(out: Path, revision: str) -> Path:
  import pyarrow.parquet as pq

  d = out / "gsm8k"
  url, pq_path = _download(GSM8K_REPO, revision, GSM8K_FILE, d)
  table = pq.read_table(pq_path).to_pylist()
  jl = d / "test.jsonl"
  with open(jl, "w") as f:
    for i, row in enumerate(table):
      answer = row["answer"].split("####")[-1].strip().replace(",", "")
      f.write(json.dumps({"id": f"gsm8k-test-{i:04d}", "subject": "gsm8k",
                          "question": row["question"], "answer": answer}) + "\n")
  manifest = {"dataset": GSM8K_REPO, "config": "main", "revision": revision, "url": url,
              "split": "test", "n": len(table), "parquet_sha256": sha256(pq_path),
              "jsonl_sha256": sha256(jl)}
  (d / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
  print(json.dumps(manifest, indent=1))
  return jl


def fetch_mmlu(out: Path, revision: str) -> Path:
  import pyarrow.parquet as pq

  d = out / "mmlu"
  url, pq_path = _download(MMLU_REPO, revision, MMLU_FILE, d)
  table = pq.read_table(pq_path).to_pylist()
  jl = d / "test.jsonl"
  with open(jl, "w") as f:
    for i, row in enumerate(table):
      f.write(json.dumps({"id": f"mmlu-test-{i:05d}", "subject": row["subject"],
                          "question": row["question"], "choices": list(row["choices"]),
                          "answer": "ABCD"[int(row["answer"])]}) + "\n")
  manifest = {"dataset": MMLU_REPO, "revision": revision, "url": url, "split": "test",
              "n": len(table), "parquet_sha256": sha256(pq_path), "jsonl_sha256": sha256(jl)}
  (d / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
  print(json.dumps(manifest, indent=1))
  return jl


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("dataset", choices=["mmlu", "gsm8k"])
  ap.add_argument("--out", type=Path, default=Path("/datasets"))
  ap.add_argument("--revision", help="override the pinned dataset revision")
  args = ap.parse_args(argv)
  if args.dataset == "mmlu":
    fetch_mmlu(args.out, args.revision or MMLU_REVISION)
  else:
    fetch_gsm8k(args.out, args.revision or GSM8K_REVISION)
  return 0


if __name__ == "__main__":
  sys.exit(main())
