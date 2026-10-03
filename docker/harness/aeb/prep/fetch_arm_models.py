"""Stage pinned, independently quantized Gemma 4 E2B packages for ARM CPU trials."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
  base = Path("/models")
  models = (
      ("justinchuby/gemma-4-e2b-it-onnx",
       "9bcf2cb1c2878b1c68a5f94db037272dfb278384",
       base / "aeb/gemma-4-e2b-onnx", ["Q4_K_M/default/**"]),
  )
  for repo, revision, directory, include in models:
    snapshot_download(repo_id=repo, revision=revision, local_dir=directory,
                      allow_patterns=include, max_workers=4)
    print(f"staged {repo}@{revision} in {directory}", flush=True)


if __name__ == "__main__":
  main()
