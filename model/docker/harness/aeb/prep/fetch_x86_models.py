"""Stage pinned Gemma 4 E2B model artifacts across all data type variants for x86 CPU benchmarks."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
  base = Path("/models")
  models = (
      (
          "google/gemma-4-E2B-it",
          "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
          base / "google/gemma-4-E2B-it",
          [
              "config.json",
              "generation_config.json",
              "tokenizer.json",
              "tokenizer_config.json",
              "model.safetensors*",
          ],
      ),
      (
          "google/gemma-4-E2B-it-qat-mobile-transformers",
          "dd693ff40353f057ca5f07e945ad867f4afbf2ec",
          base / "google/gemma-4-E2B-it-qat-mobile-transformers",
          [
              "config.json",
              "generation_config.json",
              "tokenizer.json",
              "tokenizer_config.json",
              "model.safetensors*",
          ],
      ),
      (
          "OpenVINO/gemma-4-E2B-it-int4-ov",
          "7cd9ccd5beddc5522462891b2d44668fe898cbfd",
          base / "OpenVINO/gemma-4-E2B-it-int4-ov",
          ["*.json", "*.xml", "*.bin", "tokenizer*"],
      ),
      (
          "OpenVINO/gemma-4-E2B-it-int8-ov",
          "35967cca90cd9457765575ff06e13ee9c4570d84",
          base / "OpenVINO/gemma-4-E2B-it-int8-ov",
          ["*.json", "*.xml", "*.bin", "tokenizer*"],
      ),
      (
          "OpenVINO/gemma-4-E2B-it-fp16-ov",
          "15cdb20c9881ee5139672cfea0cdef9fc3f1ac5b",
          base / "OpenVINO/gemma-4-E2B-it-fp16-ov",
          ["*.json", "*.xml", "*.bin", "tokenizer*"],
      ),
      (
          "justinchuby/gemma-4-e2b-it-onnx",
          "9bcf2cb1c2878b1c68a5f94db037272dfb278384",
          base / "aeb/gemma-4-e2b-onnx",
          ["Q4_K_M/default/**", "f16/default/**"],
      ),
      (
          "ggml-org/gemma-4-E2B-it-GGUF",
          "main",
          base / "ggml-org/gemma-4-E2B-it-GGUF",
          [
              "gemma-4-E2B-it-Q4_0.gguf",
              "gemma-4-E2B-it-Q8_0.gguf",
              "gemma-4-E2B-it-BF16.gguf",
          ],
      ),
  )
  for repo, revision, directory, include in models:
    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=directory,
        allow_patterns=include,
        max_workers=4,
    )
    print(f"staged {repo}@{revision} in {directory}", flush=True)


if __name__ == "__main__":
  main()
