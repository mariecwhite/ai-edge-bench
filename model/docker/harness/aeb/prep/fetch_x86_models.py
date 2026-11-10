"""Stage pinned Gemma 4 E2B and 31B model artifacts across all data type variants for x86 CPU benchmarks.

Usage:
  python3 -m aeb.prep.fetch_x86_models                  # stages E2B (default)
  python3 -m aeb.prep.fetch_x86_models --model 31b      # stages 31B + exports CPU .litertlm
  python3 -m aeb.prep.fetch_x86_models --model all      # stages both E2B and 31B
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def _e2b_models(base: Path) -> tuple[tuple[str, str, Path, list[str]], ...]:
  return (
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


def _31b_models(base: Path) -> tuple[tuple[str, str, Path, list[str]], ...]:
  return (
      (
          "google/gemma-4-31B-it",
          "842da3794eaa0b77d5f08bae87a17459d91ff475",
          base / "google/gemma-4-31B-it",
          [
              "config.json",
              "generation_config.json",
              "chat_template.jinja",
              "tokenizer.json",
              "tokenizer_config.json",
              "model*.safetensors*",
          ],
      ),
      (
          "google/gemma-4-31B-it-qat-q4_0-unquantized",
          "1e4d8beecacb8b7590c1d8bedd7335f687bf311f",
          base / "google/gemma-4-31B-it-qat-q4_0-unquantized",
          [
              "config.json",
              "generation_config.json",
              "chat_template.jinja",
              "tokenizer.json",
              "tokenizer_config.json",
              "model*.safetensors*",
          ],
      ),
      (
          "google/gemma-4-31B-it-qat-q4_0-gguf",
          "59dde24573e7e61570dba08b18a2e1fe246955ed",
          base / "google/gemma-4-31B-it-qat-q4_0-gguf",
          ["gemma-4-31B_q4_0-it.gguf"],
      ),
      (
          "ggml-org/gemma-4-31B-it-GGUF",
          "3d47a7e680fb9526dcac74335858f0f31acc9c27",
          base / "ggml-org/gemma-4-31B-it-GGUF",
          [
              "gemma-4-31B-it-Q4_0.gguf",
              "gemma-4-31B-it-Q8_0.gguf",
              "gemma-4-31B-it-BF16.gguf",
          ],
      ),
      (
          "OpenVINO/gemma-4-31B-it-int4-ov",
          "e7b82c4f581d68ee3990eed530f2826c12840ceb",
          base / "OpenVINO/gemma-4-31B-it-int4-ov",
          ["*.json", "*.xml", "*.bin", "tokenizer*"],
      ),
      (
          "OpenVINO/gemma-4-31B-it-int8-ov",
          "15d13daedfc9ab4da0e8799345af1360844bb31d",
          base / "OpenVINO/gemma-4-31B-it-int8-ov",
          ["*.json", "*.xml", "*.bin", "tokenizer*"],
      ),
  )


def _ensure_litert_lm_builder_compat() -> None:
  """Shim litert_lm_builder>=0.19.0 for litert_torch==0.10.0 compatibility."""
  try:
    import enum  # pylint: disable=g-import-not-at-top
    import litert_lm_builder  # pylint: disable=g-import-not-at-top
  except ImportError:
    return
  if not hasattr(litert_lm_builder, "TfLiteModelType"):

    class TfLiteModelType(str, enum.Enum):
      PREFILL_DECODE = "tf_lite_prefill_decode"
      EMBEDDER = "tf_lite_embedder"
      PER_LAYER_EMBEDDER = "tf_lite_per_layer_embedder"
      VISION_ENCODER = "tf_lite_vision_encoder"
      VISION_ADAPTER = "tf_lite_vision_adapter"
      END_OF_VISION = "tf_lite_end_of_vision"
      AUDIO_ENCODER_HW = "tf_lite_audio_encoder_hw"
      AUDIO_ADAPTER = "tf_lite_audio_adapter"
      END_OF_AUDIO = "tf_lite_end_of_audio"
      AUX = "tf_lite_aux"

    litert_lm_builder.TfLiteModelType = TfLiteModelType  # type: ignore[attr-defined]
    if hasattr(litert_lm_builder, "litertlm_builder"):
      litert_lm_builder.litertlm_builder.TfLiteModelType = TfLiteModelType  # type: ignore[attr-defined]
  builder_cls = getattr(litert_lm_builder, "LitertLmFileBuilder", None)
  if builder_cls is not None and hasattr(builder_cls, "validate_metadata"):
    builder_cls.validate_metadata = lambda self: None


def export_gemma4_31b_litertlm(source_dir: Path, output_dir: Path) -> Path:
  """Export CPU/GPU tf_lite_prefill_decode .litertlm package for Gemma 4 31B.

  Upstream litert-community/gemma-4-31B-it-litert-lm (ad2ed603ed72) publishes only
  ML Drift GPU/Web artisan packages (model_type=tf_lite_artisan_text_decoder, ops=0).
  CPU execution via XNNPACK and YNNPACK requires a tf_lite_prefill_decode flatbuffer
  exported from the official Google QAT checkpoint via litert_torch.generative.export_hf
  using the same dynamic_wi4_afp32 recipe used by litert-community/gemma-4-12B-it-litert-lm.
  """
  output_dir.mkdir(parents=True, exist_ok=True)
  target = output_dir / "gemma-4-31B-it.litertlm"
  if target.is_file() and target.stat().st_size > 1_000_000_000:
    print(f"already exported: {target}", flush=True)
    return target

  _ensure_litert_lm_builder_compat()
  from litert_torch.generative.export_hf import export  # pylint: disable=g-import-not-at-top

  export.export(
      model=str(source_dir),
      output_dir=str(output_dir),
      bundle_litert_lm=True,
      use_v2=True,
      delete_in_memory_params=True,
      quantization_recipe="dynamic_wi4_afp32",
      prefill_lengths=[128, 1024],
      cache_length=32771,
      externalize_embedder=True,
      single_token_embedder=True,
      export_vision_encoder=False,
      export_audio_encoder=False,
  )
  generated = output_dir / "model.litertlm"
  if generated.is_file() and generated != target:
    generated.rename(target)
  print(f"exported {target} from {source_dir}", flush=True)
  return target


def main(argv: list[str] | None = None) -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", choices=["e2b", "31b", "all"], default="e2b")
  ap.add_argument("--models-dir", type=Path, default=Path("/models"))
  ap.add_argument("--skip-litert-export", action="store_true")
  args = ap.parse_args(argv)

  models: list[tuple[str, str, Path, list[str]]] = []
  if args.model in ("e2b", "all"):
    models.extend(_e2b_models(args.models_dir))
  if args.model in ("31b", "all"):
    models.extend(_31b_models(args.models_dir))

  for repo, revision, directory, include in models:
    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=directory,
        allow_patterns=include,
        max_workers=4,
    )
    print(f"staged {repo}@{revision} in {directory}", flush=True)

  if args.model in ("31b", "all") and not args.skip_litert_export:
    export_gemma4_31b_litertlm(
        source_dir=args.models_dir / "google/gemma-4-31B-it-qat-q4_0-unquantized",
        output_dir=args.models_dir / "litert-community/gemma-4-31B-it-litert-lm",
    )


if __name__ == "__main__":
  main()
