"""Model-free tests for the ONNX Runtime GenAI driver."""

import json
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aeb.drivers import onnxruntime_driver as driver  # noqa: E402


class _FakeGenerator:

  def __init__(self, tokens):
    self.tokens = tokens
    self.index = 0

  def append_tokens(self, _tokens):
    pass

  def is_done(self):
    return self.index >= len(self.tokens)

  def generate_next_token(self):
    self.index += 1

  def get_next_tokens(self):
    return [self.tokens[self.index - 1]]


class _FakeParams:

  def __init__(self, _model):
    self.options = None

  def set_search_options(self, **options):
    self.options = options


class _FakeORT:

  def __init__(self, tokens):
    self.tokens = tokens
    self.params = None
    self.generator = None

  def GeneratorParams(self, model):
    self.params = _FakeParams(model)
    return self.params

  def Generator(self, _model, _params):
    self.generator = _FakeGenerator(self.tokens)
    return self.generator


class OnnxRuntimeDriverTest(unittest.TestCase):

  def test_bos_is_added_or_removed_once(self):
    self.assertEqual(driver._normalize_bos([4, 5], True), [2, 4, 5])
    self.assertEqual(driver._normalize_bos([2, 4, 5], True), [2, 4, 5])
    self.assertEqual(driver._normalize_bos([2, 4, 5], False), [4, 5])

  def test_bos_uses_the_model_configured_token_id(self):
    self.assertEqual(driver._normalize_bos([7, 4, 5], True, bos_token_id=7), [7, 4, 5])
    self.assertEqual(driver._normalize_bos([4, 5], True, bos_token_id=7), [7, 4, 5])

  def test_config_overlay_sets_threads_without_mutating_staged_model(self):
    with tempfile.TemporaryDirectory() as source:
      model_dir = Path(source)
      original = {
          "model": {
              "type": "gemma4",
              "decoder": {"session_options": {"provider_options": []}},
              "embedding": {},
              "eos_token_id": [1, 106],
          }
      }
      (model_dir / "genai_config.json").write_text(json.dumps(original))
      (model_dir / "weights.bin").write_bytes(b"weights")

      workspace, runtime_dir, updated = driver._prepare_runtime_model(model_dir, 6)
      try:
        self.assertEqual(updated["model"]["decoder"]["session_options"]["intra_op_num_threads"], 6)
        self.assertEqual(updated["model"]["decoder"]["session_options"]["inter_op_num_threads"], 1)
        self.assertEqual(updated["model"]["decoder"]["session_options"]["provider_options"], [])
        self.assertNotIn("providers", updated["model"]["decoder"]["session_options"])
        self.assertEqual(updated["model"]["embedding"]["session_options"]["intra_op_num_threads"], 6)
        self.assertEqual(json.loads((model_dir / "genai_config.json").read_text()), original)
        self.assertTrue((runtime_dir / "weights.bin").is_symlink())
      finally:
        workspace.cleanup()

  def test_generate_honors_stop_ids_and_reports_one_timestamp_per_token(self):
    fake_ort = _FakeORT([7, 8, 50, 9])
    instance = driver.Driver.__new__(driver.Driver)
    instance.args = SimpleNamespace(ctx=32)
    instance.model = object()
    instance.og = fake_ort
    instance.bos_token_id = 2
    instance.tokenizer = SimpleNamespace(
        encode=lambda _text: [2, 3],
        decode=lambda ids: ",".join(str(token) for token in ids),
    )
    instance.eos_ids = {1, 106}

    numpy_stub = types.SimpleNamespace(asarray=lambda values, dtype=None: list(values),
                                       int32="int32")
    with mock.patch.dict(sys.modules, {"numpy": numpy_stub}):
      result = instance.generate({
          "prompt": "prompt", "max_tokens": 8, "stop_ids": [1, 50, 106],
          "return_prompt_ids": True,
      })

    self.assertEqual(result["gen_ids"], [7, 8])
    self.assertEqual(result["n_gen"], 2)
    self.assertEqual(len(result["token_ns"]), 2)
    self.assertLessEqual(result["t_first_ns"], result["token_ns"][-1])
    self.assertTrue(result["stopped_on_eos"])
    self.assertEqual(result["prompt_ids"], [2, 3])
    self.assertEqual(result["text"], "7,8")
    self.assertFalse(fake_ort.params.options["do_sample"])
    self.assertEqual(fake_ort.params.options["min_length"], 0)

  def test_fixed_decode_ignores_stops_and_sets_exact_sequence_length(self):
    fake_ort = _FakeORT([1, 50, 106])
    instance = driver.Driver.__new__(driver.Driver)
    instance.args = SimpleNamespace(ctx=16)
    instance.model = object()
    instance.og = fake_ort
    instance.bos_token_id = 2
    instance.tokenizer = SimpleNamespace(
        encode=lambda _text: [2],
        decode=lambda ids: ",".join(str(token) for token in ids),
    )
    instance.eos_ids = {1, 106}

    numpy_stub = types.SimpleNamespace(asarray=lambda values, dtype=None: list(values),
                                       int32="int32")
    with mock.patch.dict(sys.modules, {"numpy": numpy_stub}):
      result = instance.generate({
          "prompt": "prompt", "max_tokens": 3, "ignore_eos": True,
          "sampling": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "seed": 9},
      })

    self.assertEqual(result["gen_ids"], [1, 50, 106])
    self.assertFalse(result["stopped_on_eos"])
    self.assertEqual(fake_ort.params.options["max_length"], 4)
    self.assertEqual(fake_ort.params.options["min_length"], 4)
    self.assertEqual(fake_ort.params.options["random_seed"], 9)
    self.assertTrue(fake_ort.params.options["do_sample"])


if __name__ == "__main__":
  unittest.main()
