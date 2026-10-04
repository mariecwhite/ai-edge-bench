"""Unit tests that need no model, dataset or accelerator.

  make test     (runs inside the aeb/tools image)
"""

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "scripts"))

from aeb import accuracy, common, stats, workload  # noqa: E402


class EncodingTest(unittest.TestCase):
  """The GGUF re-encoding must reproduce q * s exactly (up to fp16 scales)."""

  @classmethod
  def setUpClass(cls):
    try:
      import numpy  # noqa: F401
      import gguf  # noqa: F401
    except ImportError:
      raise unittest.SkipTest("numpy/gguf not installed (run in the tools image)")
    from aeb.prep import convert_qat_mobile_gguf as conv
    cls.conv = conv

  def roundtrip(self, bits, int2_as="q2_k", rows=4, cols=512):
    import numpy as np
    import gguf
    rng = np.random.default_rng(bits)
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    q = rng.integers(lo, hi + 1, size=(rows, cols)).astype(np.int8)
    s = rng.uniform(1e-3, 2e-2, size=(rows, 1)).astype(np.float32)
    self.conv.INT2_AS = int2_as
    data, qt = self.conv.encode(q, s, bits)
    deq = gguf.quants.dequantize(data, qt).reshape(q.shape)
    expect = q.astype(np.float32) * s.astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(deq, expect)
    return qt

  def test_int4_q4_0(self):
    self.assertEqual(self.roundtrip(4).name, "Q4_0")

  def test_int8_q8_0(self):
    self.assertEqual(self.roundtrip(8).name, "Q8_0")

  def test_int2_q2_k(self):
    self.assertEqual(self.roundtrip(2).name, "Q2_K")

  def test_int2_as_q4_0(self):
    self.assertEqual(self.roundtrip(2, int2_as="q4_0").name, "Q4_0")

  def test_unpack_offset_binary(self):
    import numpy as np
    # 0x1B = 0b00_01_10_11 -> values 3,2,1,0 (low bits first) minus 2.
    out = self.conv.unpack(np.array([[0x1B]], dtype=np.uint8), 2, 4)
    np.testing.assert_array_equal(out, [[1, 0, -1, -2]])
    out = self.conv.unpack(np.array([[0xF0]], dtype=np.uint8), 4, 2)
    np.testing.assert_array_equal(out, [[-8, 7]])


class AccuracyParsingTest(unittest.TestCase):

  def test_mmlu_extract(self):
    self.assertEqual(accuracy.extract(" **C**.\n"), "C")
    self.assertEqual(accuracy.extract(" B."), "B")
    self.assertEqual(accuracy.extract("The answer is (D)"), "D")
    self.assertIsNone(accuracy.extract(" $\\frac{1}{2}$"))

  def test_gsm8k_extract_and_score(self):
    self.assertEqual(accuracy.extract_gsm8k("...\nThe final answer is 1,234."), "1234")
    self.assertEqual(accuracy.extract_gsm8k("The final answer is $\\boxed{18}$"), "18")
    self.assertEqual(accuracy.extract_gsm8k("so 3 apples and 12"), "12")
    self.assertTrue(accuracy.score_gsm8k("18.0", "18"))
    self.assertFalse(accuracy.score_gsm8k(None, "18"))

  def test_stratified_is_deterministic_and_balanced(self):
    items = [{"id": f"{s}-{i}", "subject": s} for s in "abc" for i in range(10)]
    sub = accuracy.stratified(items, 6)
    self.assertEqual(sub, accuracy.stratified(items, 6))
    self.assertEqual(sorted(x["subject"] for x in sub), list("aabbcc"))

  def test_render_uses_chat_template_and_prefill(self):
    p = accuracy.render({"subject": "x_y", "question": "Q?", "choices": ["1", "2", "3", "4"]})
    self.assertTrue(p.startswith(workload.USER_PREFIX))
    self.assertTrue(p.endswith(workload.TURN_SUFFIX + accuracy.ANSWER_PREFIX))
    self.assertNotIn("<bos>", p)


class WorkloadTest(unittest.TestCase):

  def test_prompt_hits_exact_token_count(self):
    def tok(text):  # 1 token per 4 chars + bos, a stand-in tokenizer
      return [2] + list(range(len(text) // 4))
    for n in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
      text, ids = workload.build_perf_prompt(tok, n)
      self.assertEqual(len(ids), n)
      self.assertTrue(text.endswith(workload.TURN_SUFFIX))

  def test_short_prompt_preserves_original_passage_prefix(self):
    def tok(text):
      return [2] + list(range(len(text) // 4))
    text, _ = workload.build_perf_prompt(tok, 1024)
    self.assertIn(workload.passage()[:1000], text)


class StatsTest(unittest.TestCase):

  def test_describe(self):
    d = stats.describe([1, 2, 3, 4, 100])
    self.assertEqual(d["median"], 3)
    self.assertEqual((d["min"], d["max"], d["n"]), (1, 100, 5))

  def test_outliers(self):
    self.assertEqual(stats.mad_outliers([10, 10.1, 9.9, 10.05, 30]), [4])

  def test_bootstrap_ratio(self):
    rng = random.Random(1)
    a = [2 + rng.random() * 0.01 for _ in range(20)]
    b = [1 + rng.random() * 0.01 for _ in range(20)]
    r, lo, hi = stats.bootstrap_ratio_ci(a, b)
    self.assertTrue(lo <= r <= hi)
    self.assertAlmostEqual(r, 2, delta=0.05)

  def test_interp(self):
    self.assertEqual(stats.interp([0, 10], [0, 100], 5), 50)


class ReproTest(unittest.TestCase):

  def test_first_divergence(self):
    from aeb.repro import first_divergence
    self.assertIsNone(first_divergence([1, 2, 3], [1, 2, 3]))
    self.assertEqual(first_divergence([1, 2, 3], [1, 9, 3]), 1)
    self.assertEqual(first_divergence([1, 2], [1, 2, 3]), 2)

  def test_prompt_set_is_stable(self):
    from aeb.repro import prompts
    ps = prompts()
    self.assertEqual(len(ps), 12)
    self.assertEqual(len({p["id"] for p in ps}), 12)


class SuiteTest(unittest.TestCase):

  def test_sequence_sweep_exact_workloads_and_interleaving(self):
    import suite as suite_mod
    path = ROOT.parents[1] / "suites" / "gemma4-e2b-cpu-arm-runtimes.json"
    if not path.is_file():
      raise unittest.SkipTest("suites/ not mounted")
    s, ctx = suite_mod.load(path, "apple-m5-max")
    commands = suite_mod.sequence_commands(s, ctx, "apple-m5-max", 3, None, False)
    lengths = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    self.assertEqual(len(commands), 81)
    for i, n in enumerate(lengths):
      batch = commands[i * 9:(i + 1) * 9]
      frameworks = [label.split("/")[0] for label, _ in batch]
      order = [c["framework"] for c in s["configs"]]
      self.assertEqual(frameworks, order + list(reversed(order)) + order)
      for label, cmd in batch:
        self.assertTrue(label.endswith(f"-n{n}-d512"))
        self.assertEqual(cmd[cmd.index("--prompt-tokens") + 1], str(n))
        self.assertEqual(cmd[cmd.index("--gen-tokens") + 1], "512")
        self.assertEqual(cmd[cmd.index("--ctx") + 1], "16896")
        self.assertEqual(cmd[cmd.index("--track") + 1], "sweep-sequence")
        self.assertFalse(any("{" in a for a in cmd))
    self.assertEqual(s["workload"]["gen_tokens"], 256)
    self.assertEqual(ctx["ctx"], 4096)
    filtered = suite_mod.sequence_commands(s, ctx, "apple-m5-max", 1,
                                           "onnxruntime/q4_k_m", False)
    self.assertEqual(len(filtered), 9)

  def test_sequence_sweep_rejects_insufficient_context(self):
    import suite as suite_mod
    s = {"sequence_sweep": {"prompt_tokens": [16384], "gen_tokens": 512, "ctx": 4096}}
    with self.assertRaisesRegex(ValueError, "context capacity"):
      suite_mod.sequence_commands(s, {}, "apple-m5-max", 1, None, False)

  def test_driver_command_for_onnxruntime(self):
    from aeb.driver import driver_command
    self.assertEqual(driver_command("onnxruntime", ["--threads", "4"]),
                     ["python3", "-m", "aeb.drivers.onnxruntime_driver", "--threads", "4"])

  def test_suite_resolves_for_every_machine(self):
    import suite as suite_mod
    suites = ROOT.parents[1] / "suites"
    if not suites.is_dir():
      raise unittest.SkipTest("suites/ not mounted")
    for path in (suites / "gemma4-e2b-cpu.json",
                 suites / "gemma4-e2b-cpu-arm-runtimes.json"):
      data = json.loads(path.read_text())
      for machine in data["machines"]:
        s, ctx = suite_mod.load(path, machine)
        for _, cmd in suite_mod.perf_commands(s, ctx, machine, 2, None, False):
          self.assertFalse(any("{" in a for a in cmd), cmd)

  def test_unknown_suite_selection_fails(self):
    import suite as suite_mod
    with self.assertRaisesRegex(ValueError, "unknown suite config"):
      suite_mod.selected({"configs": [{"id": "llama.cpp/matched"}]}, "unknown/matched")

  def test_arm_repro_requests_greedy_and_seeded_sampling(self):
    import suite as suite_mod
    path = ROOT.parents[1] / "suites" / "gemma4-e2b-cpu-arm-runtimes.json"
    if not path.is_file():
      raise unittest.SkipTest("suites/ not mounted")
    s, ctx = suite_mod.load(path, "apple-m5-max")
    commands = suite_mod.repro_commands(s, ctx, "apple-m5-max", "onnxruntime/q4_k_m", False)
    self.assertEqual(len(commands), 7)
    self.assertEqual(sum(":greedy:" in label for label, _ in commands), 3)
    self.assertEqual(sum(":seeded:" in label for label, _ in commands), 4)


class ModelHashTest(unittest.TestCase):

  def test_directory_digest_includes_every_component(self):
    with tempfile.TemporaryDirectory() as d:
      root = Path(d)
      (root / "config.json").write_text("{}")
      (root / "weights").mkdir()
      weights = root / "weights" / "model.bin"
      weights.write_bytes(b"first")
      first = common.model_sha256(d)
      weights.write_bytes(b"second")
      weights.with_name(weights.name + ".sha256").unlink()
      self.assertNotEqual(first, common.model_sha256(d))


class SequenceReportTest(unittest.TestCase):

  def setUp(self):
    from aeb import report
    self.report = report
    self.suite = {"name": "test", "rounds": 1,
                  "workload": {"warmup": 2, "repetitions": 1},
                  "sequence_sweep": {"prompt_tokens": [64, 128], "gen_tokens": 512, "ctx": 16896},
                  "configs": [{"id": "llama.cpp/reference", "framework": "llama.cpp"}]}

  def run_data(self, n=64, gen=512, status="ok", started="2026-10-03T00:00:00Z"):
    return {"dir": Path("/run"), "meta": {
        "framework": "llama.cpp", "label": f"reference-n{n}-d512", "round": 0,
        "track": "sweep-sequence", "notes": "suite=test", "started_at": started,
        "params": {"prompt_tokens": n, "gen_tokens": 512},
        "driver_cmd": ["driver", "--ctx", "16896"],
        "workload": {"prompt_tokens": n, "prompt_ids_sha256": f"hash-{n}"}},
        "summary": {"status": status, "per_request": [
            {"measured": True, "n_prompt": n, "n_gen": gen,
             "prefill_tps": 100.0, "decode_tps": 50.0}]}}

  def test_report_preserves_missing_points_and_excludes_truncated_decode(self):
    data = self.report.sequence_data([self.run_data(), self.run_data(n=128, gen=511)], self.suite)
    self.assertEqual(data["points"]["64"]["configs"]["llama.cpp/reference"]["prefill_tps"]["median"], 100)
    invalid = data["points"]["128"]["configs"]["llama.cpp/reference"]
    self.assertIsNone(invalid["decode_tps"])
    self.assertEqual(invalid["n_requests"], 0)
    self.assertTrue(invalid["errors"])
    self.assertIn("unavailable, n=0", self.report.sequence_section(data))
    with tempfile.TemporaryDirectory() as d:
      self.report.sequence_charts(Path(d), data)
      for phase in ("prefill", "decode"):
        self.assertGreater((Path(d) / f"sequence_{phase}.png").stat().st_size, 1000)

  def test_latest_round_not_duplicate_and_since_filter(self):
    old, new = self.run_data(), self.run_data(started="2026-10-04T00:00:00Z")
    new["summary"]["per_request"][0]["decode_tps"] = 75
    data = self.report.sequence_data([old, new], self.suite)
    self.assertEqual(data["points"]["64"]["configs"]["llama.cpp/reference"]["decode_tps"]["median"], 75)
    self.assertEqual(data["points"]["64"]["configs"]["llama.cpp/reference"]["n_requests"], 1)
    self.assertEqual(self.report.sequence_data([old], self.suite, "2026-10-04"), {})

  def test_failed_generation_still_proves_tokenized_input_identity(self):
    run = self.run_data(status="error")
    run["summary"]["per_request"] = []
    data = self.report.sequence_data([run], self.suite)
    self.assertEqual(data["points"]["64"]["prompt_identity"], "identical")
    self.assertEqual(data["points"]["64"]["configs"]["llama.cpp/reference"]["n_requests"], 0)
    del run["meta"]["workload"]["prompt_ids_sha256"]
    data = self.report.sequence_data([run], self.suite)
    self.assertEqual(data["points"]["64"]["prompt_identity"], "unverified")

  def test_sequence_runs_do_not_change_headline(self):
    self.assertEqual(self.report.config_runs([self.run_data()], self.suite, None),
                     {"llama.cpp/reference": []})


if __name__ == "__main__":
  unittest.main()
