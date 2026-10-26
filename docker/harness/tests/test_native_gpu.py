"""Native Apple GPU suite: runner, workload, metrics, drivers and report.

No models, GPU or framework packages are needed: the MLX and BaseRT drivers
run against small stand-ins of their framework modules.
"""

import ctypes
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "scripts"))

from aeb import macos, monitor, perf, workload  # noqa: E402

GPU_SUITE = ROOT.parents[1] / "suites" / "gemma4-26b-a4b-gpu.json"


class NativeSuiteTest(unittest.TestCase):

  def setUp(self):
    if not GPU_SUITE.is_file():
      raise unittest.SkipTest("suites/ not mounted")
    import suite as suite_mod
    self.suite_mod = suite_mod
    self.s, self.ctx = suite_mod.load(GPU_SUITE, "apple-m5-max")

  def tearDown(self):
    self.suite_mod.EXECUTION.update(mode="container", machine_notes="", chat_template="gemma4")

  def test_perf_commands_run_on_host_with_mapped_paths(self):
    cmds = self.suite_mod.perf_commands(self.s, self.ctx, "apple-m5-max", 3, None, False)
    self.assertEqual(len(cmds), 12)
    order = [c["id"] for c in self.s["configs"]]
    self.assertEqual([cid for cid, _ in cmds], order + order[::-1] + order)
    root = str(self.suite_mod.ROOT)
    for cid, cmd in cmds:
      self.assertEqual(cmd[:2], ["env", "-i"])
      self.assertNotIn("docker", cmd)
      self.assertIn("AEB_CHAT_TEMPLATE=gemma4-empty-thought", cmd)
      self.assertIn("AEB_MACHINE=apple-m5-max", cmd)
      self.assertTrue(any(a.startswith("AEB_BUILD_INFO=") and a.endswith("build-info.json")
                          for a in cmd))
      model = cmd[cmd.index("--model") + 1]
      self.assertTrue(model.startswith(root + "/models/"), model)
      self.assertFalse(any("{" in a for a in cmd), cmd)
      self.assertFalse(any(a.startswith(("/models", "/results")) for a in cmd), cmd)
    litert = dict(cmds)["litert-lm/gpu"]
    self.assertEqual(litert[litert.index("--backend") + 1], "gpu")
    self.assertTrue(litert[litert.index("--cache-dir") + 1].startswith(root + "/results/_cache/"))
    llama = dict(cmds)["llama.cpp/metal"]
    self.assertEqual(llama[llama.index("--n-gpu-layers") + 1], "999")

  def test_accuracy_keeps_configured_threads_natively(self):
    acc = self.suite_mod.accuracy_commands(self.s, self.ctx, "apple-m5-max", None, False,
                                           ["mmlu", "gsm8k"], threads=3)
    self.assertEqual(len(acc), 8)
    for _, cmd, _ in acc:
      self.assertEqual(cmd[cmd.index("--threads") + 1], str(self.ctx["threads"]))

  def test_sequence_sweep_fits_context(self):
    cmds = self.suite_mod.sequence_commands(self.s, self.ctx, "apple-m5-max", 3, None, False)
    self.assertEqual(len(cmds), len(self.s["sequence_sweep"]["prompt_tokens"]) * 12)

  def test_host_path_mapping(self):
    hp = self.suite_mod.host_path
    root = str(self.suite_mod.ROOT)
    self.assertEqual(hp("/models/a/b.gguf"), root + "/models/a/b.gguf")
    self.assertEqual(hp("/results"), root + "/results")
    self.assertEqual(hp("/modelsx/a"), "/modelsx/a")
    self.assertEqual(hp("--threads"), "--threads")

  def test_container_suites_unchanged(self):
    path = GPU_SUITE.parent / "gemma4-e2b-cpu.json"
    s, ctx = self.suite_mod.load(path, "apple-m5-max")
    cmds = self.suite_mod.perf_commands(s, ctx, "apple-m5-max", 1, None, False)
    for _, cmd in cmds:
      self.assertEqual(cmd[:2], ["docker", "compose"])
      self.assertNotIn("AEB_CHAT_TEMPLATE=gemma4-empty-thought", " ".join(cmd))


class TemplateTest(unittest.TestCase):

  def test_variants(self):
    saved = workload.CHAT_TEMPLATE
    try:
      workload.CHAT_TEMPLATE = "gemma4"
      self.assertEqual(workload.chat("Hi"), "<|turn>user\nHi<turn|>\n<|turn>model\n")
      workload.CHAT_TEMPLATE = "gemma4-empty-thought"
      self.assertEqual(workload.chat("Hi"),
                       "<|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
    finally:
      workload.CHAT_TEMPLATE = saved


def native_sample(t_ms, cpu_us, util, footprint_kb=1000, energy_nj=0):
  return (t_ms * 1_000_000, cpu_us, 500, None, None, None, None, 8, None, None, 0, 1,
          footprint_kb, footprint_kb, 0, util, 2048, energy_nj)


class NativeMetricsTest(unittest.TestCase):

  def test_gpu_phase_utilization_footprint_energy(self):
    samples = [native_sample(t, t * 1000, 100 if 100 <= t <= 400 else 0,
                             footprint_kb=1024 * (2 if t == 300 else 1), energy_nj=t * 10**6)
               for t in range(0, 1001, 50)]
    r = {"n_prompt": 100, "n_gen": 4, "t_req_ns": 100_000_000, "t_first_ns": 200_000_000,
         "t_end_ns": 500_000_000, "token_ns": [200_000_000, 300_000_000, 400_000_000, 500_000_000]}
    m = perf.request_metrics(r, samples)
    self.assertAlmostEqual(m["gpu_util_prefill"], 1.0)
    self.assertAlmostEqual(m["gpu_util_decode"], 5 / 7)  # samples 200..400 busy, 450/500 idle
    self.assertEqual(m["footprint_max_mb"], 2.0)
    self.assertEqual(m["gpu_mem_max_mb"], 2.0)
    self.assertAlmostEqual(m["cpu_energy_j"], 0.4)
    self.assertIsNone(m["rss_anon_max_mb"])
    self.assertIsNone(m["pss_max_mb"])
    self.assertAlmostEqual(m["cpu_cores_decode"], 1e6 / monitor.CLK_TCK, places=6)

  def test_linux_samples_have_no_gpu_metrics(self):
    samples = [(t * 1_000_000, t, 500, 400, 100, 0, 600, 8, None, None, 0, 1) +
               (None,) * 6 for t in range(0, 1001, 50)]
    r = {"n_prompt": 10, "n_gen": 2, "t_req_ns": 0, "t_first_ns": 100_000_000,
         "t_end_ns": 200_000_000, "token_ns": [100_000_000, 200_000_000]}
    m = perf.request_metrics(r, samples)
    self.assertNotIn("gpu_util_prefill", m)
    self.assertEqual(m["rss_anon_max_mb"], 400 / 1024)
    self.assertIsNone(m["footprint_max_mb"])

  def test_idle_gpu_util_excludes_requests(self):
    samples = [native_sample(t, 0, 90 if 1000 <= t <= 2000 else 10) for t in range(0, 6001, 50)]
    reqs = [{"ok": True, "t_req_ns": 1_000_000_000, "t_end_ns": 2_000_000_000}]
    self.assertAlmostEqual(perf._idle_gpu_util(samples, reqs, 0), 0.10)
    self.assertIsNone(perf._idle_gpu_util(samples, reqs, None))

  def test_peak_handles_missing_linux_columns(self):
    s = native_sample(0, 0, 0, footprint_kb=4096)
    self.assertIsNone(perf._peak(s, [s], 6))
    self.assertEqual(perf._peak(s, [s], 13), 4.0)


class DriverRegistryTest(unittest.TestCase):

  def test_native_drivers_use_harness_interpreter(self):
    from aeb.driver import driver_command
    self.assertEqual(driver_command("mlx", ["--ctx", "8"])[:3],
                     [sys.executable, "-m", "aeb.drivers.mlx_driver"])
    self.assertEqual(driver_command("basert", [])[:3],
                     [sys.executable, "-m", "aeb.drivers.basert_driver"])


class FakeMlxTest(unittest.TestCase):
  """mlx_driver against a stand-in mlx / mlx_lm (EOS and length handling)."""

  def setUp(self):
    self.saved = {k: sys.modules.get(k) for k in
                  ("mlx", "mlx.core", "mlx_lm", "mlx_lm.sample_utils", "aeb.drivers.mlx_driver")}
    core = types.ModuleType("mlx.core")
    core.device_info = lambda: {"device_name": "fake", "architecture": "x", "memory_size": 1,
                                "max_recommended_working_set_size": 1}
    core.set_wired_limit = lambda n: None
    core.eval = lambda *a: None
    core.reset_peak_memory = lambda: None
    core.get_peak_memory = lambda: 123
    core.default_device = lambda: "gpu"
    core.random = types.SimpleNamespace(seed=lambda s: None)
    mlx = types.ModuleType("mlx")
    mlx.core = core

    class Tok:
      eos_token_ids = {1, 106}
      bos_token_id = 2

      def encode(self, text, add_special_tokens=True):
        # Like the Gemma 4 MLX tokenizer: add_special_tokens does not add <bos>.
        return [10] * len(text.split())

    class Model:
      def parameters(self):
        return {}

    script = [5, 6, 106, 7, 8, 9]

    def stream_generate(model, tok, prompt, max_tokens, sampler, prefill_step_size):
      for n, t in enumerate(script):
        if t in tok.eos_token_ids:
          yield types.SimpleNamespace(token=t, text="", finish_reason="stop",
                                      prompt_tps=1.0, generation_tps=1.0)
          return
        last = (n + 1) == max_tokens
        yield types.SimpleNamespace(token=t, text=f"<{t}>",
                                    finish_reason="length" if last else None,
                                    prompt_tps=1.0, generation_tps=1.0)
        if last:
          return

    lm = types.ModuleType("mlx_lm")
    lm.load = lambda path: (Model(), Tok())
    lm.stream_generate = stream_generate
    su = types.ModuleType("mlx_lm.sample_utils")
    su.make_sampler = lambda **kw: kw
    sys.modules.update({"mlx": mlx, "mlx.core": core, "mlx_lm": lm, "mlx_lm.sample_utils": su})
    sys.modules.pop("aeb.drivers.mlx_driver", None)
    self.mod = importlib.import_module("aeb.drivers.mlx_driver")
    args = types.SimpleNamespace(model="m", ctx=64, prefill_step_size=2048)
    self.d = self.mod.Driver(args)

  def tearDown(self):
    for k, v in self.saved.items():
      if v is None:
        sys.modules.pop(k, None)
      else:
        sys.modules[k] = v

  def test_stops_on_stop_token_without_counting_it(self):
    r = self.d.generate({"prompt": "a b c", "max_tokens": 5, "stop_ids": [106]})
    self.assertEqual(r["gen_ids"], [5, 6])
    self.assertTrue(r["stopped_on_eos"])
    self.assertEqual(r["n_prompt"], 4)
    self.assertEqual(len(r["token_ns"]), 2)
    self.assertEqual(self.d.tokenizer.eos_token_ids, {1, 106})  # restored

  def test_bos_added_once(self):
    self.assertEqual(self.d.tokenize("a b", True), [2, 10, 10])
    self.assertEqual(self.d.tokenize("a b", False), [10, 10])

  def test_ignore_eos_decodes_exactly_max_tokens(self):
    r = self.d.generate({"prompt": "a", "max_tokens": 5, "ignore_eos": True})
    self.assertEqual(r["gen_ids"], [5, 6, 106, 7, 8])
    self.assertFalse(r["stopped_on_eos"])
    self.assertEqual(r["text"], "<5><6><106><7><8>")

  def test_context_limit(self):
    with self.assertRaisesRegex(RuntimeError, "exceeds context"):
      self.d.generate({"prompt": "a", "max_tokens": 64})


class FakeBaseRTTest(unittest.TestCase):
  """basert_driver against a stand-in baseRT binding (stop/ignore-EOS handling)."""

  def setUp(self):
    self.saved = {k: sys.modules.get(k) for k in ("baseRT", "aeb.drivers.basert_driver")}
    cb_type = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_void_p)

    class Sampling(ctypes.Structure):
      _fields_ = [("temperature", ctypes.c_float), ("top_k", ctypes.c_int),
                  ("top_p", ctypes.c_float), ("min_p", ctypes.c_float),
                  ("repeat_penalty", ctypes.c_float), ("presence_penalty", ctypes.c_float),
                  ("frequency_penalty", ctypes.c_float), ("seed", ctypes.c_uint32),
                  ("n_logit_bias", ctypes.c_int32),
                  ("logit_bias_tokens", ctypes.POINTER(ctypes.c_int32)),
                  ("logit_bias_values", ctypes.POINTER(ctypes.c_float))]

    calls = self.calls = []

    class Fn:
      def __init__(self, f):
        self.f, self.restype, self.argtypes = f, None, None

      def __call__(self, *a):
        return self.f(*a)

    script = {"generate": [5, 106, 7, 8], "continue": [9, 50, 11, 12, 13]}

    def run(kind, arr, n, max_tokens, cb):
      # Like the engine: a stop id ends generation and is not passed to the callback.
      calls.append({"kind": kind, "prompt": list(arr[:n]), "max": max_tokens})
      count = 0
      while script[kind] and count < max_tokens:
        t = script[kind].pop(0)  # the model keeps going from where it stopped
        count += 1
        if t in (1, 50, 106) or not cb(t, b"x", None):
          break
      return types.SimpleNamespace(prompt_tokens=n, generated_tokens=count, prefill_time_ms=1.0,
                                   decode_time_ms=1.0, prefill_tokens_per_sec=1.0,
                                   decode_tokens_per_sec=1.0)

    def generate(h, arr, n, max_tokens, cfg, cb, ud):
      return run("generate", arr, n, max_tokens, cb)

    def generate_continue(h, arr, n, max_tokens, cfg, cb, ud):
      return run("continue", arr, n, max_tokens, cb)

    def encode(h, text, buf, cap):
      ids = [2] + [10] * len(text.split())
      for i, t in enumerate(ids):
        buf[i] = t
      return len(ids)

    lib = types.SimpleNamespace(
        baseRT_version_string=Fn(lambda: b"0.4.0"), baseRT_eos_token_id=Fn(lambda h: 1),
        baseRT_bos_id=Fn(lambda h: 2), baseRT_add_bos=Fn(lambda h: 1),
        baseRT_is_eos_token=Fn(lambda h, t: int(t in (1, 106))), baseRT_reset=Fn(lambda h: None),
        baseRT_has_speculator=Fn(lambda h: 0), baseRT_get_speculation=Fn(lambda h: False),
        baseRT_encode=encode, baseRT_generate=generate,
        baseRT_generate_continue=generate_continue)

    class Model:
      def __init__(self, path, max_context=0):
        self._handle = 1

      @property
      def config(self):  # a property in the real binding
        return types.SimpleNamespace(vocab_size=200)

    fake = types.ModuleType("baseRT")
    fake._get_lib = lambda: lib
    fake.Model = Model
    fake.get_error = lambda: None
    fake.BASERT_TOKEN_CALLBACK = cb_type
    fake.BaseRTSamplingConfig = Sampling
    sys.modules["baseRT"] = fake
    sys.modules.pop("aeb.drivers.basert_driver", None)
    mod = importlib.import_module("aeb.drivers.basert_driver")
    self.d = mod.Driver(types.SimpleNamespace(model="m.base", ctx=64))

  def tearDown(self):
    for k, v in self.saved.items():
      if v is None:
        sys.modules.pop(k, None)
      else:
        sys.modules[k] = v

  def test_eos_ids_and_tokenize(self):
    self.assertEqual(self.d.eos_ids, [1, 106])
    self.assertEqual(self.d.tokenize("a b", True), [2, 10, 10])
    self.assertEqual(self.d.tokenize("a b", False), [10, 10])

  def test_stop_token_ends_generation(self):
    r = self.d.generate({"prompt": "a", "max_tokens": 4, "stop_ids": [50]})
    self.assertEqual(r["gen_ids"], [5])
    self.assertTrue(r["stopped_on_eos"])
    self.assertEqual(r["eos_resumed"], 0)
    self.assertEqual([c["kind"] for c in self.calls], ["generate"])
    self.assertEqual(self.calls[-1]["prompt"], [2, 10])

  def test_ignore_eos_feeds_stop_token_back(self):
    r = self.d.generate({"prompt": "a", "max_tokens": 7, "ignore_eos": True, "stop_ids": [50]})
    # 5, <stop> (counted, fed back as the primary EOS 1), 9, <stop>, 11, 12, 13
    self.assertEqual(r["gen_ids"], [5, 1, 9, 1, 11, 12, 13])
    self.assertEqual(r["n_gen"], 7)
    self.assertEqual(len(r["token_ns"]), 7)
    self.assertEqual(r["eos_resumed"], 2)
    self.assertFalse(r["stopped_on_eos"])
    self.assertEqual([(c["kind"], c["prompt"], c["max"]) for c in self.calls],
                     [("generate", [2, 10], 7), ("continue", [1], 5), ("continue", [1], 3)])

  def test_ignore_eos_without_stop_is_one_call(self):
    r = self.d.generate({"prompt": "a", "max_tokens": 1, "ignore_eos": True})
    self.assertEqual(r["gen_ids"], [5])
    self.assertEqual(len(self.calls), 1)


class GpuCheckTest(unittest.TestCase):

  def setUp(self):
    import datetime as dt
    from aeb import gpucheck
    self.g, self.dt = gpucheck, dt
    utc = dt.timezone.utc
    t = lambda s: dt.datetime(2026, 10, 22, 19, 22, s, tzinfo=utc)
    self.t = t
    self.errors = [
        {"t": t(33), "process": "python3.12", "pid": 30852,
         "kind": "kIOGPUCommandBufferCallbackErrorHang", "message": "Caused GPU Hang Error"},
        {"t": t(33), "process": "WindowServer", "pid": 173,
         "kind": "kIOGPUCommandBufferCallbackErrorInnocentVictim", "message": "victim"},
        {"t": t(40), "process": "python3.12", "pid": 30852,
         "kind": "kIOGPUCommandBufferCallbackErrorSubmissionsIgnored",
         "message": "Ignored (for causing prior/excessive GPU errors)"},
    ]

  def test_attributes_errors_to_the_run_window_and_pid(self):
    c = self.g.summarize(self.errors, self.t(30), self.t(50), pid=30852)
    self.assertEqual(c["errors"], 2)
    self.assertEqual(c["recoveries_other_processes"], 1)
    self.assertTrue(c["root_cause"].startswith("kIOGPUCommandBufferCallbackErrorHang"))
    self.assertIn("2 aborted Metal command buffers", self.g.error_text(c))

  def test_other_windows_and_pids_are_clean(self):
    self.assertEqual(self.g.summarize(self.errors, self.t(0), self.t(30), pid=30852)["errors"], 0)
    self.assertEqual(self.g.summarize(self.errors, self.t(30), self.t(50), pid=1)["errors"], 0)
    self.assertIsNone(self.g.error_text({"errors": 0}))

  def test_without_pid_matches_driver_process_names(self):
    self.assertEqual(self.g.summarize(self.errors, self.t(30), self.t(50))["errors"], 2)

  def test_report_marks_checked_runs_as_errors(self):
    try:
      from aeb import report
    except ImportError:
      raise unittest.SkipTest("matplotlib not installed")
    with tempfile.TemporaryDirectory() as tmp:
      d = Path(tmp)
      (d / "gpu_check.json").write_text(json.dumps(
          self.g.summarize(self.errors, self.t(30), self.t(50), pid=30852)))
      meta, summ = report._gpu_checked(d, {"framework": "x"}, {"status": "ok"})
      self.assertEqual(summ["status"], "error")
      self.assertIn("GPU error", meta["error"])
      (d / "gpu_check.json").write_text(json.dumps({"errors": 0}))
      self.assertEqual(report._gpu_checked(d, {}, {"status": "ok"})[1]["status"], "ok")


@unittest.skipUnless(macos.IS_DARWIN, "macOS only")
class MacProbeTest(unittest.TestCase):

  def test_self_probes(self):
    import os
    r = macos.proc_rusage(os.getpid())
    self.assertGreater(r["cpu_ns"], 0)
    self.assertGreater(r["footprint_bytes"], 0)
    self.assertGreaterEqual(r["footprint_max_bytes"], r["footprint_bytes"])
    self.assertGreaterEqual(macos.proc_threads(os.getpid()), 1)
    busy, total = macos.host_cpu_ticks()
    self.assertGreater(total, busy)
    s = monitor.Monitor(os.getpid()).sample()
    self.assertEqual(len(s), len(monitor.FIELDS))
    self.assertIsNone(macos.proc_rusage(10**7))


class NativeReportTest(unittest.TestCase):
  """Renders a native-suite report from a synthetic run directory."""

  def test_native_report_renders(self):
    try:
      from aeb import report
    except ImportError:
      raise unittest.SkipTest("matplotlib not installed")
    if not GPU_SUITE.is_file():
      raise unittest.SkipTest("suites/ not mounted")
    suite = json.loads(GPU_SUITE.read_text())
    with tempfile.TemporaryDirectory() as tmp:
      tmp = Path(tmp)
      results, out = tmp / "results", tmp / "reports"
      report_dir = out / f"2026-10-22-{suite['name']}-m"
      report_dir.mkdir(parents=True)
      (report_dir / "thread_scaling.png").write_bytes(b"stale chart")
      for k, c in enumerate(suite["configs"]):
        fw, label = c["framework"], c["id"].split("/", 1)[1]
        for rnd in range(3):
          d = results / "m" / "perf" / fw / label / f"run{rnd}"
          d.mkdir(parents=True)
          reqs = []
          for i in range(3):
            t0 = (i + 1) * 2_000_000_000
            toks = [t0 + 500_000_000 + j * 10_000_000 for j in range(256)]
            reqs.append({"ok": True, "index": i, "measured": i > 0, "n_prompt": 1024,
                         "n_gen": 256, "t_req_ns": t0, "t_first_ns": toks[0],
                         "t_end_ns": toks[-1], "token_ns": toks, "text": "same"})
          (d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
          samples = [native_sample(t, t * 900, 95 if t % 2000 > 400 else 5)
                     for t in range(0, 9000, 50)]
          with open(d / "timeseries.csv", "w") as f:
            f.write(",".join(monitor.FIELDS) + "\n")
            for s in samples:
              f.write(",".join("" if v is None else str(v) for v in s) + "\n")
          per = [dict(perf.request_metrics(r, samples), index=r["index"], measured=r["measured"])
                 for r in reqs]
          meta = {"framework": fw, "label": label, "track": c["track"], "round": rnd,
                  "notes": f"suite={suite['name']}", "started_at": f"2026-10-22T10:0{k}:{rnd}0Z",
                  "machine_notes": "native", "driver_cmd": ["python", "-m", "x", "--model",
                                                            str(tmp / "models" / "m.bin")],
                  "model_path": str(tmp / "models" / "m.bin"), "model_sha256": "ab" * 32,
                  "build": {"ref": "v1", "commit": None, "harness_sha256": "cd" * 32,
                            "repo_rev": "r"}, "clk_tck": monitor.CLK_TCK,
                  "workload": {"prompt_ids_sha256": "ef" * 32}}
          summ = {"status": "ok", "per_request": per, "metrics": {}, "load_ms": 1000,
                  "peak_footprint_mb_lifetime": 100, "gpu_util_idle": 0.05,
                  "idle_after_load": {"rss_mb": 1, "footprint_mb": 2}, "outliers_e2e": []}
          (d / "meta.json").write_text(json.dumps(meta))
          (d / "summary.json").write_text(json.dumps(summ))
          (d / "sysinfo.json").write_text(json.dumps({"host": {"os": "macOS 27"}, "cpu": {},
                                                      "gpu": {"cores": "40"}, "power": {}}))
      rc = report.main(["--machine", "m", "--suite", str(GPU_SUITE), "--results", str(results),
                        "--out", str(out), "--date", "2026-10-22"])
      self.assertEqual(rc, 0)
      md = (out / f"2026-10-22-{suite['name']}-m" / "README.md").read_text()
      self.assertIn("Host (native, no container)", md)
      self.assertIn("GPU utilization, decode (system counter)", md)
      self.assertIn("Ratio to `litert-lm/gpu`", md)
      self.assertIn("![GPU](gpu.png)", md)
      self.assertNotIn(str(tmp), md)
      self.assertNotIn("Container memory.peak", md)
      page = (out / f"2026-10-22-{suite['name']}-m" / "README.html").read_text(encoding="utf-8")
      self.assertIn("<!DOCTYPE html>", page)
      self.assertIn("<table>", page)
      self.assertIn('alt="GPU"', page)
      self.assertIn('src="data:image/png;base64,', page)
      self.assertEqual(page, (out / f"2026-10-22-{suite['name']}-m" / "index.html").read_text())
      self.assertIn("Host (native, no container)", page)
      self.assertNotIn(str(tmp), page)
      sections = ["## Summary", "## Results", "### Throughput", "### Latency", "### GPU",
                  "### CPU", "### Memory", "### Memory, CPU and GPU over time",
                  "### Detailed metrics", "### Generated text on the timing prompt",
                  "## Environment", "## Configurations", "## Data-quality checks",
                  "## Caveats", "## Reproducing these results"]
      positions = [md.index(section + "\n") for section in sections]
      self.assertEqual(positions, sorted(positions))
      self.assertNotIn("### Prefix caching", md)
      self.assertNotIn("### Thread scaling", md)
      self.assertFalse((report_dir / "thread_scaling.png").exists())
      self.assertIn("GPU memory Δ (MB)", md)
      self.assertNotIn("int2_int4_int8 / int8 / int8", md)
      self.assertNotIn("**`matched`**:", md)
      for heading in ("### 1. Exact framework & repository versions used in this report",
                      "### 2. Building the native framework stack (pinned)",
                      "### 3. Machine setup & running the full benchmark suite",
                      "### 4. Single-configuration performance commands"):
        self.assertIn(heading, md)
      self.assertIn("native-report DATE=2026-10-22", md)
      self.assertNotIn("docker compose", md)
      for png in ("gpu.png", "timeline.png", "memory.png", "throughput.png"):
        self.assertTrue((out / f"2026-10-22-{suite['name']}-m" / png).is_file(), png)
      cpu_suite = dict(suite, execution="container",
                       configs=[c for c in suite["configs"] if c["framework"] in ("llama.cpp", "litert-lm")])
      cpu_path = tmp / "cpu-suite.json"
      cpu_path.write_text(json.dumps(cpu_suite))
      cpu_out = tmp / "cpu-reports"
      self.assertEqual(report.main(["--machine", "m", "--suite", str(cpu_path),
                                    "--results", str(results), "--out", str(cpu_out),
                                    "--date", "2026-10-22"]), 0)
      cpu_dir = cpu_out / f"2026-10-22-{suite['name']}-m"
      cpu_md = (cpu_dir / "README.md").read_text()
      cpu_page = (cpu_dir / "README.html").read_text()
      self.assertIn("Peak RSS (MB)", cpu_md)
      self.assertNotIn("### GPU\n", cpu_md)
      self.assertNotIn("### Prefix caching", cpu_md)
      self.assertNotIn("### Thread scaling", cpu_md)
      self.assertEqual(page.split("<style>")[1].split("</style>")[0],
                       cpu_page.split("<style>")[1].split("</style>")[0])
      self.assertEqual(cpu_page, (cpu_dir / "index.html").read_text())
      report.HEADLINE = report.CPU_HEADLINE


if __name__ == "__main__":
  unittest.main()
