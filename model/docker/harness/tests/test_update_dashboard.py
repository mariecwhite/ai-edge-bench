"""Unit tests for model/scripts/update_dashboard.py."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT.parents[1] / "scripts", Path("/opt/scripts")):
  if candidate.is_dir() and str(candidate) not in sys.path:
    sys.path.insert(0, str(candidate))

import update_dashboard  # noqa: E402


class UpdateDashboardTest(unittest.TestCase):

  def test_multiday_reported_runs_and_delegate_scoped_accuracy(self):
    with tempfile.TemporaryDirectory() as tmp:
      tmp_p = Path(tmp)
      suites_dir = tmp_p / "suites"
      reports_dir = tmp_p / "reports"
      results_dir = tmp_p / "results"
      suites_dir.mkdir()
      reports_dir.mkdir()
      results_dir.mkdir()

      suite = {
          "name": "gemma4-31b-cpu-sapphire-rapids",
          "description": "Gemma 4 31B CPU suite",
          "rounds": 1,
          "workload": {"prompt_tokens": 1024, "gen_tokens": 128, "ctx": 4096, "repetitions": 2},
          "configs": [
              {"id": "litert-lm/fastest", "framework": "litert-lm", "track": "fastest", "dtype": "int4"},
              {"id": "litert-lm/matched", "framework": "litert-lm", "track": "matched", "dtype": "int4"},
              {"id": "litert-lm/ynnpack-fastest", "framework": "litert-lm", "track": "reference", "dtype": "int4"},
              {"id": "litert-lm/ynnpack", "framework": "litert-lm", "track": "reference", "dtype": "int4"},
              {"id": "xla/bf16-onednn", "framework": "xla", "track": "reference", "dtype": "bf16", "disabled": True},
          ],
      }
      (suites_dir / "gemma4-31b-cpu-sapphire-rapids.json").write_text(json.dumps(suite), encoding="utf-8")

      shared_sha = "abc123sha256"
      run_dir_name = "20261117T150000Z-112233"

      # Raw perf run on 2026-11-17 that was later included in the 2026-11-18 report
      raw_run = results_dir / "sapphire-rapids" / "perf" / "litert-lm" / "fastest" / run_dir_name
      raw_run.mkdir(parents=True)
      (raw_run / "meta.json").write_text(
          json.dumps({
              "framework": "litert-lm",
              "label": "fastest",
              "machine": "sapphire-rapids",
              "track": "fastest",
              "notes": "suite=gemma4-31b-cpu-sapphire-rapids",
              "started_at": "2026-11-17T15:00:00Z",
              "model_path": "/models/litert-community/gemma-4-31B-it.litertlm",
              "model_sha256": shared_sha,
              "driver_ready": {"info": {"threads": 56}},
              "build": {"ref": "main", "commit": "b5391c5"},
          }),
          encoding="utf-8",
      )
      (raw_run / "summary.json").write_text(
          json.dumps({
              "status": "ok",
              "per_request": [
                  {"measured": True, "e2e_ms": 30000.0, "ttft_ms": 6500.0, "prefill_tps": 157.0, "decode_tps": 5.35}
              ],
          }),
          encoding="utf-8",
      )

      # Disabled config raw run on 2026-11-19 must also be ignored
      dis_run = results_dir / "sapphire-rapids" / "perf" / "xla" / "bf16-onednn" / "20261119T010000Z-dead00"
      dis_run.mkdir(parents=True)
      (dis_run / "meta.json").write_text(
          json.dumps({
              "framework": "xla",
              "label": "bf16-onednn",
              "machine": "sapphire-rapids",
              "track": "reference",
              "notes": "suite=gemma4-31b-cpu-sapphire-rapids",
              "started_at": "2026-11-19T01:00:00Z",
          }),
          encoding="utf-8",
      )
      (dis_run / "summary.json").write_text(
          json.dumps({
              "status": "ok",
              "per_request": [{"measured": True, "e2e_ms": 1000.0, "prefill_tps": 100.0, "decode_tps": 1.0}],
          }),
          encoding="utf-8",
      )

      rep_id = "2026-11-18-gemma4-31b-cpu-sapphire-rapids-sapphire-rapids"
      rep_dir = reports_dir / rep_id
      rep_dir.mkdir(parents=True)
      (rep_dir / "index.html").write_text("<html></html>", encoding="utf-8")
      (rep_dir / "README.md").write_text("# report", encoding="utf-8")

      def _cfg_entry(cid, threads, prefill, decode, r_name):
        return {
            "runs": [{
                "run_dir": f"/results/sapphire-rapids/perf/{cid}/{r_name}",
                "meta": {
                    "framework": "litert-lm",
                    "started_at": "2026-11-17T15:00:00Z",
                    "model_path": "/models/litert-community/gemma-4-31B-it.litertlm",
                    "model_sha256": shared_sha,
                    "driver_ready": {"info": {"threads": threads}},
                    "build": {"ref": "main", "commit": "b5391c5"},
                },
            }],
            "aggregate": {
                "e2e_ms": {"n": 2, "median": 30000.0, "min": 29900.0, "max": 30100.0, "cv": 0.003},
                "ttft_ms": {"n": 2, "median": 6500.0, "min": 6400.0, "max": 6600.0, "cv": 0.01},
                "prefill_tps": {"n": 2, "median": prefill, "min": prefill, "max": prefill, "cv": 0.0},
                "decode_tps": {"n": 2, "median": decode, "min": decode, "max": decode, "cv": 0.0},
            },
        }

      rep_data = {
          "report_id": rep_id,
          "suite": suite,
          "machine": "sapphire-rapids",
          "configs": {
              "litert-lm/fastest": _cfg_entry("litert-lm/fastest", 56, 157.0, 5.35, run_dir_name),
              "litert-lm/matched": _cfg_entry("litert-lm/matched", 32, 125.2, 5.56, "20261117T151000Z-223344"),
              "litert-lm/ynnpack-fastest": _cfg_entry("litert-lm/ynnpack-fastest", 56, 106.2, 3.97, "20261117T152000Z-334455"),
              "litert-lm/ynnpack": _cfg_entry("litert-lm/ynnpack", 32, 76.9, 3.71, "20261117T153000Z-445566"),
          },
          "prefix_cache": {
              "configs": {
                  "litert-lm/fastest": {
                      "warm_ttft_ms": {"median": 1377.5},
                      "warm_effective_prefill_tps": {"median": 743.4},
                      "ttft_speedup": [4.74, 4.4, 4.85],
                  }
              }
          },
          "accuracy": [
              {"id": "litert-lm/fastest", "task": "mmlu", "accuracy": 0.71, "n": 200},
              {"id": "litert-lm/fastest", "task": "gsm8k", "accuracy": 0.94, "n": 50},
              {"id": "litert-lm/ynnpack-fastest", "task": "mmlu", "accuracy": 0.715, "n": 200},
              {"id": "litert-lm/ynnpack-fastest", "task": "gsm8k", "accuracy": 0.92, "n": 50},
          ],
      }
      (rep_dir / "data.json").write_text(json.dumps(rep_data), encoding="utf-8")

      dash = update_dashboard.collect_dashboard_data(suites_dir, reports_dir, results_dir)
      self.assertEqual(len(dash["suites"]), 1)
      s_out = dash["suites"][0]
      # Must NOT include a phantom 2026-11-17 report or disabled 2026-11-19 run
      self.assertEqual([r["date"] for r in s_out["reports"]], ["2026-11-18"])

      by_id = {c["id"]: c for c in s_out["configs"]}
      self.assertNotIn("xla/bf16-onednn", by_id)
      self.assertEqual(by_id["litert-lm/matched"]["checkpoint"], "qat-q4_0")
      self.assertEqual(by_id["litert-lm/matched"]["weights_acts_kv"], "int4_int8 / fp32 / fp32")
      # XNNPACK matched must inherit XNNPACK accuracy (0.71 / 0.94), not YNNPACK (0.715 / 0.92)
      self.assertEqual(by_id["litert-lm/matched"]["points"][0]["mmlu"], 0.71)
      self.assertEqual(by_id["litert-lm/matched"]["points"][0]["gsm8k"], 0.94)
      self.assertEqual(by_id["litert-lm/ynnpack"]["points"][0]["mmlu"], 0.715)
      self.assertEqual(by_id["litert-lm/ynnpack"]["points"][0]["gsm8k"], 0.92)
      self.assertEqual(by_id["litert-lm/fastest"]["points"][0]["warm_ttft_ms"], 1377.5)

      html_out = update_dashboard.render_dashboard_html(dash)
      md_out = update_dashboard.render_dashboard_markdown(dash)
      self.assertIn("Warm TTFT (ms)", html_out)
      self.assertIn("Warm TTFT (ms) ↓", md_out)


if __name__ == "__main__":
  unittest.main()
