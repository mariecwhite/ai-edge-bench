"""Offline HTML rendering shared by native and container reports."""

import unittest
import base64
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

from aeb import report
from aeb.report import _config_dtype_info, markdown_to_html, summary_by_dtype_section


class PageParser(HTMLParser):
  def __init__(self, text):
    super().__init__(convert_charrefs=True)
    self.elements = []
    self.text = []
    self.feed(text)

  def handle_starttag(self, tag, attrs):
    self.elements.append((tag, dict(attrs)))

  def handle_data(self, data):
    self.text.append(data)


class ReportHtmlTest(unittest.TestCase):
  def test_markdown_features_and_local_assets(self):
    page = markdown_to_html("""# GPU report

| Framework | Speed |
| --- | --- |
| MLX | 118.5 |

![GPU](gpu.png)

[Methodology](../../docs/benchmark_methodology.md)

[Markdown source](README.md) / [Raw data](data.json)

Use `a < b & c`.

```sh
echo "<tag> & value"
```

1. First
2. Second
""", Path("."), embed_images=False)
    parsed = PageParser(page)
    tags = [tag for tag, _ in parsed.elements]
    for tag in ("html", "head", "title", "main", "h1", "table", "th", "td", "pre", "code", "ol"):
      self.assertIn(tag, tags)
    self.assertIn(("html", {"lang": "en"}), parsed.elements)
    self.assertIn(("h1", {"id": "gpu-report"}), parsed.elements)
    self.assertIn(("img", {"alt": "GPU", "src": "gpu.png", "loading": "lazy"}), parsed.elements)
    links = [attrs["href"] for tag, attrs in parsed.elements if tag == "a"]
    self.assertIn("../../docs/benchmark_methodology.md", links)
    self.assertIn("README.md", links)
    self.assertIn("data.json", links)
    self.assertIn("GPU report", parsed.text)
    self.assertIn("a < b & c", parsed.text)
    self.assertIn('echo "<tag> & value"', parsed.text)
    self.assertNotIn("tag", tags)
    self.assertNotIn("report", tags)
    self.assertNotIn("script", tags)
    self.assertNotIn("link", tags)
    self.assertIn('name="viewport"', page)

  def test_conversion_has_no_state_between_reports(self):
    first = markdown_to_html("# Same\n\n# Same", Path("."))
    second = markdown_to_html("# Same", Path("."))
    self.assertIn('id="same_1"', first)
    self.assertIn('id="same"', second)
    self.assertNotIn('id="same_1"', second)

  def test_embedded_images_and_standard_table_highlighting(self):
    image = b"\x89PNG\r\n\x1a\n"
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      (root / "gpu.png").write_bytes(image)
      page = markdown_to_html("""# GPU <report> & "results"

| Framework | Speed |
| --- | ---: |
| **INT4 / Mixed Track** | |
| 🏆 **`MLX`** | **118.5** |
| BaseRT | 77.6 |

![GPU](gpu.png)
""", root)
    parsed = PageParser(page)
    self.assertIn('GPU <report> & "results"', parsed.text)
    self.assertIn(("tr", {"class": "group-row"}), parsed.elements)
    self.assertIn(("tr", {"class": "best-row"}), parsed.elements)
    self.assertIn(("div", {"class": "table-wrap"}), parsed.elements)
    images = [attrs["src"] for tag, attrs in parsed.elements if tag == "img"]
    self.assertEqual(images, ["data:image/png;base64," + base64.b64encode(image).decode("ascii")])
    self.assertNotIn("<report>", page)
    self.assertIn("--bg: #ffffff", page)

  def test_escaped_pipes_do_not_split_table_cells(self):
    page = markdown_to_html("| Text | Value |\n| --- | --- |\n| a\\|b | `x\\|y` |", Path("."))
    parsed = PageParser(page)
    self.assertEqual(sum(tag == "td" for tag, _ in parsed.elements), 2)
    self.assertIn("a|b", parsed.text)
    self.assertIn("x|y", parsed.text)


class ReportSummaryTest(unittest.TestCase):
  def test_native_precision_requires_explicit_metadata(self):
    suite = {"execution": "native", "configs": [{"id": "litert-lm/gpu"}]}
    meta = {"framework": "litert-lm", "model_path": "/models/gpu.litertlm",
            "driver_ready": {"info": {"threads": 6}}}
    self.assertEqual(_config_dtype_info(suite, "litert-lm/gpu", meta),
                     ("int4", "`not exposed`", "`not exposed / not exposed / not exposed`", "6"))
    suite["configs"][0]["report"] = {"checkpoint": "it (GPU)", "weights": "4-bit",
                                     "kv_cache": "fp16", "thread_usage": "host"}
    self.assertEqual(_config_dtype_info(suite, "litert-lm/gpu", meta),
                     ("int4", "`it (GPU)`", "`4-bit / not exposed / fp16`", "6 (host)"))

  def test_native_summary_memory_ordering_and_best_values(self):
    suite = {"execution": "native", "workload": {"prompt_tokens": 1024, "gen_tokens": 256},
             "configs": [{"id": cid, "track": "reference", "dtype": "int4"}
                         for cid in ("mlx/metal", "basert/metal")]}
    runs = {c["id"]: [{"meta": {"framework": c["id"].split("/")[0]}}] for c in suite["configs"]}
    agg = {"mlx/metal": {"e2e_ms": {"median": 2000}, "decode_tps": {"median": 120},
                         "gpu_mem_delta_mb": {"median": 15000}, "rss_max_mb": {"median": 999}},
           "basert/metal": {"e2e_ms": {"median": 3000}, "decode_tps": {"median": 80},
                            "gpu_mem_delta_mb": {"median": 14000}, "rss_max_mb": {"median": 888}}}
    md = summary_by_dtype_section(suite, ["basert/metal", "mlx/metal"], runs, agg, [])
    self.assertLess(md.index("| 🏆 **`mlx/metal`**"), md.index("| `basert/metal`"))
    self.assertIn("GPU memory Δ (MB)", md)
    mlx_row = next(line for line in md.splitlines() if line.startswith("| 🏆"))
    basert_row = next(line for line in md.splitlines() if line.startswith("| `basert"))
    self.assertIn("| 15,000 |", mlx_row)
    self.assertNotIn("**15,000**", mlx_row)
    self.assertIn("| **14,000** |", basert_row)
    self.assertIn("**120.00**", mlx_row)
    self.assertNotIn("999", md)
    self.assertNotIn("888", md)
    self.assertNotIn("**`matched`**:", md)
    self.assertNotIn("**`fastest`**:", md)
    self.assertIn("**`reference`**:", md)
    self.assertNotIn("pareto.png", md)


class ReportChartTest(unittest.TestCase):
  def test_thread_sweeps_must_match_model_platform_and_suite(self):
    suite = {"name": "gpu"}
    headline = {"framework": "llama.cpp", "model_sha256": "26b", "platform": "darwin"}
    cfg_runs = {"llama.cpp/metal": [{"meta": headline}]}

    def run(**meta):
      return {"meta": dict(headline, track="sweep-threads", started_at="2026-10-22", **meta),
              "summary": {"status": "ok"}}

    untagged = run()
    tagged = run(notes="suite=gpu")
    cpu_model = run(model_sha256="e2b", platform="linux")
    cpu_platform = run(platform="linux")
    wrong_suite = run(notes="suite=gpu-other")
    wrong_tagged_model = run(notes="suite=gpu", model_sha256="e2b")
    no_model = run(model_sha256=None)
    failed = run()
    failed["summary"]["status"] = "error"
    runs = [untagged, tagged, cpu_model, cpu_platform, wrong_suite, wrong_tagged_model, no_model, failed]
    self.assertEqual(report.thread_scaling_runs(runs, suite, cfg_runs), [untagged, tagged])
    self.assertEqual(report.thread_scaling_runs(runs, suite, cfg_runs, "2026-10-23"), [])
    legacy = {"meta": {"framework": "llama.cpp", "model_sha256": "e2b",
                       "track": "sweep-threads"}, "summary": {"status": "ok"}}
    self.assertEqual(report.thread_scaling_runs([legacy], suite, {"cpu": [legacy]}), [legacy])

  def test_native_timeline_combines_configs_and_uses_recorded_clock(self):
    cfg_ids = ["mlx/metal", "basert/metal"]
    runs = {cid: [{"dir": Path(cid), "meta": {"clk_tck": 1_000_000}}] for cid in cfg_ids}
    samples = {"t_ns": [0, 1_000_000_000, 2_000_000_000],
               "cpu_ticks": [0, 500_000, 1_000_000], "rss_kb": [1024, 2048, 3072],
               "footprint_kb": [512, 1024, None], "gpu_util_pct": [0, 50, 100]}
    figures = []
    subplots = report.plt.subplots

    def capture_subplots(*args, **kwargs):
      fig, axes = subplots(*args, **kwargs)
      figures.append((fig, axes))
      return fig, axes

    with tempfile.TemporaryDirectory() as tmp, \
         patch.object(report, "read_timeseries", return_value=samples), \
         patch.object(report.plt, "subplots", side_effect=capture_subplots):
      path = Path(tmp) / "timeline.png"
      report.native_timeline_chart(path, cfg_ids, runs)
      self.assertTrue(path.is_file())
    self.assertEqual(len(figures), 1)
    _, axes = figures[0]
    self.assertEqual(len(axes), 3)
    self.assertEqual([len(ax.lines) for ax in axes], [4, 2, 2])
    self.assertEqual(list(axes[0].lines[0].get_ydata())[:2], [0.5, 1.0])
    self.assertEqual(list(axes[1].lines[0].get_ydata()), [0.5, 0.5])
    self.assertEqual(list(axes[2].lines[0].get_ydata()), [0, 50, 100])
    self.assertEqual(axes[0].lines[0].get_color(), axes[1].lines[0].get_color())
    self.assertEqual(axes[1].lines[0].get_color(), axes[2].lines[0].get_color())
    self.assertEqual([line.get_label() for line in axes[1].lines], cfg_ids)


if __name__ == "__main__":
  unittest.main()
