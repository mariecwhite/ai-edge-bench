"""Unit tests for kernel/scripts/extract_litert_lm_kernels.py."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "kernel" / "scripts"))

import extract_litert_lm_kernels as ext  # noqa: E402


SAMPLE_XNNPACK_TFLITE_LOG = """\
I0000 00:00:1794223443.642315 2224572 litert_lm_lib.cc:461] BenchmarkInfo:
  Init Phases (7):
    - Init Conversation: 0.11 ms
    - Init Executor: 246.83 ms
  TimeToFirstToken: 0.425 s
  PrefillSpeed: 298.50 tokens/sec
  DecodeSpeed: 8.12 tokens/sec
--------------------------------------------------
  Profile Summary:
====================================================== Summary by node type ======================================================
                                                 [Node type]    [count]   [avg ms]    [avg %]    [cdf %]   [mem KB] [times called]
         Delegate/Fully Connected (NC, QDU8, F32, QC4W) GEMM        420     12.400    84.500%    84.500%      0.000          54180
               Delegate/Batch Matrix Multiply (NC, F32) GEMM        120      0.950     6.100%    90.600%      0.000          15480
                                       Delegate/Convert (NC)        540      0.045     3.200%    93.800%      0.000          69660
                                                    SELECT_V2        120      0.110     1.800%    95.600%      0.000          15480
                                   Delegate/Softmax (NC, F32)         60      0.210     1.500%    97.100%      0.000           7740
                              Delegate/Normalize (ND) Reduce2        300      0.032     1.300%    98.400%      0.000          38700
                                         DYNAMIC_UPDATE_SLICE        120      0.080     1.600%   100.000%      0.000          15480

========================================================= Delegate Statistics =========================================================
Op Name                                                                               Count    Avg(us)    Min(us)    Max(us)  Total(us)
Delegate/Batch Matrix Multiply (NC, F32) GEMM                                         15480      950.0         80       9200   14706000
Delegate/Convert (NC)                                                                 69660       45.0          1        420    3134700
Delegate/Fully Connected (NC, QDU8, F32, QC4W) GEMM                                   54180    12400.0        210     185000  671832000
Delegate/Normalize (ND) Reduce2                                                       38700       32.0          2        150    1238400
Delegate/Softmax (NC, F32)                                                             7740      210.0         15        890    1625400

Timings (microseconds): count=1680 curr=795000000
Memory (bytes): count=0
1680 nodes observed
"""

SAMPLE_YNNPACK_TFLITE_LOG = """\
I0000 00:00:1794223443.642315 2224572 litert_lm_lib.cc:461] BenchmarkInfo:
  TimeToFirstToken: 0.280 s
  PrefillSpeed: 452.10 tokens/sec
  DecodeSpeed: 9.45 tokens/sec
--------------------------------------------------
====================================================== Summary by node type ======================================================
                                                 [Node type]    [count]   [avg ms]    [avg %]    [cdf %]   [mem KB] [times called]
                                             YNNPackDelegate        360      1.850    94.200%    94.200%      0.000          46440
                                                   SELECT_V2        120      0.095     4.100%    98.300%      0.000          15480
                                        DYNAMIC_UPDATE_SLICE        120      0.040     1.700%   100.000%      0.000          15480

Timings (microseconds): count=600 curr=620000000
"""

SAMPLE_PERF_REPORT = """\
# Overhead  Command                  Shared Object               Symbol
    62.40%  litert_lm_advan  litert_lm_advanced_main     [.] xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512vnni
    21.60%  litert_lm_advan  litert_lm_advanced_main     [.] xnn_qd8_f32_qc4w_gemm_minmax_ukernel_32x64c4__amxint8
     6.00%  litert_lm_advan  litert_lm_advanced_main     [.] xnn_f32_gemm_minmax_ukernel_1x16__avx512f_broadcast
     5.00%  litert_lm_advan  litert_lm_advanced_main     [.] xnn_qs8_packq_f32qp8_ukernel__scalar_u1
     5.00%  litert_lm_advan  [kernel.kallsyms]           [k] clear_page_erms
"""


class ExtractLitertLmKernelsTest(unittest.TestCase):

  def test_parse_xnnpack_tflite_profiler_output(self) -> None:
    graph_rows, delegate_rows = ext.parse_tflite_profiler_output(SAMPLE_XNNPACK_TFLITE_LOG)
    self.assertEqual(len(graph_rows), 7)
    self.assertEqual(len(delegate_rows), 5)
    top = graph_rows[0]
    self.assertEqual(top.node_type, "Delegate/Fully Connected (NC, QDU8, F32, QC4W) GEMM")
    self.assertEqual(top.count, 420)
    self.assertEqual(top.times_called, 54180)
    self.assertAlmostEqual(top.pct, 84.5)
    self.assertAlmostEqual(top.total_ms, 671832.0)

  def test_parse_ynnpack_tflite_profiler_output(self) -> None:
    graph_rows, delegate_rows = ext.parse_tflite_profiler_output(SAMPLE_YNNPACK_TFLITE_LOG)
    self.assertEqual(len(graph_rows), 3)
    self.assertEqual(len(delegate_rows), 1)
    self.assertEqual(delegate_rows[0].node_type, "YNNPackDelegate")
    self.assertAlmostEqual(delegate_rows[0].pct, 94.2)
    self.assertAlmostEqual(delegate_rows[0].total_ms, 584040.0)

  def test_parse_perf_report_normalizes_and_excludes_kernel_page_clearing(self) -> None:
    stats, isa_map, cat_map = ext.parse_perf_report(SAMPLE_PERF_REPORT)
    self.assertEqual(len(stats), 4)
    # 62.4 / 95.0 = 65.68%
    self.assertAlmostEqual(stats[0].pct_of_kernel_cycles, 65.68, places=2)
    self.assertEqual(stats[0].isa, "AVX-512 VNNI")
    self.assertEqual(stats[1].isa, "AMX-INT8")
    self.assertAlmostEqual(sum(isa_map.values()), 100.0, places=1)
    self.assertIn("INT4 Weight-Only Linear GEMM/GEMV", cat_map)

  def test_from_logs_and_markdown_generation(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      raw_dir = Path(tmp) / "raw"
      out_dir = Path(tmp) / "out"
      raw_dir.mkdir(parents=True)
      (raw_dir / "xnnpack_p128_d128_tflite.log").write_text(SAMPLE_XNNPACK_TFLITE_LOG)
      (raw_dir / "xnnpack_p128_d128_perf.txt").write_text(SAMPLE_PERF_REPORT)
      (raw_dir / "ynnpack_p128_d128_tflite.log").write_text(SAMPLE_YNNPACK_TFLITE_LOG)

      rc = ext.main([
          "--from-logs",
          str(raw_dir),
          "--shapes",
          "p128_d128",
          "--out-dir",
          str(out_dir),
      ])
      self.assertEqual(rc, 0)
      md = (out_dir / "kernels_gemma4_31b_sapphire_rapids.md").read_text()
      self.assertIn("Delegate/Fully Connected (NC, QDU8, F32, QC4W) GEMM", md)
      self.assertIn("84.50%", md)
      self.assertIn("YNNPackDelegate", md)
      self.assertIn("94.20%", md)
      self.assertIn("xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512vnni", md)
      self.assertIn('title "Prefill / Encode Throughput per Shape (XNNPACK vs. YNNPACK)"', md)
      self.assertIn('title "Decode Throughput per Shape (XNNPACK vs. YNNPACK)"', md)
      self.assertIn('x-axis "Shape" ["p128_d128"', md)
      self.assertIn('bar "XNNPACK"', md)
      self.assertIn('bar "YNNPACK"', md)
      self.assertIn('y-axis "tok/s"', md)

  def test_multimodal_shapes_and_ynnpack_amx_int2_classification(self) -> None:
    modalities = {s.id: s.modality for s in ext.DEFAULT_SHAPES}
    for vid in ("vis_70", "vis_140", "vis_280", "vis_560", "vis_1120"):
      self.assertEqual(modalities.get(vid), "vision")
    self.assertEqual(modalities.get("audio_204"), "audio")

    self.assertEqual(
        ext.classify_isa("ynn::dot_int8_int8_int32_32x32x64_16x16x4_amxint8"),
        "AMX-INT8",
    )
    self.assertEqual(
        ext.classify_kernel_category("ynn::dot_int8_int8_int32_32x32x64_16x16x4_amxint8"),
        "INT8 Quantized Linear / Conv / Embedding GEMM/GEMV",
    )
    self.assertEqual(
        ext.classify_isa("ynn::dot_uint8_int2_int32_4x64x16_1x16x16_avx512vnni"),
        "AVX-512 VNNI",
    )
    self.assertEqual(
        ext.classify_kernel_category("ynn::dot_uint8_int2_int32_4x64x16_1x16x16_avx512vnni"),
        "INT2 Weight-Only Linear GEMM/GEMV",
    )


if __name__ == "__main__":
  unittest.main()
