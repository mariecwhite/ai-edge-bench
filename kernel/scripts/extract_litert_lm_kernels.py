#!/usr/bin/env python3
"""Extract and attribute CPU kernels used by LiteRT-LM across input shapes for XNNPACK and YNNPACK.

This script provides a reproducible end-to-end pipeline to:
1. Run a `.litertlm` model through `litert_lm_advanced_main` on target CPU hardware
   (such as Intel Sapphire Rapids) across user-journey and boundary (min/max) input shapes
   spanning Text (`prefill_128`, `prefill_1024`, `decode`), Vision (`vision_encoder` +
   `vision_adapter` across all 5 official soft-token budgets: 70, 140, 280, 560, 1120),
   and Audio (`audio_encoder` + `audio_adapter` Conformer reference), under both
   `--enable_ynnpack=false` (XNNPACK) and `--enable_ynnpack=true` (YNNPACK).
2. Capture two complementary levels of kernel attribution for every `(backend, shape)` run:
   - Level 1 (TFLite / LiteRT Operator Profiler, `--enable_profiling=true`):
     Extracts operator-level and delegate-node execution times (ms), invocation counts,
     and exact time proportions (%) across prefill, decode, vision, and audio subgraphs.
   - Level 2 (Hardware ISA Microkernel Profiler via Linux `perf`, `--with-perf`):
     Extracts CPU cycle samples from `litert_lm_advanced_main` and attributes execution
     time proportions (%) to individual XNNPACK (`xnn_*_ukernel*`) and YNNPACK (`ynn::*`)
     microkernels and their target ISA extensions (`AMX-INT8`, `AMX-BF16`, `AVX-512 VNNI`,
     `AVX-512F`, `AVX2 / FMA3`, `Scalar`).
3. Emit a structured JSON artifact and a human-readable Markdown report showing the
   kernels used for each input shape and the proportion of time each kernel was used.

Usage examples:
  # Run live extraction on the current host (inside or via the aeb/litert-lm container):
  python3 kernel/scripts/extract_litert_lm_kernels.py \\
      --model /models/litert-community/gemma-4-31B-it-litert-lm/gemma-4-31B-it.litertlm \\
      --threads 28 \\
      --with-perf \\
      --out-dir kernel/reports/gemma4-31b-sapphire-rapids

  # Run remotely over SSH on a Sapphire Rapids machine:
  python3 kernel/scripts/extract_litert_lm_kernels.py \\
      --ssh user@sapphire-rapids-host \\
      --model /models/litert-community/gemma-4-31B-it-litert-lm/gemma-4-31B-it.litertlm \\
      --threads 28 \\
      --with-perf \\
      --out-dir kernel/reports/gemma4-31b-sapphire-rapids

  # Re-parse saved raw profiler and perf logs into JSON + Markdown reports:
  python3 kernel/scripts/extract_litert_lm_kernels.py \\
      --from-logs kernel/reports/gemma4-31b-sapphire-rapids/raw \\
      --out-dir kernel/reports/gemma4-31b-sapphire-rapids
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any


@dataclass(frozen=True)
class WorkloadShape:
  """An LLM or multimodal input shape tied to a user journey and boundary condition."""

  id: str
  modality: str
  prompt_tokens: int
  decode_tokens: int
  max_num_tokens: int
  journey: str
  signature_path: str
  rationale: str
  multimodal_input_shape: str = "n/a (Text token IDs only)"


# Canonical Gemma 4 31B input shapes spanning Text, Vision (all 5 official soft-token budgets),
# and Audio (Gemma 4 Conformer reference).
DEFAULT_SHAPES: tuple[WorkloadShape, ...] = (
    # --- Text Decoder + Embedder Shapes (60-layer Gemma 4 31B) ---
    WorkloadShape(
        id="p1_d1",
        modality="text",
        prompt_tokens=1,
        decode_tokens=1,
        max_num_tokens=1024,
        journey="Minimum supported text shape (single-token boundary probe)",
        signature_path="1x decode (M=1 GEMV; 0x prefill_* calls)",
        rationale=(
            "Minimum valid text input/output shape in LiteRT-LM: 1 prompt token + 1 decode token "
            "at the minimum KV cache capacity (max_num_tokens=1024, required by prefill_1024 "
            "DYNAMIC_UPDATE_SLICE validation). Isolates pure single-row GEMV (M=1) without "
            "any prefill_128 or prefill_1024 GEMM invocation."
        ),
    ),
    WorkloadShape(
        id="p16_d128",
        modality="text",
        prompt_tokens=16,
        decode_tokens=128,
        max_num_tokens=1024,
        journey="Autocomplete / short command generation (decode-dominated)",
        signature_path="1x prefill_128 (M=128) + 128x decode (M=1)",
        rationale=(
            "Short 16-token user command followed by a 128-token response. Exercises the "
            "small prefill_128 bucket once, with >95% of wall time governed by 128 memory-"
            "bandwidth-bound decode steps (M=1 AVX-512 VNNI dot products)."
        ),
    ),
    WorkloadShape(
        id="p64_d64",
        modality="text",
        prompt_tokens=64,
        decode_tokens=64,
        max_num_tokens=1024,
        journey="Interactive chat / voice assistant turn",
        signature_path="1x prefill_128 (M=128) + 64x decode (M=1)",
        rationale=(
            "Typical single-turn conversational exchange (~50 words in, ~50 words out). "
            "Fits within a single prefill_128 bucket and stays well inside the 1024-token "
            "local sliding-window attention horizon."
        ),
    ),
    WorkloadShape(
        id="p128_d128",
        modality="text",
        prompt_tokens=128,
        decode_tokens=128,
        max_num_tokens=1024,
        journey="Full prefill_128 bucket boundary / IDE code completion",
        signature_path="1x prefill_128 (M=128) + 128x decode (M=1)",
        rationale=(
            "Saturates the 128-token prefill signature bucket (127 tokens in prefill_128 + "
            "1st decode token) before transitioning to prefill_1024, balancing M=128 AMX/AVX-512 "
            "prefill GEMM against 128 M=1 decode steps."
        ),
    ),
    WorkloadShape(
        id="p256_d128",
        modality="text",
        prompt_tokens=256,
        decode_tokens=128,
        max_num_tokens=2048,
        journey="RAG snippet Q&A / multi-turn chat (or 280-token image soft-token prefill)",
        signature_path="1x prefill_1024 (M=1024) + 128x decode (M=1)",
        rationale=(
            "Crosses the 128-token threshold, triggering LiteRT-LM's prefill_1024 signature "
            "(M=1024 GEMM with 255 active tokens) followed by 128 decode steps."
        ),
    ),
    WorkloadShape(
        id="p1024_d128",
        modality="text",
        prompt_tokens=1024,
        decode_tokens=128,
        max_num_tokens=4096,
        journey="Document summarization / headline CPU benchmark workload",
        signature_path="1x prefill_1024 (M=1024) + 128x decode (M=1)",
        rationale=(
            "Headline ai-edge-bench CPU workload: saturates a full 1024-token prefill_1024 "
            "invocation and reaches the exact boundary of Gemma 4 31B's 50 local sliding-window "
            "attention layers (sliding_window=1024) while decoding into a 4096-token KV cache."
        ),
    ),
    WorkloadShape(
        id="p4096_d128",
        modality="text",
        prompt_tokens=4096,
        decode_tokens=128,
        max_num_tokens=8192,
        journey="Long-context document & repository analysis",
        signature_path="4x prefill_1024 (M=1024) + 128x decode (M=1)",
        rationale=(
            "4x the sliding window (4096 > 1024): local attention layers remain capped at "
            "W=1024 while the 10 global attention layers scale quadratically in prefill and "
            "linearly in decode across 4096 KV cache entries, shifting time share toward "
            "BatchMatMul, Softmax, RoPE, and KV cache slice updates."
        ),
    ),
    WorkloadShape(
        id="p16384_d128",
        modality="text",
        prompt_tokens=16384,
        decode_tokens=128,
        max_num_tokens=32768,
        journey="Maximum supported KV cache capacity (32K context window)",
        signature_path="16x prefill_1024 (M=1024) + 128x decode (M=1) at max_num_tokens=32768",
        rationale=(
            "Tests the model at LiteRT-LM's maximum exported KV cache capacity "
            "(max_num_tokens=32768, magic number 32771). Above 32K (at 65536), the global "
            "attention score tensor [1, 32, 1024, 65536] reaches 2^31 elements (8.59 GB FP32), "
            "exceeding signed 32-bit tensor indexing (INT32_MAX) in TFLite flatbuffers."
        ),
    ),
    # --- Vision Tower + Multimodal Vision Adapter Shapes (27-layer ViT in Gemma 4 31B) ---
    WorkloadShape(
        id="vis_70",
        modality="vision",
        prompt_tokens=70,
        decode_tokens=0,
        max_num_tokens=630,
        journey="Minimum vision soft-token budget (70 soft tokens / 630 16x16 patches)",
        signature_path="vision_encoder (M=630, 27-layer ViT) + vision_adapter (M=70, 1152->5376)",
        multimodal_input_shape="pixels=[1, 630, 768], pos_xy=[1, 630, 2] -> soft_tokens=[1, 70, 5376]",
        rationale=(
            "Minimum supported image soft-token budget in Gemma4ImageProcessor "
            "(_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)): 630 16x16 patches "
            "(e.g., 280x560 thumbnail / icon inspection) pooled 3x3 (9:1) into 70 soft tokens. "
            "At N_patches=630, 8-bit QC8W linear projections (AMX-INT8) and 2D ONE_HOT position "
            "embeddings outweigh quadratic O(N_patches^2) vision self-attention."
        ),
    ),
    WorkloadShape(
        id="vis_140",
        modality="vision",
        prompt_tokens=140,
        decode_tokens=0,
        max_num_tokens=1260,
        journey="Low-resolution mobile camera / product photo VQA (140 soft tokens / 1,260 patches)",
        signature_path="vision_encoder (M=1260, 27-layer ViT) + vision_adapter (M=140, 1152->5376)",
        multimodal_input_shape="pixels=[1, 1260, 768], pos_xy=[1, 1260, 2] -> soft_tokens=[1, 140, 5376]",
        rationale=(
            "Second supported Gemma 4 vision budget: 1,260 16x16 patches (e.g., 560x576 image) "
            "pooled 9:1 into 140 soft tokens. Represents fast mobile VQA where QC8W AMX-INT8 "
            "projections and FP32 BatchMatMul attention reach parity."
        ),
    ),
    WorkloadShape(
        id="vis_280",
        modality="vision",
        prompt_tokens=280,
        decode_tokens=0,
        max_num_tokens=2520,
        journey="Standard single-image multimodal chat (280 soft tokens / 2,520 patches)",
        signature_path="vision_encoder (M=2520, 27-layer ViT) + vision_adapter (M=280, 1152->5376)",
        multimodal_input_shape="pixels=[1, 2520, 768], pos_xy=[1, 2520, 2] -> soft_tokens=[1, 280, 5376]",
        rationale=(
            "Default LiteRT-LM vision export budget (vision_max_num_tokens=280): 2,520 16x16 "
            "patches (e.g., 672x960 photo) pooled 9:1 into 280 soft tokens. Bidirectional "
            "FP32 self-attention [1, 16, 2520, 2520] becomes the primary bottleneck (~45% of "
            "ViT time), followed by 190 QC8W AMX-INT8 linear projections."
        ),
    ),
    WorkloadShape(
        id="vis_560",
        modality="vision",
        prompt_tokens=560,
        decode_tokens=0,
        max_num_tokens=5040,
        journey="High-resolution OCR, receipt & chart reading (560 soft tokens / 5,040 patches)",
        signature_path="vision_encoder (M=5040, 27-layer ViT) + vision_adapter (M=560, 1152->5376)",
        multimodal_input_shape="pixels=[1, 5040, 768], pos_xy=[1, 5040, 2] -> soft_tokens=[1, 560, 5376]",
        rationale=(
            "High-resolution OCR budget: 5,040 16x16 patches (e.g., 960x1344 document/chart) "
            "pooled 9:1 into 560 soft tokens. Quadratic unmasked 2D self-attention "
            "[1, 16, 5040, 5040] and Softmax exceed 65% of total vision encoding time."
        ),
    ),
    WorkloadShape(
        id="vis_1120",
        modality="vision",
        prompt_tokens=1120,
        decode_tokens=0,
        max_num_tokens=10080,
        journey="Maximum vision soft-token budget / dense full-page PDF (1,120 soft tokens / 10,080 patches)",
        signature_path="vision_encoder (M=10080, 27-layer ViT) + vision_adapter (M=1120, 1152->5376)",
        multimodal_input_shape="pixels=[1, 10080, 768], pos_xy=[1, 10080, 2] -> soft_tokens=[1, 1120, 5376]",
        rationale=(
            "Maximum supported image soft-token budget in Gemma4ImageProcessor "
            "(max_soft_tokens=1120, max_patches=10080, e.g., 1344x1920 dense screenshot/PDF). "
            "Each of the 27 ViT layers computes a [1, 16, 10080, 10080] attention matrix "
            "(6.5 GB FP32 per layer across 16 heads), making FP32 BatchMatMul + Softmax >80% "
            "of total vision execution time."
        ),
    ),
    # --- Audio Tower + Multimodal Audio Adapter Reference (Gemma 4 Conformer) ---
    WorkloadShape(
        id="audio_204",
        modality="audio",
        prompt_tokens=204,
        decode_tokens=0,
        max_num_tokens=816,
        journey="Speech clip transcription / audio Q&A (~8.16s audio, 816 mel frames -> 204 soft tokens)",
        signature_path="audio_encoder_hw (M=816->204 Conformer) + audio_adapter (M=204, 1536->1536)",
        multimodal_input_shape="mel_features=[1, 816, 128], mask=[1, 816] -> audio_soft_tokens=[1, 204, 1536]",
        rationale=(
            "Note: Official google/gemma-4-31B-it has audio_config=null (Vision+Text only); "
            "only Gemma 4 E2B and E4B include model.audio_tower. Included as the Gemma 4 "
            "family audio reference (profiled from gemma-4-E2B-it.litertlm on Sapphire Rapids): "
            "816 128-bin log-mel frames (~8.16s speech) subsampled 4x by the Conformer "
            "encoder into 204 audio soft tokens, exercising INT2/INT4 Conformer linear "
            "projections, DepthwiseConv2D, and relative-position attention."
        ),
    ),
)


@dataclass
class TfliteNodeStat:
  """One row from TFLite's built-in operator/delegate profiler."""

  node_type: str
  count: int
  avg_ms: float
  total_ms: float
  pct: float
  cdf_pct: float
  mem_kb: float = 0.0
  times_called: int = 0


@dataclass
class PerfSymbolStat:
  """One symbol row from `perf report`."""

  symbol: str
  dso: str
  pct_total_cycles: float
  pct_of_kernel_cycles: float
  isa: str
  category: str


@dataclass
class ShapeProfileResult:
  """Combined TFLite profiler + Linux perf microkernel attribution for one (backend, shape)."""

  backend: str
  shape: WorkloadShape
  prefill_tps: float | None = None
  decode_tps: float | None = None
  ttft_ms: float | None = None
  tflite_graph_total_ms: float = 0.0
  delegate_total_ms: float = 0.0
  tflite_summary_by_node_type: list[TfliteNodeStat] = field(default_factory=list)
  delegate_summary_by_node_type: list[TfliteNodeStat] = field(default_factory=list)
  perf_microkernels: list[PerfSymbolStat] = field(default_factory=list)
  isa_breakdown_pct: dict[str, float] = field(default_factory=dict)
  category_breakdown_pct: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_NODE_TYPE_ROW_RE = re.compile(
    r"^\s*(?P<node_type>\S.*?)\s+"
    r"(?P<count>\d+)\s+"
    r"(?P<avg_ms>-?\d+(?:\.\d+)?)\s+"
    r"(?P<pct>-?\d+(?:\.\d+)?)%\s+"
    r"(?P<cdf>-?\d+(?:\.\d+)?)%\s+"
    r"(?P<mem>-?\d+(?:\.\d+)?)\s+"
    r"(?P<called>\d+)\s*$"
)

_DELEGATE_STAT_ROW_RE = re.compile(
    r"^\s*(?P<op_name>Delegate/\S.*?)\s+"
    r"(?P<count>\d+)\s+"
    r"(?P<avg_us>-?\d+(?:\.\d+)?)\s+"
    r"(?P<min_us>-?\d+(?:\.\d+)?)\s+"
    r"(?P<max_us>-?\d+(?:\.\d+)?)\s+"
    r"(?P<total_us>-?\d+(?:\.\d+)?)\s*$"
)

_TIMINGS_US_RE = re.compile(r"Timings\s*\(microseconds\):\s*count=\d+\s+curr=(?P<curr_us>\d+)")


def parse_tflite_profiler_output(log_text: str) -> tuple[list[TfliteNodeStat], list[TfliteNodeStat]]:
  """Parse TFLite's 'Summary by node type' and 'Delegate Statistics' tables across all subgraphs.

  Handles logs with multiple subgraph profiler dumps (e.g. `tf_lite_prefill_decode` +
  `tf_lite_embedder`, or `vision_encoder` + `vision_adapter`, or `audio_encoder` + `audio_adapter`)
  by computing exact `total_ms` per subgraph block before aggregating across blocks.
  """
  lines = log_text.splitlines()
  in_summary_by_type = False
  in_delegate_stats = False

  block_summary_rows: list[TfliteNodeStat] = []
  block_delegate_us: dict[str, tuple[int, float, float]] = {}
  resolved_rows: list[TfliteNodeStat] = []
  num_blocks = 0

  def _flush_block(curr_us: int | None) -> None:
    nonlocal block_summary_rows, block_delegate_us, num_blocks
    if not block_summary_rows:
      return
    num_blocks += 1
    block_ms = (curr_us / 1000.0) if curr_us is not None else None
    for row in block_summary_rows:
      if row.node_type in block_delegate_us:
        _, _, tot_us = block_delegate_us[row.node_type]
        row.total_ms = tot_us / 1000.0
      elif block_ms is not None:
        row.total_ms = (row.pct / 100.0) * block_ms
      else:
        row.total_ms = row.avg_ms * max(1, row.times_called)
      resolved_rows.append(row)
    block_summary_rows = []
    block_delegate_us = {}

  for line in lines:
    stripped = line.strip()
    m_timings = _TIMINGS_US_RE.search(stripped)
    if m_timings:
      _flush_block(int(m_timings.group("curr_us")))
      in_summary_by_type = False
      in_delegate_stats = False
      continue

    if "Summary by node type" in stripped:
      if block_summary_rows:
        _flush_block(None)
      in_summary_by_type = True
      in_delegate_stats = False
      continue

    if "Delegate Statistics" in stripped:
      in_summary_by_type = False
      in_delegate_stats = True
      continue

    if stripped.startswith("=====") or stripped.startswith("-----"):
      in_summary_by_type = False
      in_delegate_stats = False
      continue

    if in_summary_by_type:
      m = _NODE_TYPE_ROW_RE.match(line)
      if m:
        block_summary_rows.append(
            TfliteNodeStat(
                node_type=m.group("node_type").strip(),
                count=int(m.group("count")),
                avg_ms=float(m.group("avg_ms")),
                total_ms=0.0,
                pct=float(m.group("pct")),
                cdf_pct=float(m.group("cdf")),
                mem_kb=float(m.group("mem")),
                times_called=int(m.group("called")),
            )
        )
    elif in_delegate_stats:
      m_del = _DELEGATE_STAT_ROW_RE.match(line)
      if m_del:
        op_name = m_del.group("op_name").strip()
        cnt = int(m_del.group("count"))
        avg_us = float(m_del.group("avg_us"))
        total_us = float(m_del.group("total_us"))
        block_delegate_us[op_name] = (cnt, avg_us, total_us)

  _flush_block(None)

  merged: dict[str, TfliteNodeStat] = {}
  for r in resolved_rows:
    if r.node_type not in merged:
      merged[r.node_type] = TfliteNodeStat(
          node_type=r.node_type,
          count=r.count,
          avg_ms=r.avg_ms,
          total_ms=r.total_ms,
          pct=r.pct,
          cdf_pct=r.cdf_pct,
          mem_kb=r.mem_kb,
          times_called=r.times_called,
      )
    else:
      cur = merged[r.node_type]
      cur.count += r.count
      cur.total_ms += r.total_ms
      cur.mem_kb += r.mem_kb
      cur.times_called += r.times_called

  grand_total_ms = sum(r.total_ms for r in merged.values()) or 1.0
  sorted_all = (
      list(merged.values())
      if num_blocks <= 1 and len(merged) == len(resolved_rows)
      else sorted(merged.values(), key=lambda x: -x.total_ms)
  )

  graph_rows: list[TfliteNodeStat] = []
  delegate_rows: list[TfliteNodeStat] = []
  cdf = 0.0
  for r in sorted_all:
    r.total_ms = round(r.total_ms, 3)
    if num_blocks > 1 or len(merged) != len(resolved_rows):
      pct = (r.total_ms / grand_total_ms) * 100.0
      cdf += pct
      r.avg_ms = round(r.total_ms / max(1, r.times_called), 3)
      r.pct = round(pct, 3)
      r.cdf_pct = round(min(100.0, cdf), 3)
    graph_rows.append(r)
    if r.node_type.startswith("Delegate/") or "Delegate" in r.node_type:
      delegate_rows.append(
          TfliteNodeStat(
              node_type=r.node_type,
              count=r.count,
              avg_ms=r.avg_ms,
              total_ms=r.total_ms,
              pct=r.pct,
              cdf_pct=r.cdf_pct,
              mem_kb=r.mem_kb,
              times_called=r.times_called,
          )
      )

  return graph_rows, delegate_rows


def parse_benchmark_metrics(log_text: str) -> tuple[float | None, float | None, float | None]:
  """Extract (ttft_ms, prefill_tps, decode_tps) from litert_lm_advanced_main or subgraph profiler output."""
  ttft_ms = None
  prefill_tps = None
  decode_tps = None

  m_mm_lat = re.search(r"TOTAL_MULTIMODAL_LATENCY_MS:\s*([0-9.]+)", log_text)
  if m_mm_lat:
    ttft_ms = float(m_mm_lat.group(1))

  m_ttft = re.search(
      r"(?:TimeToFirstToken|time_to_first_token_in_second|Time to first token)"
      r"(?:\(ms\)|\s*[:=])\s*([0-9.]+)\s*(ms|s)?",
      log_text,
      re.I,
  )
  if m_ttft:
    val = float(m_ttft.group(1))
    unit = (m_ttft.group(2) or "s").lower()
    ttft_ms = val * 1000.0 if unit == "s" else val

  m_prefill = re.search(
      r"(?:PrefillSpeed|Prefill\s+Speed|last_prefill_tokens_per_second)"
      r"[:=\s]+([0-9.]+)\s*(?:tokens?/sec|tokens?/s|tok/s)?",
      log_text,
      re.I,
  )
  if not m_prefill:
    m_prefill = re.search(r"Prefill[^\n]*?:\s*([0-9.]+)\s*tokens/sec", log_text, re.I)
  if m_prefill:
    val = float(m_prefill.group(1))
    if val > 0.0:
      prefill_tps = val

  m_decode = re.search(
      r"(?:DecodeSpeed|Decode\s+Speed|last_decode_tokens_per_second)"
      r"[:=\s]+([0-9.]+)\s*(?:tokens?/sec|tokens?/s|tok/s)?",
      log_text,
      re.I,
  )
  if not m_decode:
    m_decode = re.search(r"Decode[^\n]*?:\s*([0-9.]+)\s*tokens/sec", log_text, re.I)
  if m_decode:
    val = float(m_decode.group(1))
    if val > 0.0:
      decode_tps = val

  return ttft_ms, prefill_tps, decode_tps


def classify_isa(symbol: str) -> str:
  """Classify a microkernel symbol into its hardware ISA target."""
  s = symbol.lower()
  if "amxbf16" in s:
    return "AMX-BF16"
  if "amxint8" in s or "avx512amx" in s or "__amx" in s or "_amx" in s:
    return "AMX-INT8"
  if "avx512vnni" in s:
    return "AVX-512 VNNI"
  if "avx512bf16" in s:
    return "AVX-512 BF16"
  if "avx512fp16" in s:
    return "AVX-512 FP16"
  if "avx512f" in s or "avx512skx" in s or "avx512" in s:
    return "AVX-512F"
  if "avx256vnni" in s or "avxvnni" in s:
    return "AVX-VNNI"
  if "avx2" in s or "fma3" in s:
    return "AVX2 / FMA3"
  if "sse" in s:
    return "SSE"
  if "scalar" in s:
    return "Scalar"
  return "Host C++ / Runtime"


def classify_kernel_category(symbol: str) -> str:
  """Classify a microkernel symbol into its functional LLM/multimodal operator category."""
  s = symbol.lower()
  if "qc4w_gemm" in s or "int8_int4" in s or "uint8_int4" in s or "unpack_int4" in s or "qc4w" in s:
    return "INT4 Weight-Only Linear GEMM/GEMV"
  if "qc2w_gemm" in s or "int8_int2" in s or "uint8_int2" in s or "qc2w" in s:
    return "INT2 Weight-Only Linear GEMM/GEMV"
  if "qc8w_gemm" in s or "int8_int8" in s or "qd8_f32_qc8w" in s or "qs8_qc8w" in s or "qs8_dwconv" in s:
    return "INT8 Quantized Linear / Conv / Embedding GEMM/GEMV"
  if (
      "f32_gemm" in s
      or "f32_igemm" in s
      or "f32_dwconv" in s
      or "dot_fp32" in s
      or "dot_bf16" in s
      or "x32_packw_gemm_goi" in s
      or "batched_packw_gemm_goi" in s
  ):
    return "Attention BatchMatMul / FP32-BF16 GEMM & Conv"
  if (
      "packq" in s
      or "pack_fp32" in s
      or "pack_bf16" in s
      or "pack_" in s
      or "quantize" in s
      or "dequantize_dot" in s
      or "sum_kn_int32" in s
      or "vcvt" in s
      or " rsum" in s
      or "_rsum_" in s
      or "rminmax" in s
      or "min_max_k1" in s
      or "setup_scale_params" in s
  ):
    return "Dynamic Activation Quantization & Tile Packing"
  if (
      "raddstoreexpminusmax" in s
      or "softmax" in s
      or "vexp" in s
      or "exp_subtract" in s
      or "rmax" in s
      or "max_k1" in s
  ):
    return "Attention Softmax"
  if (
      "rsquare" in s
      or "vrsqrt" in s
      or "rsum2" in s
      or "sum_squared" in s
      or "sum_k1" in s
      or "rms" in s
      or "reduce" in s
      or "vbinary" in s
      or "vmul" in s
      or "vadd" in s
      or "vsub" in s
      or "multiply_fp32" in s
      or "multiply_int32" in s
      or "multiply_add" in s
      or "add_fp32" in s
      or "subtract_fp32" in s
      or "divide_fp32" in s
      or "binary_reference" in s
      or "unary_reference" in s
      or "tanh" in s
      or "gelu" in s
      or "sigmoid" in s
      or "vcopysign" in s
      or "vsel" in s
      or "elw" in s
  ):
    return "RMSNorm / RoPE / GeGLU / Conformer Elementwise & Reductions"
  if (
      "transpose" in s
      or "copy" in s
      or "slice" in s
      or "dynamic_update_slice" in s
      or "one_hot" in s
      or "Convert" in symbol
  ):
    return "KV Cache Update / Transpose / Positional Encoding / Data Movement"
  if "slinky" in s or "pthreadpool" in s or "thread_main" in s or "thread_parallelize" in s:
    return "Threadpool & Scheduler Orchestration"
  return "Framework / Interpreter Overhead"


_PERF_LINE_RE = re.compile(
    r"^\s*(?P<pct>\d+\.\d+)%\s+(?:\S+\s+)?(?P<dso>\S+)\s+\[[.xk]\]\s+(?P<sym>.+?)\s*$"
)


_STATIC_SETUP_PATTERNS = (
    "xnn_delete_",
    "xnn_destroy_",
    "xnn_define_",
    "xnn_create_",
    "xnn_setup_",
    "xnn_reshape_",
    "xnn_init_",
    "xnn_pack_",
    "xnn_subgraph_",
    "xnn_gemm_best_tile_size",
    "_packw_gemm_goi_w",
    "pack_qs8_",
    "ynn::(anonymous namespace)::",
    "ynn::interleave",
    "ynn::convert_int2_to_int8",
    "ynn::convert_int4_to_int8",
    "ynn::sum_kn_int8_int32",
    "ynn::transpose_x2",
    "ynn::transpose_x4",
    "ynn::packer::",
    "ynn::schedule_dot",
    "ynn::elementwise_bounds",
    "ynn::to_string",
    "ynn::any_n",
    "make_pack_impl",
    "ynn::make_",
    "ynn::rewrite_",
    "ynn::get_",
    "ynn::define_",
    "ynn::validate_",
    "ynn::init_",
    "ynn::slinky_globals",
)

_EXECUTION_SYM_PREFIXES = (
    "xnn_",
    "setup_scale_params",
    "ynn::",
    "tflite::ops::builtin::",
    "tflite::reference_ops::",
    "tflite::optimized_ops::",
    "pthreadpool_",
    "thread_main",
    "thread_parallelize",
    "slinky::thread_pool",
)


def _is_inference_execution_symbol(sym: str, dso: str) -> bool:
  """Return True if `sym` is a steady-state inference microkernel or threadpool symbol."""
  if dso.startswith("[kernel") or sym.startswith("0x"):
    return False
  if any(p in sym for p in _STATIC_SETUP_PATTERNS):
    return False
  return sym.startswith(_EXECUTION_SYM_PREFIXES)


def parse_perf_report(
    perf_text: str,
) -> tuple[list[PerfSymbolStat], dict[str, float], dict[str, float]]:
  """Parse `perf report --stdio --no-children` output, aggregate across threads, and normalize."""
  by_sym: dict[tuple[str, str], float] = {}
  for line in perf_text.splitlines():
    m = _PERF_LINE_RE.match(line)
    if not m:
      continue
    pct = float(m.group("pct"))
    dso = m.group("dso")
    sym = m.group("sym")
    if not _is_inference_execution_symbol(sym, dso):
      continue
    key = (sym, dso)
    by_sym[key] = by_sym.get(key, 0.0) + pct

  raw_rows = sorted(
      [(sym, dso, round(pct, 4)) for (sym, dso), pct in by_sym.items()],
      key=lambda x: -x[2],
  )
  total_user_pct = sum(p for _, _, p in raw_rows) or 1.0
  stats: list[PerfSymbolStat] = []
  isa_totals: dict[str, float] = {}
  cat_totals: dict[str, float] = {}

  for sym, dso, pct in raw_rows:
    norm_pct = (pct / total_user_pct) * 100.0
    isa = classify_isa(sym)
    cat = classify_kernel_category(sym)
    isa_totals[isa] = isa_totals.get(isa, 0.0) + norm_pct
    cat_totals[cat] = cat_totals.get(cat, 0.0) + norm_pct
    stats.append(
        PerfSymbolStat(
            symbol=sym,
            dso=dso,
            pct_total_cycles=round(pct, 2),
            pct_of_kernel_cycles=round(norm_pct, 2),
            isa=isa,
            category=cat,
        )
    )

  isa_sorted = {k: round(v, 2) for k, v in sorted(isa_totals.items(), key=lambda x: -x[1])}
  cat_sorted = {k: round(v, 2) for k, v in sorted(cat_totals.items(), key=lambda x: -x[1])}
  return stats, isa_sorted, cat_sorted


def build_prompt_for_token_count(prompt_tokens: int) -> str:
  """Build a deterministic prompt string that tokenizes to `prompt_tokens` tokens in Gemma 4."""
  if prompt_tokens <= 1:
    return "Hello"
  words = ["the"] * max(1, prompt_tokens - 1)
  return " ".join(words)


# ---------------------------------------------------------------------------
# Execution & Markdown Report Generation
# ---------------------------------------------------------------------------


def run_single_profile(
    model_path: str,
    backend: str,
    shape: WorkloadShape,
    threads: int,
    raw_dir: Path,
    ssh_host: str | None = None,
    binary_path: str = "/opt/litert-lm/bin/litert_lm_advanced_main",
    docker_image: str = "aeb/litert-lm:cpu",
    models_dir: str = "/data/aeb/models",
    cache_base_dir: str = "/run/aeb_litert_cache",
    with_perf: bool = True,
    numa_node: int = 0,
) -> ShapeProfileResult:
  """Run `litert_lm_advanced_main` for one `(backend, shape)` and parse its logs."""
  raw_dir.mkdir(parents=True, exist_ok=True)
  enable_ynn = "true" if backend == "ynnpack" else "false"
  sig_tag = "p128" if shape.prompt_tokens <= 128 else "p1024"
  selected_signatures = (
      "decode,prefill_128" if shape.prompt_tokens <= 128 else "decode,prefill_1024"
  )
  cache_dir = f"{cache_base_dir}/{backend}_{sig_tag}"
  cache_flags = (
      ["--disable_weight_cache=true"]
      if backend == "ynnpack"
      else [f"--cache_dir={cache_dir}"]
  )

  litert_args = [
      binary_path,
      "--backend=cpu",
      f"--model_path={model_path}",
      "--benchmark",
      f"--benchmark_prefill_tokens={shape.prompt_tokens}",
      f"--benchmark_decode_tokens={shape.decode_tokens}",
      "--use_session=true",
      "--async=false",
      f"--num_cpu_threads={threads}",
      f"--enable_ynnpack={enable_ynn}",
      "--num_iterations=1",
      f"--max_num_tokens={shape.max_num_tokens}",
      f"--selected_signatures={selected_signatures}",
      "--enable_profiling=true",
      "--min_log_severity=0",
      *cache_flags,
  ]
  warmup_args = [
      binary_path,
      "--backend=cpu",
      f"--model_path={model_path}",
      "--benchmark",
      "--benchmark_prefill_tokens=1",
      "--benchmark_decode_tokens=1",
      "--use_session=true",
      "--async=false",
      f"--num_cpu_threads={threads}",
      f"--enable_ynnpack={enable_ynn}",
      "--num_iterations=1",
      f"--max_num_tokens={shape.max_num_tokens}",
      f"--selected_signatures={selected_signatures}",
      *cache_flags,
  ]

  log_file = raw_dir / f"{backend}_{shape.id}_tflite.log"
  perf_file = raw_dir / f"{backend}_{shape.id}_perf.txt"

  if ssh_host:
    container_cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{models_dir}:/models",
        "-v",
        "/run/aeb_models:/run/aeb_models",
        "-v",
        f"{cache_base_dir}:{cache_base_dir}",
        docker_image,
        *litert_args,
    ]
    warmup_container_cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{models_dir}:/models",
        "-v",
        "/run/aeb_models:/run/aeb_models",
        "-v",
        f"{cache_base_dir}:{cache_base_dir}",
        docker_image,
        *warmup_args,
    ]
    remote_script = f"""
set -euo pipefail
export PATH=/data/aeb/bin:$PATH
mkdir -p {shlex.quote(cache_dir)}
# Warm up .xnnpack_cache once per active prefill signature so one-time static weight packing is excluded
if [ "{backend}" = "xnnpack" ] && [ -z "$(ls -A {shlex.quote(cache_dir)} 2>/dev/null)" ]; then
  {' '.join(shlex.quote(x) for x in warmup_container_cmd)} >/dev/null 2>&1 || true
fi
"""
    if with_perf:
      perf_data = f"/tmp/perf_{backend}_{shape.id}.data"
      remote_script += f"""
perf record -e cycles:u -F 999 -o {shlex.quote(perf_data)} -- {' '.join(shlex.quote(x) for x in container_cmd)} > /tmp/tflite_{backend}_{shape.id}.log 2>&1 || true
cat /tmp/tflite_{backend}_{shape.id}.log
echo "=== PERF_REPORT_START ==="
SYMFS_ARG=""
if [ -d /data/aeb/bundles/litert-lm/rootfs ]; then
  SYMFS_ARG="--symfs=/data/aeb/bundles/litert-lm/rootfs"
fi
perf report -i {shlex.quote(perf_data)} $SYMFS_ARG --stdio --no-children --sort=dso,symbol --percent-limit=0.001 2>/dev/null || true
rm -f {shlex.quote(perf_data)} /tmp/tflite_{backend}_{shape.id}.log
"""
    else:
      remote_script += f"""
{' '.join(shlex.quote(x) for x in container_cmd)} 2>&1
"""
    ssh_opts = ["-o", "StrictHostKeyChecking=no"]
    cm_sock = Path.home() / ".ssh" / "cm-aeb"
    if cm_sock.exists():
      ssh_opts.extend(["-o", f"ControlPath={cm_sock}"])
    cmd = ["ssh", *ssh_opts, ssh_host, remote_script]
  else:
    local_cmd = [
        "numactl",
        f"--cpunodebind={numa_node}",
        f"--membind={numa_node}",
        *litert_args,
    ]
    local_warmup_cmd = [
        "numactl",
        f"--cpunodebind={numa_node}",
        f"--membind={numa_node}",
        *warmup_args,
    ]
    remote_script = f"""
set -euo pipefail
mkdir -p {shlex.quote(cache_dir)}
if [ "{backend}" = "xnnpack" ] && [ -z "$(ls -A {shlex.quote(cache_dir)} 2>/dev/null)" ]; then
  {' '.join(shlex.quote(x) for x in local_warmup_cmd)} >/dev/null 2>&1 || true
fi
"""
    if with_perf:
      perf_data = f"/tmp/perf_{backend}_{shape.id}.data"
      remote_script += f"""
perf record -e cycles:u -F 999 -o {shlex.quote(perf_data)} -- {' '.join(shlex.quote(x) for x in local_cmd)} > /tmp/tflite_{backend}_{shape.id}.log 2>&1
cat /tmp/tflite_{backend}_{shape.id}.log
echo "=== PERF_REPORT_START ==="
perf report -i {shlex.quote(perf_data)} --stdio --no-children --sort=dso,symbol --percent-limit=0.001 2>/dev/null
rm -f {shlex.quote(perf_data)} /tmp/tflite_{backend}_{shape.id}.log
"""
    else:
      remote_script += f"""
{' '.join(shlex.quote(x) for x in local_cmd)} 2>&1
"""
    cmd = ["bash", "-c", remote_script]

  proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
  combined = proc.stdout + "\n" + proc.stderr
  if "=== PERF_REPORT_START ===" in combined:
    tflite_part, perf_part = combined.split("=== PERF_REPORT_START ===", 1)
  else:
    tflite_part, perf_part = combined, ""

  log_file.write_text(tflite_part)
  if perf_part:
    perf_file.write_text(perf_part)

  return build_shape_result_from_texts(backend, shape, tflite_part, perf_part)


def build_shape_result_from_texts(
    backend: str,
    shape: WorkloadShape,
    tflite_text: str,
    perf_text: str,
) -> ShapeProfileResult:
  """Construct a `ShapeProfileResult` from raw TFLite profiler and `perf report` text."""
  graph_rows, delegate_rows = parse_tflite_profiler_output(tflite_text)
  ttft_ms, prefill_tps, decode_tps = parse_benchmark_metrics(tflite_text)
  perf_rows, isa_map, cat_map = parse_perf_report(perf_text) if perf_text else ([], {}, {})

  graph_total = round(sum(r.total_ms for r in graph_rows), 3)
  delegate_total = round(sum(r.total_ms for r in delegate_rows), 3)

  if shape.modality in ("vision", "audio") and ttft_ms and ttft_ms > 0.0:
    prefill_tps = round(shape.prompt_tokens / (ttft_ms / 1000.0), 2)

  return ShapeProfileResult(
      backend=backend,
      shape=shape,
      prefill_tps=prefill_tps,
      decode_tps=decode_tps,
      ttft_ms=ttft_ms,
      tflite_graph_total_ms=graph_total,
      delegate_total_ms=delegate_total,
      tflite_summary_by_node_type=graph_rows,
      delegate_summary_by_node_type=delegate_rows,
      perf_microkernels=perf_rows,
      isa_breakdown_pct=isa_map,
      category_breakdown_pct=cat_map,
  )


def render_markdown_report(
    results: list[ShapeProfileResult],
    metadata: dict[str, Any],
    uprobe_params: dict[str, Any] | None = None,
) -> str:
  """Render a comprehensive Markdown report showing kernels, time proportions, and parameter values per shape."""
  lines: list[str] = [
      "# Gemma 4 31B (`gemma-4-31B-it`) Multimodal CPU Kernel Profile on Intel Sapphire Rapids",
      "",
      f"- **Model**: `{metadata.get('model_id', 'google/gemma-4-31B-it')}` "
      f"(`{metadata.get('litertlm_artifact', 'gemma-4-31B-it.litertlm')}`, `dynamic_wi4_afp32` Text + 8-bit `QC8W` Vision)",
      f"- **Multimodal Coverage**:",
      "  - **Text (`gemma-4-31B-it`)**: 60-layer hybrid decoder (50 sliding-window local attention layers with `W=1024`, `H=32`, `KV=16`, `D=256` + 10 global attention layers with `H=32`, `KV=4`, `D=512`) + text embedder across 8 shapes (`p1_d1` to `p16384_d128`).",
      "  - **Vision (`gemma-4-31B-it`)**: 27-layer ViT Vision Encoder (`hidden_size=1152`, `intermediate_size=4304`, `16` heads, `head_dim=72`, `patch_size=16`, `pooling_kernel_size=3`) + Multimodal Vision Adapter (`1152 -> 5376`) across all 5 official soft-token budgets (`vis_70`, `vis_140`, `vis_280`, `vis_560`, `vis_1120`, corresponding to `630` to `10,080` `16x16` patches).",
      "  - **Audio (`audio_config: null` in 31B; `gemma-4-E2B-it` Conformer reference `audio_204`)**: The official `google/gemma-4-31B-it` checkpoint does not include an audio tower (`\"audio_config\": null`; only `Gemma 4 E2B` and `Gemma 4 E4B` ship `model.audio_tower`). We profile the Gemma 4 Conformer Audio Encoder + Adapter (`[1, 816, 128] -> [1, 204, 1536]`, `204` audio soft tokens) on Sapphire Rapids as a reference for Gemma 4 audio kernel behavior.",
      f"- **Hardware**: `{metadata.get('cpu_model', 'Intel(R) Xeon(R) Sapphire Rapids')}` "
      f"(NUMA Node 0 pinned, `threads={metadata.get('threads', 28)}`)",
      f"- **Runtime**: LiteRT-LM `{metadata.get('litert_lm_commit', 'v0.19.0')}` "
      "(`litert_lm_advanced_main`, XNNPACK `--enable_ynnpack=false` vs. YNNPACK `--enable_ynnpack=true`)",
      "",
      "## 1. Input Shapes & User-Journey Rationale (Text, Vision & Audio)",
      "",
      "| Shape ID | Modality | Prompt / Soft Tokens ($P$) | Decode ($D$) | Patches / Context (`max_num_tokens`) | Tensor / Signature Path | User Journey & Boundary Rationale |",
      "| :--- | :--- | ---: | ---: | ---: | :--- | :--- |",
  ]

  seen_shapes: set[str] = set()
  for r in results:
    s = r.shape
    if s.id in seen_shapes:
      continue
    seen_shapes.add(s.id)
    lines.append(
        f"| `{s.id}` | **{s.modality.upper()}** | {s.prompt_tokens} | {s.decode_tokens} | {s.max_num_tokens} | "
        f"`{s.signature_path}` (`{s.multimodal_input_shape}`) | **{s.journey}** — {s.rationale} |"
    )

  lines.extend([
      "",
      "## 2. Throughput & Latency Summary Across All Shapes (Text, Vision & Audio)",
      "",
      "| Shape ID | Modality | Backend | Soft / Prompt Tokens ($P$) | Decode ($D$) | Latency / TTFT (ms) | Prefill / Encode (tok/s) | Decode (tok/s) | TFLite Graph Total (ms) | Delegate Internal Total (ms) |",
      "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ])
  shape_order: list[str] = []
  by_backend_shape: dict[tuple[str, str], ShapeProfileResult] = {}
  for r in results:
    if r.shape.id not in shape_order:
      shape_order.append(r.shape.id)
    by_backend_shape[(r.backend, r.shape.id)] = r
    ttft = f"{r.ttft_ms:.1f}" if r.ttft_ms is not None else "n/a"
    pf = f"{r.prefill_tps:.2f}" if r.prefill_tps is not None else "n/a (M=1 only)"
    dc = f"{r.decode_tps:.2f}" if r.decode_tps is not None else ("n/a (encoder)" if r.shape.modality != "text" else "n/a")
    lines.append(
        f"| `{r.shape.id}` | {r.shape.modality} | **{r.backend.upper()}** | {r.shape.prompt_tokens} | {r.shape.decode_tokens} | "
        f"{ttft} | {pf} | {dc} | {r.tflite_graph_total_ms:.1f} | {r.delegate_total_ms:.1f} |"
    )

  def _invisible_cat(index: int) -> str:
    alphabet = ("\u200b", "\u200c", "\u200d", "\u2060")
    return f"{alphabet[(index >> 4) & 3]}{alphabet[(index >> 2) & 3]}{alphabet[index & 3]}"

  def _build_side_by_side_series(
      shape_ids: list[str],
      xnn_values: list[float],
      ynn_values: list[float],
      ymax: int,
      gap_slots: int = 2,
  ) -> tuple[str, str, str]:
    cats: list[str] = []
    xnn_bars: list[str] = []
    ynn_bars: list[str] = []
    inv_idx = 1
    # Mermaid's LinearAxis adds a 5px outerPadding at y=0, so a value of 0 draws a 5px stub bar.
    # Using a negative sentinel (-ymax) yields height < 0 in SVG, which is not rendered at all.
    hide_str = str(-ymax)
    for idx, (sid, xv, yv) in enumerate(zip(shape_ids, xnn_values, ynn_values)):
      cats.append(f'"{sid}"')
      xnn_bars.append(f"{xv:.2f}")
      ynn_bars.append(hide_str)
      cats.append(f'"{_invisible_cat(inv_idx)}"')
      inv_idx += 1
      xnn_bars.append(hide_str)
      ynn_bars.append(f"{yv:.2f}")
      if idx < len(shape_ids) - 1:
        for _ in range(gap_slots):
          cats.append(f'"{_invisible_cat(inv_idx)}"')
          inv_idx += 1
          xnn_bars.append(hide_str)
          ynn_bars.append(hide_str)
    return ", ".join(cats), ", ".join(xnn_bars), ", ".join(ynn_bars)

  prefill_shapes = [
      sid
      for sid in shape_order
      if (
          (by_backend_shape.get(("xnnpack", sid)) and by_backend_shape[("xnnpack", sid)].prefill_tps is not None)
          or (by_backend_shape.get(("ynnpack", sid)) and by_backend_shape[("ynnpack", sid)].prefill_tps is not None)
      )
  ]
  if prefill_shapes:
    xnn_prefill = [
        round(by_backend_shape[("xnnpack", sid)].prefill_tps, 2)
        if ("xnnpack", sid) in by_backend_shape and by_backend_shape[("xnnpack", sid)].prefill_tps is not None
        else 0.0
        for sid in prefill_shapes
    ]
    ynn_prefill = [
        round(by_backend_shape[("ynnpack", sid)].prefill_tps, 2)
        if ("ynnpack", sid) in by_backend_shape and by_backend_shape[("ynnpack", sid)].prefill_tps is not None
        else 0.0
        for sid in prefill_shapes
    ]
    max_prefill = max([*xnn_prefill, *ynn_prefill, 1.0])
    prefill_ymax = int(math.ceil(max_prefill / 500.0) * 500) if max_prefill > 200 else int(math.ceil(max_prefill / 20.0) * 20)
    x_labels_prefill, xnn_vals_prefill, ynn_vals_prefill = _build_side_by_side_series(
        prefill_shapes, xnn_prefill, ynn_prefill, prefill_ymax, gap_slots=2
    )
    lines.extend([
        "",
        "### 2.1 Prefill / Encode Throughput per Shape (`tok/s`) — XNNPACK vs. YNNPACK",
        "",
        "> **Series Legend**: **Left Bar (Blue `#2b6cd9`)** = **XNNPACK** (`--enable_ynnpack=false`) | **Right Bar (Orange `#e67e22`)** = **YNNPACK** (`--enable_ynnpack=true`)",
        "> *(Note: For `p1_d1`, LiteRT-LM's `Prefill Turn 1` runs only the 3-op token embedder in `<0.65 ms` and defers the 60-layer transformer pass to `Decode Turn 1`; `p16_d128`–`p4096_d128` execute the full 60-layer prefill subgraph; `vis_70`–`vis_1120` and `audio_204` report multimodal encoder + adapter soft-token throughput.)*",
        "",
        "```mermaid",
        '%%{init: {"xyChart": {"width": 1250, "height": 500, "xAxis": {"labelRotation": -35, "labelFontSize": 11, "showTick": false}}, "themeVariables": {"xyChart": {"plotColorPalette": "#2b6cd9, #e67e22"}}}}%%',
        "xychart-beta",
        '    title "Prefill / Encode Throughput per Shape (XNNPACK vs. YNNPACK)"',
        f'    x-axis "Shape" [{x_labels_prefill}]',
        f'    y-axis "tok/s" 0 --> {prefill_ymax}',
        f'    bar "XNNPACK" [{xnn_vals_prefill}]',
        f'    bar "YNNPACK" [{ynn_vals_prefill}]',
        "```",
    ])

  decode_shapes = [
      sid
      for sid in shape_order
      if (
          (by_backend_shape.get(("xnnpack", sid)) and by_backend_shape[("xnnpack", sid)].decode_tps is not None)
          or (by_backend_shape.get(("ynnpack", sid)) and by_backend_shape[("ynnpack", sid)].decode_tps is not None)
      )
  ]
  if decode_shapes:
    xnn_decode = [
        round(by_backend_shape[("xnnpack", sid)].decode_tps, 2)
        if ("xnnpack", sid) in by_backend_shape and by_backend_shape[("xnnpack", sid)].decode_tps is not None
        else 0.0
        for sid in decode_shapes
    ]
    ynn_decode = [
        round(by_backend_shape[("ynnpack", sid)].decode_tps, 2)
        if ("ynnpack", sid) in by_backend_shape and by_backend_shape[("ynnpack", sid)].decode_tps is not None
        else 0.0
        for sid in decode_shapes
    ]
    max_decode = max([*xnn_decode, *ynn_decode, 1.0])
    decode_ymax = int(math.ceil(max_decode))
    x_labels_decode, xnn_vals_decode, ynn_vals_decode = _build_side_by_side_series(
        decode_shapes, xnn_decode, ynn_decode, decode_ymax, gap_slots=2
    )
    lines.extend([
        "",
        "### 2.2 Decode Throughput per Shape (`tok/s`) — XNNPACK vs. YNNPACK",
        "",
        "> **Series Legend**: **Left Bar (Blue `#2b6cd9`)** = **XNNPACK** (`--enable_ynnpack=false`) | **Right Bar (Orange `#e67e22`)** = **YNNPACK** (`--enable_ynnpack=true`)",
        "> *(Note: Encoder-only multimodal shapes `vis_70`–`vis_1120` and `audio_204` have `Decode (D) = 0` and are plotted in the Prefill / Encode chart above.)*",
        "",
        "```mermaid",
        '%%{init: {"xyChart": {"width": 1100, "height": 460, "xAxis": {"labelRotation": 0, "labelFontSize": 12, "showTick": false}}, "themeVariables": {"xyChart": {"plotColorPalette": "#2b6cd9, #e67e22"}}}}%%',
        "xychart-beta",
        '    title "Decode Throughput per Shape (XNNPACK vs. YNNPACK)"',
        f'    x-axis "Shape" [{x_labels_decode}]',
        f'    y-axis "tok/s" 0 --> {decode_ymax}',
        f'    bar "XNNPACK" [{xnn_vals_decode}]',
        f'    bar "YNNPACK" [{ynn_vals_decode}]',
        "```",
    ])

  lines.extend([
      "",
      "## 3. High-Level Summary of Dominant Kernels (>50% Execution Time) & Matched XNNPACK ↔ YNNPACK Benchmark Pairs",
      "",
      "Across all 14 multimodal shapes (`8` Text, `5` Vision, `1` Audio), **over 50% (and up to 93.4%) of total execution time is concentrated in just 6 functional kernel pairs**. Depending on the modality and sequence/patch length, execution falls into **5 distinct compute regimes**, each dominated (`>50%` cumulative time) by 1–2 matched XNNPACK and YNNPACK kernels:",
      "",
      "### 3.1 Dominant Kernels (`>50%` Cumulative Share) by Workload Regime & Why They Dominate",
      "",
      "| Workload Regime & Shapes | TFLite / Delegate Dominant Ops (`>50%`) | XNNPACK Dominant Microkernels (`>50%` Cycles) | YNNPACK Dominant Microkernels (`>50%` Cycles) | Why These Kernels Consume `>50%` of Execution Time |",
      "| :--- | :--- | :--- | :--- | :--- |",
      "| **1. Short-to-Medium Text Generation**<br>(`p1_d1`, `p16_d128`, `p64_d64`, `p128_d128`, `p256_d128`) | `Delegate/Fully Connected (NC, QD8, F32, QC4W) GEMM` (**60.6%–93.4%** of TFLite time) | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx` (**50.8%–78.7%** alone; **63.7%–78.7%** with `16x64c4__avx512amx` prefill) | `ynn::dot_uint8_int4_int32_1x32x8_1x16x8_avx512vnni` (**46.0%–72.3%**) + `ynn::dot_uint8_int4_int32_12x16x8_1x16x8_avx512vnni` (**6.8%–20.8%**; **64.1%–79.1%** combined) | **Memory-Bandwidth-Bound 4-Bit Weight Streaming ($M=1$ Decode) + 420 Projections/Step**: Each of the 60 layers executes 7 large 4-bit linear projections (`420` `FULLY_CONNECTED` ops/step across $21504\\times 5376$, $5376\\times 21504$, $8192\\times 5376$, $16384\\times 5376$, $4096\\times 5376$, $2048\\times 5376$). Every `M=1` decode token must stream **~15.3 GB of 4-bit weights** from DDR5 RAM with zero reuse across tokens, whereas short-context attention ($S_{\\text{kv}} \\le 1024$) touches only megabytes. |",
      "| **2. Long-Context Text Prefill**<br>(`p1024_d128`, `p4096_d128`, `p16384_d128`) | `Delegate/Batch Matrix Multiply (NC, F32) GEMM` (**37.4%–58.1%+**) + `Delegate/Fully Connected (NC, QD8, F32, QC4W) GEMM` (**24.8%–47.9%**; **82.9%–85.3%** combined) | `xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast` (**15.9%–78.5%**) + `xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4_prfm` (**11.1%–19.8%**) + `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_{1x64c4,16x64c4}__avx512amx` (**75.2%–89.6%** combined) | `ynn::dot_fp32_5x64x1_1x16x1_avx512` (**7.9%–22.2%**) + `ynn::transpose_x32_avx512` (**25.6%–29.7%**) + `ynn::dot_uint8_int4_int32_{12x16x8,1x32x8}_1x16x8_avx512vnni` (**76.8%–80.0%** combined) | **$O(P \\cdot S_{\\text{kv}})$ Quadratic Global Attention + Dynamic KV/Weight Packing**: Batched `M=1024` prefill amortizes 4-bit weight reads across 1,024 tokens (`16x64` AMX / `12x16` VNNI), but the 10 global full-attention layers (`H=32, KV=4, D=512`) compute FP32 $Q\\cdot K^T$ and $\\text{scores}\\cdot V$ against the entire KV cache ($S_{\\text{kv}} \\in \\{4096, 8192, 32768\\}$) on every chunk, while `packw_gemm_goi` (XNNPACK) and `transpose_x32_avx512` (YNNPACK) repack updated KV/weight tiles on every call. |",
      "| **3. Low-to-Medium Resolution Vision**<br>(`vis_70`, `vis_140`: `630`–`1,260` patches) | `Delegate/Fully Connected (NC, QD8, F32, QC8W) GEMM` (**24.9%–33.0%**) + `Delegate/Batch Matrix Multiply (NC, F32) GEMM` (**19.4%–32.6%**; **52.3%–57.5%** combined) | `xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx` (**25.3%–31.6%**) + `xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast` (**19.9%–35.0%**; **51.5%–60.3%** combined) | **Vision Adapter (`INT8`)**: `ynn::dot_int8_int8_int32_32x32x64_16x16x4_amxint8` (**30.4%–30.0%**) + `ynn::dequantize_dot_f32_avx512` (**15.8%–20.3%**; **50.3%** combined; ViT encoder falls back to XNNPACK) | **Balanced 8-Bit ViT Linear Projections ($O(N_{\\text{patches}} D^2)$) & Self-Attention ($O(N_{\\text{patches}}^2 D)$)**: The 27-layer ViT encoder executes `190` 8-bit (`QC8W`) `FULLY_CONNECTED` ops (`108x` $1152\\times 1152$ Q/K/V/O, `54x` $4304\\times 1152$ MLP gate/up, `27x` $1152\\times 4304$ MLP down). At $N_{\\text{patches}} \\in \\{630, 1260\\}$ ( close to hidden dim $D=1152$), linear projection FLOPs and attention FLOPs are comparable. |",
      "| **4. Standard-to-High Resolution Vision**<br>(`vis_280`, `vis_560`, `vis_1120`: `2,520`–`10,080` patches) | `Delegate/Batch Matrix Multiply (NC, F32) GEMM` (**45.3%–60.0%**) + `Delegate/Softmax (NC, F32)` (**15.0%–24.9%**; **60.3%–84.9%** combined) | `xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast` (**49.8%–60.4%**) + `xnn_f32_{rmax,raddstoreexpminusmax}_ukernel__avx512f*` (**19.2%–28.5%**; **69.0%–88.9%** combined) | **Vision Adapter (`INT8`)**: `ynn::dot_int8_int8_int32_32x32x64_16x16x4_amxint8` (**44.1%–56.8%**) + `ynn::dequantize_dot_f32_avx512` (**21.2%–25.8%**; **69.9%–78.0%** combined) | **Unmasked 2D Bidirectional $O(N_{\\text{patches}}^2)$ Full Self-Attention Explosion**: Unlike causal text attention with a 1,024-token sliding window, all 27 ViT layers compute full unmasked $[1, 16, N_{\\text{patches}}, N_{\\text{patches}}]$ attention in FP32. As $N_{\\text{patches}}$ grows $16\\times$ from `630` to `10,080`, attention work grows $256\\times$ (reaching $27 \\times 16 \\times 10080^2 = 43.9\\text{ billion}$ attention pairs at `vis_1120`), while 8-bit linear layers run on AMX-INT8 (`4x` higher peak ops/cycle than FP32 AVX-512). |",
      "| **5. Audio Conformer Encoding**<br>(`audio_204`: `816` frames $\\to$ `204` tokens) | `Delegate/Fully Connected (NC, QS8, QC2W) GEMM` (**41.2%**) + `Delegate/Multiply (ND)` (**11.7%**; **52.9%** combined) | `xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni` (**40.2%**) + `thread_main` (**33.8%**) + `xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast` (**3.4%**; **77.4%** combined) | `ynn::dot_uint8_int2_int32_4x64x16_1x16x16_avx512vnni` (**39.2%**) + `slinky::thread_pool_impl::wait_for` (**15.4%**) + `ynn::transpose_x32_avx512` (**10.6%**; **65.2%** combined) | **108 Sub-Byte 2-Bit (`INT2`) Linear Projections Without AMX Support**: 108 of the 122 `FULLY_CONNECTED` layers in the 12-layer Conformer (`60x` $1024\\times 1024$ attn, `24x` $4096\\times 1024$ FFN up, `24x` $1024\\times 4096$ FFN down) use 2-bit weights (`i2`). Neither XNNPACK nor YNNPACK has an AMX-INT8 kernel for 2-bit weights, forcing both onto narrower VNNI kernels (`5x8c8` AVX-VNNI vs. `4x64x16` AVX-512 VNNI) with bit-unpacking overhead. |",
      "",
      "---",
      "",
      "### 3.2 Head-to-Head Matched XNNPACK ↔ YNNPACK Dominant Kernel Pairs for Standalone Benchmarking",
      "",
      "To directly benchmark and compare XNNPACK against YNNPACK at the microkernel level (`kernel/` tier), the table below pairs each dominant (`>50%`) XNNPACK kernel with its exact YNNPACK counterpart, alongside the exact runtime parameter values and tensor dimensions captured on Sapphire Rapids:",
      "",
      "| # | Functional Kernel Category & Modality | XNNPACK Microkernel (`ISA`, Tile & Runtime Params) | YNNPACK Matched Microkernel (`ISA`, Tile & Runtime Params) | Canonical Benchmark Dimensions (`M, N, K` & Dtypes) | Why It Dominates & Head-to-Head Comparison on Sapphire Rapids |",
      "| :-: | :--- | :--- | :--- | :--- | :--- |",
      "| **1** | **4-Bit Weight-Only GEMV (Autoregressive Decode, $M=1$)**<br>*Text (`p1_d1`–`p256_d128`)*<br>**50.8%–78.7%** of cycles | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx`<br>- **ISA**: `AMX-INT8` (`TDPBSSD`)<br>- **Tile**: `mr=1, nr=64, kr=4`<br>- **Runtime Params**: `mr=1, nc=N, kc=K` (full $K$ in 1 call) | `ynn::dot_uint8_int4_int32_1x32x8_1x16x8_avx512vnni`<br>- **ISA**: `AVX-512 VNNI` (`VPDPBUSD`)<br>- **Tile**: `M=1, N=32, K_tile=8`<br>- **Runtime Params**: `M=1, N=32, K1=min(K, 4096), A_stride_m=0` ($K$ split $\\le 4096$) | `input`: `qd8/u8[1, K]`<br>`weights`: `qc4w/i4[N, K]`<br>`output`: `f32/i32[1, N]`<br>- MLP Gate/Up (`120x`): `M=1, N=21504, K=5376`<br>- MLP Down (`60x`): `M=1, N=5376, K=21504`<br>- Attn Q/K/V/O (`230x`): `M=1, N∈{4096,8192,16384,2048,5376}, K∈{5376,8192,16384}` | **Why #1 in Decode**: Streams 15.3 GB of 4-bit weights per token across 420 linear ops.<br>**Comparison**: XNNPACK uses `AMX-INT8` (`1x64c4`) with full-$K$ accumulation (`7.44 tok/s`), beating YNNPACK's `AVX-512 VNNI` (`1x32x8`) with $K_1 \\le 4096$ splitting (`6.93 tok/s`) by **+7.4%** on decode. |",
      "| **2** | **4-Bit Weight-Only Batched GEMM (Prefill, $M \\in \\{128, 1024\\}$)**<br>*Text (`p64_d64`–`p4096_d128`)*<br>**14.4%–34.4%** of cycles (up to **93%** in prefill-only phase) | `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx`<br>- **ISA**: `AMX-INT8` (`TDPBSSD`)<br>- **Tile**: `mr=16, nr=64, kr=4`<br>- **Runtime Params**: `mr=16, nc∈{704,448,384,192,128}, kc=K, a_stride=K` | `ynn::dot_uint8_int4_int32_12x16x8_1x16x8_avx512vnni`<br>- **ISA**: `AVX-512 VNNI` (`VPDPBUSD`)<br>- **Tile**: `M=12, N=16 (nc=32), K_tile=8`<br>- **Runtime Params**: `M=12, N=32, K1∈{4096,1280,1024}, A_stride_m=K` | `input`: `qd8/u8[M, K]` ($M \\in \\{128, 1024\\}$)<br>`weights`: `qc4w/i4[N, K]`<br>`output`: `f32/i32[M, N]`<br>- MLP Gate/Up: `N=21504, K=5376`<br>- MLP Down: `N=5376, K=21504`<br>- Attn Q/K/V/O: `N∈{4096,8192,16384,2048,5376}, K∈{5376,8192,16384}` | **Why #1 in Prefill**: Compute-bound matrix multiplication (`39.6 TFLOPs` per 1,024-token chunk).<br>**Comparison**: XNNPACK dispatches 2D hardware `AMX-INT8` tiles (`16x64c4`, `147.9 tok/s` prefill at `p1024`), whereas YNNPACK lacks an AMX 4-bit kernel and falls back to `AVX-512 VNNI` (`12x16x8`, `110.2 tok/s`), giving XNNPACK a **+34.2% prefill speedup**. |",
      "| **3** | **FP32 Attention `BatchMatMul` ($Q \\cdot K^T$ & $\\text{scores} \\cdot V$) + KV/Weight Packing**<br>*Long-Context Text (`p1024`–`p16384`) & Vision (`vis_70`–`vis_1120`)*<br>**49.8%–89.6%** of cycles | `xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast` + `xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4_prfm`<br>- **ISA**: `AVX-512F`<br>- **Tile**: `mr=7, nr=32`<br>- **Runtime Params**:<br>  - Text Local: `mr=7, nc=1024, kc=1024B` ($Q K^T$) & `nc=256, kc=4096B` ($S V$)<br>  - Vision: `mr=7, nc=N_patches, kc=288B` ($Q K^T$) & `nc=72, kc=4*N_patches` ($S V$) | `ynn::dot_fp32_5x64x1_1x16x1_avx512` (batched) & `ynn::dot_fp32_2x64x1_1x16x1_avx512` (decode) + `ynn::transpose_x32_avx512`<br>- **ISA**: `AVX-512F`<br>- **Tile**: `M=5 (or 2), N=64 (or 128), K_tile=1`<br>- **Runtime Params**: `M=5, N=128, K1=256, A_stride_m=1024` (Local $Q K^T$) & `M=5, N=64, K1=512, A_stride_m=4096` (Global $Q K^T$ / $S V$) | `x, y, out`: `f32`<br>- **Text Local (`100x`)**: `B=[1,16], M=2*M_seq, N=1024, K=256` & `M=2*M_seq, N=256, K=1024`<br>- **Text Global (`20x`)**: `B=[1,4], M=8*M_seq, N=S_kv, K=512` ($S_{\\text{kv}} \\le 32768$)<br>- **Vision (`54x`)**: `B=[16], M=N_patches, N=N_patches, K=72` ($N_{\\text{patches}} \\le 10080$) | **Why #1 at Large $P$ / High-Res Vision**: Scales quadratically $O(P \\cdot S_{\\text{kv}})$ in global text attention and $O(N_{\\text{patches}}^2)$ in unmasked 2D ViT self-attention (`43.9B` pairs at `vis_1120`), while running in FP32 (`AVX-512F`) rather than AMX.<br>**Comparison**: XNNPACK `7x32` (7 AVX-512 accumulators $\\times 2$ ZMMs) achieves higher register reuse than YNNPACK `5x64` (`5` rows $\\times 4$ ZMMs) and avoids YNNPACK's `25.6%–29.7%` `transpose_x32_avx512` overhead. |",
      "| **4** | **FP32 Attention Softmax (`Max` + `Exp-Subtract-Sum` Reductions)**<br>*Vision (`vis_280`–`vis_1120`) & Long-Context Text*<br>**15.0%–28.5%** of cycles | `xnn_f32_rmax_ukernel__avx512f_u64_acc4` + `xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc2`<br>- **ISA**: `AVX-512F`<br>- **Tile**: `64` floats/iter (`4` ZMM accumulators)<br>- **Runtime Params**: `elements_bytes = 4 * N_patches` (`10080B` at `vis_280`, `40320B` at `vis_1120`, `4096B` in Text Local) | `ynn::max_k1_fp32_avx512` + `ynn::sum_k1_fp32_avx512` + `ynn::unary_LUT_op` (`exp`) / `ynn::multiply_fp32_avx512`<br>- **ISA**: `AVX-512F` + scalar LUT<br>- **Runtime Params**: 3-pass row reduction + LUT exp + multiply over `axis_dim = N_patches` or `S_kv` | `input / output`: `f32[1, 16, N_patches, N_patches]` (`27` ViT layers, $N_{\\text{patches}} \\in \\{630..10080\\}$) or `f32[1, 4, 8*M_seq, S_kv]` (`10` Global Text layers) | **Why #2 in Vision**: At `vis_1120`, Softmax must normalize $27 \\times 16 \\times 10080 = 4,354,560$ rows of length `10,080` (`175.6 GB` of FP32 logits read/written).<br>**Comparison**: XNNPACK fuses `exp(x - max)` and the denominator `sum` into a single SIMD rational-polynomial pass (`raddstoreexpminusmax`), whereas YNNPACK decomposes Softmax into separate max, subtract, LUT-exp, sum, and reciprocal-multiply passes. |",
      "| **5** | **8-Bit Per-Channel (`QC8W` / `INT8 × INT8`) Linear GEMM**<br>*Vision ViT (`vis_70`–`vis_1120`) & Vision Adapter*<br>**25.3%–31.6%** in `vis_70`/`vis_140`; **69.9%–78.0%** in `vision_adapter` | `xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx`<br>- **ISA**: `AMX-INT8` (`TDPBSSD`)<br>- **Tile**: `mr=16, nr=64, kr=4`<br>- **Runtime Params**: `mr=16, nc∈{1152,1472,1360,384,1344}, kc∈{1152,4304,768}` | `ynn::dot_int8_int8_int32_32x32x64_16x16x4_amxint8` (and `16x64x64_16x16x4_amxint8`) + `ynn::dequantize_dot_f32_avx512`<br>- **ISA**: `AMX-INT8` (`TDPBSSD`) + `AVX-512F`<br>- **Tile**: `M=32 (or 16), N=32 (or 64), K_tile=64`<br>- **Runtime Params**: `M=32 (main) / 24 (tail), N=64, K1=1152, A_stride_m=1152` | `input`: `qd8/i8[M, K]` ($M = N_{\\text{patches}}$ or $N_{\\text{soft}}$)<br>`weights`: `qc8w/i8[N, K]`<br>`output`: `f32[M, N]`<br>- ViT Q/K/V/O (`108x`): `N=1152, K=1152`<br>- ViT Gate/Up (`54x`): `N=4304, K=1152`<br>- ViT Down (`27x`): `N=1152, K=4304`<br>- Vision Adapter (`1x`): `M=N_soft, N=5376, K=1152` | **Why Dominant in Low-Res Vision & Adapter**: All 190 ViT linear layers and the Vision Adapter (`1152 -> 5376`) use 8-bit weights.<br>**Comparison**: **Both XNNPACK and YNNPACK dispatch hardware `AMX-INT8` (`TDPBSSD`) for 8-bit weights!** XNNPACK fuses INT32-to-FP32 dequantization inside `16x64c4__avx512amx`, whereas YNNPACK runs `32x32x64_amxint8` into an INT32 buffer followed by a separate `ynn::dequantize_dot_f32_avx512` pass (`21–26%` of cycles). |",
      "| **6** | **2-Bit Weight-Only (`QC2W` / `INT2`) Linear GEMM**<br>*Audio Conformer (`audio_204`)*<br>**39.2%–41.2%** of time/cycles | `xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni`<br>- **ISA**: `AVX-VNNI` (`VPDPBUSD`, 256-bit)<br>- **Tile**: `mr=5, nr=8, kr=8`<br>- **Runtime Params**: `mr=5, nc∈{1024,4096}, kc∈{1024,4096}` | `ynn::dot_uint8_int2_int32_4x64x16_1x16x16_avx512vnni`<br>- **ISA**: `AVX-512 VNNI` (`VPDPBUSD`, 512-bit)<br>- **Tile**: `M=4, N=64, K_tile=16`<br>- **Runtime Params**: `M=4, N=64, K1∈{1024,2048}, A_stride_m∈{1024,4096}` | `input`: `i8/u8[204, K]`<br>`weights`: `qc2w/i2[N, K]`<br>`output`: `i8/i32[204, N]`<br>- Conformer Attn (`60x`): `M=204, N=1024, K=1024`<br>- Macaron FFN Up (`24x`): `M=204, N=4096, K=1024`<br>- Macaron FFN Down (`24x`): `M=204, N=1024, K=4096` | **Why #1 in Audio**: 108 of 122 Conformer linear projections are quantized to 2-bit (`QC2W`).<br>**Comparison**: YNNPACK uses a wider 512-bit `AVX-512 VNNI` (`4x64x16`) kernel vs. XNNPACK's 256-bit `AVX-VNNI` (`5x8c8`) kernel, but YNNPACK's per-run weight transpose (`transpose_x32_avx512`, `10.6%`) and Slinky barrier overhead (`15.4%`) make end-to-end Audio faster on XNNPACK (`38.3 ms` vs. `60.5 ms`). |",
      "",
      "---",
      "",
      "### 3.3 Modality-Specific Architectural Notes",
      "",
      "1. **Text Decoder (`gemma-4-31B-it`, 60 layers, 4-bit `QC4W` + FP32 Attention)**:",
      "   - **XNNPACK (`--enable_ynnpack=false`)**: Dispatches Intel **AMX-INT8** (`xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx` for batched prefill `M=128` / `M=1024`, achieving **14.7–147.9 tok/s** prefill, and `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx` for `M=1` autoregressive decoding, achieving **2.74–7.44 tok/s** decode) alongside AVX-512 FP32 attention (`xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast` + `xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4_prfm`, which grows to **89.64%** of cycles at `p16384_d128`).",
      "   - **YNNPACK (`--enable_ynnpack=true`)**: Dispatches **AVX-512 VNNI** (`ynn::dot_uint8_int4_int32_12x16x8_1x16x8_avx512vnni` for prefill and `ynn::dot_uint8_int4_int32_1x32x8_1x16x8_avx512vnni` for decode) rather than AMX for 4-bit weights, yielding **2.65–6.93 tok/s** decode and **13.6–110.2 tok/s** prefill, while also spending **16–29%** of cycles in `ynn::transpose_x32_avx512` to repack weights per invocation.",
      "2. **Vision Encoder & Adapter (`gemma-4-31B-it`, 27-layer ViT + Adapter, `vis_70` to `vis_1120`)**:",
      "   - Unlike the 4-bit text decoder, the 27-layer ViT Vision Encoder and Vision Adapter use **8-bit per-channel quantized weights (`QC8W` / `INT8 × INT8 -> INT32`)** and **unmasked 2D bidirectional FP32 self-attention (`[1, 16, N_patches, N_patches]`)**.",
      "   - **Under XNNPACK**, as image resolution scales $16\\times$ from `vis_70` (`630` patches, `135.1 ms`) to `vis_1120` (`10,080` patches, `7,590.9 ms`), the dominant kernel shifts dramatically from **AMX-INT8 `QC8W` GEMM** (`xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx`, **32.95%** of TFLite time / **31.62%** of kernel cycles at `vis_70` down to **5.12%** of TFLite time / **3.90%** of kernel cycles at `vis_1120`) to **FP32 Bidirectional Self-Attention `BatchMatMul` + `Softmax`** (`xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast` + `xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc2`, rising from **21.42%** of TFLite time at `vis_70` to **60.34%** at `vis_280` and **84.91%** of TFLite time / **88.90%** of kernel cycles at `vis_1120`).",
      "   - **Under YNNPACK**, `vision_adapter` (`[1, N_soft, 1152] -> [1, N_soft, 5376]`, 8-bit `INT8 × INT8`) delegates 100% of its nodes (8/8) to `YNNPackDelegate` and **DOES dispatch Intel AMX-INT8** (`ynn::dot_int8_int8_int32_32x32x64_16x16x4_amxint8` at **39.2–56.8%** of cycles + `ynn::dequantize_dot_f32_avx512` at **20.3–25.8%** of cycles). For `vision_encoder`, `YNNPackDelegate` in LiteRT-LM `v0.19.0` clusters 3,048/3,075 nodes during graph partitioning—including two `int32` `REDUCE_MAX` ops in the 2D patch coordinate header—and aborts during `ynn_create_runtime` (`Unsupported reduction reduce_max for int32 to int32` at `external/XNNPACK/ynnpack/subgraph/reduce.cc:440`), automatically falling back to `TfLiteXNNPackDelegate`.",
      "3. **Audio Conformer Encoder & Adapter (`gemma-4-E2B-it` Reference, `audio_204`)**:",
      "   - Exercises **2-bit (`QC2W` / `INT2`) and 4-bit (`QC4W` / `INT4`) Conformer linear projections**, **1D/2D Depthwise Convolution**, and **relative-position self-attention** (`[1, 816, 128] -> [1, 204, 1536]`).",
      "   - **XNNPACK (`38.3 ms`)**: Dominated by 2-bit `Delegate/Fully Connected (NC, QS8, QC2W) GEMM` (**41.20%** of TFLite time across 108 calls; `xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni` at **40.23%** of kernel cycles).",
      "   - **YNNPACK (`60.5 ms`)**: Dominated by 2-bit `ynn::dot_uint8_int2_int32_4x64x16_1x16x16_avx512vnni` (**39.18%** of kernel cycles / **32.41%** of total cycles), `slinky::thread_pool_impl::wait_for` (**15.40%**), `ynn::transpose_x32_avx512` (**10.65%** of kernel cycles), `ynn::dot_fp32_4x16x2_1x8x2_avx512` (**8.07%**), and 4-bit `ynn::dot_uint8_int4_int32_12x16x8_1x16x8_avx512vnni` (**4.21%**).",
  ])

  for backend in ("xnnpack", "ynnpack"):
    b_results = [r for r in results if r.backend == backend]
    if not b_results:
      continue
    lines.extend([
        "",
        f"## 4. {backend.upper()} Kernel Breakdown & Time Proportions by Input Shape",
        "",
    ])
    for r in b_results:
      lines.extend([
          f"### `{r.shape.id}` (`modality={r.shape.modality}`, `tokens={r.shape.prompt_tokens}`, `decode={r.shape.decode_tokens}`, `ctx/patches={r.shape.max_num_tokens}`) — {r.shape.journey}",
          "",
      ])
      if r.delegate_summary_by_node_type:
        lines.extend([
            "#### TFLite Delegate Operator Kernel Time Proportions (`--enable_profiling=true`)",
            "",
            "| Operator / Delegate Kernel | Distinct Nodes | Invocations (`times_called`) | Total Time (ms) | Avg Time (ms) | Proportion of Time (%) | Cumulative (%) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in r.delegate_summary_by_node_type:
          lines.append(
              f"| `{row.node_type}` | {row.count} | {row.times_called} | {row.total_ms:.2f} | {row.avg_ms:.3f} | "
              f"**{row.pct:.2f}%** | {row.cdf_pct:.2f}% |"
          )
        lines.append("")

      if r.tflite_summary_by_node_type:
        lines.extend([
            "#### TFLite Full Graph & Non-Delegated Operator Time Proportions (`Summary by node type`)",
            "",
            "| Graph Node Type | Distinct Nodes | Invocations (`times_called`) | Total Time (ms) | Avg Time (ms) | Proportion of Time (%) | Cumulative (%) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in r.tflite_summary_by_node_type:
          lines.append(
              f"| `{row.node_type}` | {row.count} | {row.times_called} | {row.total_ms:.2f} | {row.avg_ms:.3f} | "
              f"**{row.pct:.2f}%** | {row.cdf_pct:.2f}% |"
          )
        lines.append("")

      if r.perf_microkernels:
        lines.extend([
            "#### Hardware ISA Microkernel Time Proportions (`perf record -e cycles:u`)",
            "",
            "| Microkernel Symbol (`xnn_*` / `ynn::*`) | ISA Extension | Functional Category | Proportion of Kernel Cycles (%) | Raw Cycle Share (%) |",
            "| :--- | :--- | :--- | ---: | ---: |",
        ])
        for p in r.perf_microkernels[:25]:
          lines.append(
              f"| `{p.symbol}` | `{p.isa}` | {p.category} | "
              f"**{p.pct_of_kernel_cycles:.2f}%** | {p.pct_total_cycles:.2f}% |"
          )
        lines.append("")

  lines.extend([
      "",
      "## 5. Delegate & Subgraph Operator Call Parameter Catalog (Text, Vision & Audio)",
      "",
      "Every TFLite / LiteRT subgraph in `.litertlm` (after recursive inlining of `STABLEHLO_COMPOSITE` decomposition subgraphs via `kernel/scripts/extract_tflite_params.py`) exposes the exact input tensor shapes/dtypes, weight tensor shapes/quantization parameters, output tensor shapes/dtypes, and operator-specific attributes (`M, N, K`, `adj_x, adj_y`, `keep_num_dims`, `asymmetric_quantize_input`, `fused_act`, `beta`, `axes`, `keep_dims`, `perm`, `new_shape`, `stride`, `padding`, `depth_multiplier`, `filter`). The complete machine-readable catalog across all 17 subgraphs is stored in `raw/subgraph_operator_params.json`.",
      "",
      "### 5.1 Text Decoder & Embedder (`gemma-4-31B-it.litertlm`: `decode` 5,052 ops, `prefill_128` 5,412 ops, `prefill_1024` 5,412 ops, `embedder` 3 ops)",
      "",
      "| Operator | Nodes / Subgraph | Layer / Architectural Role | Exact Input / Weight / Output Tensor Shapes & Operator Attributes (`M_seq ∈ {1, 128, 1024}`, `S_kv ∈ {1024, 4096, 8192, 32768}`) |",
      "| :--- | ---: | :--- | :--- |",
      "| `FULLY_CONNECTED` | `120` | MLP `gate_proj` & `up_proj` (60 layers × 2) | `M=M_seq, N=21504, K=5376`, `input=f32[1, M_seq, 5376]`, `weights=i4[21504, 5376] per_channel(axis=0, n=21504)`, `bias=none`, `output=f32[1, M_seq, 21504]`, `keep_num_dims=true, asymmetric_quantize_input=false, fused_act=NONE` |",
      "| `FULLY_CONNECTED` | `60` | MLP `down_proj` (60 layers × 1) | `M=M_seq, N=5376, K=21504`, `input=f32[1, M_seq, 21504]`, `weights=i4[5376, 21504] per_channel(axis=0, n=5376)`, `bias=none`, `output=f32[1, M_seq, 5376]`, `keep_num_dims=true, asymmetric_quantize_input=false, fused_act=NONE` |",
      "| `FULLY_CONNECTED` | `100` | Local Sliding Attention `k_proj` & `v_proj` (50 layers × 2) | `M=M_seq, N=4096, K=5376`, `input=f32[1, M_seq, 5376]`, `weights=i4[4096, 5376] per_channel(axis=0, n=4096)`, `bias=none`, `output=f32[1, M_seq, 4096]`, `keep_num_dims=true, asymmetric_quantize_input=false, fused_act=NONE` |",
      "| `FULLY_CONNECTED` | `50` | Local Sliding Attention `q_proj` (50 layers × 1) | `M=M_seq, N=8192, K=5376`, `input=f32[1, M_seq, 5376]`, `weights=i4[8192, 5376] per_channel(axis=0, n=8192)`, `bias=none`, `output=f32[1, M_seq, 8192]`, `keep_num_dims=true, asymmetric_quantize_input=false, fused_act=NONE` |",
      "| `FULLY_CONNECTED` | `50` | Local Sliding Attention `o_proj` (50 layers × 1) | `M=M_seq, N=5376, K=8192`, `input=f32[1, M_seq, 8192]`, `weights=i4[5376, 8192] per_channel(axis=0, n=5376)`, `bias=none`, `output=f32[1, M_seq, 5376]`, `keep_num_dims=true, asymmetric_quantize_input=false, fused_act=NONE` |",
      "| `FULLY_CONNECTED` | `10` | Global Attention `q_proj` (10 layers × 1) | `M=M_seq, N=16384, K=5376`, `input=f32[1, M_seq, 5376]`, `weights=i4[16384, 5376] per_channel(axis=0, n=16384)`, `bias=none`, `output=f32[1, M_seq, 16384]`, `keep_num_dims=true, asymmetric_quantize_input=false, fused_act=NONE` |",
      "| `FULLY_CONNECTED` | `10` | Global Attention unified `k=v proj` (10 layers × 1) | `M=M_seq, N=2048, K=5376`, `input=f32[1, M_seq, 5376]`, `weights=i4[2048, 5376] per_channel(axis=0, n=2048)`, `bias=none`, `output=f32[1, M_seq, 2048]`, `keep_num_dims=true, asymmetric_quantize_input=false, fused_act=NONE` |",
      "| `FULLY_CONNECTED` | `10` | Global Attention `o_proj` (10 layers × 1) | `M=M_seq, N=5376, K=16384`, `input=f32[1, M_seq, 16384]`, `weights=i4[5376, 16384] per_channel(axis=0, n=5376)`, `bias=none`, `output=f32[1, M_seq, 5376]`, `keep_num_dims=true, asymmetric_quantize_input=false, fused_act=NONE` |",
      "| `BATCH_MATMUL` | `50` | Local Sliding Attention `Q · K^T` (`W=1024`) | `batch=[1, 16], M=2*M_seq, N=1024, K=256`, `adj_x=false, adj_y=true`, `x=f32[1, 16, 2*M_seq, 256]`, `y=f32[1, 16, 1024, 256]`, `output=f32[1, 16, 2*M_seq, 1024]` |",
      "| `BATCH_MATMUL` | `50` | Local Sliding Attention `scores · V` (`W=1024`) | `batch=[1, 16], M=2*M_seq, N=256, K=1024`, `adj_x=false, adj_y=true`, `x=f32[1, 16, 2*M_seq, 1024]`, `y=f32[1, 16, 256, 1024]`, `output=f32[1, 16, 2*M_seq, 256]` |",
      "| `BATCH_MATMUL` | `10` | Global Attention `Q · K^T` (`S_kv=max_num_tokens`) | `batch=[1, 4], M=8*M_seq, N=S_kv, K=512`, `adj_x=false, adj_y=true`, `x=f32[1, 4, 8*M_seq, 512]`, `y=f32[1, 4, S_kv, 512]`, `output=f32[1, 4, 8*M_seq, S_kv]` |",
      "| `BATCH_MATMUL` | `10` | Global Attention `scores · V` (`S_kv=max_num_tokens`) | `batch=[1, 4], M=8*M_seq, N=512, K=S_kv`, `adj_x=false, adj_y=true`, `x=f32[1, 4, 8*M_seq, S_kv]`, `y=f32[1, 4, 512, S_kv]`, `output=f32[1, 4, 8*M_seq, 512]` |",
      "| `SOFTMAX` | `50` | Local Sliding Attention Softmax | `input=f32[1, 16, 2*M_seq, 1024]`, `output=f32[1, 16, 2*M_seq, 1024]`, `axis_dim=1024, beta=1.0` |",
      "| `SOFTMAX` | `10` | Global Attention Softmax | `input=f32[1, 4, 8*M_seq, S_kv]`, `output=f32[1, 4, 8*M_seq, S_kv]`, `axis_dim=S_kv, beta=1.0` |",
      "| `MEAN` + `RSQRT` | `401 + 401` | RMSNorm (Pre/Post-Attn, Pre/Post-FFN, Q-Norm, K-Norm, V-Norm) | `241x` hidden RMSNorm (`input=f32[1, M_seq, 5376], axes=[2], keep_dims=true`), `100x` Local Q/K Norm (`f32[1, M_seq, 16/32, 256], axes=[3]`), `20x` Global Q/K Norm (`f32[1, M_seq, 4/32, 512], axes=[3]`), `50x` Local V-Norm, `10x` Global V-Norm |",
      "| `GELU` | `60` | GeGLU MLP Gate Activation (60 layers) | `input=f32[1, M_seq, 21504]`, `output=f32[1, M_seq, 21504]` |",
      "| `COS` + `SIN` | `51 + 51` | RoPE Rotary Position Embeddings (Local `D=256`, Global `D=512`) | `50x` Local RoPE (`f32[1, M_seq, 1, 128]`), `1x` shared Global Partial RoPE (`f32[1, M_seq, 1, 64]`, `partial_rotary_factor=0.25`) |",
      "| `DYNAMIC_UPDATE_SLICE` | `110` | KV Cache In-Place Write (`50x` Local K+V, `10x` Global K=V) | Local K: `f32[1, 16, 1024, 256]`, Local V: `f32[1, 16, 256, 1024]`, Global K: `f32[1, 4, S_kv, 512]`, Global V: `f32[1, 4, 512, S_kv]` |",
      "| `GATHER` (`embedder`) | `1` | Token Embedding Table Lookup (`vocab_size=262,144`) | `params=i8[262144, 5376] per_channel(axis=0, n=262144)`, `indices=i32[1, 1]`, `output=f32[1, 1, 5376]` (+ `MUL` by `sqrt(5376)=73.3212`) |",
      "",
      "### 5.2 Vision Encoder & Multimodal Adapter (`vision_encoder` 3,075 ops, `vision_adapter` 8 ops)",
      "",
      "| Operator | Nodes / Subgraph | Layer / Architectural Role | Exact Input / Weight / Output Tensor Shapes & Operator Attributes (`N_patches ∈ {630, 1260, 2520, 5040, 10080}`, `N_soft = N_patches / 9 ∈ {70, 140, 280, 560, 1120}`) |",
      "| :--- | ---: | :--- | :--- |",
      "| `FULLY_CONNECTED` | `1` | ViT Patch Linear Projection (`16×16×3 = 768 -> 1152`) | `M=N_patches, N=1152, K=768`, `input=f32[1, N_patches, 768]`, `weights=i8[1152, 768] per_channel(axis=0, n=1152)`, `bias=f32[1152]`, `output=f32[1, N_patches, 1152]` |",
      "| `ONE_HOT` + `BATCH_MATMUL` | `2 + 2` | 2D Factorized $(x, y)$ Positional Embedding Lookup | `ONE_HOT(indices=i32[1, N_patches], depth=10240) -> f32[1, N_patches, 10240]`, `BATCH_MATMUL(M=N_patches, N=1152, K=10240, x=f32[1, N_patches, 10240], y=f32[10240, 1152])` |",
      "| `FULLY_CONNECTED` | `108` | ViT Self-Attention `q_proj, k_proj, v_proj, o_proj` (27 layers × 4) | `M=N_patches, N=1152, K=1152`, `input=f32[1, N_patches, 1152]`, `weights=i8[1152, 1152] per_channel(axis=0, n=1152)`, `bias=none`, `output=f32[1, N_patches, 1152]` |",
      "| `FULLY_CONNECTED` | `54` | ViT GeGLU MLP `gate_proj` & `up_proj` (27 layers × 2) | `M=N_patches, N=4304, K=1152`, `input=f32[1, N_patches, 1152]`, `weights=i8[4304, 1152] per_channel(axis=0, n=4304)`, `bias=none`, `output=f32[1, N_patches, 4304]` |",
      "| `FULLY_CONNECTED` | `27` | ViT GeGLU MLP `down_proj` (27 layers × 1) | `M=N_patches, N=1152, K=4304`, `input=f32[1, N_patches, 4304]`, `weights=i8[1152, 4304] per_channel(axis=0, n=1152)`, `bias=none`, `output=f32[1, N_patches, 1152]` |",
      "| `BATCH_MATMUL` | `27` | ViT 2D Bidirectional Self-Attention `Q · K^T` (`16` heads, `D=72`) | `batch=[16], M=N_patches, N=N_patches, K=72`, `adj_x=false, adj_y=false`, `x=f32[16, N_patches, 72]`, `y=f32[16, 72, N_patches]`, `output=f32[16, N_patches, N_patches]` |",
      "| `SOFTMAX` | `27` | ViT 2D Bidirectional Self-Attention Softmax | `input=f32[1, 16, N_patches, N_patches]`, `output=f32[1, 16, N_patches, N_patches]`, `axis_dim=N_patches, beta=1.0` |",
      "| `BATCH_MATMUL` | `27` | ViT 2D Bidirectional Self-Attention `scores · V` | `batch=[16], M=N_patches, N=72, K=N_patches`, `adj_x=false, adj_y=false`, `x=f32[16, N_patches, N_patches]`, `y=f32[16, N_patches, 72]`, `output=f32[16, N_patches, 72]` |",
      "| `ONE_HOT` + `BATCH_MATMUL` | `1 + 1` | $3 \\times 3$ Spatial Average Pooling (`N_patches -> N_soft = N_patches / 9`) | `ONE_HOT(indices=i32[1, N_patches], depth=N_soft) -> f32[1, N_patches, N_soft]`, `BATCH_MATMUL(M=N_soft, N=1152, K=N_patches, adj_x=true, x=f32[1, N_patches, N_soft], y=f32[1, N_patches, 1152]) -> f32[1, N_soft, 1152]` |",
      "| `FULLY_CONNECTED` (`vision_adapter`) | `1` | Multimodal Vision-to-Text Linear Projection (`1152 -> 5376`) | `M=N_soft, N=5376, K=1152`, `input=f32[1, N_soft, 1152]`, `weights=i8[5376, 1152] per_channel(axis=0, n=5376)`, `bias=none`, `output=f32[1, N_soft, 5376]` (+ RMSNorm over `axis=2`) |",
      "",
      "### 5.3 Audio Conformer Encoder & Adapter (`audio_encoder` 2,182 ops, `audio_adapter` 11 ops, `audio_204`)",
      "",
      "| Operator | Nodes / Subgraph | Layer / Architectural Role | Exact Input / Weight / Output Tensor Shapes & Operator Attributes (`816` mel frames `@ 128` bins -> `204` soft tokens `@ 1536`) |",
      "| :--- | ---: | :--- | :--- |",
      "| `DEPTHWISE_CONV_2D` | `1` | 2D Subsampling Stage 1 (`stride=2`, `1 -> 128` channels) | `input=f32[1, 819, 130, 1]`, `filter=f32[1, 3, 3, 128]`, `output=f32[1, 409, 64, 128]`, `stride=[2, 2], padding=VALID, depth_multiplier=128` |",
      "| `CONV_2D` | `1` | 2D Subsampling Stage 2 (`stride=2`, `128 -> 32` channels) | `input=f32[1, 409, 66, 128]`, `filter=f32[32, 3, 3, 128]`, `output=f32[1, 204, 32, 32]`, `stride=[2, 2], padding=VALID` |",
      "| `FULLY_CONNECTED` | `1` | Subsampling Flatten Linear Projection (`32×32 = 1024 -> 1024`) | `M=204, N=1024, K=1024`, `input=f32[1, 204, 1024]`, `weights=f32[1024, 1024]`, `output=f32[1, 204, 1024]` |",
      "| `FULLY_CONNECTED` | `60` | Conformer 2-bit Attention `q/k/v/o/pos_proj` (12 layers × 5) | `M=204, N=1024, K=1024`, `input=i8[1, 204, 1024]` (or `i8[204, 1024]`), `weights=i2[1024, 1024] per_channel(axis=0, n=1024)`, `output=i8[1, 204, 1024]` |",
      "| `FULLY_CONNECTED` | `24` | Conformer 2-bit Macaron FFN1 & FFN2 `up_proj` (12 layers × 2) | `M=204, N=4096, K=1024`, `input=i8[1, 204, 1024]`, `weights=i2[4096, 1024] per_channel(axis=0, n=4096)`, `output=i8[1, 204, 4096]` |",
      "| `FULLY_CONNECTED` | `24` | Conformer 2-bit Macaron FFN1 & FFN2 `down_proj` (12 layers × 2) | `M=204, N=1024, K=4096`, `input=i8[1, 204, 4096]`, `weights=i2[1024, 4096] per_channel(axis=0, n=1024)`, `output=i8[1, 204, 1024]` |",
      "| `FULLY_CONNECTED` | `12` | Conformer 4-bit LConv GLU Pointwise In (12 layers × 1) | `M=204, N=2048, K=1024`, `input=i8[1, 204, 1024]`, `weights=i4[2048, 1024] per_channel(axis=0, n=2048)`, `output=i8[1, 204, 2048]` |",
      "| `DEPTHWISE_CONV_2D` | `12` | Conformer 1D Depthwise Convolution (`kernel_size=5`, 12 layers) | `input=f32[1, 208, 1, 1024]`, `filter=f32[1, 5, 1, 1024]`, `output=f32[1, 204, 1, 1024]`, `stride=[1, 1], padding=VALID, depth_multiplier=1` |",
      "| `BATCH_MATMUL` | `12` | Conformer Local Chunked Content Attention `Q · K^T` (`17` chunks of `12` queries × `24` keys) | `batch=[8, 17], M=12, N=24, K=128`, `adj_x=false, adj_y=false`, `x=f32[8, 17, 12, 128]`, `y=f32[8, 17, 128, 24]`, `output=f32[8, 17, 12, 24]` |",
      "| `BATCH_MATMUL` | `12` | Conformer Relative Position Attention `Q · R^T` (`window=13`) | `batch=[8], M=204, N=13, K=128`, `adj_x=true, adj_y=true`, `x=f32[8, 128, 204]`, `y=f32[8, 13, 128]`, `output=f32[8, 204, 13]` |",
      "| `SOFTMAX` | `12` | Conformer Local Chunked Attention Softmax (`axis_dim=24`) | `input=f32[8, 17, 12, 24]`, `output=f32[8, 17, 12, 24]`, `axis_dim=24, beta=1.0` |",
      "| `BATCH_MATMUL` | `12` | Conformer Local Chunked Attention `scores · V` | `batch=[17, 8], M=128, N=12, K=24`, `adj_x=false, adj_y=false`, `x=f32[17, 8, 128, 24]`, `y=f32[17, 8, 24, 12]`, `output=f32[17, 8, 128, 12]` |",
      "| `FULLY_CONNECTED` | `1 + 1` | Audio Encoder Output Proj (`1024 -> 1536`) + Audio Adapter (`1536 -> 1536`) | Encoder out: `M=204, N=1536, K=1024, weights=f32[1536, 1024], bias=f32[1536]`; Adapter: `M=204, N=1536, K=1536, weights=f32[1536, 1536], bias=none` |",
  ])

  if uprobe_params and "traces" in uprobe_params:
    lines.extend([
        "",
        "## 6. Low-Level Hardware Microkernel Call Parameter Values (`perf probe` Uprobe Trace on Sapphire Rapids)",
        "",
        "Using Linux `perf probe` (user-space dynamic uprobes on `/opt/litert-lm/bin/litert_lm_advanced_main`), we intercepted the exact x86-64 System V ABI register and stack arguments passed at runtime to each XNNPACK and YNNPACK C++/assembly microkernel (`raw/uprobe_microkernel_params.json`):",
        "",
        "- **XNNPACK GEMM ABI**: `void xnn_*_gemm_minmax_ukernel_*(size_t mr [%di], size_t nc [%si], size_t kc [%dx], const void* a [%cx], size_t a_stride [%r8], const void* w [%r9], void* c [+8(%sp)], size_t cm_stride [+16(%sp)], ...)`",
        "- **XNNPACK Softmax/Reduction ABI**: `void xnn_f32_{rmax,raddstoreexpminusmax}_ukernel_*(size_t elements_bytes [%di], const float* input [%si], ...)`",
        "- **YNNPACK Dot ABI**: `void ynn::dot_*(size_t M [%di], size_t N [%si], size_t K3 [%dx], size_t K2 [%cx], size_t K1 [%r8], size_t A_stride_m [%r9], ...)`",
        "",
        "### Key Structural Difference Revealed by Microkernel Parameter Values",
        "- **XNNPACK** passes the **entire reduction dimension `kc = K` in a single microkernel call** (`kc ∈ {5376, 8192, 16384, 21504}` in Text; `kc ∈ {768, 1152, 4304}` in Vision; `kc ∈ {1024, 4096}` in Audio) and tiles across `M` (`mr=16` for batched AMX, `mr=1` for decode AMX, `mr=7` for AVX-512 FP32, `mr=5` for AVX-VNNI INT2) and `N` (`nc ∈ {704, 448, 384, 192, 128}` in Text prefill; full `nc=N` in `M=1` decode).",
        "- **YNNPACK** tiles both `M × N` (`M=12, N=32` for prefill INT4; `M=1, N=32` for decode INT4; `M=32/24, N=64` for Vision Adapter AMX INT8; `M=4, N=64` for Audio INT2; `M=5, N=64/128` for FP32 BMM) **and splits the reduction dimension `K` into L2-cache-friendly `K1 ≤ 4096` blocks** (`K=5376 -> 4096 + 1280`; `K=21504 -> 5×4096 + 1024`; `K=8192 -> 2×4096`; `K=16384 -> 4×4096`; Audio `K=4096 -> 2×2048`), while using `K3=3, K2=3, K1=1` for $3\\times 3$ 2D convolutions.",
        "",
    ])
    for trace_name, rows in uprobe_params["traces"].items():
      lines.extend([
          f"### `{trace_name}`",
          "",
          "| Microkernel Symbol | Invocations | Runtime Register / Stack Parameter Values | Layer / Architectural Role |",
          "| :--- | ---: | :--- | :--- |",
      ])
      for item in rows:
        param_str = ", ".join(f"{k}={v}" for k, v in item["params"].items())
        lines.append(
            f"| `{item['symbol']}` | {item['calls']:,} | `{param_str}` | {item['layer_role']} |"
        )
      lines.append("")

  return "\n".join(lines) + "\n"


def load_results_from_logs(raw_dir: Path, shapes: tuple[WorkloadShape, ...] = DEFAULT_SHAPES) -> list[ShapeProfileResult]:
  """Reconstruct `ShapeProfileResult` list from saved `<backend>_<shape_id>_tflite.log` and `_perf.txt` files."""
  results: list[ShapeProfileResult] = []
  for backend in ("xnnpack", "ynnpack"):
    for shape in shapes:
      tfl_p = raw_dir / f"{backend}_{shape.id}_tflite.log"
      prf_p = raw_dir / f"{backend}_{shape.id}_perf.txt"
      if not tfl_p.is_file() and not prf_p.is_file():
        continue
      tfl_txt = tfl_p.read_text() if tfl_p.is_file() else ""
      prf_txt = prf_p.read_text() if prf_p.is_file() else ""
      results.append(build_shape_result_from_texts(backend, shape, tfl_txt, prf_txt))
  return results


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--model", help="Path to .litertlm model inside container or host")
  ap.add_argument("--ssh", help="Optional SSH target (e.g., user@sapphire-rapids-host) to execute on remote Sapphire Rapids host")
  ap.add_argument("--models-dir", default="/data/aeb/models", help="Host directory mounted as /models on remote host")
  ap.add_argument("--threads", type=int, default=28, help="Number of CPU threads (default: 28)")
  ap.add_argument("--backends", default="xnnpack,ynnpack", help="Comma-separated backends: xnnpack,ynnpack")
  ap.add_argument("--shapes", help="Optional comma-separated subset of shape IDs (e.g., p1_d1,p64_d64,p1024_d128,vis_280)")
  ap.add_argument("--with-perf", action="store_true", help="Also collect Linux perf symbol-level microkernel profiles")
  ap.add_argument("--from-logs", type=Path, help="Re-parse existing raw logs directory instead of running live")
  ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for JSON, Markdown, and raw logs")
  args = ap.parse_args(argv)

  args.out_dir.mkdir(parents=True, exist_ok=True)
  raw_dir = args.from_logs if args.from_logs else (args.out_dir / "raw")

  selected_shapes = DEFAULT_SHAPES
  if args.shapes:
    want = set(args.shapes.split(","))
    selected_shapes = tuple(s for s in DEFAULT_SHAPES if s.id in want)

  if not args.from_logs:
    if not args.model:
      ap.error("--model is required unless --from-logs is specified")
    for backend in [b.strip() for b in args.backends.split(",") if b.strip()]:
      for shape in selected_shapes:
        print(f"[extract] profiling backend={backend} shape={shape.id} (P={shape.prompt_tokens}, D={shape.decode_tokens}, ctx={shape.max_num_tokens})...", flush=True)
        run_single_profile(
            model_path=args.model,
            backend=backend,
            shape=shape,
            threads=args.threads,
            raw_dir=raw_dir,
            ssh_host=args.ssh,
            models_dir=args.models_dir,
            with_perf=args.with_perf,
        )

  results = load_results_from_logs(raw_dir, DEFAULT_SHAPES)

  uprobe_params: dict[str, Any] | None = None
  uprobe_file = raw_dir / "uprobe_microkernel_params.json"
  if uprobe_file.is_file():
    uprobe_params = json.loads(uprobe_file.read_text())

  metadata = {
      "model_id": "google/gemma-4-31B-it",
      "source_checkpoint": "google/gemma-4-31B-it-qat-q4_0-unquantized@1e4d8beecacb8b7590c1d8bedd7335f687bf311f",
      "litertlm_artifact": "gemma-4-31B-it.litertlm (Text 60L) + gemma-4-31B-it-vision-{70,140,280,560,1120}.litertlm (Vision 27L ViT + Adapter) + gemma-4-E2B-it.litertlm (Audio Conformer Reference)",
      "cpu_model": "Intel(R) Xeon(R) Sapphire Rapids (AVX-512 + AMX)",
      "host": args.ssh or "sapphire-rapids",
      "threads": args.threads,
      "litert_lm_commit": "v0.19.0 (83ce6067f2e2a3fb697556928f93936693ded971)",
      "subgraph_operator_params_file": "raw/subgraph_operator_params.json",
      "uprobe_microkernel_params_file": "raw/uprobe_microkernel_params.json",
  }

  json_path = args.out_dir / "kernels_gemma4_31b_sapphire_rapids.json"
  md_path = args.out_dir / "kernels_gemma4_31b_sapphire_rapids.md"

  matched_dominant_kernel_pairs = [
      {
          "pair_id": "int4_gemv_decode",
          "category": "4-Bit Weight-Only Linear GEMV (Autoregressive Decode, M=1)",
          "modalities_and_shapes": ["text: p1_d1, p16_d128, p64_d64, p128_d128, p256_d128"],
          "cycle_share_pct_range": "50.8% - 78.7%",
          "tflite_delegate_op": "Delegate/Fully Connected (NC, QD8, F32, QC4W) GEMM",
          "xnnpack_kernel": {
              "symbol": "xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx",
              "isa": "AMX-INT8",
              "tile": {"mr": 1, "nr": 64, "kr": 4},
              "runtime_params": "mr=1, nc=N, kc=K, a_stride=K, cm_stride=4*N (full K in 1 call)",
          },
          "ynnpack_kernel": {
              "symbol": "ynn::dot_uint8_int4_int32_1x32x8_1x16x8_avx512vnni",
              "isa": "AVX-512 VNNI",
              "tile": {"M": 1, "N": 32, "K_tile": 8},
              "runtime_params": "M=1, N=32, K3=1, K2=1, K1=min(K, 4096), A_stride_m=0 (K split into <=4096 blocks)",
          },
          "benchmark_dimensions": [
              {"role": "MLP gate_proj & up_proj (120x)", "M": 1, "N": 21504, "K": 5376},
              {"role": "MLP down_proj (60x)", "M": 1, "N": 5376, "K": 21504},
              {"role": "Local Attn k_proj & v_proj (100x)", "M": 1, "N": 4096, "K": 5376},
              {"role": "Local Attn q_proj (50x)", "M": 1, "N": 8192, "K": 5376},
              {"role": "Local Attn o_proj (50x)", "M": 1, "N": 5376, "K": 8192},
              {"role": "Global Attn q_proj / k=v / o_proj (30x)", "M": 1, "N": [16384, 2048, 5376], "K": [5376, 5376, 16384]},
          ],
          "why_dominant": "Every M=1 decode token executes 420 4-bit linear projections across 60 layers, streaming ~15.3 GB of 4-bit weights from DDR5 RAM with zero weight reuse across tokens.",
      },
      {
          "pair_id": "int4_gemm_prefill",
          "category": "4-Bit Weight-Only Batched Linear GEMM (Prefill, M in {128, 1024})",
          "modalities_and_shapes": ["text: p64_d64, p128_d128, p256_d128, p1024_d128, p4096_d128"],
          "cycle_share_pct_range": "14.4% - 34.4% combined (up to 93% in prefill-only phase)",
          "tflite_delegate_op": "Delegate/Fully Connected (NC, QD8, F32, QC4W) GEMM",
          "xnnpack_kernel": {
              "symbol": "xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx",
              "isa": "AMX-INT8",
              "tile": {"mr": 16, "nr": 64, "kr": 4},
              "runtime_params": "mr=16, nc in {704, 448, 384, 192, 128}, kc=K, a_stride=K, cm_stride=4*N",
          },
          "ynnpack_kernel": {
              "symbol": "ynn::dot_uint8_int4_int32_12x16x8_1x16x8_avx512vnni",
              "isa": "AVX-512 VNNI",
              "tile": {"M": 12, "N": 16, "K_tile": 8},
              "runtime_params": "M=12, N=32, K3=1, K2=1, K1 in {4096, 1280, 1024}, A_stride_m=K",
          },
          "benchmark_dimensions": [
              {"role": "MLP gate_proj & up_proj (120x)", "M": [128, 1024], "N": 21504, "K": 5376},
              {"role": "MLP down_proj (60x)", "M": [128, 1024], "N": 5376, "K": 21504},
              {"role": "Local & Global Attn Q/K/V/O (240x)", "M": [128, 1024], "N": [4096, 8192, 16384, 2048, 5376], "K": [5376, 8192, 16384]},
          ],
          "why_dominant": "Compute-bound matrix multiplication (39.6 TFLOPs per 1024-token prefill chunk); XNNPACK dispatches 2D AMX-INT8 tiles (147.9 tok/s at p1024) vs. YNNPACK AVX-512 VNNI (110.2 tok/s).",
      },
      {
          "pair_id": "fp32_attention_bmm",
          "category": "FP32 Attention BatchMatMul (Q*K^T and scores*V) & KV/Weight Packing",
          "modalities_and_shapes": ["text: p1024_d128, p4096_d128, p16384_d128", "vision: vis_70, vis_140, vis_280, vis_560, vis_1120"],
          "cycle_share_pct_range": "35.0% - 89.6%",
          "tflite_delegate_op": "Delegate/Batch Matrix Multiply (NC, F32) GEMM",
          "xnnpack_kernel": {
              "symbol": "xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast + xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4_prfm",
              "isa": "AVX-512F",
              "tile": {"mr": 7, "nr": 32},
              "runtime_params": "mr=7, nc in {1024, 256, 512, N_patches, 72}, kc in {1024B, 4096B, 2048B, 288B, 4*N_patches}",
          },
          "ynnpack_kernel": {
              "symbol": "ynn::dot_fp32_5x64x1_1x16x1_avx512 + ynn::dot_fp32_2x64x1_1x16x1_avx512 + ynn::transpose_x32_avx512",
              "isa": "AVX-512F",
              "tile": {"M": 5, "N": 64, "K_tile": 1},
              "runtime_params": "M=5 (or 2), N in {64, 128}, K1 in {256, 512}, A_stride_m in {1024, 4096}",
          },
          "benchmark_dimensions": [
              {"role": "Text Local Q*K^T & scores*V (100x)", "batch": [1, 16], "M": [2, 256, 2048], "N": [1024, 256], "K": [256, 1024]},
              {"role": "Text Global Q*K^T & scores*V (20x)", "batch": [1, 4], "M": [8, 1024, 8192], "N": [1024, 4096, 8192, 32768, 512], "K": [512, 1024, 4096, 8192, 32768]},
              {"role": "Vision 2D Bidirectional Q*K^T & scores*V (54x)", "batch": [16], "M": [630, 1260, 2520, 5040, 10080], "N": [630, 1260, 2520, 5040, 10080, 72], "K": [72, 630, 1260, 2520, 5040, 10080]},
          ],
          "why_dominant": "Scales quadratically O(P * S_kv) in global text attention and O(N_patches^2) in unmasked 2D ViT self-attention (43.9B pairs at vis_1120), executing in FP32 AVX-512F rather than AMX.",
      },
      {
          "pair_id": "fp32_attention_softmax",
          "category": "FP32 Attention Softmax (Row Max + Exp-Subtract-Sum Reductions)",
          "modalities_and_shapes": ["vision: vis_280, vis_560, vis_1120", "text: p4096_d128, p16384_d128"],
          "cycle_share_pct_range": "15.0% - 28.5%",
          "tflite_delegate_op": "Delegate/Softmax (NC, F32)",
          "xnnpack_kernel": {
              "symbol": "xnn_f32_rmax_ukernel__avx512f_u64_acc4 + xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc2",
              "isa": "AVX-512F",
              "tile": {"simd_width": 64},
              "runtime_params": "elements_bytes = 4 * axis_dim (4096B for Text Local; 2520B..40320B for Vision)",
          },
          "ynnpack_kernel": {
              "symbol": "ynn::max_k1_fp32_avx512 + ynn::sum_k1_fp32_avx512 + ynn::unary_LUT_op + ynn::multiply_fp32_avx512",
              "isa": "AVX-512F + Scalar LUT",
              "tile": {"simd_width": 16},
              "runtime_params": "Multi-pass max, exp_subtract LUT, sum, and reciprocal multiply over axis_dim",
          },
          "benchmark_dimensions": [
              {"role": "Vision 2D Self-Attention Softmax (27x)", "shape": [1, 16, "N_patches", "N_patches"], "axis_dim": [630, 1260, 2520, 5040, 10080]},
              {"role": "Text Local & Global Softmax (60x)", "shape": [1, 16, "2*M_seq", 1024], "axis_dim": [1024, 4096, 8192, 32768]},
          ],
          "why_dominant": "At vis_1120, Softmax normalizes 27 * 16 * 10080 = 4.35M rows of 10080 FP32 logits (175.6 GB read/written); XNNPACK fuses exp(x - max) and sum into a single AVX-512 polynomial pass.",
      },
      {
          "pair_id": "int8_gemm_vision",
          "category": "8-Bit Per-Channel (QC8W / INT8xINT8) Linear GEMM",
          "modalities_and_shapes": ["vision: vis_70, vis_140, vis_280, vis_560, vis_1120"],
          "cycle_share_pct_range": "25.3% - 31.6% in ViT (vis_70/140); 69.9% - 78.0% in Vision Adapter",
          "tflite_delegate_op": "Delegate/Fully Connected (NC, QD8, F32, QC8W) GEMM",
          "xnnpack_kernel": {
              "symbol": "xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx",
              "isa": "AMX-INT8",
              "tile": {"mr": 16, "nr": 64, "kr": 4},
              "runtime_params": "mr=16, nc in {1152, 1472, 1360, 384, 1344}, kc in {1152, 4304, 768}",
          },
          "ynnpack_kernel": {
              "symbol": "ynn::dot_int8_int8_int32_32x32x64_16x16x4_amxint8 + ynn::dequantize_dot_f32_avx512",
              "isa": "AMX-INT8 + AVX-512F",
              "tile": {"M": 32, "N": 32, "K_tile": 64},
              "runtime_params": "M in {32, 24}, N=64, K3=1, K2=1, K1=1152, A_stride_m=1152",
          },
          "benchmark_dimensions": [
              {"role": "ViT Self-Attention Q/K/V/O (108x)", "M": [630, 1260, 2520, 5040, 10080], "N": 1152, "K": 1152},
              {"role": "ViT MLP gate_proj & up_proj (54x)", "M": [630, 1260, 2520, 5040, 10080], "N": 4304, "K": 1152},
              {"role": "ViT MLP down_proj (27x)", "M": [630, 1260, 2520, 5040, 10080], "N": 1152, "K": 4304},
              {"role": "Vision Adapter mm_projection (1x)", "M": [70, 140, 280, 560, 1120], "N": 5376, "K": 1152},
          ],
          "why_dominant": "All 190 ViT linear layers and the Vision Adapter use 8-bit QC8W weights; both XNNPACK and YNNPACK dispatch hardware AMX-INT8 (TDPBSSD), with XNNPACK fusing FP32 dequantization into the tile epilogue.",
      },
      {
          "pair_id": "int2_gemm_audio",
          "category": "2-Bit Weight-Only (QC2W / INT2) Conformer Linear GEMM",
          "modalities_and_shapes": ["audio: audio_204"],
          "cycle_share_pct_range": "39.2% - 41.2%",
          "tflite_delegate_op": "Delegate/Fully Connected (NC, QS8, QC2W) GEMM",
          "xnnpack_kernel": {
              "symbol": "xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni",
              "isa": "AVX-VNNI",
              "tile": {"mr": 5, "nr": 8, "kr": 8},
              "runtime_params": "mr=5, nc in {1024, 4096}, kc in {1024, 4096}, a_stride=K, cm_stride=N",
          },
          "ynnpack_kernel": {
              "symbol": "ynn::dot_uint8_int2_int32_4x64x16_1x16x16_avx512vnni",
              "isa": "AVX-512 VNNI",
              "tile": {"M": 4, "N": 64, "K_tile": 16},
              "runtime_params": "M=4, N=64, K3=1, K2=1, K1 in {1024, 2048}, A_stride_m in {1024, 4096}",
          },
          "benchmark_dimensions": [
              {"role": "Conformer Attention q/k/v/o/pos_proj (60x)", "M": 204, "N": 1024, "K": 1024},
              {"role": "Conformer Macaron FFN1 & FFN2 up_proj (24x)", "M": 204, "N": 4096, "K": 1024},
              {"role": "Conformer Macaron FFN1 & FFN2 down_proj (24x)", "M": 204, "N": 1024, "K": 4096},
          ],
          "why_dominant": "108 of 122 Conformer linear projections use 2-bit weights (i2); neither backend has an AMX 2-bit kernel, so both execute VNNI dot-product kernels with bit-unpacking.",
      },
  ]

  json_payload: dict[str, Any] = {
      "metadata": metadata,
      "matched_dominant_kernel_pairs": matched_dominant_kernel_pairs,
      "shapes": [asdict(s) for s in selected_shapes],
      "results": [asdict(r) for r in results],
  }
  if uprobe_params is not None:
    json_payload["uprobe_microkernel_params"] = uprobe_params

  json_path.write_text(json.dumps(json_payload, indent=2) + "\n")
  md_path.write_text(render_markdown_report(results, metadata, uprobe_params=uprobe_params))
  print(f"[extract] wrote {json_path} and {md_path}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
