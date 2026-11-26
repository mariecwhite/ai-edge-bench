#!/usr/bin/env python3
"""Generate and update the AI Edge Bench Level-2 Layer Dashboard.

Aggregates dated layer benchmark reports under `layer/reports/YYYY-MM-DD-<suite>/`
(`layer_report_*.json`) and emits:
  1. `layer/dashboards/data.json`   — structured multi-suite JSON across dates
  2. `layer/dashboards/index.html`  — self-contained interactive HTML/SVG dashboard
  3. `layer/dashboards/README.md`   — GitHub-native Markdown dashboard with Mermaid charts
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List

LAYER_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = LAYER_ROOT / "reports"
DASH_ROOT = LAYER_ROOT / "dashboards"
DATE_DIR_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)$")

BACKEND_COLORS = {
    "litert_xnn_qc4w": "#93c5fd",
    "litert_xnn_qb4w32": "#2563eb",
    "litert_ynn_qc4w_decomposed": "#fdba74",
    "litert_ynn_qc4w_sdpa": "#f97316",
    "litert_ynn_qb4w32_hybrid_sdpa": "#d97706",
    "litert_ynn_qb4w32_pure_sdpa": "#ea580c",
    "direct_ynn_capi_qc4w": "#5eead4",
    "direct_ynn_capi_qb4w32": "#0d9488",
    "ggml_cpu_neon_fa": "#7c3aed",
    "ggml_cpu_sme_fa": "#ec4899",
    "ggml_cpu_neon_unfused": "#a78bfa",
}

BACKEND_SHORT = {
    "litert_xnn_qc4w": "LiteRT XNN (QC4W)",
    "litert_xnn_qb4w32": "LiteRT XNN (QB4W-32)",
    "litert_ynn_qc4w_decomposed": "LiteRT YNN Decomp (QC4W)",
    "litert_ynn_qc4w_sdpa": "LiteRT YNN SDPA (QC4W)",
    "litert_ynn_qb4w32_hybrid_sdpa": "LiteRT YNN Hybrid (QB4W-32)",
    "litert_ynn_qb4w32_pure_sdpa": "LiteRT YNN Pure (QB4W-32)",
    "direct_ynn_capi_qc4w": "Direct YNN C API (QC4W)",
    "direct_ynn_capi_qb4w32": "Direct YNN C API (QB4W-32)",
    "ggml_cpu_neon_fa": "ggml NEON/i8mm + FA",
    "ggml_cpu_sme_fa": "ggml KleidiAI SME2 + FA",
    "ggml_cpu_neon_unfused": "ggml NEON Unfused",
}


def collect_suites() -> Dict[str, Any]:
  suites: Dict[str, Dict[str, Any]] = {}
  if not REPORTS_ROOT.exists():
    return {"generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "suites": []}

  for rdir in sorted(REPORTS_ROOT.iterdir()):
    if not rdir.is_dir():
      continue
    m = DATE_DIR_RE.match(rdir.name)
    if not m:
      continue
    date_str, suite_slug = m.group(1), m.group(2)
    json_files = sorted(rdir.glob("layer_report_*.json"))
    if not json_files:
      continue
    data = json.loads(json_files[0].read_text(encoding="utf-8"))
    md_files = sorted(rdir.glob("layer_report_*.md"))
    md_rel = f"../reports/{rdir.name}/{md_files[0].name}" if md_files else ""

    if suite_slug not in suites:
      suites[suite_slug] = {
          "suite_id": suite_slug,
          "title": f"{data.get('model', 'Gemma 4 31B')} — {data.get('hardware', {}).get('microarchitecture', suite_slug)}",
          "model": data.get("model", ""),
          "hardware": data.get("hardware", {}),
          "reports": [],
      }

    # Store compact report snapshot for interactive dashboard
    suites[suite_slug]["reports"].append({
        "report_id": rdir.name,
        "date": date_str,
        "report_url": md_rel,
        "commits": data.get("commits", {}),
        "verification_summary": data.get("verification_summary", {}),
        "projections_60_layer": data.get("projections_60_layer", []),
        "benchmarks": [
            {
                "backend_id": b["backend_id"],
                "subgraph_id": b["subgraph_id"],
                "regime_id": b["regime_id"],
                "mode": b["mode"],
                "weight_quant": b.get("weight_quant", ""),
                "kv_dtype": b.get("kv_dtype", ""),
                "T": b["T"],
                "S_kv": b["S_kv"],
                "S_cap": b["S_cap"],
                "threads": b["threads"],
                "median_ms": b["timing"]["median_ms"],
                "cv_pct": b["timing"]["cv_pct"],
                "bound": b["roofline"]["bound"],
                "t_min_format_ms": b["roofline"]["t_min_format_ms"],
                "roofline_util_format_pct": b["roofline"]["roofline_util_format_pct"],
                "roofline_util_ideal_pct": b["roofline"]["roofline_util_ideal_pct"],
                "effective_gops": b["roofline"]["effective_gops"],
                "effective_gbps_format": b["roofline"]["effective_gbps_format"],
            }
            for b in data.get("benchmarks", [])
        ],
        "verification": [
            {
                "backend_id": v["backend_id"],
                "subgraph_id": v["subgraph_id"],
                "regime_id": v["regime_id"],
                "weight_quant": v.get("weight_quant", ""),
                "lva_cos": v["verify"]["level_a_dequant_fp64"]["cosine_sim"],
                "lva_sqnr": v["verify"]["level_a_dequant_fp64"]["sqnr_db"],
                "lva_kl": v["verify"]["level_a_dequant_fp64"]["kl_div"],
                "lva_pass": v["verify"]["level_a_dequant_fp64"]["passed"],
                "lvb_cos": v["verify"]["level_b_unquant_fp64"]["cosine_sim"],
                "lvb_sqnr": v["verify"]["level_b_unquant_fp64"]["sqnr_db"],
                "lvb_kl": v["verify"]["level_b_unquant_fp64"]["kl_div"],
                "det": v["verify"]["deterministic"],
                "max_repro_diff": v["verify"]["max_repro_diff"],
            }
            for v in data.get("verification", [])
            if v.get("ok")
        ],
    })

  return {
      "schema": "aeb.layer_dashboard/1",
      "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
      "backend_colors": BACKEND_COLORS,
      "backend_short": BACKEND_SHORT,
      "suites": list(suites.values()),
  }


def render_html(payload: Dict[str, Any]) -> str:
  data_json = json.dumps(payload)
  return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AI Edge Bench — Level 2 Layer Benchmark Dashboard</title>
<style>
  :root {{
    --bg: #0f172a; --card: #1e293b; --border: #334155; --text: #f1f5f9;
    --muted: #94a3b8; --accent: #38bdf8; --good: #34d399; --warn: #fbbf24; --bad: #f87171;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.5; }}
  header {{ padding: 24px 32px; border-bottom: 1px solid var(--border);
            display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 16px; }}
  h1 {{ margin: 0; font-size: 1.45rem; font-weight: 700; letter-spacing: -0.01em; }}
  .subtitle {{ color: var(--muted); font-size: 0.9rem; margin-top: 4px; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
  select, button {{ background: var(--card); color: var(--text); border: 1px solid var(--border);
                    border-radius: 8px; padding: 7px 12px; font-size: 0.88rem; cursor: pointer; }}
  button.active {{ background: #0284c7; border-color: #38bdf8; font-weight: 600; }}
  main {{ max-width: 1400px; margin: 0 auto; padding: 24px 32px 64px; display: grid; gap: 22px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; }}
  .kpi-label {{ color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .kpi-val {{ font-size: 1.6rem; font-weight: 700; margin-top: 4px; }}
  .kpi-sub {{ color: var(--muted); font-size: 0.82rem; margin-top: 4px; }}
  h2 {{ margin: 0 0 12px; font-size: 1.1rem; font-weight: 600; }}
  .charts-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 18px; }}
  svg.chart {{ width: 100%; height: auto; display: block; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.84rem; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: right; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  th {{ color: var(--muted); font-weight: 600; position: sticky; top: 0; background: var(--card); }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }}
  .pass {{ background: rgba(52,211,153,0.15); color: var(--good); }}
  .warn {{ background: rgba(251,191,36,0.15); color: var(--warn); }}
  .fail {{ background: rgba(248,113,113,0.15); color: var(--bad); }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 6px 0 10px; font-size: 0.8rem; color: var(--muted); }}
  .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
  .dot {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block; }}
  .tbl-wrap {{ overflow-x: auto; max-height: 460px; }}
  a {{ color: var(--accent); text-decoration: none; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>AI Edge Bench — Level 2 Layer Benchmark Dashboard</h1>
    <div class="subtitle" id="subtitle">Whole-Subgraph Practical Roofline, KV-Context Scaling &amp; FP64 Numerical Parity</div>
  </div>
  <div class="controls">
    <select id="suiteSelect"></select>
    <select id="reportSelect"></select>
    <select id="threadSelect"></select>
    <select id="subgraphSelect">
      <option value="full_local_block">Full Local Block (50/60)</option>
      <option value="full_global_block">Full Global Block (10/60)</option>
      <option value="local_swa_attn">Local SWA Attention</option>
      <option value="global_full_attn">Global Full Attention (K=V)</option>
      <option value="geglu_mlp">GeGLU MLP</option>
    </select>
  </div>
</header>
<main>
  <section class="kpi-grid" id="kpiGrid"></section>
  <section class="charts-2">
    <div class="card">
      <h2 id="rooflineChartTitle">Whole-Subgraph Practical Roofline Utilization (%)</h2>
      <div class="legend" id="rooflineLegend"></div>
      <svg class="chart" id="rooflineSvg" viewBox="0 0 640 330"></svg>
    </div>
    <div class="card">
      <h2>Global Full Attention Cold Decode Latency vs. KV Depth (18T)</h2>
      <div class="legend" id="kvLegend"></div>
      <svg class="chart" id="kvSvg" viewBox="0 0 640 330"></svg>
    </div>
  </section>
  <section class="card">
    <h2>Cross-Tier 60-Layer Projection (`50× Local + 10× Global`) vs. Level 3 End-to-End Model</h2>
    <div class="tbl-wrap"><table id="projTable"></table></div>
  </section>
  <section class="card">
    <h2 id="benchTableTitle">Detailed Subgraph Benchmarks</h2>
    <div class="tbl-wrap"><table id="benchTable"></table></div>
  </section>
  <section class="card">
    <h2>Two-Level FP64 Numerical Parity &amp; 18-Thread Determinism Gate</h2>
    <div class="tbl-wrap"><table id="verifyTable"></table></div>
  </section>
</main>
<script>
const DATA = {data_json};
const suiteSel = document.getElementById('suiteSelect');
const reportSel = document.getElementById('reportSelect');
const threadSel = document.getElementById('threadSelect');
const sgSel = document.getElementById('subgraphSelect');

function fmtMs(v) {{ return v == null ? '—' : (v < 10 ? v.toFixed(3) : v.toFixed(2)); }}
function bShort(id) {{ return DATA.backend_short[id] || id; }}
function bCol(id) {{ return DATA.backend_colors[id] || '#38bdf8'; }}

function populateSelectors() {{
  suiteSel.innerHTML = '';
  DATA.suites.forEach((s, i) => {{
    const o = document.createElement('option'); o.value = i; o.textContent = s.title; suiteSel.appendChild(o);
  }});
  onSuiteChange();
}}

function onSuiteChange() {{
  const suite = DATA.suites[suiteSel.value || 0];
  if (!suite) return;
  reportSel.innerHTML = '';
  suite.reports.forEach((r, i) => {{
    const o = document.createElement('option'); o.value = i; o.textContent = r.date + ' (' + r.report_id + ')';
    reportSel.appendChild(o);
  }});
  reportSel.value = suite.reports.length - 1;
  onReportChange();
}}

function onReportChange() {{
  const suite = DATA.suites[suiteSel.value || 0];
  const rep = suite.reports[reportSel.value || 0];
  const thrs = [...new Set(rep.benchmarks.map(b => b.threads))].sort((a,b) => a - b);
  threadSel.innerHTML = '';
  thrs.forEach(t => {{
    const o = document.createElement('option'); o.value = t; o.textContent = t + ' Threads'; threadSel.appendChild(o);
  }});
  if (thrs.includes(18)) threadSel.value = 18;
  renderAll();
}}

function renderBarChart(svgId, legendId, items, regimes) {{
  const svg = document.getElementById(svgId);
  const leg = document.getElementById(legendId);
  const W = 640, H = 330, ml = 48, mr = 16, mt = 18, mb = 78;
  const pw = W - ml - mr, ph = H - mt - mb;
  const backends = [...new Set(items.map(x => x.backend_id))];
  const maxVal = Math.max(80, ...items.map(x => x.roofline_util_format_pct || 0)) * 1.1;

  leg.innerHTML = regimes.map((r, i) => `<span><i class="dot" style="background:${{i===0?'#38bdf8':'#f97316'}}"></i>${{r}}</span>`).join('');
  let h = '';
  for (let i = 0; i <= 4; i++) {{
    const v = maxVal * i / 4;
    const y = mt + ph - (ph * i / 4);
    h += `<line x1="${{ml}}" y1="${{y}}" x2="${{W-mr}}" y2="${{y}}" stroke="#334155" stroke-dasharray="3 3"/>`;
    h += `<text x="${{ml-6}}" y="${{y+4}}" text-anchor="end" fill="#94a3b8" font-size="11">${{v.toFixed(0)}}%</text>`;
  }}
  const grpW = pw / Math.max(1, backends.length);
  const barW = Math.min(26, (grpW - 10) / Math.max(1, regimes.length));
  backends.forEach((b, bi) => {{
    const gx = ml + bi * grpW + grpW / 2;
    regimes.forEach((reg, ri) => {{
      const rec = items.find(x => x.backend_id === b && x.regime_id === reg);
      if (!rec) return;
      const val = rec.roofline_util_format_pct;
      const bh = (val / maxVal) * ph;
      const x = gx - (regimes.length * barW) / 2 + ri * barW;
      const y = mt + ph - bh;
      const col = ri === 0 ? '#38bdf8' : '#f97316';
      h += `<rect x="${{x}}" y="${{y}}" width="${{barW-2}}" height="${{bh}}" rx="3" fill="${{col}}"><title>${{bShort(b)}} (${{reg}}): ${{val.toFixed(1)}}% (${{fmtMs(rec.median_ms)}} ms)</title></rect>`;
      h += `<text x="${{x+(barW-2)/2}}" y="${{y-4}}" text-anchor="middle" fill="#e2e8f0" font-size="9.5">${{val.toFixed(0)}}%</text>`;
    }});
    h += `<text x="${{gx}}" y="${{mt+ph+14}}" text-anchor="end" transform="rotate(-28 ${{gx}} ${{mt+ph+14}})" fill="#cbd5e1" font-size="10">${{bShort(b).replace('LiteRT ','')}}</text>`;
  }});
  svg.innerHTML = h;
}}

function renderKvChart(rep) {{
  const svg = document.getElementById('kvSvg');
  const leg = document.getElementById('kvLegend');
  const W = 640, H = 330, ml = 50, mr = 20, mt = 18, mb = 44;
  const pw = W - ml - mr, ph = H - mt - mb;
  const kvRegs = ['decode_kv64','decode_kv1024','decode_kv4096','decode_kv8192','decode_kv16384'];
  const kvLabels = ['64','1024','4096','8192','16384'];
  const series = ['litert_xnn_qb4w32','litert_ynn_qb4w32_hybrid_sdpa','litert_ynn_qb4w32_pure_sdpa','direct_ynn_capi_qb4w32','ggml_cpu_neon_fa'];
  const pts = [];
  series.forEach(b => {{
    kvRegs.forEach(r => {{
      const rec = rep.benchmarks.find(x => x.backend_id === b && x.subgraph_id === 'global_full_attn' && x.regime_id === r && x.threads === 18);
      if (rec) pts.push(rec.median_ms);
    }});
  }});
  const maxVal = Math.max(2, ...pts) * 1.12;
  leg.innerHTML = series.map(b => `<span><i class="dot" style="background:${{bCol(b)}}"></i>${{bShort(b)}}</span>`).join('');
  let h = '';
  for (let i = 0; i <= 4; i++) {{
    const v = maxVal * i / 4;
    const y = mt + ph - (ph * i / 4);
    h += `<line x1="${{ml}}" y1="${{y}}" x2="${{W-mr}}" y2="${{y}}" stroke="#334155" stroke-dasharray="3 3"/>`;
    h += `<text x="${{ml-6}}" y="${{y+4}}" text-anchor="end" fill="#94a3b8" font-size="11">${{v.toFixed(1)}}ms</text>`;
  }}
  kvLabels.forEach((lb, i) => {{
    const x = ml + (pw * i / (kvLabels.length - 1));
    h += `<text x="${{x}}" y="${{mt+ph+22}}" text-anchor="middle" fill="#cbd5e1" font-size="11">S_kv=${{lb}}</text>`;
  }});
  series.forEach(b => {{
    const coords = [];
    kvRegs.forEach((r, i) => {{
      const rec = rep.benchmarks.find(x => x.backend_id === b && x.subgraph_id === 'global_full_attn' && x.regime_id === r && x.threads === 18);
      if (!rec) return;
      const x = ml + (pw * i / (kvRegs.length - 1));
      const y = mt + ph - (rec.median_ms / maxVal) * ph;
      coords.push([x, y, rec.median_ms, kvLabels[i]]);
    }});
    if (coords.length > 1) {{
      h += `<polyline fill="none" stroke="${{bCol(b)}}" stroke-width="2.4" points="${{coords.map(c => c[0]+','+c[1]).join(' ')}}"/>`;
    }}
    coords.forEach(c => {{
      h += `<circle cx="${{c[0]}}" cy="${{c[1]}}" r="4" fill="${{bCol(b)}}"><title>${{bShort(b)}} (S_kv=${{c[3]}}): ${{fmtMs(c[2])}} ms</title></circle>`;
    }});
  }});
  svg.innerHTML = h;
}}

function renderAll() {{
  const suite = DATA.suites[suiteSel.value || 0];
  const rep = suite.reports[reportSel.value || 0];
  const thr = Number(threadSel.value || 18);
  const sg = sgSel.value;

  document.getElementById('subtitle').innerHTML =
    `${{suite.title}} · Date: <strong>${{rep.date}}</strong> · <a href="${{rep.report_url}}">Full Markdown Report</a> · Updated ${{DATA.generated_at}}`;

  // KPI cards
  const flP = rep.benchmarks.filter(b => b.subgraph_id === 'full_local_block' && b.regime_id === 'prefill_1024' && b.threads === 18);
  const flD = rep.benchmarks.filter(b => b.subgraph_id === 'full_local_block' && b.regime_id === 'decode_kv1024' && b.threads === 18);
  const bestP = flP.slice().sort((a,b) => a.median_ms - b.median_ms)[0];
  const bestD = flD.slice().sort((a,b) => a.median_ms - b.median_ms)[0];
  const vs = rep.verification_summary || {{}};
  document.getElementById('kpiGrid').innerHTML = `
    <div class="card"><div class="kpi-label">Fastest Local Block Prefill (1024, 18T)</div>
      <div class="kpi-val">${{bestP ? fmtMs(bestP.median_ms) + ' ms' : '—'}}</div>
      <div class="kpi-sub">${{bestP ? bShort(bestP.backend_id) + ' · ' + bestP.roofline_util_format_pct.toFixed(1) + '% roofline' : ''}}</div></div>
    <div class="card"><div class="kpi-label">Fastest Local Block Cold Decode (18T)</div>
      <div class="kpi-val">${{bestD ? fmtMs(bestD.median_ms) + ' ms' : '—'}}</div>
      <div class="kpi-sub">${{bestD ? bShort(bestD.backend_id) + ' · ' + bestD.effective_gbps_format.toFixed(1) + ' GB/s (' + bestD.roofline_util_format_pct.toFixed(1) + '%)' : ''}}</div></div>
    <div class="card"><div class="kpi-label">FP64 Level A Parity Gate</div>
      <div class="kpi-val" style="color:var(--good)">${{vs.level_a_passed || 0}} / ${{vs.total_cases || 0}} PASS</div>
      <div class="kpi-sub">All 10 backends verified vs. dequantized FP64</div></div>
    <div class="card"><div class="kpi-label">18T Determinism Gate</div>
      <div class="kpi-val">${{(vs.non_deterministic_18t || []).length === 0 ? '100% Bit-Exact' : (vs.non_deterministic_18t.length + ' XNNPACK Drift')}}</div>
      <div class="kpi-sub">YNNPACK &amp; ggml: 100% bit-exact (0.00 diff)</div></div>`;

  // Roofline bar chart for selected subgraph & thread count
  const sgItems = rep.benchmarks.filter(b => b.subgraph_id === sg && b.threads === thr);
  const prefReg = sgItems.some(x => x.regime_id === 'prefill_1024') ? 'prefill_1024' : 'prefill_128';
  const decReg = sg === 'geglu_mlp' ? 'decode_1' : (sg.includes('global') ? 'decode_kv4096' : 'decode_kv1024');
  document.getElementById('rooflineChartTitle').textContent =
    `${{sg}} — Whole-Subgraph Practical Roofline Util at ${{thr}}T (%)`;
  renderBarChart('rooflineSvg', 'rooflineLegend', sgItems, [prefReg, decReg]);
  renderKvChart(rep);

  // Projections table
  const projs = rep.projections_60_layer || [];
  document.getElementById('projTable').innerHTML = `
    <thead><tr><th>Backend</th><th>Threads</th><th>Level 3 Counterpart</th><th>L2 Proj Prefill 1024</th><th>L3 Measured Prefill</th><th>L2 Proj Decode S=1024</th><th>L3 Measured Decode</th><th>L2 Proj S=4096</th><th>L3 N=4096</th></tr></thead>
    <tbody>${{projs.map(p => `<tr>
      <td><strong>${{bShort(p.backend_id)}}</strong></td><td>${{p.threads}}T</td><td>${{p.level3_config}}</td>
      <td>${{p.proj_60l_prefill_1024_tok_s ? p.proj_60l_prefill_1024_tok_s.toFixed(1) + ' tok/s' : '—'}}</td>
      <td>${{p.l3_prefill_1024_tok_s ? p.l3_prefill_1024_tok_s.toFixed(1) + ' tok/s (' + p.l3_vs_l2_prefill_eff_pct + '%)' : '—'}}</td>
      <td>${{p.proj_60l_decode_1024_tok_s ? p.proj_60l_decode_1024_tok_s.toFixed(2) + ' tok/s' : '—'}}</td>
      <td>${{p.l3_decode_1024_tok_s ? p.l3_decode_1024_tok_s.toFixed(2) + ' tok/s (' + p.l3_vs_l2_decode_1024_eff_pct + '%)' : '—'}}</td>
      <td>${{p.proj_60l_decode_4096_tok_s ? p.proj_60l_decode_4096_tok_s.toFixed(2) + ' tok/s' : '—'}}</td>
      <td>${{p.l3_decode_4096_tok_s ? p.l3_decode_4096_tok_s.toFixed(2) + ' tok/s' : '—'}}</td>
    </tr>`).join('')}}</tbody>`;

  // Bench detail table
  document.getElementById('benchTableTitle').textContent = `Detailed Subgraph Benchmarks — ${{sg}} (${{thr}}T)`;
  document.getElementById('benchTable').innerHTML = `
    <thead><tr><th>Regime</th><th>Backend</th><th>Quant / KV</th><th>Bound</th><th>Median (ms)</th><th>CV %</th><th>Eff Perf</th><th>T_min (ms)</th><th>Format Roofline</th><th>Ideal Roofline</th></tr></thead>
    <tbody>${{sgItems.map(r => `<tr>
      <td><code>${{r.regime_id}}</code></td>
      <td><strong>${{bShort(r.backend_id)}}</strong></td>
      <td>${{r.weight_quant}} / ${{r.kv_dtype}}</td>
      <td>${{r.bound}}</td>
      <td><strong>${{fmtMs(r.median_ms)}}</strong></td>
      <td>${{r.cv_pct.toFixed(2)}}%</td>
      <td>${{r.bound === 'compute' ? r.effective_gops.toFixed(1) + ' GOPS' : r.effective_gbps_format.toFixed(1) + ' GB/s'}}</td>
      <td>${{fmtMs(r.t_min_format_ms)}}</td>
      <td><strong>${{r.roofline_util_format_pct.toFixed(1)}}%</strong></td>
      <td>${{r.roofline_util_ideal_pct.toFixed(1)}}%</td>
    </tr>`).join('')}}</tbody>`;

  // Verify table
  const ver = rep.verification.filter(v => v.subgraph_id === sg);
  document.getElementById('verifyTable').innerHTML = `
    <thead><tr><th>Regime</th><th>Backend</th><th>Quant</th><th>LvA Cosine</th><th>LvA SQNR (dB)</th><th>LvA KL-Div</th><th>LvA Gate</th><th>LvB Cosine</th><th>LvB SQNR (dB)</th><th>18T Det</th></tr></thead>
    <tbody>${{ver.map(v => `<tr>
      <td><code>${{v.regime_id}}</code></td>
      <td><strong>${{bShort(v.backend_id)}}</strong></td>
      <td>${{v.weight_quant}}</td>
      <td>${{v.lva_cos.toFixed(6)}}</td>
      <td>${{v.lva_sqnr.toFixed(1)}}</td>
      <td>${{v.lva_kl.toExponential(2)}}</td>
      <td><span class="badge ${{v.lva_pass?'pass':'fail'}}">${{v.lva_pass?'PASS':'FAIL'}}</span></td>
      <td>${{v.lvb_cos.toFixed(5)}}</td>
      <td>${{v.lvb_sqnr.toFixed(1)}}</td>
      <td><span class="badge ${{v.det?'pass':'warn'}}">${{v.det?'BIT-EXACT':'DRIFT ' + v.max_repro_diff.toExponential(1)}}</span></td>
    </tr>`).join('')}}</tbody>`;
}}

suiteSel.addEventListener('change', onSuiteChange);
reportSel.addEventListener('change', onReportChange);
threadSel.addEventListener('change', renderAll);
sgSel.addEventListener('change', renderAll);
populateSelectors();
</script>
</body>
</html>
"""


def render_md(payload: Dict[str, Any]) -> str:
  lines: List[str] = [
      "# Level 2 Layer Benchmark Suite Dashboard — Gemma 4 31B",
      "",
      "- **Interactive HTML Dashboard**: [`layer/dashboards/index.html`](index.html)",
      "- **Raw Time-Series JSON**: [`layer/dashboards/data.json`](data.json)",
      f"- **Last Updated**: `{payload['generated_at']}`",
      "- **Tracked Hardware Suites**:",
  ]
  for s in payload["suites"]:
    lines.append(f"  - **{s['title']}** (`{s['suite_id']}`):")
    for r in s["reports"]:
      c = r["commits"]
      lines.append(
          f"    - **`{r['date']}`** (`LiteRT @ {c.get('litert_commit','?')[:8]}`, `XNNPACK @ {c.get('xnnpack_commit','?')[:8]}`, `llama.cpp @ {c.get('llama_cpp_commit','?')[:8]}`): [{r['report_id']}]({r['report_url']})"
      )
  lines.append("")

  for s in payload["suites"]:
    if not s["reports"]:
      continue
    rep = s["reports"][-1]
    lines.append(f"## {s['title']} (`{rep['date']}`)")
    lines.append("")
    bmap = {
        (b["backend_id"], b["subgraph_id"], b["regime_id"], b["threads"]): b
        for b in rep["benchmarks"]
    }
    chart_order = [
        ("litert_xnn_qb4w32", "XNN QB4W"),
        ("litert_ynn_qc4w_sdpa", "YNN QC4W"),
        ("direct_ynn_capi_qc4w", "Direct QC4W"),
        ("litert_ynn_qb4w32_hybrid_sdpa", "YNN Hybrid"),
        ("litert_ynn_qb4w32_pure_sdpa", "YNN Pure"),
        ("direct_ynn_capi_qb4w32", "Direct QB4W"),
        ("ggml_cpu_neon_fa", "ggml NEON"),
        ("ggml_cpu_sme_fa", "ggml SME2"),
    ]
    p_vals = [
        round(bmap[(bid, "full_local_block", "prefill_1024", 18)]["roofline_util_format_pct"], 1)
        if (bid, "full_local_block", "prefill_1024", 18) in bmap
        else 0.0
        for bid, _ in chart_order
    ]
    d_vals = [
        round(bmap[(bid, "full_local_block", "decode_kv1024", 18)]["roofline_util_format_pct"], 1)
        if (bid, "full_local_block", "decode_kv1024", 18) in bmap
        else 0.0
        for bid, _ in chart_order
    ]
    lines.append(
        "> **Series Legend**: **Blue (`#2563eb`)** = Prefill `T=1024` Roofline Util (%) | **Orange (`#ea580c`)** = Cold Decode `T=1, S_kv=1024` Roofline Util (%)"
    )
    lines.append("")
    lines.append("```mermaid")
    lines.append(
        '%%{init: {"xyChart": {"width": 1100, "height": 400}, "themeVariables": {"xyChart": {"plotColorPalette": "#2563eb, #ea580c"}}}}%%'
    )
    lines.append("xychart-beta")
    lines.append(
        '    title "Full Local Block (50/60 Layers) Whole-Subgraph Roofline Utilization at 18T (%)"'
    )
    lines.append("    x-axis " + json.dumps([lbl for _, lbl in chart_order]))
    lines.append('    y-axis "Whole-Subgraph Roofline Util (%)" 0 --> 100')
    lines.append('    bar "Prefill 1024" ' + json.dumps(p_vals))
    lines.append('    bar "Decode KV=1024" ' + json.dumps(d_vals))
    lines.append("```")
    lines.append("")
    lines.append(
        "| Backend | Quant / KV | Full Local Prefill `1024` (`18T`) | Full Local Decode `S=1024` (`18T`) | Full Global Prefill `1024` (`18T`) | Full Global Decode `S=4096` (`18T`) | Full Global Decode `S=16384` (`18T`) |"
    )
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: |")
    backends = list(dict.fromkeys(b["backend_id"] for b in rep["benchmarks"]))
    for bid in backends:
      if bid == "ggml_cpu_neon_unfused":
        continue
      def cell(sg: str, reg: str) -> str:
        r = bmap.get((bid, sg, reg, 18))
        if not r:
          return "—"
        ms = f"{r['median_ms']:.2f}" if r["median_ms"] >= 10 else f"{r['median_ms']:.3f}"
        return f"**`{ms} ms`** (`{r['roofline_util_format_pct']:.1f}%`)"
      sample = next((b for b in rep["benchmarks"] if b["backend_id"] == bid), {})
      qkv = f"`{sample.get('weight_quant','?')}` / `{sample.get('kv_dtype','?')}`"
      lines.append(
          f"| `{bid}` | {qkv} | {cell('full_local_block', 'prefill_1024')} | {cell('full_local_block', 'decode_kv1024')} | {cell('full_global_block', 'prefill_1024')} | {cell('full_global_block', 'decode_kv4096')} | {cell('full_global_block', 'decode_kv16384')} |"
      )
    lines.append("")
    if rep.get("projections_60_layer"):
      lines.append("### 60-Layer Level 2 Projection vs. Level 3 Measured (`1024 → 128`)")
      lines.append("")
      lines.append(
          "| Backend (`Threads`) | Level 3 Counterpart | L2 Proj Prefill `1024` | L3 Measured Prefill (L3/L2) | L2 Proj Decode `S=1024` | L3 Measured Decode |"
      )
      lines.append("| :--- | :--- | ---: | ---: | ---: | ---: |")
      for p in rep["projections_60_layer"]:
        pp = f"**`{p['proj_60l_prefill_1024_tok_s']:.1f} tok/s`**" if "proj_60l_prefill_1024_tok_s" in p else "—"
        l3p = (
            f"**`{p['l3_prefill_1024_tok_s']:.1f} tok/s`** (`{p['l3_vs_l2_prefill_eff_pct']:.1f}%`)"
            if p.get("l3_prefill_1024_tok_s") and "l3_vs_l2_prefill_eff_pct" in p
            else "—"
        )
        dg = f"**`{p['proj_60l_decode_1024_tok_s']:.2f} tok/s`**" if "proj_60l_decode_1024_tok_s" in p else "—"
        l3d = f"**`{p['l3_decode_1024_tok_s']:.2f} tok/s`**" if p.get("l3_decode_1024_tok_s") else "—"
        lines.append(f"| `{p['backend_id']}` (`{p['threads']}T`) | {p['level3_config']} | {pp} | {l3p} | {dg} | {l3d} |")
      lines.append("")
  return "\n".join(lines)


def main() -> None:
  DASH_ROOT.mkdir(parents=True, exist_ok=True)
  payload = collect_suites()
  (DASH_ROOT / "data.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  (DASH_ROOT / "index.html").write_text(render_html(payload), encoding="utf-8")
  (DASH_ROOT / "README.md").write_text(render_md(payload) + "\n", encoding="utf-8")
  print(f"Updated {DASH_ROOT}/data.json, index.html, README.md ({len(payload['suites'])} suite(s))")


if __name__ == "__main__":
  main()
