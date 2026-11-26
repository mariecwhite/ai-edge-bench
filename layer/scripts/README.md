# Layer Runner, Roofline & Dashboard Scripts (`layer/scripts/`)

Orchestration, whole-subgraph roofline calculation, and dashboard generation scripts for Level 2 (`layer/`):

- [generate_layer_report.py](generate_layer_report.py) — Combines Level 1 empirical hardware ceilings (`kernel/reports/.../ceilings_*.json`), Level 2 raw benchmark & two-level FP64 verification JSONs (`layer/reports/<date>-<suite>/raw/`), and Level 3 end-to-end model metrics (`model/reports/.../data.json`) to compute Whole-Subgraph Practical Roofline Utilization (%), 60-layer L2→L3 projections, and emit `layer_report_*.json` and `layer_report_*.md` (with Mermaid charts).
- [update_dashboard.py](update_dashboard.py) — Aggregates dated `layer/reports/YYYY-MM-DD-<suite>/` reports and emits the interactive HTML/SVG dashboard ([../dashboards/index.html](../dashboards/index.html)), structured JSON ([../dashboards/data.json](../dashboards/data.json)), and GitHub Markdown dashboard ([../dashboards/README.md](../dashboards/README.md)).

## Usage

```bash
# Regenerate the 2026-11-26 Gemma 4 31B Apple M5 Max report and dashboard:
make -C layer report-arm64

# Update layer/dashboards/ across all dated reports:
make layer-dashboard
```
