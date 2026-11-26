# AI Edge Bench — Level 2: Layer Tier Makefile

REPORT_DIR_ARM64 ?= reports/2026-11-26-gemma4-31b-apple-m5-max
CEILINGS_ARM64 ?= ../kernel/reports/2026-11-25-gemma4-31b-apple-m5-max/ceilings_apple_m5_max.json
SUITE_ARM64 ?= suites/gemma4-31b-cpu-apple-m5-max.json
MODEL_DATA_ARM64 ?= ../model/reports/2026-11-26-gemma4-31b-cpu-apple-m5-max-apple-m5-max/data.json

.DEFAULT_GOAL := help
.PHONY: help bench report-arm64 dashboard test

help:
	@echo "AI Edge Bench - Layer Tier (Level 2)"
	@echo
	@echo "  make report-arm64  regenerate Apple M5 Max layer roofline report & dashboard"
	@echo "  make dashboard     update layer/dashboards/index.html, data.json, and README.md"
	@echo "  make test          validate C++ syntax (where headers present), report generation & dashboard"

bench:
	@echo "[layer] See layer/subgraphs/README.md to build and run bench_litert_layers and bench_ggml_layers."

report-arm64:
	python3 scripts/generate_layer_report.py \
	  --report-dir $(REPORT_DIR_ARM64) \
	  --ceilings $(CEILINGS_ARM64) \
	  --suite $(SUITE_ARM64) \
	  --model-data $(MODEL_DATA_ARM64)
	python3 scripts/update_dashboard.py

dashboard:
	python3 scripts/update_dashboard.py

test:
	python3 -m py_compile scripts/generate_layer_report.py scripts/update_dashboard.py dashboards/update_dashboard.py
	python3 -c 'import json; json.load(open("$(SUITE_ARM64)"))'
	@if [ -d "$(REPORT_DIR_ARM64)/raw" ]; then \
	  python3 scripts/generate_layer_report.py \
	    --report-dir $(REPORT_DIR_ARM64) \
	    --ceilings $(CEILINGS_ARM64) \
	    --suite $(SUITE_ARM64) \
	    --model-data $(MODEL_DATA_ARM64); \
	  python3 scripts/update_dashboard.py; \
	fi
	@echo "[layer] All syntax checks, suite JSON validation, report generation, and dashboard updates passed (ok)"
