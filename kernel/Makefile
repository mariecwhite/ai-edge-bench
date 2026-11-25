# AI Edge Bench — Level 1: Kernel Tier Makefile

CC ?= gcc
CFLAGS ?= -O3 -march=sapphirerapids -fopenmp
THREADS ?= 28,56
TRIALS ?= 7
REPORT_DIR ?= reports/2026-11-17-gemma4-31b-sapphire-rapids
REPORT_DIR_ARM64 ?= reports/2026-11-25-gemma4-31b-apple-m5-max
THREADS_ARM64 ?= 6,18
ARM64_CFLAGS ?= -O3 -march=armv8.2-a+sme+sme2+fp16+bf16+i8mm+dotprod -pthread
LITERT_BIN ?= /opt/litert-lm/bin/litert_lm_advanced_main
LITERT_LIB_DIR ?= /opt/litert-lm/lib:/opt/litert-lm/bin

.DEFAULT_GOAL := help
.PHONY: help probes probes-arm64 bench verify report report-arm64 dashboard test

help:
	@echo "AI Edge Bench - Kernel Tier (Level 1)"
	@echo
	@echo "  make probes        compile & run repeatable x86_64 compute and memory-hierarchy ceiling probes"
	@echo "  make probes-arm64  compile & run repeatable ARM64/Apple Silicon SME2+NEON ceiling probes"
	@echo "  make bench         compile & run repeatable XNNPACK/YNNPACK microkernel benchmarks"
	@echo "  make verify        compile & run XNNPACK/YNNPACK microkernel numerical precision verification"
	@echo "  make report        generate JSON + Markdown x86_64 kernel report and update dashboard"
	@echo "  make report-arm64  generate JSON + Markdown ARM64 kernel report and update dashboard"
	@echo "  make dashboard     update kernel/dashboards/index.html, data.json, and README.md"
	@echo "  make test          validate probe, benchmark, and precision harness compilation and report generation"

probes:
	mkdir -p $(REPORT_DIR)
	$(CC) $(CFLAGS) probes/cpu_ceilings.c -o /tmp/cpu_ceilings -lm
	/tmp/cpu_ceilings --threads $(THREADS) --trials $(TRIALS) --output $(REPORT_DIR)/ceilings_sapphire_rapids.json

probes-arm64:
	mkdir -p $(REPORT_DIR_ARM64)
	clang $(ARM64_CFLAGS) probes/cpu_ceilings_arm64.c -o /tmp/cpu_ceilings_arm64 -lm
	/tmp/cpu_ceilings_arm64 --threads $(THREADS_ARM64) --trials $(TRIALS) --output $(REPORT_DIR_ARM64)/ceilings_apple_m5_max.json

bench:
	mkdir -p $(REPORT_DIR)
	$(CC) -shared -fPIC $(CFLAGS) scripts/bench_litert_microkernels.c -o /tmp/libbench_ukernels.so -lm
	LD_LIBRARY_PATH=$(LITERT_LIB_DIR) \
	  LD_PRELOAD=/tmp/libbench_ukernels.so \
	  AEB_BENCH_THREADS=$(THREADS) \
	  AEB_BENCH_OUTPUT=$(REPORT_DIR)/ukernel_benchmarks_sapphire_rapids.json \
	  $(LITERT_BIN)

verify:
	mkdir -p $(REPORT_DIR)
	$(CC) -shared -fPIC $(CFLAGS) scripts/verify_litert_microkernels.c -o /tmp/libverify_ukernels.so -lm
	LD_LIBRARY_PATH=$(LITERT_LIB_DIR) \
	  LD_PRELOAD=/tmp/libverify_ukernels.so \
	  AEB_VERIFY_OUTPUT=$(REPORT_DIR)/ukernel_precision_sapphire_rapids.json \
	  $(LITERT_BIN)

report:
	python3 scripts/generate_kernel_report.py --report-dir $(REPORT_DIR)
	python3 scripts/update_dashboard.py

report-arm64:
	python3 scripts/generate_kernel_report.py --report-dir $(REPORT_DIR_ARM64)
	python3 scripts/update_dashboard.py

dashboard:
	python3 scripts/update_dashboard.py

test:
	@if [ "$$(uname -m)" = "arm64" ] || [ "$$(uname -m)" = "aarch64" ]; then \
	  clang -O2 -march=armv8.2-a+sme+sme2+fp16+bf16+i8mm+dotprod -pthread -fsyntax-only probes/cpu_ceilings_arm64.c; \
	  clang -O2 -march=armv8.2-a+sme+sme2+fp16+bf16+i8mm+dotprod -pthread -fsyntax-only scripts/bench_litert_microkernels_arm64.c; \
	  clang -O2 -march=armv8.2-a+sme+sme2+fp16+bf16+i8mm+dotprod -pthread -fsyntax-only scripts/verify_litert_microkernels_arm64.c; \
	else \
	  $(CC) -O2 -fopenmp -fsyntax-only probes/cpu_ceilings.c; \
	  $(CC) -O2 -fopenmp -fsyntax-only scripts/bench_litert_microkernels.c; \
	  $(CC) -O2 -fopenmp -fsyntax-only scripts/verify_litert_microkernels.c; \
	fi
	python3 scripts/generate_kernel_report.py --report-dir $(REPORT_DIR)
	python3 scripts/generate_kernel_report.py --report-dir $(REPORT_DIR_ARM64)
	python3 scripts/update_dashboard.py
	@echo "[kernel] All syntax checks, kernel report generation, and dashboard updates passed (ok)"
