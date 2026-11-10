# AI Edge Bench — Level 1: Kernel Tier Makefile

CC ?= gcc
CFLAGS ?= -O3 -march=sapphirerapids -fopenmp
THREADS ?= 28
TRIALS ?= 7
REPORT_DIR ?= reports/gemma4-31b-sapphire-rapids
LITERT_BIN ?= /opt/litert-lm/bin/litert_lm_advanced_main
LITERT_LIB_DIR ?= /opt/litert-lm/lib:/opt/litert-lm/bin

.DEFAULT_GOAL := help
.PHONY: help probes bench verify report test

help:
	@echo "AI Edge Bench - Kernel Tier (Level 1)"
	@echo
	@echo "  make probes        compile & run repeatable compute and memory-hierarchy ceiling probes"
	@echo "  make bench         compile & run repeatable XNNPACK/YNNPACK microkernel benchmarks"
	@echo "  make verify        compile & run XNNPACK/YNNPACK microkernel numerical precision verification"
	@echo "  make report        generate JSON + Markdown kernel report (roofline + precision)"
	@echo "  make test          validate probe, benchmark, and precision harness compilation and report generation"

probes:
	$(CC) $(CFLAGS) probes/cpu_ceilings.c -o /tmp/cpu_ceilings -lm
	/tmp/cpu_ceilings --threads $(THREADS) --trials $(TRIALS) --output $(REPORT_DIR)/ceilings_sapphire_rapids.json

bench:
	$(CC) -shared -fPIC $(CFLAGS) scripts/bench_litert_microkernels.c -o /tmp/libbench_ukernels.so -lm
	LD_LIBRARY_PATH=$(LITERT_LIB_DIR) \
	  LD_PRELOAD=/tmp/libbench_ukernels.so \
	  AEB_BENCH_OUTPUT=$(REPORT_DIR)/ukernel_benchmarks_sapphire_rapids.json \
	  $(LITERT_BIN)

verify:
	$(CC) -shared -fPIC $(CFLAGS) scripts/verify_litert_microkernels.c -o /tmp/libverify_ukernels.so -lm
	LD_LIBRARY_PATH=$(LITERT_LIB_DIR) \
	  LD_PRELOAD=/tmp/libverify_ukernels.so \
	  AEB_VERIFY_OUTPUT=$(REPORT_DIR)/ukernel_precision_sapphire_rapids.json \
	  $(LITERT_BIN)

report:
	python3 scripts/generate_kernel_report.py

test:
	$(CC) -O2 -fopenmp -fsyntax-only probes/cpu_ceilings.c
	$(CC) -O2 -fopenmp -fsyntax-only scripts/bench_litert_microkernels.c
	$(CC) -O2 -fopenmp -fsyntax-only scripts/verify_litert_microkernels.c
	python3 scripts/generate_kernel_report.py
	@echo "[kernel] All syntax checks and kernel report generation passed (ok)"
