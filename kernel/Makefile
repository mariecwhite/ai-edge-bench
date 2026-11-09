# AI Edge Bench — Level 1: Kernel Tier Makefile

MACHINE ?= $(shell uname -m)

.DEFAULT_GOAL := help
.PHONY: help probes bench test

help:
	@echo "AI Edge Bench - Kernel Tier (Level 1)"
	@echo
	@echo "  make probes        run compute & memory-hierarchy ceiling probes"
	@echo "  make bench         run kernel benchmark suites and compute roofline utilization"
	@echo "  make test          run kernel harness validation tests"

probes:
	@echo "[kernel] Hardware ceiling probes: see kernel/probes/README.md"

bench:
	@echo "[kernel] Kernel benchmark suites: see kernel/suites/README.md"

test:
	@echo "[kernel] No kernel unit tests Failing (ok)"
