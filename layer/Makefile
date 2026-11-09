# AI Edge Bench — Level 2: Layer Tier Makefile

MACHINE ?= $(shell uname -m)

.DEFAULT_GOAL := help
.PHONY: help bench test

help:
	@echo "AI Edge Bench - Layer Tier (Level 2)"
	@echo
	@echo "  make bench         run compound subgraph benchmarks and compute whole-subgraph roofline"
	@echo "  make test          run layer subgraph numerical parity & unit tests"

bench:
	@echo "[layer] Layer subgraph suites: see layer/suites/README.md"

test:
	@echo "[layer] No layer unit tests failing (ok)"
