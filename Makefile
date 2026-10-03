# AI Edge Bench container harness.
#
#   make base                      # build the shared harness image (do this first)
#   make MACHINE=dgx-spark images  # build the framework images for a machine
#   make MACHINE=dgx-spark models  # stage model weights into ./models
#   make MACHINE=dgx-spark bench   # run the full CPU comparison
#
# MACHINE must match a file in compose/machines/.

MACHINE ?= epyc-7443p
SUITE ?= suites/gemma4-e2b-cpu.json
COMPOSE := docker compose -f compose.yaml -f compose/machines/$(MACHINE).yaml
# Report date and optional run filter for `make report`.
DATE ?= $(shell date -u +%Y-%m-%d)
SINCE ?=

.DEFAULT_GOAL := help
.PHONY: help base images models bench bench-llama-cpp bench-litert-lm \
        bench-gpu sysinfo config machines clean-results tools matched-models \
        datasets suite-perf suite-accuracy suite-repro report test diagnose-activations \
        arm-images arm-models

help:
	@echo "AI Edge Bench - MACHINE=$(MACHINE)"
	@echo
	@echo "  make base                       build the shared aeb/base image"
	@echo "  make MACHINE=<m> images         build llama.cpp + LiteRT-LM images"
	@echo "  make MACHINE=apple-m5-max arm-images build optional ONNX Runtime image"
	@echo "  make MACHINE=apple-m5-max arm-models stage pinned ONNX E2B package"
	@echo "  make MACHINE=<m> models         download model weights into ./models"
	@echo "  make MACHINE=<m> bench          run every CPU benchmark"
	@echo "  make MACHINE=<m> bench-gpu      run the machine's accelerator benchmark"
	@echo "  make MACHINE=<m> sysinfo        print captured machine metadata"
	@echo "  make MACHINE=<m> config         render the merged compose config"
	@echo "  make machines                   list available machine overlays"
	@echo
	@echo "  Framework-neutral harness (docs/harness.md):"
	@echo "  make tools                      build the offline tools image"
	@echo "  make MACHINE=<m> matched-models build matched-weights GGUFs from the QAT checkpoint"
	@echo "  make datasets                   stage pinned MMLU + GSM8K into ./datasets"
	@echo "  make MACHINE=<m> suite-perf     prime caches, then interleaved timing rounds"
	@echo "  make MACHINE=<m> suite-accuracy accuracy gate for every suite config"
	@echo "  make MACHINE=<m> suite-repro    output reproducibility (greedy + seeded)"
	@echo "  make MACHINE=<m> report         reports/<date>-<suite>-<machine>/ + history"
	@echo "  make test                       harness unit tests (no models needed)"
	@echo "  make MACHINE=<m> diagnose-activations  PyTorch reference with/without activation quantization"

machines:
	@ls compose/machines/*.yaml | xargs -n1 basename | sed 's/\.yaml$$//'

# The framework images copy /opt/aeb out of this image, so it must exist first.
base:
	$(COMPOSE) --profile base build base

images: base
	$(COMPOSE) --profile cpu build

arm-images: base
	$(COMPOSE) --profile onnxruntime build onnxruntime

arm-models: tools
	$(COMPOSE) --profile tools run --rm tools aeb.prep.fetch_arm_models

models: base
	$(COMPOSE) --profile models run --rm fetch-models

# Services are run one at a time and never with `up`: concurrent benchmark
# containers would contend for the same cores and invalidate the numbers.
bench-llama-cpp:
	$(COMPOSE) run --rm llama-cpp

bench-litert-lm:
	$(COMPOSE) run --rm litert-lm-xnnpack
	$(COMPOSE) run --rm litert-lm-ynnpack

bench: bench-llama-cpp bench-litert-lm

bench-gpu:
	@$(COMPOSE) config --services | grep -qx llama-cpp-cuda \
	  && $(COMPOSE) run --rm llama-cpp-cuda \
	  || { $(COMPOSE) config --services | grep -qx llama-cpp-vulkan \
	       && $(COMPOSE) run --rm llama-cpp-vulkan \
	       || echo "no accelerator service defined for MACHINE=$(MACHINE)"; }

sysinfo: base
	$(COMPOSE) --profile base run --rm base aeb-sysinfo

config:
	$(COMPOSE) config

tools: base
	$(COMPOSE) --profile tools build tools

matched-models: tools
	$(COMPOSE) --profile tools run --rm --entrypoint aeb-prep-matched-gguf tools

datasets: tools
	$(COMPOSE) --profile tools run --rm tools aeb.datasets mmlu --out /datasets
	$(COMPOSE) --profile tools run --rm tools aeb.datasets gsm8k --out /datasets

# One benchmark container at a time; never in parallel (see scripts/suite.py).
suite-perf:
	python3 scripts/suite.py perf --machine $(MACHINE) --suite $(SUITE)

# Accuracy for the original two frameworks was verified invariant across
# thread counts. The suite runner keeps the configured count for new engines.
suite-accuracy:
	python3 scripts/suite.py accuracy --machine $(MACHINE) --suite $(SUITE) \
	  --parallel 2 --shards 2 --threads half

suite-repro:
	python3 scripts/suite.py repro --machine $(MACHINE) --suite $(SUITE) --cooldown-s 2

report:
	$(COMPOSE) --profile tools run --rm -v $(CURDIR)/suites:/suites:ro tools aeb.report \
	  --machine $(MACHINE) --suite /suites/$(notdir $(SUITE)) --date $(DATE) \
	  $(if $(SINCE),--since $(SINCE),) \
	  $(if $(wildcard reports/notes/$(notdir $(basename $(SUITE)))-$(MACHINE).md),--extra-notes /reports/notes/$(notdir $(basename $(SUITE)))-$(MACHINE).md,)

test:
	$(COMPOSE) --profile tools run --rm -v $(CURDIR)/scripts:/opt/scripts:ro \
	  -v $(CURDIR)/suites:/opt/suites:ro --entrypoint python3 tools \
	  -m unittest discover -s /opt/aeb/harness/tests -p 'test*.py'

# Optional: explains accuracy gaps between frameworks that share the mobile QAT
# weights by rerunning the PyTorch reference with and without its static int8
# activation quantization on items where the matched configs disagree.
diagnose-activations: tools
	docker build -t aeb/tools-hf:latest -f docker/tools/hf.Dockerfile docker
	docker run --rm -v $(CURDIR)/models:/models:ro -v $(CURDIR)/results:/results \
	  -v $(CURDIR)/datasets:/datasets:ro aeb/tools-hf:latest python3 -m aeb.prep.hf_reference \
	  --qat-dir /models/google/gemma-4-E2B-it-qat-mobile-transformers \
	  --a /results/$$($(COMPOSE) --profile tools run --rm -T --entrypoint printenv tools AEB_MACHINE | tail -1)/accuracy-mmlu/llama.cpp/matched \
	  --b /results/$$($(COMPOSE) --profile tools run --rm -T --entrypoint printenv tools AEB_MACHINE | tail -1)/accuracy-mmlu/litert-lm/matched \
	  --out /results/$$($(COMPOSE) --profile tools run --rm -T --entrypoint printenv tools AEB_MACHINE | tail -1)/diagnostics/hf-reference-mmlu.json

clean-results:
	rm -rf results/*
