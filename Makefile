# AI Edge Bench — Three-Tier Benchmarking Architecture (Kernel -> Layer -> Model)
#
#   make base                       # build shared aeb/base container (docker/base/)
#   make kernel-probes              # run Level 1 hardware ceiling probes
#   make kernel-bench               # run Level 1 microkernel / shader benchmarks
#   make layer-bench                # run Level 2 compound subgraph benchmarks
#   make MACHINE=dgx-spark bench    # run Level 3 model benchmarks (delegates to model/)
#   make test                       # run unit tests across kernel, layer, and model

MACHINE ?= epyc-7443p
AEB_BASE_IMAGE ?= aeb/base:latest
AEB_BASE_OS_IMAGE ?= ubuntu:24.04

MODEL_TARGETS := images models bench bench-llama-cpp bench-litert-lm \
                 bench-gpu sysinfo config machines clean-results tools matched-models \
                 datasets suite-perf suite-prefix-cache suite-accuracy suite-repro report \
                 dashboard diagnose-activations arm-images arm-models x86-images x86-models \
                 suite-sequence native gpu-models native-report native-crosscheck

.DEFAULT_GOAL := help
.PHONY: help base test kernel-probes kernel-bench kernel-test \
        layer-bench layer-test model-test $(MODEL_TARGETS)

help:
	@echo "AI Edge Bench (Kernel -> Layer -> Model) - MACHINE=$(MACHINE)"
	@echo
	@echo "  Shared Base Container (docker/base/):"
	@echo "    make base                       build shared $(AEB_BASE_IMAGE) image"
	@echo
	@echo "  Level 1 — Kernel Tier (kernel/):"
	@echo "    make kernel-probes              run compute & memory-hierarchy ceiling probes"
	@echo "    make kernel-bench               run microkernel & GPU shader suites"
	@echo "    make kernel-test                run kernel validation tests"
	@echo
	@echo "  Level 2 — Layer Tier (layer/):"
	@echo "    make layer-bench                run compound subgraph suites (Tensor API & runtimes)"
	@echo "    make layer-test                 run layer subgraph tests"
	@echo
	@echo "  Level 3 — Model Tier (model/):"
	@echo "    make MACHINE=<m> images         build model framework images (llama.cpp, LiteRT-LM, ...)"
	@echo "    make MACHINE=<m> models         download model weights into model/models/"
	@echo "    make MACHINE=<m> bench          run CPU model benchmarks"
	@echo "    make MACHINE=<m> suite-perf     interleaved model performance suite"
	@echo "    make MACHINE=<m> suite-accuracy MMLU / GSM8K model accuracy gate"
	@echo "    make MACHINE=<m> suite-repro    output reproducibility suite"
	@echo "    make MACHINE=<m> report         generate model report, charts & dashboard"
	@echo "    make dashboard                  update model/dashboards/index.html"
	@echo "    make -C model help              show all model-tier targets"
	@echo
	@echo "  Testing:"
	@echo "    make test                       run unit tests across all tiers"

base:
	docker build -t $(AEB_BASE_IMAGE) --build-arg BASE_IMAGE=$(AEB_BASE_OS_IMAGE) -f docker/base/Dockerfile docker

kernel-probes:
	$(MAKE) -C kernel MACHINE=$(MACHINE) probes

kernel-bench:
	$(MAKE) -C kernel MACHINE=$(MACHINE) bench

kernel-test:
	$(MAKE) -C kernel MACHINE=$(MACHINE) test

layer-bench:
	$(MAKE) -C layer MACHINE=$(MACHINE) bench

layer-test:
	$(MAKE) -C layer MACHINE=$(MACHINE) test

model-test:
	$(MAKE) -C model MACHINE=$(MACHINE) test

test: kernel-test layer-test model-test

$(MODEL_TARGETS):
	$(MAKE) -C model MACHINE=$(MACHINE) $(if $(SUITE),SUITE=$(SUITE),) $(if $(DATE),DATE=$(DATE),) $(if $(SINCE),SINCE=$(SINCE),) $@
