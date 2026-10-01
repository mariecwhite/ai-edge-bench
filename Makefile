# AI Edge Bench container harness.
#
#   make base                      # build the shared harness image (do this first)
#   make MACHINE=dgx-spark images  # build the framework images for a machine
#   make MACHINE=dgx-spark models  # stage model weights into ./models
#   make MACHINE=dgx-spark bench   # run the full CPU comparison
#
# MACHINE must match a file in compose/machines/.

MACHINE ?= epyc-7443p
COMPOSE := docker compose -f compose.yaml -f compose/machines/$(MACHINE).yaml

.DEFAULT_GOAL := help
.PHONY: help base images models bench bench-llama-cpp bench-litert-lm \
        bench-gpu sysinfo config machines clean-results

help:
	@echo "AI Edge Bench - MACHINE=$(MACHINE)"
	@echo
	@echo "  make base                       build the shared aeb/base image"
	@echo "  make MACHINE=<m> images         build llama.cpp + LiteRT-LM images"
	@echo "  make MACHINE=<m> models         download model weights into ./models"
	@echo "  make MACHINE=<m> bench          run every CPU benchmark"
	@echo "  make MACHINE=<m> bench-gpu      run the machine's accelerator benchmark"
	@echo "  make MACHINE=<m> sysinfo        print captured machine metadata"
	@echo "  make MACHINE=<m> config         render the merged compose config"
	@echo "  make machines                   list available machine overlays"

machines:
	@ls compose/machines/*.yaml | xargs -n1 basename | sed 's/\.yaml$$//'

# The framework images copy /opt/aeb out of this image, so it must exist first.
base:
	$(COMPOSE) --profile base build base

images: base
	$(COMPOSE) --profile cpu build

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

clean-results:
	rm -rf results/*
