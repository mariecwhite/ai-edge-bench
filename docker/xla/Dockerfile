# syntax=docker/dockerfile:1.7
#
# OpenXLA (JAX/XLA:CPU) benchmark image.
# Supports both default XLA:CPU (YNNPACK __ynn_fusion) and XLA:CPU with oneDNN
# (XLA_FLAGS="--xla_cpu_experimental_onednn_custom_call=true --xla_cpu_experimental_ynn_fusion_type=invalid").

ARG AEB_BASE_IMAGE=aeb/base:latest
FROM ${AEB_BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG JAX_VERSION=0.11.2
ARG FLAX_VERSION=0.12.10
ARG QWIX_VERSION=0.1.8
ARG TUNIX_VERSION=0.1.7
ARG SAFETENSORS_VERSION=0.7.0
ARG JAXTYPING_VERSION=0.3.10
ARG TOKENIZERS_VERSION=0.22.1

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 python3-pip \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --no-cache-dir --only-binary=:all: \
      "jax==${JAX_VERSION}" \
      "jaxlib==${JAX_VERSION}" \
      "flax==${FLAX_VERSION}" \
      "qwix==${QWIX_VERSION}" \
      "safetensors==${SAFETENSORS_VERSION}" \
      "jaxtyping==${JAXTYPING_VERSION}" \
      "etils[epath]" \
      "huggingface_hub" \
      "numpy==2.3.3" \
      "tokenizers==${TOKENIZERS_VERSION}" \
 && python3 -m pip install --break-system-packages --no-cache-dir --no-deps \
      "google-tunix==${TUNIX_VERSION}" \
 && python3 -c "import pathlib, importlib.util; p = pathlib.Path(importlib.util.find_spec('tunix').origin); p.write_text('__version__ = \"${TUNIX_VERSION}\"\\n')"

RUN printf '{"framework":"xla","ref":"jax==%s","commit":"google-tunix==%s","runtime":"jaxlib==%s","tunix":"google-tunix==%s","qwix":"qwix==%s","flax":"flax==%s","tokenizers":"tokenizers==%s","built_at":"%s"}\n' \
      "${JAX_VERSION}" "${TUNIX_VERSION}" "${JAX_VERSION}" "${TUNIX_VERSION}" "${QWIX_VERSION}" "${FLAX_VERSION}" "${TOKENIZERS_VERSION}" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > /opt/aeb/build-info.json

ENV PYTHONPATH="/opt/aeb/harness" \
    AEB_MODELS_DIR=/models \
    AEB_RESULTS_DIR=/results \
    TZ=UTC

VOLUME ["/models", "/results"]
WORKDIR /work
CMD ["aeb-sysinfo"]
