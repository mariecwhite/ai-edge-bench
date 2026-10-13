# syntax=docker/dockerfile:1.7
#
# OpenVINO GenAI benchmark image (CPU plugin with oneDNN / AMX / AVX-512).

ARG AEB_BASE_IMAGE=aeb/base:latest
FROM ${AEB_BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG OPENVINO_VERSION=2026.4.1.0
ARG OPENVINO_GENAI_VERSION=2026.4.1.0

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 python3-pip \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --no-cache-dir --only-binary=:all: \
      "openvino==${OPENVINO_VERSION}" \
      "openvino-tokenizers==${OPENVINO_VERSION}" \
      "openvino-genai==${OPENVINO_GENAI_VERSION}" \
      "numpy==2.3.3"

RUN printf '{"framework":"openvino","ref":"openvino-genai==%s","commit":"openvino==%s","runtime":"openvino==%s","genai":"openvino-genai==%s","built_at":"%s"}\n' \
      "${OPENVINO_GENAI_VERSION}" "${OPENVINO_VERSION}" "${OPENVINO_VERSION}" "${OPENVINO_GENAI_VERSION}" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > /opt/aeb/build-info.json

ENV PYTHONPATH="/opt/aeb/harness" \
    AEB_MODELS_DIR=/models \
    AEB_RESULTS_DIR=/results \
    TZ=UTC

VOLUME ["/models", "/results"]
WORKDIR /work
CMD ["aeb-sysinfo"]
