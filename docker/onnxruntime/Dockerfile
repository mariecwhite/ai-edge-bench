# syntax=docker/dockerfile:1.7
#
# ONNX Runtime GenAI benchmark image (CPU execution provider).
# Build from the repository root so docker/base and docker/harness are in
# context. The CPU wheels are published for linux/amd64 and linux/arm64.

ARG AEB_BASE_IMAGE=aeb/base:latest
FROM ${AEB_BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG ONNXRUNTIME_VERSION=1.30.0
ARG ONNXRUNTIME_GENAI_VERSION=0.17.1

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Install binary wheels only. This avoids accidental source builds and makes
# unsupported Python/platform combinations fail at image build time.
RUN python3 -m pip install --break-system-packages --no-cache-dir --only-binary=:all: \
      "onnxruntime==${ONNXRUNTIME_VERSION}" \
      "onnxruntime-genai==${ONNXRUNTIME_GENAI_VERSION}"

RUN printf '{"framework":"onnxruntime","ref":"onnxruntime-genai==%s","runtime":"onnxruntime==%s","genai":"onnxruntime-genai==%s","execution_provider":"CPUExecutionProvider","built_at":"%s"}\n' \
      "${ONNXRUNTIME_GENAI_VERSION}" "${ONNXRUNTIME_VERSION}" "${ONNXRUNTIME_GENAI_VERSION}" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > /opt/aeb/build-info.json

ENV PYTHONPATH="/opt/aeb/harness" \
    AEB_MODELS_DIR=/models \
    AEB_RESULTS_DIR=/results \
    TZ=UTC

VOLUME ["/models", "/results"]
WORKDIR /work
CMD ["aeb-sysinfo"]
