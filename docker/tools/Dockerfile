# syntax=docker/dockerfile:1.7
#
# aeb/tools - offline helpers that never run during a timed benchmark:
# model preparation (GGUF re-encoding of the LiteRT-LM QAT checkpoint), dataset
# staging, and report/chart generation. Kept out of the framework images so
# those stay minimal and stdlib-only.

ARG AEB_BASE_IMAGE=aeb/base:latest
FROM ${AEB_BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3-venv git \
 && rm -rf /var/lib/apt/lists/*

# gguf-py is pinned to the same llama.cpp ref as the benchmark image so the
# writer emits exactly the tensor types and metadata that build understands.
ARG LLAMA_CPP_REF=b11259
RUN python3 -m venv /opt/aeb/venv \
 && /opt/aeb/venv/bin/pip install --no-cache-dir \
      "numpy==2.3.3" "matplotlib==3.10.6" "pyarrow==21.0.0" "tflite==2.18.0" \
      "litert-lm==0.17.1" \
      "gguf @ git+https://github.com/ggml-org/llama.cpp@${LLAMA_CPP_REF}#subdirectory=gguf-py"

ENV PATH="/opt/aeb/venv/bin:${PATH}"
CMD ["python3", "-m", "aeb.report", "--help"]
