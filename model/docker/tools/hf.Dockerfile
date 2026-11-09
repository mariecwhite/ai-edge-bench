# syntax=docker/dockerfile:1.7
#
# aeb/tools-hf - aeb/tools plus the PyTorch reference stack, only for the
# optional activation-quantization diagnostic (python3 -m aeb.prep.hf_reference).
# Large (~1 GB); not needed for benchmarking or reports.
ARG AEB_TOOLS_IMAGE=aeb/tools:latest
FROM ${AEB_TOOLS_IMAGE}
RUN /opt/aeb/venv/bin/pip install --no-cache-dir torch==2.9.0 \
      --index-url https://download.pytorch.org/whl/cpu \
 && /opt/aeb/venv/bin/pip install --no-cache-dir "transformers==5.18.0" accelerate jinja2
