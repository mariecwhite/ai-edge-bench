# syntax=docker/dockerfile:1.7
#
# aeb/base - the composable base layer shared across kernel, layer, and model
# benchmark containers.
#
# This image carries no inference framework. It provides the shared environment
# scripts (/opt/aeb/bin), the /models and /results contract, and the system
# metadata collection (aeb-sysinfo) used to make runs reproducible. Tier-specific
# images copy /opt/aeb out of this image or layer their own harness on top.

ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      jq \
      file \
      procps \
      util-linux \
      coreutils \
      hostname \
      tzdata \
      python3 \
 && rm -rf /var/lib/apt/lists/*

COPY bin/ /opt/aeb/bin/
RUN chmod -R 0755 /opt/aeb/bin /opt/aeb/harness 2>/dev/null || chmod -R 0755 /opt/aeb/bin

ENV PATH="/opt/aeb/bin:${PATH}" \
    PYTHONPATH="/opt/aeb/harness" \
    AEB_MODELS_DIR=/models \
    AEB_RESULTS_DIR=/results \
    TZ=UTC

# Bind-mount or volume-mount both of these from the host. Models are cached and
# shared between framework containers; results accumulate one directory per run.
VOLUME ["/models", "/results"]
WORKDIR /work

ENTRYPOINT ["/opt/aeb/bin/aeb-entrypoint"]
CMD ["aeb-sysinfo"]
