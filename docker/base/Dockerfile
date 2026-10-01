# syntax=docker/dockerfile:1.7
#
# aeb/base - the composable layer shared by every benchmark container.
#
# This image carries no inference framework. It provides the harness scripts,
# the /models and /results contract, and the metadata collection used to make a
# run reproducible. Framework images copy /opt/aeb out of this image, which
# lets them keep a framework-specific runtime base (e.g. nvidia/cuda).

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
# Framework-neutral orchestration (stdlib-only Python): runners, resource
# monitor, accuracy evals and the litert-lm driver. See harness/README.md.
COPY harness/ /opt/aeb/harness/
RUN chmod -R 0755 /opt/aeb/bin

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
