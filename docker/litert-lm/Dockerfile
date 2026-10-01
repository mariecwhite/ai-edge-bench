# syntax=docker/dockerfile:1.7
#
# LiteRT-LM benchmark image (CPU backend).
#
# Builds //runtime/engine:litert_lm_main and, best-effort,
# :litert_lm_advanced_main from source with Bazel. The Bazel config is a build
# argument so the same file serves linux/amd64 and linux/arm64.
#
# YNNPACK: the runtime flag --enable_ynnpack only has an effect when the binary
# was compiled with --define=litert_enable_ynnpack=true. That define is already
# on for --config=linux_arm64; we pass it explicitly so the with/without
# comparison is available on x86-64 too. XNNPACK itself is always the CPU
# delegate and has no upstream off switch.

ARG AEB_BASE_IMAGE=aeb/base:latest
ARG BUILDER_BASE=ubuntu:24.04
ARG RUNTIME_BASE=ubuntu:24.04

FROM ${AEB_BASE_IMAGE} AS aebbase

# ---------------------------------------------------------------- build stage
FROM ${BUILDER_BASE} AS build

ARG DEBIAN_FRONTEND=noninteractive

# Package set mirrors the upstream CI image (.github/workflows/ml-ci-build.yml).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential gcc clang cmake make pkg-config \
      openjdk-17-jdk-headless \
      git git-lfs wget curl unzip tar ca-certificates \
      python3 python3-dev python3-pip python3-numpy \
      ccache zlib1g-dev libssl-dev libpng-dev libcurl4-openssl-dev libfftw3-dev \
      golang-go npm patchelf \
 && rm -rf /var/lib/apt/lists/*

# Bazelisk honours the repository's .bazelversion (7.6.1 at v0.17.1).
RUN npm install -g @bazel/bazelisk

# llguidance is built through rules_rust, so a host Rust toolchain is required.
ARG RUST_VERSION=1.84.0
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "${RUST_VERSION}" --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

ARG LITERT_LM_REPO=https://github.com/google-ai-edge/LiteRT-LM
ARG LITERT_LM_REF=v0.17.1

WORKDIR /src
RUN git init -q . \
 && git remote add origin "${LITERT_LM_REPO}" \
 && git fetch --depth 1 origin "${LITERT_LM_REF}" \
 && git checkout -q --detach FETCH_HEAD \
 && git lfs install --local \
 && git lfs pull \
 && git rev-parse HEAD > /src/.litert-lm-commit

# linux_x86_64 | linux_arm64. Do not use the x86-64 config on Arm: it adds
# -mavx2. Note the x86-64 config requires an AVX2-capable host (Zen 3+ is fine).
ARG BAZEL_CONFIG=linux_x86_64
ARG BAZEL_EXTRA_FLAGS="--define=litert_enable_ynnpack=true"
# Bazel + the LiteRT/TFLite/XLA/gRPC graph is memory hungry. Upstream's rule of
# thumb is one job per 8 GB of RAM; 0 means "let Bazel decide".
ARG BAZEL_JOBS=0

ENV CC=clang CXX=clang++
# Bazel fetches very large external archives (LLVM, XLA, TFLite). A single
# flaky download otherwise throws away the whole multi-hour build, so retry.
# Already-fetched repositories stay in Bazel's cache, so a retry resumes rather
# than starting over.
RUN set -eu; \
    for attempt in 1 2 3; do \
      echo "=== bazel build attempt ${attempt}/3 ==="; \
      if bazelisk build \
           --config="${BAZEL_CONFIG}" \
           --deleted_packages=swift \
           --java_runtime_version=remotejdk_17 \
           $( [ "${BAZEL_JOBS}" -gt 0 ] && echo "--jobs=${BAZEL_JOBS}" ) \
           ${BAZEL_EXTRA_FLAGS} \
           //runtime/engine:litert_lm_main; then \
        break; \
      fi; \
      [ "${attempt}" = 3 ] && { echo "bazel build failed after 3 attempts" >&2; exit 1; }; \
      echo "transient failure; retrying in 30s"; sleep 30; \
    done \
 && mkdir -p /opt/litert-lm/bin \
 && cp -L bazel-bin/runtime/engine/litert_lm_main /opt/litert-lm/bin/

# The advanced binary implements the full --benchmark_* flag behaviour. It is
# not covered by upstream CI on every platform, so a failure here must not fail
# the image; the harness falls back to litert_lm_main.
RUN bazelisk build \
      --config="${BAZEL_CONFIG}" \
      --deleted_packages=swift \
      --java_runtime_version=remotejdk_17 \
      $( [ "${BAZEL_JOBS}" -gt 0 ] && echo "--jobs=${BAZEL_JOBS}" ) \
      ${BAZEL_EXTRA_FLAGS} \
      //runtime/engine:litert_lm_advanced_main \
      && cp -L bazel-bin/runtime/engine/litert_lm_advanced_main /opt/litert-lm/bin/ \
      || echo "litert_lm_advanced_main unavailable for ${BAZEL_CONFIG}; using litert_lm_main"

# The C API shared library backs the in-process harness driver
# (harness/aeb/drivers/litert_lm_driver.py). It is built from the same pinned
# source and Bazel flags as the CLI binaries above, rather than taken from the
# litert-lm-api wheel, so both measurement paths run identical kernels. The
# Python bindings are pure ctypes and are copied from the same checkout.
RUN set -eu; \
    for attempt in 1 2 3; do \
      if bazelisk build \
           --config="${BAZEL_CONFIG}" \
           --deleted_packages=swift \
           --java_runtime_version=remotejdk_17 \
           $( [ "${BAZEL_JOBS}" -gt 0 ] && echo "--jobs=${BAZEL_JOBS}" ) \
           ${BAZEL_EXTRA_FLAGS} \
           //c:litert-lm; then \
        break; \
      fi; \
      [ "${attempt}" = 3 ] && { echo "bazel build //c:litert-lm failed" >&2; exit 1; }; \
      sleep 30; \
    done \
 && mkdir -p /opt/litert-lm/python/litert_lm \
 && cp python/litert_lm/*.py /opt/litert-lm/python/litert_lm/ \
 && rm -f /opt/litert-lm/python/litert_lm/*_test.py \
 && cp -L bazel-bin/c/liblitert-lm.so /opt/litert-lm/python/litert_lm/liblitert-lm.so

# Runtime accelerators and helper providers are shipped as prebuilt shared
# objects in the source tree and loaded dynamically by litert_lm_* binaries.
# Copy only the matching architecture; mixing both overwrites same-named .so
# files with the wrong ELF class.
RUN case "${BAZEL_CONFIG}" in \
     linux_arm64) cp -L prebuilt/linux_arm64/*.so /opt/litert-lm/bin/ ;; \
     linux_x86_64) cp -L prebuilt/linux_x86_64/*.so /opt/litert-lm/bin/ ;; \
     *) echo "unsupported BAZEL_CONFIG for prebuilt runtime libs: ${BAZEL_CONFIG}" >&2; exit 1 ;; \
   esac

# -------------------------------------------------------------- runtime stage
FROM ${RUNTIME_BASE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG EXTRA_RUNTIME_PACKAGES=""

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl jq file procps util-linux hostname tzdata \
      libgomp1 libcurl4 libssl3 zlib1g python3 \
      ${EXTRA_RUNTIME_PACKAGES} \
 && rm -rf /var/lib/apt/lists/*

COPY --from=aebbase /opt/aeb /opt/aeb
COPY --from=build /opt/litert-lm /opt/litert-lm

ARG LITERT_LM_REF
ARG BAZEL_CONFIG
ARG BAZEL_EXTRA_FLAGS
COPY --from=build /src/.litert-lm-commit /tmp/commit
RUN printf '{"framework":"litert-lm","ref":"%s","commit":"%s","bazel_config":"%s","bazel_extra_flags":"%s","binaries":"%s","built_at":"%s"}\n' \
      "${LITERT_LM_REF}" "$(cat /tmp/commit)" "${BAZEL_CONFIG}" "${BAZEL_EXTRA_FLAGS}" \
      "$(ls /opt/litert-lm/bin | tr '\n' ' ')" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > /opt/aeb/build-info.json \
 && rm -f /tmp/commit

ENV PATH="/opt/aeb/bin:/opt/litert-lm/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/litert-lm/bin" \
    PYTHONPATH="/opt/aeb/harness:/opt/litert-lm/python" \
    AEB_MODELS_DIR=/models \
    AEB_RESULTS_DIR=/results \
    TZ=UTC

VOLUME ["/models", "/results"]
WORKDIR /work

ENTRYPOINT ["/opt/aeb/bin/aeb-entrypoint"]
CMD ["aeb-bench-litert-lm"]
