# syntax=docker/dockerfile:1.7
#
# llama.cpp benchmark image.
#
# The same Dockerfile produces the CPU, CUDA and Vulkan variants: the base
# images, extra packages and CMake flags are all build arguments, so a machine
# overlay in compose/machines/ selects the backend without forking this file.

ARG AEB_BASE_IMAGE=aeb/base:latest
ARG BUILDER_BASE=ubuntu:24.04
ARG RUNTIME_BASE=ubuntu:24.04

FROM ${AEB_BASE_IMAGE} AS aebbase

# ---------------------------------------------------------------- build stage
FROM ${BUILDER_BASE} AS build

ARG DEBIAN_FRONTEND=noninteractive
# Extra toolchain packages for accelerated variants, e.g. Vulkan needs
# "libvulkan-dev glslang-tools". Empty for the plain CPU build.
ARG EXTRA_BUILD_PACKAGES=""

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential gcc-14 g++-14 cmake git ca-certificates pkg-config \
      libssl-dev libcurl4-openssl-dev ccache \
      ${EXTRA_BUILD_PACKAGES} \
 && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14

# Pin the upstream revision. Bump deliberately: changing it changes results.
ARG LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp
ARG LLAMA_CPP_REF=b11259

WORKDIR /src
RUN git init -q . \
 && git remote add origin "${LLAMA_CPP_REPO}" \
 && git fetch --depth 1 origin "${LLAMA_CPP_REF}" \
 && git checkout -q --detach FETCH_HEAD \
 && git rev-parse HEAD > /src/.llama-cpp-commit

# GGML_NATIVE=OFF keeps the binary portable across the benchmark fleet;
# GGML_CPU_ALL_VARIANTS builds every ISA variant and picks the best at runtime,
# so one image is valid on Zen 3, Zen 5 and Arm without re-tuning per host.
#
# Only llama-bench is needed, so the server, examples and the unified `llama`
# app are all off. LLAMA_BUILD_APP must be off explicitly: the app target links
# llama-server-impl and llama-cli-impl, so it fails to link when the server is
# disabled. LLAMA_BUILD_TOOLS stays on because llama-bench is a tool.
ARG CMAKE_EXTRA_FLAGS=""
ARG CMAKE_BUILD_JOBS=0
RUN cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=OFF \
      -DGGML_BACKEND_DL=ON \
      -DGGML_CPU_ALL_VARIANTS=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DLLAMA_CURL=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=OFF \
      -DLLAMA_BUILD_APP=OFF \
      ${CMAKE_EXTRA_FLAGS} \
 && cmake --build build --config Release \
      -j "$( [ "${CMAKE_BUILD_JOBS}" -gt 0 ] && echo "${CMAKE_BUILD_JOBS}" || nproc )"

# Collect the binary and every dynamically loaded ggml backend into one dir.
# cp -a preserves the SONAME symlink chains (libggml.so -> .so.0 -> .so.0.x.y);
# dereferencing them drops the .so.0 name the loader actually resolves.
RUN mkdir -p /opt/llama.cpp/bin \
 && cp -a build/bin/llama-bench /opt/llama.cpp/bin/ \
 && cp -a build/bin/*.so* /opt/llama.cpp/bin/

# In-process driver for the AI Edge Bench harness (JSON lines over stdio). It
# links the same libllama/libggml built above, so it runs identical kernels to
# llama-bench.
COPY llama-cpp/aeb-llama-driver.cpp /src/aeb-llama-driver.cpp
RUN ${CXX} -O2 -std=c++17 /src/aeb-llama-driver.cpp \
      -Iinclude -Iggml/include -Ivendor \
      -Lbuild/bin -lllama -lggml -lggml-base -Wl,-rpath,'$ORIGIN' \
      -o /opt/llama.cpp/bin/aeb-llama-driver

# -------------------------------------------------------------- runtime stage
FROM ${RUNTIME_BASE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
# e.g. "libvulkan1 mesa-vulkan-drivers vulkan-tools" for the Vulkan variant.
ARG EXTRA_RUNTIME_PACKAGES=""

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl jq file procps util-linux hostname tzdata \
      libgomp1 libcurl4 libssl3 python3 \
      ${EXTRA_RUNTIME_PACKAGES} \
 && rm -rf /var/lib/apt/lists/*

COPY --from=aebbase /opt/aeb /opt/aeb
COPY --from=build /opt/llama.cpp /opt/llama.cpp

ARG LLAMA_CPP_REF
ARG CMAKE_EXTRA_FLAGS
ARG AEB_VARIANT=cpu
COPY --from=build /src/.llama-cpp-commit /tmp/commit
RUN printf '{"framework":"llama.cpp","ref":"%s","commit":"%s","variant":"%s","cmake_extra_flags":"%s","built_at":"%s"}\n' \
      "${LLAMA_CPP_REF}" "$(cat /tmp/commit)" "${AEB_VARIANT}" "${CMAKE_EXTRA_FLAGS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > /opt/aeb/build-info.json \
 && rm -f /tmp/commit

ENV PATH="/opt/aeb/bin:/opt/llama.cpp/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/llama.cpp/bin" \
    PYTHONPATH="/opt/aeb/harness" \
    AEB_MODELS_DIR=/models \
    AEB_RESULTS_DIR=/results \
    AEB_VARIANT=${AEB_VARIANT} \
    TZ=UTC

VOLUME ["/models", "/results"]
WORKDIR /work

ENTRYPOINT ["/opt/aeb/bin/aeb-entrypoint"]
CMD ["aeb-bench-llama-cpp"]
