#!/usr/bin/env bash
# Build the native (host) framework stack for the Apple GPU suite.
#
#   scripts/native-apple-gpu.sh            # idempotent; reuses finished steps
#
# Containers on macOS run in a Linux VM without Metal, so GPU results must run
# on the host. Everything lands in build/native/ (git-ignored):
#
#   venv/                 Python 3.12 (uv) with pinned MLX, LiteRT-LM, report deps
#   llama.cpp/bin/        llama.cpp (Metal, embedded shaders) + aeb-llama-driver
#   basert/engine/        BaseRT prebuilt engine (checksum-verified)
#   <framework>/build-info.json   recorded in every run (AEB_BUILD_INFO)
#
# Requirements: Apple silicon, macOS 14+, Xcode Command Line Tools, Homebrew
# `cmake` and `uv`. The full Xcode Metal compiler is not needed: llama.cpp
# embeds its shader source and Metal compiles it at load.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/build/native"
LLAMA_CPP_REF="${AEB_LLAMA_CPP_REF:-b11259}"
BASERT_VERSION="0.4.0"
BASERT_SHA256="c9b49ea1906fa871c646df3f62fb8237b53a817f786356323ff04bda1e6e79f2"
JOBS="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 8)"

log() { printf '[native] %s\n' "$*" >&2; }
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

[[ "$(uname -s)/$(uname -m)" == "Darwin/arm64" ]] || { log "Apple silicon macOS only"; exit 1; }
for c in cmake uv clang++ git curl shasum; do
  command -v "$c" >/dev/null || { log "missing $c (brew install cmake uv; xcode-select --install)"; exit 1; }
done
mkdir -p "${OUT}"

# ---------------------------------------------------------------- Python
if [[ ! -x "${OUT}/venv/bin/python" ]]; then
  log "creating Python 3.12 venv"
  uv venv --python 3.12 "${OUT}/venv"
fi
VIRTUAL_ENV="${OUT}/venv" uv pip install -q -r "${ROOT}/scripts/native-apple-gpu-requirements.txt"
PY="${OUT}/venv/bin/python"

pyver() { "${PY}" -c "from importlib import metadata; print(metadata.version('$1'))"; }

mkdir -p "${OUT}/mlx" "${OUT}/litert-lm"
printf '{"framework":"mlx","ref":"mlx-lm==%s","commit":null,"variant":"metal","mlx":"%s","mlx_metal":"%s","built_at":"%s"}\n' \
  "$(pyver mlx-lm)" "$(pyver mlx)" "$(pyver mlx-metal)" "$(now)" > "${OUT}/mlx/build-info.json"
printf '{"framework":"litert-lm","ref":"litert-lm-api==%s","commit":null,"variant":"gpu (ML Drift, Metal), PyPI macosx_12_0_arm64 wheel","built_at":"%s"}\n' \
  "$(pyver litert-lm-api)" "$(now)" > "${OUT}/litert-lm/build-info.json"

# ------------------------------------------------------------- llama.cpp
SRC="${OUT}/llama.cpp-src"
if [[ ! -d "${SRC}" ]]; then
  git clone --depth 1 --branch "${LLAMA_CPP_REF}" https://github.com/ggml-org/llama.cpp.git "${SRC}"
fi
CMAKE_FLAGS="-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF"
if [[ ! -f "${SRC}/build/bin/llama-bench" ]]; then
  log "building llama.cpp ${LLAMA_CPP_REF} (Metal)"
  # shellcheck disable=SC2086
  cmake -S "${SRC}" -B "${SRC}/build" -DCMAKE_BUILD_TYPE=Release ${CMAKE_FLAGS}
  cmake --build "${SRC}/build" -j "${JOBS}" --target llama llama-bench
fi
mkdir -p "${OUT}/llama.cpp/bin"
cp -a "${SRC}"/build/bin/*.dylib "${SRC}/build/bin/llama-bench" "${OUT}/llama.cpp/bin/"
# Same driver source as the container image, linked against the Metal build.
clang++ -O2 -std=c++17 "${ROOT}/docker/llama-cpp/aeb-llama-driver.cpp" \
  -I"${SRC}/include" -I"${SRC}/ggml/include" -I"${SRC}/vendor" \
  -L"${SRC}/build/bin" -lllama -lggml -lggml-base -Wl,-rpath,@loader_path \
  -o "${OUT}/llama.cpp/bin/aeb-llama-driver"
printf '{"framework":"llama.cpp","ref":"%s","commit":"%s","variant":"metal","cmake_extra_flags":"%s","built_at":"%s"}\n' \
  "${LLAMA_CPP_REF}" "$(git -C "${SRC}" rev-parse HEAD)" "${CMAKE_FLAGS}" "$(now)" > "${OUT}/llama.cpp/build-info.json"

# ---------------------------------------------------------------- BaseRT
B="${OUT}/basert"
mkdir -p "${B}"
if [[ ! -f "${B}/engine/libbaseRT.dylib" ]]; then
  log "fetching BaseRT ${BASERT_VERSION} engine"
  curl -sSfL -o "${B}/engine.tar.gz" \
    "https://github.com/basecompute/baseRT/releases/download/v${BASERT_VERSION}/basert-engine-macos-arm64-${BASERT_VERSION}.tar.gz"
  echo "${BASERT_SHA256}  ${B}/engine.tar.gz" | shasum -a 256 -c -
  mkdir -p "${B}/engine" && tar -xzf "${B}/engine.tar.gz" -C "${B}/engine"
fi
if [[ ! -d "${B}/baseRT-${BASERT_VERSION}" ]]; then
  # Apache-2.0 Python binding and headers matching the engine release.
  curl -sSfL "https://github.com/basecompute/baseRT/archive/refs/tags/v${BASERT_VERSION}.tar.gz" | tar -xz -C "${B}"
fi
VIRTUAL_ENV="${OUT}/venv" uv pip install -q "${B}/baseRT-${BASERT_VERSION}/bindings/python"
BASERT_LIB_PATH="${B}/engine/libbaseRT.dylib" "${PY}" -c "import baseRT" >/dev/null
printf '{"framework":"basert","ref":"v%s","commit":"%s","variant":"metal (prebuilt proprietary engine)","engine_sha256":"%s","built_at":"%s"}\n' \
  "${BASERT_VERSION}" "24badc8dd173e4111b64996a718d0e056405ea62" "${BASERT_SHA256}" "$(now)" > "${B}/build-info.json"

log "ready: ${OUT}"
