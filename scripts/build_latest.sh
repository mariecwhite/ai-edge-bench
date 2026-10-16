#!/usr/bin/env bash
# Pull and build all benchmark container images using the latest upstream versions
# of llama.cpp, LiteRT-LM, OpenXLA (JAX/Flax/Qwix/Tunix), OpenVINO GenAI, and ONNX Runtime GenAI.
#
# Usage:
#   ./scripts/build_latest.sh [machine] [user@hostname]
#
# Examples:
#   ./scripts/build_latest.sh
#   ./scripts/build_latest.sh sapphire-rapids user@amx-host

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MACHINE="${1:-${AEB_MACHINE:-sapphire-rapids}}"
REMOTE_HOST="${2:-${AEB_REMOTE_HOST:-<user@hostname>}}"
COMPOSE_OVERLAY="compose/machines/${MACHINE}.yaml"

if [[ ! -f "${COMPOSE_OVERLAY}" ]]; then
  echo "ERROR: Machine overlay not found: ${COMPOSE_OVERLAY}" >&2
  echo "Available machines:" >&2
  ls compose/machines/*.yaml | xargs -n1 basename | sed 's/\.yaml$//; s/^/  - /' >&2
  exit 1
fi

echo "=================================================================="
echo "[1/4] Resolving latest upstream versions for all frameworks ..."
echo "=================================================================="

export AEB_LLAMA_CPP_REF="${AEB_LLAMA_CPP_REF:-$(git ls-remote --tags --sort='-v:refname' https://github.com/ggml-org/llama.cpp.git 'b*' | head -n1 | sed 's|.*refs/tags/||')}"
export AEB_LITERTLM_REF="${AEB_LITERTLM_REF:-$(git ls-remote --tags --sort='-v:refname' https://github.com/google-ai-edge/LiteRT-LM.git 'v*' | head -n1 | sed 's|.*refs/tags/||; s/\^{}//')}"

read -r JAX_VER FLAX_VER QWIX_VER TUNIX_VER SAFETENSORS_VER JAXTYPING_VER TOKENIZERS_VER OV_VER OV_GENAI_VER ORT_VER ORT_GENAI_VER < <(python3 - << 'PY'
import json
import urllib.request

def latest(pkg: str) -> str:
  with urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=30) as r:
    return json.load(r)["info"]["version"]

print(
    latest("jax"),
    latest("flax"),
    latest("qwix"),
    latest("google-tunix"),
    latest("safetensors"),
    latest("jaxtyping"),
    latest("tokenizers"),
    latest("openvino"),
    latest("openvino-genai"),
    latest("onnxruntime"),
    latest("onnxruntime-genai"),
)
PY
)

echo "  llama.cpp          : ${AEB_LLAMA_CPP_REF}"
echo "  litert-lm          : ${AEB_LITERTLM_REF}"
echo "  jax / jaxlib       : ${JAX_VER}"
echo "  flax               : ${FLAX_VER}"
echo "  qwix               : ${QWIX_VER}"
echo "  google-tunix       : ${TUNIX_VER}"
echo "  safetensors        : ${SAFETENSORS_VER}"
echo "  jaxtyping          : ${JAXTYPING_VER}"
echo "  tokenizers         : ${TOKENIZERS_VER}"
echo "  openvino           : ${OV_VER}"
echo "  openvino-genai     : ${OV_GENAI_VER}"
echo "  onnxruntime        : ${ORT_VER}"
echo "  onnxruntime-genai  : ${ORT_GENAI_VER}"

COMPOSE=(docker compose -f compose.yaml -f "${COMPOSE_OVERLAY}")

echo "=================================================================="
echo "[2/4] Building base and tools images (MACHINE=${MACHINE}) ..."
echo "=================================================================="
"${COMPOSE[@]}" --profile base build --pull base
"${COMPOSE[@]}" --profile tools build --no-cache tools

echo "=================================================================="
echo "[3/4] Building llama.cpp (${AEB_LLAMA_CPP_REF}) and LiteRT-LM (${AEB_LITERTLM_REF}) ..."
echo "=================================================================="
"${COMPOSE[@]}" --profile cpu build --no-cache llama-cpp litert-lm-xnnpack

echo "=================================================================="
echo "[4/4] Building OpenXLA, OpenVINO, and ONNX Runtime images ..."
echo "=================================================================="
"${COMPOSE[@]}" --profile x86-sota build --no-cache \
  --build-arg JAX_VERSION="${JAX_VER}" \
  --build-arg FLAX_VERSION="${FLAX_VER}" \
  --build-arg QWIX_VERSION="${QWIX_VER}" \
  --build-arg TUNIX_VERSION="${TUNIX_VER}" \
  --build-arg SAFETENSORS_VERSION="${SAFETENSORS_VER}" \
  --build-arg JAXTYPING_VERSION="${JAXTYPING_VER}" \
  --build-arg TOKENIZERS_VERSION="${TOKENIZERS_VER}" \
  xla

"${COMPOSE[@]}" --profile x86-sota build --no-cache \
  --build-arg OPENVINO_VERSION="${OV_VER}" \
  --build-arg OPENVINO_GENAI_VERSION="${OV_GENAI_VER}" \
  openvino

"${COMPOSE[@]}" --profile x86-sota build --no-cache \
  --build-arg ONNXRUNTIME_VERSION="${ORT_VER}" \
  --build-arg ONNXRUNTIME_GENAI_VERSION="${ORT_GENAI_VER}" \
  onnxruntime

cat <<EOF

==================================================================
✅ All framework images built with latest upstream versions!
==================================================================
Built versions:
  - llama.cpp         : ${AEB_LLAMA_CPP_REF}
  - litert-lm         : ${AEB_LITERTLM_REF}
  - xla               : jax==${JAX_VER}, google-tunix==${TUNIX_VER}, qwix==${QWIX_VER}, flax==${FLAX_VER}
  - openvino          : openvino==${OV_VER}, openvino-genai==${OV_GENAI_VER}
  - onnxruntime       : onnxruntime==${ORT_VER}, onnxruntime-genai==${ORT_GENAI_VER}

Next steps to sync, run, and fetch results:

1. Sync repo, models, datasets, and updated container bundles to the target machine:
   ./scripts/setup_amx_machine.sh ${REMOTE_HOST} --bundles

2. Archive any previous results/caches on the target machine and launch the full benchmark suite:
   ssh ${REMOTE_HOST} '
     set -e
     if [[ -d /data/aeb/results/${MACHINE} ]]; then
       mv /data/aeb/results/${MACHINE} "/data/aeb/results/${MACHINE}.bak-\$(date -u +%Y%m%dT%H%M%SZ)"
     fi
     rm -rf /data/aeb/results/_cache
     nohup /data/aeb/repo/scripts/run_amx_benchmarks.sh > /data/aeb/bench.log 2>&1 &
     echo "Started PID \$! — monitor with: tail -f /data/aeb/bench.log"
   '

   # Monitor live progress from your workstation:
   ssh ${REMOTE_HOST} 'tail -f /data/aeb/bench.log'

3. Once the run completes, fetch the results and generated report back to your workstation:
   ./scripts/fetch_results.sh ${REMOTE_HOST}
==================================================================
EOF
