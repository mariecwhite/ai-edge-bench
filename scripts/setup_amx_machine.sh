#!/usr/bin/env bash
# Bootstrap and sync repo, container bundles, datasets, and models to any x86 machine.
#
# Usage:
#   ./scripts/setup_amx_machine.sh <user@hostname> [--bundles]
#
# Features:
#   - Auto-bootstraps /data/aeb on the remote host.
#   - Automatically detects if any of the 7 container rootfs bundles are missing on the remote
#     machine and pushes + unpacks them with SHA-256 verification.
#   - Checks remote model files using pure POSIX `stat` + `sha256sum` (no remote Python needed for check):
#     if a file on the remote machine is already complete and matches the local SHA-256, it is skipped.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

HOST="${1:-${AEB_REMOTE_HOST:-}}"
if [[ -z "${HOST}" ]]; then
  echo "Usage: ./scripts/setup_amx_machine.sh <user@hostname> [--bundles]" >&2
  exit 1
fi
REMOTE_DIR="${AEB_REMOTE_DIR:-/data/aeb}"
BWLIMIT="${AEB_RSYNC_BWLIMIT:-20000}" # KB/s (~20 MB/s default bandwidth limit)

# Reuse a ControlMaster socket (~/.ssh/cm-aeb) if active; otherwise authenticate interactively.
SSH_OPTS="${AEB_SSH_OPTS:--o StrictHostKeyChecking=no}"
SOCK="${HOME}/.ssh/cm-aeb"
if [[ -S "${SOCK}" ]]; then
  SSH_OPTS="${SSH_OPTS} -o ControlPath=${SOCK}"
fi

echo "==> [0/4] Bootstrapping ${REMOTE_DIR} on ${HOST} ..."
RUNTIME_MODE="$(ssh ${SSH_OPTS} "${HOST}" "
  set -e
  if [[ ! -e ${REMOTE_DIR} ]]; then
    mkdir -p ${REMOTE_DIR}
  fi
  mkdir -p ${REMOTE_DIR}/{repo,datasets,models,bundles,results,reports,bin}
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    if docker image inspect aeb/xla:cpu >/dev/null 2>&1; then
      echo 'docker-ready'
    else
      echo 'docker-fresh'
    fi
  else
    all_ok=1
    for b in base tools llama-cpp litert-lm onnxruntime openvino xla; do
      if [[ ! -x ${REMOTE_DIR}/bundles/\${b}/rootfs/usr/bin/env ]]; then
        all_ok=0
        break
      fi
    done
    if [[ ! -f ${REMOTE_DIR}/bundles/xla/rootfs/usr/local/lib/python3.12/dist-packages/tunix/__init__.py ]]; then
      all_ok=0
    fi
    if [[ \"\${all_ok}\" -eq 1 ]]; then
      echo 'bundles-ready'
    else
      echo 'bundles-fresh'
    fi
  fi
")"
echo "    Remote runtime status on ${HOST}: ${RUNTIME_MODE}"

echo "==> [1/4] Syncing repository code and datasets to ${HOST}:${REMOTE_DIR}/ ..."
rsync -avP --checksum --exclude='.git' --exclude='.cache' --exclude='models' --exclude='datasets' --exclude='results' --exclude='reports' \
  -e "ssh ${SSH_OPTS}" ./ "${HOST}:${REMOTE_DIR}/repo/"

rsync -avP --checksum -e "ssh ${SSH_OPTS}" ./datasets/ "${HOST}:${REMOTE_DIR}/datasets/"
rsync -avP --checksum -e "ssh ${SSH_OPTS}" ./reports/notes/ "${HOST}:${REMOTE_DIR}/reports/notes/"

if [[ "${RUNTIME_MODE}" == "bundles-fresh" || "${2:-}" == "--bundles" ]]; then
  echo "==> [2/4] Checking & syncing 7 container bundles to ${HOST}:${REMOTE_DIR}/bundles/ ..."
  mkdir -p "${ROOT}/.cache/bundles"
  for pair in \
    "base:aeb/base:latest" \
    "tools:aeb/tools:latest" \
    "llama-cpp:aeb/llama-cpp:cpu" \
    "litert-lm:aeb/litert-lm:cpu" \
    "onnxruntime:aeb/onnxruntime:cpu" \
    "openvino:aeb/openvino:cpu" \
    "xla:aeb/xla:cpu"; do
    name="${pair%%:*}"
    img="${pair#*:}"
    tarball="${ROOT}/.cache/bundles/${name}.tar.gz"
    sha_file="${tarball}.sha256"
    id_file="${tarball}.image_id"
    cur_img_id="$(docker image inspect -f '{{.Id}}' "${img}" 2>/dev/null || true)"
    cached_img_id="$(cat "${id_file}" 2>/dev/null || true)"
    if [[ ! -s "${tarball}" || ( -n "${cur_img_id}" && -n "${cached_img_id}" && "${cur_img_id}" != "${cached_img_id}" ) ]]; then
      echo "  -> exporting ${img} to ${tarball} ..."
      cid="$(docker create "${img}")"
      docker export "${cid}" | pigz -1 > "${tarball}.tmp"
      docker rm "${cid}" >/dev/null
      mv "${tarball}.tmp" "${tarball}"
      sha256sum "${tarball}" | awk '{print $1}' > "${sha_file}"
      [[ -n "${cur_img_id}" ]] && echo "${cur_img_id}" > "${id_file}"
    elif [[ ! -s "${sha_file}" ]]; then
      sha256sum "${tarball}" | awk '{print $1}' > "${sha_file}"
      [[ -n "${cur_img_id}" ]] && echo "${cur_img_id}" > "${id_file}"
    elif [[ -n "${cur_img_id}" && ! -s "${id_file}" ]]; then
      echo "${cur_img_id}" > "${id_file}"
    fi
    local_sha="$(cat "${sha_file}")"
    local_size="$(stat -c '%s' "${tarball}")"

    remote_status="$(ssh ${SSH_OPTS} "${HOST}" "
      if [[ '${name}' == 'xla' && ! -f ${REMOTE_DIR}/bundles/xla/rootfs/usr/local/lib/python3.12/dist-packages/tunix/__init__.py ]]; then
        rm -f ${REMOTE_DIR}/bundles/xla/.verified_sha256
      fi
      if [[ -x ${REMOTE_DIR}/bundles/${name}/rootfs/usr/bin/env && -f ${REMOTE_DIR}/bundles/${name}/.verified_sha256 ]]; then
        if [[ \"\$(cat ${REMOTE_DIR}/bundles/${name}/.verified_sha256)\" == '${local_sha}' ]]; then
          echo 'UNPACKED_OK'
          exit 0
        fi
      elif [[ -x ${REMOTE_DIR}/bundles/${name}/rootfs/usr/bin/env && ! -f ${REMOTE_DIR}/bundles/${name}/.verified_sha256 && '${name}' != 'xla' ]]; then
        echo '${local_sha}' > ${REMOTE_DIR}/bundles/${name}/.verified_sha256
        echo 'UNPACKED_OK'
        exit 0
      fi
      if [[ -f ${REMOTE_DIR}/bundles/${name}.tar.gz ]]; then
        rsize=\$(stat -c '%s' ${REMOTE_DIR}/bundles/${name}.tar.gz 2>/dev/null || echo 0)
        if [[ \"\${rsize}\" == '${local_size}' ]]; then
          rsha=\$(sha256sum ${REMOTE_DIR}/bundles/${name}.tar.gz | awk '{print \$1}')
          if [[ \"\${rsha}\" == '${local_sha}' ]]; then
            echo 'TAR_OK'
            exit 0
          fi
        fi
      fi
      echo 'NEED'
    ")"

    if [[ "${remote_status}" == "UNPACKED_OK" ]]; then
      echo "  [SKIP] bundle ${name} already unpacked & verified (${local_sha:0:12}...)"
      continue
    elif [[ "${remote_status}" == "TAR_OK" ]]; then
      echo "  [SKIP] ${name}.tar.gz already complete & hash-verified (${local_sha:0:12}...), unpacking ..."
    else
      echo "  [PUSH] syncing ${name}.tar.gz (${local_sha:0:12}...) ..."
      rsync -avP --partial --inplace --bwlimit="${BWLIMIT}" -e "ssh ${SSH_OPTS}" \
        "${tarball}" "${HOST}:${REMOTE_DIR}/bundles/${name}.tar.gz"
    fi

    ssh ${SSH_OPTS} "${HOST}" "
      set -e
      rsha=\$(sha256sum ${REMOTE_DIR}/bundles/${name}.tar.gz | awk '{print \$1}')
      if [[ \"\${rsha}\" != '${local_sha}' ]]; then
        echo 'ERROR: SHA-256 mismatch for ${name}.tar.gz on remote!' >&2
        exit 1
      fi
      rm -rf ${REMOTE_DIR}/bundles/${name}/rootfs
      mkdir -p ${REMOTE_DIR}/bundles/${name}/rootfs
      tar -xzf ${REMOTE_DIR}/bundles/${name}.tar.gz -C ${REMOTE_DIR}/bundles/${name}/rootfs
      echo '${local_sha}' > ${REMOTE_DIR}/bundles/${name}/.verified_sha256
      rm -f ${REMOTE_DIR}/bundles/${name}.tar.gz
    "
  done
elif [[ "${RUNTIME_MODE}" == "docker-fresh" ]]; then
  echo "==> [2/4] Fresh Docker daemon machine detected: saving & loading 7 Docker images ..."
  mkdir -p "${ROOT}/.cache/bundles"
  img_tar="${ROOT}/.cache/bundles/aeb-all-images.tar.gz"
  if [[ ! -s "${img_tar}" ]]; then
    docker save aeb/base:latest aeb/tools:latest aeb/llama-cpp:cpu aeb/litert-lm:cpu \
      aeb/onnxruntime:cpu aeb/openvino:cpu aeb/xla:cpu | pigz -1 > "${img_tar}.tmp"
    mv "${img_tar}.tmp" "${img_tar}"
  fi
  rsync -avP --partial --inplace --bwlimit="${BWLIMIT}" -e "ssh ${SSH_OPTS}" \
    "${img_tar}" "${HOST}:${REMOTE_DIR}/bundles/aeb-all-images.tar.gz"
  ssh ${SSH_OPTS} "${HOST}" "docker load -i ${REMOTE_DIR}/bundles/aeb-all-images.tar.gz"
else
  echo "==> [2/4] Container images/bundles already present on ${HOST} (${RUNTIME_MODE}); skipping."
fi

echo "==> [3/4] Installing runtime wrappers on ${HOST} ..."
ssh ${SSH_OPTS} "${HOST}" "
  mkdir -p ${REMOTE_DIR}/bin ${REMOTE_DIR}/results ${REMOTE_DIR}/reports
  chmod +x ${REMOTE_DIR}/repo/scripts/dock_shim.py ${REMOTE_DIR}/repo/scripts/run_amx_benchmarks.sh
  for b in base tools llama-cpp litert-lm onnxruntime openvino xla; do
    if [[ -d ${REMOTE_DIR}/bundles/\${b}/rootfs/opt/aeb/harness ]]; then
      cp -r ${REMOTE_DIR}/repo/docker/harness/. ${REMOTE_DIR}/bundles/\${b}/rootfs/opt/aeb/harness/
    fi
  done
  if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
    cat > ${REMOTE_DIR}/bin/python3 << 'EOF'
#!/bin/bash
exec /usr/bin/env PYTHONHOME=${REMOTE_DIR}/bundles/base/rootfs/usr \\
  ${REMOTE_DIR}/bundles/base/rootfs/lib64/ld-linux-x86-64.so.2 \\
  --library-path ${REMOTE_DIR}/bundles/base/rootfs/lib/x86_64-linux-gnu:${REMOTE_DIR}/bundles/base/rootfs/usr/lib/x86_64-linux-gnu \\
  ${REMOTE_DIR}/bundles/base/rootfs/usr/bin/python3.12 \"\$@\"
EOF
    chmod +x ${REMOTE_DIR}/bin/python3
    ln -sf ${REMOTE_DIR}/repo/scripts/dock_shim.py ${REMOTE_DIR}/bin/docker
  fi
"

echo "==> [4/4] Verifying local & remote model file SHA-256 hashes ..."
MANIFEST_TSV="$(python3 - << 'PY'
import hashlib, json
from pathlib import Path

targets = [
    "models/litert-community",
    "models/google/gemma-4-E2B-it",
    "models/google/gemma-4-E2B-it-qat-mobile-transformers",
    "models/aeb/litert-parity.json",
    "models/aeb/gemma-4-E2B-it-qat-mobile-matched.gguf",
    "models/aeb/gemma-4-E2B-it-qat-mobile-matched.report.json",
    "models/aeb/gemma-4-E2B-it-qat-mobile-matched-q4.gguf",
    "models/aeb/gemma-4-E2B-it-qat-mobile-matched-q4.report.json",
    "models/ggml-org/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_0.gguf",
    "models/ggml-org/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf",
    "models/ggml-org/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-BF16.gguf",
    "models/OpenVINO/gemma-4-E2B-it-int4-ov",
    "models/OpenVINO/gemma-4-E2B-it-int8-ov",
    "models/OpenVINO/gemma-4-E2B-it-fp16-ov",
    "models/aeb/gemma-4-e2b-onnx/Q4_K_M",
]

files = []
for t in targets:
  p = Path(t)
  if p.is_file():
    files.append(p)
  elif p.is_dir():
    for f in sorted(p.rglob("*")):
      if f.is_file() and not f.name.endswith(".sha256") and ".cache" not in f.parts:
        files.append(f)

manifest = {}
for f in files:
  sidecar = Path(str(f) + ".sha256")
  digest = None
  if sidecar.is_file() and sidecar.stat().st_mtime >= f.stat().st_mtime:
    val = sidecar.read_text().strip().split()[0]
    if len(val) == 64:
      digest = val
  if not digest:
    h = hashlib.sha256()
    with f.open("rb") as fp:
      for chunk in iter(lambda: fp.read(16 * 1024 * 1024), b""):
        h.update(chunk)
    digest = h.hexdigest()
    sidecar.write_text(f"{digest}  {f.name}\n")
  rel = f.relative_to("models").as_posix()
  sz = f.stat().st_size
  manifest[rel] = {"size": sz, "sha256": digest}
  print(f"{rel}\t{sz}\t{digest}")

Path("models/aeb/sync-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
)"

echo "    Checking remote files on ${HOST}:${REMOTE_DIR}/models/ against SHA-256 manifest ..."
# Pure POSIX bash + stat + sha256sum on the remote host (no remote Python required!)
NEED_FILES="$(ssh ${SSH_OPTS} "${HOST}" "
  set -e
  mkdir -p '${REMOTE_DIR}/models'
  while IFS=\$'\t' read -r rel exp_size exp_sha; do
    [[ -z \"\${rel}\" ]] && continue
    fpath='${REMOTE_DIR}/models/'\"\${rel}\"
    sidecar=\"\${fpath}.sha256\"
    if [[ ! -f \"\${fpath}\" ]]; then
      mkdir -p \"\$(dirname \"\${fpath}\")\"
      printf 'NEED\t%s\t%s\tmissing\n' \"\${rel}\" \"\${exp_sha}\"
      continue
    fi
    actual_size=\$(stat -c '%s' \"\${fpath}\" 2>/dev/null || echo 0)
    if [[ \"\${actual_size}\" != \"\${exp_size}\" ]]; then
      printf 'NEED\t%s\t%s\tsize_mismatch(%s!=%s)\n' \"\${rel}\" \"\${exp_sha}\" \"\${actual_size}\" \"\${exp_size}\"
      continue
    fi
    actual_sha=''
    if [[ -f \"\${sidecar}\" && \"\${sidecar}\" -nt \"\${fpath}\" ]]; then
      actual_sha=\$(awk '{print \$1}' \"\${sidecar}\" 2>/dev/null || true)
    fi
    if [[ \"\${actual_sha}\" != \"\${exp_sha}\" ]]; then
      actual_sha=\$(sha256sum \"\${fpath}\" | awk '{print \$1}')
      if [[ \"\${actual_sha}\" == \"\${exp_sha}\" ]]; then
        echo \"\${actual_sha}  \$(basename \"\${rel}\")\" > \"\${sidecar}\"
      fi
    fi
    if [[ \"\${actual_sha}\" == \"\${exp_sha}\" ]]; then
      printf 'OK\t%s\t%s\tverified\n' \"\${rel}\" \"\${exp_sha}\"
    else
      printf 'NEED\t%s\t%s\thash_mismatch\n' \"\${rel}\" \"\${exp_sha}\"
    fi
  done
" <<< "${MANIFEST_TSV}")"

TO_PUSH=()
while IFS=$'\t' read -r status rel sha reason; do
  [[ -z "${status}" ]] && continue
  if [[ "${status}" == "OK" ]]; then
    echo "  [SKIP] models/${rel} already complete & hash-verified (sha256=${sha:0:12}...)"
  else
    echo "  [QUEUE] models/${rel} needs push (${reason}, expected sha256=${sha:0:12}...)"
    TO_PUSH+=("${rel}:${sha}")
  fi
done <<< "${NEED_FILES}"

if [[ "${#TO_PUSH[@]}" -eq 0 ]]; then
  echo "==> All model files on ${HOST}:${REMOTE_DIR}/models/ are already complete and SHA-256 verified!"
else
  echo "==> Pushing ${#TO_PUSH[@]} incomplete/missing model files to ${HOST}:${REMOTE_DIR}/models/ ..."
  for item in "${TO_PUSH[@]}"; do
    rel="${item%%:*}"
    sha="${item#*:}"
    rel_dir="$(dirname "${rel}")"
    echo "  [PUSH] models/${rel} (expected sha256=${sha:0:12}...) ..."
    rsync -avP --partial --inplace --bwlimit="${BWLIMIT}" -e "ssh ${SSH_OPTS}" \
      "./models/${rel}" "${HOST}:${REMOTE_DIR}/models/${rel}"
    # Verify remote SHA-256 hash after push and write remote .sha256 sidecar
    ssh ${SSH_OPTS} "${HOST}" "
      set -e
      actual=\$(sha256sum '${REMOTE_DIR}/models/${rel}' | awk '{print \$1}')
      if [[ \"\${actual}\" != '${sha}' ]]; then
        echo 'ERROR: Remote SHA-256 mismatch after rsync for models/${rel}: got '\${actual}', expected ${sha}' >&2
        exit 1
      fi
      echo '${sha}  $(basename "${rel}")' > '${REMOTE_DIR}/models/${rel}.sha256'
    "
    echo "  [VERIFIED] models/${rel} (sha256=${sha:0:12}...)"
  done
fi

echo "==> Sync complete! You can now SSH into ${HOST} and run:"
echo "    nohup /data/aeb/repo/scripts/run_amx_benchmarks.sh > /data/aeb/bench.log 2>&1 &"
