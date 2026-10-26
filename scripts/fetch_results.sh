#!/usr/bin/env bash
# Fetch benchmark results, updated suite configs, and generated reports from a
# remote benchmark host back to the local repository.
#
# Usage:
#   ./scripts/fetch_results.sh <user@hostname> [--machine <machine>] [--suite <suite.json>]
#

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

HOST="${AEB_REMOTE_HOST:-}"
MACHINE="${AEB_MACHINE:-}"
SUITE="${AEB_SUITE:-}"
REMOTE_DIR="${AEB_REMOTE_DIR:-/data/aeb}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host|-H)
      HOST="$2"
      shift 2
      ;;
    --machine|-m)
      MACHINE="$2"
      shift 2
      ;;
    --suite|-s)
      SUITE="$2"
      shift 2
      ;;
    --remote-dir|-d)
      REMOTE_DIR="$2"
      shift 2
      ;;
    --help|-h)
      sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      if [[ -z "${HOST}" ]]; then
        HOST="$1"
      elif [[ -z "${MACHINE}" ]]; then
        MACHINE="$1"
      elif [[ -z "${SUITE}" ]]; then
        SUITE="$1"
      else
        echo "Unknown argument: $1" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "${HOST}" ]]; then
  echo "Usage: ./scripts/fetch_results.sh <user@hostname> [--machine <machine>] [--suite <suite.json>]" >&2
  exit 1
fi

SSH_OPTS="${AEB_SSH_OPTS:--o StrictHostKeyChecking=no}"
SOCK="${HOME}/.ssh/cm-aeb"
if [[ -S "${SOCK}" ]]; then
  SSH_OPTS="${SSH_OPTS} -o ControlPath=${SOCK}"
fi

mkdir -p ./suites ./results ./reports
for dir in ./results ./reports; do
  if [[ ! -w "${dir}" ]] && command -v docker >/dev/null 2>&1; then
    docker run --rm -v "${ROOT}/${dir#./}:/target" aeb/base:latest chown -R "$(id -u):$(id -g)" /target || true
  fi
done

echo "==> Fetching suite configs, results, and reports from ${HOST}:${REMOTE_DIR}/ ..."

if [[ -n "${SUITE}" ]]; then
  SUITE_FILE="$(basename "${SUITE}")"
  [[ "${SUITE_FILE}" == *.json ]] || SUITE_FILE="${SUITE_FILE}.json"
  rsync -avP -e "ssh ${SSH_OPTS}" \
    "${HOST}:${REMOTE_DIR}/repo/suites/${SUITE_FILE}" \
    "./suites/${SUITE_FILE}"
else
  rsync -avP --include='*.json' --exclude='*' -e "ssh ${SSH_OPTS}" \
    "${HOST}:${REMOTE_DIR}/repo/suites/" \
    ./suites/
fi

if [[ -n "${MACHINE}" ]]; then
  mkdir -p "./results/${MACHINE}"
  rsync -avP --exclude='_cache' --exclude='*.bak*' -e "ssh ${SSH_OPTS}" \
    "${HOST}:${REMOTE_DIR}/results/${MACHINE}/" \
    "./results/${MACHINE}/"
else
  rsync -avP --exclude='_cache' --exclude='*.bak*' -e "ssh ${SSH_OPTS}" \
    "${HOST}:${REMOTE_DIR}/results/" \
    ./results/
fi

rsync -avP --exclude='2026-10-12-*' -e "ssh ${SSH_OPTS}" \
  "${HOST}:${REMOTE_DIR}/reports/" \
  ./reports/

python3 - << 'PY'
import json
import re
from pathlib import Path

XEON_RE = re.compile(
    r"(Intel\(R\)\s+Xeon\(R\))(?:\s+(?:Platinum|Gold|Silver|Bronze|Max))?"
    r"\s+(?!Sapphire\s+Rapids\b)[0-9A-Za-z-]+(?:\s+CPU)?(?:\s*@\s*[0-9.]+GHz)?"
)
KERNEL_RE = re.compile(r"^([A-Za-z]+\s+\d+\.\d+(?:\.\d+)?).*$")

for p in sorted(Path("results").rglob("sysinfo.json")):
  try:
    d = json.loads(p.read_text())
  except Exception:
    continue
  changed = False
  m = d.get("machine", {})
  m_label = m.get("label") or "container"
  c = d.get("container", {})
  if c.get("hostname") != m_label:
    c["hostname"] = m_label
    changed = True
  k_str = str(c.get("kernel", ""))
  new_k = KERNEL_RE.sub(r"\1", k_str)
  if new_k != k_str:
    c["kernel"] = new_k
    changed = True
  if m_label == "sapphire-rapids":
    new_m_notes = XEON_RE.sub(r"\1 Sapphire Rapids", str(m.get("notes", "")))
    if new_m_notes != str(m.get("notes", "")):
      m["notes"] = new_m_notes
      changed = True
    cpu = d.get("cpu", {})
    new_cpu_model = XEON_RE.sub(r"\1 Sapphire Rapids", str(cpu.get("model", "")))
    if new_cpu_model != str(cpu.get("model", "")):
      cpu["model"] = new_cpu_model
      changed = True
  if changed:
    p.write_text(json.dumps(d, indent=2) + "\n")

for p in sorted(Path("results").rglob("meta.json")):
  try:
    d = json.loads(p.read_text())
  except Exception:
    continue
  if d.get("machine") == "sapphire-rapids":
    new_notes = XEON_RE.sub(r"\1 Sapphire Rapids", str(d.get("machine_notes", "")))
    if new_notes != str(d.get("machine_notes", "")):
      d["machine_notes"] = new_notes
      p.write_text(json.dumps(d, indent=1) + "\n")
PY

echo "==> Updating dashboards/index.html ..."
python3 "${ROOT}/scripts/update_dashboard.py"

echo "==> Done! Reports are available under ./reports/ and dashboard under ./dashboards/index.html:"
if [[ -n "${MACHINE}" ]]; then
  ls -la ./reports/*"${MACHINE}"* 2>/dev/null || ls -la ./reports/
else
  ls -la ./reports/
fi
