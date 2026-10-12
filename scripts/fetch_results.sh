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
  rsync -avP --exclude='_cache' -e "ssh ${SSH_OPTS}" \
    "${HOST}:${REMOTE_DIR}/results/${MACHINE}/" \
    "./results/${MACHINE}/"
else
  rsync -avP --exclude='_cache' -e "ssh ${SSH_OPTS}" \
    "${HOST}:${REMOTE_DIR}/results/" \
    ./results/
fi

rsync -avP -e "ssh ${SSH_OPTS}" \
  "${HOST}:${REMOTE_DIR}/reports/" \
  ./reports/

echo "==> Done! Reports are available under ./reports/:"
if [[ -n "${MACHINE}" ]]; then
  ls -la ./reports/*"${MACHINE}"* 2>/dev/null || ls -la ./reports/
else
  ls -la ./reports/
fi
