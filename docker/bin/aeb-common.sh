#!/usr/bin/env bash
# Shared helpers for the AI Edge Bench container harness.
# shellcheck shell=bash

set -euo pipefail

AEB_MODELS_DIR="${AEB_MODELS_DIR:-/models}"
AEB_RESULTS_DIR="${AEB_RESULTS_DIR:-/results}"

aeb::log() { printf '[aeb] %s\n' "$*" >&2; }
aeb::die() { printf '[aeb] error: %s\n' "$*" >&2; exit 1; }

aeb::require_cmd() {
  command -v "$1" >/dev/null 2>&1 || aeb::die "required command not found: $1"
}

# Normalise the many spellings of true/false accepted from compose env files.
aeb::bool() {
  case "${1,,}" in
    1|true|yes|on) printf 'true' ;;
    0|false|no|off|'') printf 'false' ;;
    *) aeb::die "not a boolean: $1" ;;
  esac
}

aeb::run_id() {
  printf '%s-%s' "$(date -u +%Y%m%dT%H%M%SZ)" "$(tr -dc 'a-f0-9' </dev/urandom | head -c 6)"
}

# Physical cores as seen inside the container, clamped by the cgroup CPU quota
# so a `--cpus` limit does not get oversubscribed by the framework's own
# auto-detection.
aeb::default_threads() {
  local cores quota
  cores="$(nproc 2>/dev/null || echo 1)"
  if [[ -r /sys/fs/cgroup/cpu.max ]]; then
    read -r quota period < /sys/fs/cgroup/cpu.max || true
    if [[ "${quota:-max}" != "max" && -n "${period:-}" && "${period}" -gt 0 ]]; then
      local allowed=$(( quota / period ))
      (( allowed < 1 )) && allowed=1
      (( allowed < cores )) && cores="${allowed}"
    fi
  fi
  printf '%s' "${cores}"
}

# Create and echo the directory for a single run.
# usage: aeb::new_run_dir <framework> <variant> <run_id>
aeb::new_run_dir() {
  local framework="$1" variant="$2" run_id="$3"
  local machine="${AEB_MACHINE:-unknown-machine}"
  local dir="${AEB_RESULTS_DIR}/${machine}/${framework}/${variant}/${run_id}"
  mkdir -p "${dir}" || aeb::die "cannot create ${dir} (is ${AEB_RESULTS_DIR} mounted writable?)"
  printf '%s' "${dir}"
}

aeb::sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    aeb::die "sha256sum not available"
  fi
}

# Read the build manifest that each framework image writes at build time.
aeb::build_info() {
  if [[ -r /opt/aeb/build-info.json ]]; then
    cat /opt/aeb/build-info.json
  else
    printf '{}'
  fi
}
