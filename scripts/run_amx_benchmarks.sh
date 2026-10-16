#!/usr/bin/env bash
# End-to-end benchmark runner for Gemma 4 E2B (FP32, BF16, INT8, INT4/INT2) on x86 Sapphire Rapids.
#
# Run directly on the target machine:
#   nohup /data/aeb/repo/scripts/run_amx_benchmarks.sh > /data/aeb/bench.log 2>&1 &
#   tail -f /data/aeb/bench.log
#
# Optional phase filter:
#   /data/aeb/repo/scripts/run_amx_benchmarks.sh smoke
#   /data/aeb/repo/scripts/run_amx_benchmarks.sh sweep
#   /data/aeb/repo/scripts/run_amx_benchmarks.sh perf
#   /data/aeb/repo/scripts/run_amx_benchmarks.sh prefix-cache
#   /data/aeb/repo/scripts/run_amx_benchmarks.sh accuracy
#   /data/aeb/repo/scripts/run_amx_benchmarks.sh sequence
#   /data/aeb/repo/scripts/run_amx_benchmarks.sh repro
#   /data/aeb/repo/scripts/run_amx_benchmarks.sh report

set -euo pipefail

export AEB_DATA_ROOT="${AEB_DATA_ROOT:-/data/aeb}"
export PATH="${AEB_DATA_ROOT}/bin:${PATH}"
export AEB_MACHINE="sapphire-rapids"
if [[ -z "${AEB_MACHINE_NOTES:-}" ]]; then
  CPU_MODEL="$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo 2>/dev/null | sed -E 's/Platinum [0-9]{4}C|[0-9]{4}[A-Z]-C/Sapphire Rapids/g' || echo 'Intel Xeon AMX CPU')"
  SOCKETS="$(lscpu 2>/dev/null | awk -F': +' '/^Socket\(s\):/{print $2; exit}' || echo '1')"
  CORES_PER_SOCKET="$(lscpu 2>/dev/null | awk -F': +' '/^Core\(s\) per socket:/{print $2; exit}' || echo '40')"
  export AEB_MACHINE_NOTES="${CPU_MODEL} (${SOCKETS}x ${CORES_PER_SOCKET}C) pinned to NUMA node 0 (${CORES_PER_SOCKET} physical cores)"
fi

REPO_DIR="${AEB_DATA_ROOT}/repo"
SUITE="suites/gemma4-e2b-cpu-sapphire-rapids.json"
PHASE="${1:-all}"

cd "${REPO_DIR}"

# Terminate any prior run_amx_benchmarks.sh / suite.py / dock_shim.py / driver processes
# so a restarted benchmark run never collides with an older background run:
SELF_PID="$$"
PARENT_PID="${PPID:-0}"
for pid in $(pgrep -f "scripts/run_amx_benchmarks\.sh|scripts/suite\.py|dock_shim\.py|aeb\.(perf|accuracy|repro)|_driver\.py" 2>/dev/null || true); do
  if [[ "${pid}" != "${SELF_PID}" && "${pid}" != "${PARENT_PID}" ]]; then
    echo "  [cleanup] stopping previous benchmark process PID ${pid}"
    kill -TERM "${pid}" 2>/dev/null || true
  fi
done
sleep 1
for pid in $(pgrep -f "scripts/run_amx_benchmarks\.sh|scripts/suite\.py|dock_shim\.py|aeb\.(perf|accuracy|repro)|_driver\.py" 2>/dev/null || true); do
  if [[ "${pid}" != "${SELF_PID}" && "${pid}" != "${PARENT_PID}" ]]; then
    kill -9 "${pid}" 2>/dev/null || true
  fi
done

mkdir -p "${AEB_DATA_ROOT}/results" "${AEB_DATA_ROOT}/reports/notes"
if [[ -d "${REPO_DIR}/reports" && ! -L "${REPO_DIR}/reports" ]]; then
  cp -rn "${REPO_DIR}/reports/." "${AEB_DATA_ROOT}/reports/"
  rm -rf "${REPO_DIR}/reports"
fi
ln -sfn "${AEB_DATA_ROOT}/models" "${REPO_DIR}/models"
ln -sfn "${AEB_DATA_ROOT}/datasets" "${REPO_DIR}/datasets"
ln -sfn "${AEB_DATA_ROOT}/results" "${REPO_DIR}/results"
ln -sfn "${AEB_DATA_ROOT}/reports" "${REPO_DIR}/reports"

# Clean up any partial F32 GGUF temp files if a prior conversion was interrupted:
rm -f "${AEB_DATA_ROOT}/models/aeb/gemma-4-E2B-it-F32.gguf" \
      "${AEB_DATA_ROOT}/models/aeb/gemma-4-E2B-it-F32.gguf.tmp"

# Automatically filter out any suite config whose model/GGUF artifact is not already present on disk:
python3 - << 'PY'
import json
import os
from pathlib import Path

data_root = os.environ.get("AEB_DATA_ROOT", "/data/aeb")
suite_path = Path("suites/gemma4-e2b-cpu-sapphire-rapids.json")
suite = json.loads(suite_path.read_text())
models = suite.get("models", {})

kept_configs = []
for cfg in suite.get("configs", []):
  args = cfg.get("args", [])
  missing_model = None
  for a in args:
    if isinstance(a, str) and a.startswith("{") and a.endswith("}"):
      key = a[1:-1]
      if key in models:
        host_path = Path(models[key].replace("/models/", f"{data_root}/models/", 1))
        if not host_path.exists():
          missing_model = str(host_path)
          break
  if missing_model:
    print(f"  [SKIP-CONFIG] Excluding {cfg['id']}: model artifact not present at {missing_model}")
  else:
    kept_configs.append(cfg)

if len(kept_configs) != len(suite.get("configs", [])):
  suite["configs"] = kept_configs
  suite_path.write_text(json.dumps(suite, indent=2) + "\n")
PY

run_phase() {
  [[ "${PHASE}" == "all" || "${PHASE}" == "$1" ]]
}

has_complete_perf() {
  local fw="$1"
  local label="$2"
  local track="$3"
  local min_reps="${4:-1}"
  python3 - "${AEB_DATA_ROOT}/results/${AEB_MACHINE}/perf/${fw}/${label}" "${track}" "${min_reps}" << 'PY'
import json
import sys
from pathlib import Path

cfg_dir = Path(sys.argv[1])
track = sys.argv[2]
min_reps = int(sys.argv[3])
if not cfg_dir.is_dir():
  sys.exit(1)
for summ_p in sorted(cfg_dir.glob("*/summary.json")):
  meta_p = summ_p.parent / "meta.json"
  if not meta_p.is_file():
    continue
  try:
    summ = json.loads(summ_p.read_text())
    meta = json.loads(meta_p.read_text())
  except Exception:
    continue
  if (summ.get("status") in ("ok", "warning")
      and int(summ.get("n_measured", 0)) >= min_reps
      and meta.get("track") == track):
    sys.exit(0)
sys.exit(1)
PY
}

if run_phase "smoke"; then
  echo "=================================================================="
  echo "[Phase 1/7] Unit tests, sysinfo, and 5-framework smoke validation"
  echo "=================================================================="
  make MACHINE=sapphire-rapids test
  if compgen -G "${AEB_DATA_ROOT}/results/${AEB_MACHINE}/sysinfo/*.json" > /dev/null; then
    echo "  [skip] sysinfo already exists in ${AEB_DATA_ROOT}/results/${AEB_MACHINE}/sysinfo/"
  else
    make MACHINE=sapphire-rapids sysinfo
  fi

  echo "==> Smoke-testing all 5 framework containers (128 prompt -> 8 decode tokens) ..."
  SMOKE_CFGS=(
    "llama.cpp/matched"
    "litert-lm/fastest"
    "xla/matched-ynnpack"
    "openvino/int4"
    "onnxruntime/q4_k_m"
  )
  for cfg in "${SMOKE_CFGS[@]}"; do
    echo "  -> smoke ${cfg}"
    python3 scripts/suite.py perf --machine sapphire-rapids --suite "${SUITE}" \
      --only "${cfg}" --rounds 0 --dev \
      --results "${AEB_DATA_ROOT}/results" --skip-existing
  done
fi

if run_phase "sweep"; then
  echo "=================================================================="
  echo "[Phase 2/7] NUMA Node 0 Thread-Scaling Sweep (t in 8,16,32,40)"
  echo "=================================================================="
  THREADS_LIST=(8 16 32 40)
  for t in "${THREADS_LIST[@]}"; do
    echo "--- Thread count t=${t} ---"
    # 1. llama.cpp matched (Q2_K/Q4_0/Q8_0) & Q4_0 (only if GGUF files exist)
    if [[ -s "${AEB_DATA_ROOT}/models/aeb/gemma-4-E2B-it-qat-mobile-matched.gguf" ]]; then
      if has_complete_perf "llama.cpp" "matched-t${t}" "sweep-threads" 2; then
        echo "  [skip] existing sweep-threads llama.cpp/matched-t${t}"
      else
        docker compose -f compose.yaml -f compose/machines/sapphire-rapids.yaml --profile harness run --rm -T \
          harness-llama-cpp aeb.perf --framework llama.cpp --label "matched-t${t}" --track sweep-threads \
          --prompt-tokens 1024 --gen-tokens 64 --warmup 1 --repetitions 2 --cooldown-s 1 -- \
          --model /models/aeb/gemma-4-E2B-it-qat-mobile-matched.gguf --threads "${t}" --ctx 4096 \
          --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
      fi
    fi

    if [[ -s "${AEB_DATA_ROOT}/models/aeb/gemma-4-E2B-it-qat-mobile-matched-q4.gguf" ]]; then
      if has_complete_perf "llama.cpp" "q4_0-t${t}" "sweep-threads" 2; then
        echo "  [skip] existing sweep-threads llama.cpp/q4_0-t${t}"
      else
        docker compose -f compose.yaml -f compose/machines/sapphire-rapids.yaml --profile harness run --rm -T \
          harness-llama-cpp aeb.perf --framework llama.cpp --label "q4_0-t${t}" --track sweep-threads \
          --prompt-tokens 1024 --gen-tokens 64 --warmup 1 --repetitions 2 --cooldown-s 1 -- \
          --model /models/aeb/gemma-4-E2B-it-qat-mobile-matched-q4.gguf --threads "${t}" --ctx 4096 \
          --flash-attn on --ubatch 1024
      fi
    fi

    # 2. LiteRT-LM XNNPACK & YNNPACK
    if has_complete_perf "litert-lm" "matched-t${t}" "sweep-threads" 2; then
      echo "  [skip] existing sweep-threads litert-lm/matched-t${t}"
    else
      docker compose -f compose.yaml -f compose/machines/sapphire-rapids.yaml --profile harness run --rm -T \
        harness-litert-lm aeb.perf --framework litert-lm --label "matched-t${t}" --track sweep-threads \
        --prompt-tokens 1024 --gen-tokens 64 --warmup 1 --repetitions 2 --cooldown-s 1 -- \
        --model /models/litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm \
        --threads "${t}" --ctx 4096 --cache-dir "/results/_cache/litert-lm-xnnpack-t${t}"
    fi

    if has_complete_perf "litert-lm" "ynnpack-t${t}" "sweep-threads" 2; then
      echo "  [skip] existing sweep-threads litert-lm/ynnpack-t${t}"
    else
      docker compose -f compose.yaml -f compose/machines/sapphire-rapids.yaml --profile harness run --rm -T \
        harness-litert-lm aeb.perf --framework litert-lm --label "ynnpack-t${t}" --track sweep-threads \
        --prompt-tokens 1024 --gen-tokens 64 --warmup 1 --repetitions 2 --cooldown-s 1 -- \
        --model /models/litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm \
        --threads "${t}" --ctx 4096 --ynnpack
    fi

    # 3. OpenXLA (YNNPACK)
    if has_complete_perf "xla" "ynnpack-bf16-t${t}" "sweep-threads" 2; then
      echo "  [skip] existing sweep-threads xla/ynnpack-bf16-t${t}"
    else
      docker compose -f compose.yaml -f compose/machines/sapphire-rapids.yaml --profile harness run --rm -T \
        -v "${REPO_DIR}/docker/harness:/opt/aeb/harness:ro" \
        harness-xla aeb.perf --framework xla --label "ynnpack-bf16-t${t}" --track sweep-threads \
        --prompt-tokens 1024 --gen-tokens 64 --warmup 1 --repetitions 2 --cooldown-s 1 -- \
        --model /models/google/gemma-4-E2B-it --threads "${t}" --ctx 4096 --dtype bf16 \
        --cache-dir "/results/_cache/xla-bf16-ynnpack"
    fi

    # 4. OpenVINO GenAI INT4
    if has_complete_perf "openvino" "int4-t${t}" "sweep-threads" 2; then
      echo "  [skip] existing sweep-threads openvino/int4-t${t}"
    else
      docker compose -f compose.yaml -f compose/machines/sapphire-rapids.yaml --profile harness run --rm -T \
        harness-openvino aeb.perf --framework openvino --label "int4-t${t}" --track sweep-threads \
        --prompt-tokens 1024 --gen-tokens 64 --warmup 1 --repetitions 2 --cooldown-s 1 -- \
        --model /models/OpenVINO/gemma-4-E2B-it-int4-ov --threads "${t}" --ctx 4096 --dtype int4
    fi

    # 5. ONNX Runtime GenAI Q4_K_M
    if has_complete_perf "onnxruntime" "q4_k_m-t${t}" "sweep-threads" 2; then
      echo "  [skip] existing sweep-threads onnxruntime/q4_k_m-t${t}"
    else
      docker compose -f compose.yaml -f compose/machines/sapphire-rapids.yaml --profile harness run --rm -T \
        harness-onnxruntime aeb.perf --framework onnxruntime --label "q4_k_m-t${t}" --track sweep-threads \
        --prompt-tokens 1024 --gen-tokens 64 --warmup 1 --repetitions 2 --cooldown-s 1 -- \
        --model /models/aeb/gemma-4-e2b-onnx/Q4_K_M/default --threads "${t}" --ctx 4096
    fi
  done

  echo "==> Selecting empirically fastest thread counts per framework from sweep ..."
  python3 - << 'PY'
import json
import os
from pathlib import Path

data_root = os.environ.get("AEB_DATA_ROOT", "/data/aeb")
results_root = Path(f"{data_root}/results/sapphire-rapids/perf")
suite_path = Path("suites/gemma4-e2b-cpu-sapphire-rapids.json")
suite = json.loads(suite_path.read_text())

mapping = {
    ("llama.cpp", "q4_0"): "llama.cpp",
    ("litert-lm", "ynnpack"): "litert-lm",
    ("xla", "ynnpack-bf16"): "xla",
    ("openvino", "int4"): "openvino",
    ("onnxruntime", "q4_k_m"): "onnxruntime",
}
best = {}
for meta_file in sorted(results_root.glob("*/*/*/meta.json")):
  run_dir = meta_file.parent
  summary_file = run_dir / "summary.json"
  if not summary_file.is_file():
    continue
  meta = json.loads(meta_file.read_text())
  summary = json.loads(summary_file.read_text())
  if meta.get("track") != "sweep-threads" or summary.get("status") == "error":
    continue
  label = meta["label"]
  name, _, t_str = label.rpartition("-t")
  key = (meta["framework"], name)
  if key not in mapping:
    continue
  e2e = (summary.get("metrics", {}).get("e2e_ms") or {}).get("median")
  if e2e is None:
    continue
  t = int(t_str)
  target = mapping[key]
  if target not in best or e2e < best[target][0]:
    best[target] = (e2e, t)

fastest = suite["machines"]["sapphire-rapids"]["fastest"]
for target, (e2e, t) in sorted(best.items()):
  print(f"  [sweep] {target}: best threads = {t} (1024->64 median e2e = {e2e:.1f} ms)")
  fastest[target]["threads"] = t

suite_path.write_text(json.dumps(suite, indent=2) + "\n")
PY
fi

if run_phase "perf"; then
  echo "=================================================================="
  echo "[Phase 3/8] Full Interleaved Performance Suite (Stateless Cold KV, 2 rounds x 3 reps)"
  echo "=================================================================="
  python3 scripts/suite.py perf --machine sapphire-rapids --suite "${SUITE}" \
    --cooldown-s 2 --dev --results "${AEB_DATA_ROOT}/results" --skip-existing \
    || echo "[WARN] Some perf runs exited non-zero; continuing to next phase..."
fi

if run_phase "prefix-cache"; then
  echo "=================================================================="
  echo "[Phase 4/8] Cached-Prefix Performance Suite (Warm KV, 896 cached + 128 suffix -> 128 gen)"
  echo "=================================================================="
  python3 scripts/suite.py prefix-cache --machine sapphire-rapids --suite "${SUITE}" \
    --cooldown-s 2 --dev --results "${AEB_DATA_ROOT}/results" --skip-existing \
    || echo "[WARN] Some prefix-cache runs exited non-zero; continuing to next phase..."
fi

if run_phase "accuracy"; then
  echo "=================================================================="
  echo "[Phase 5/8] Generative Accuracy Suite (Full MMLU-2000 + GSM8K-250 on fastest configs)"
  echo "=================================================================="
  python3 scripts/suite.py accuracy --machine sapphire-rapids --suite "${SUITE}" \
    --parallel 2 --shards 2 --threads half --dev \
    --results "${AEB_DATA_ROOT}/results" --skip-existing \
    || echo "[WARN] Some accuracy runs exited non-zero; continuing to next phase..."
fi

if run_phase "sequence"; then
  echo "=================================================================="
  echo "[Phase 6/8] Input-Length Sequence Sweep (fastest accurate per framework+dtype)"
  echo "=================================================================="
  python3 scripts/suite.py sequence --machine sapphire-rapids --suite "${SUITE}" \
    --cooldown-s 2 --dev --results "${AEB_DATA_ROOT}/results" --skip-existing \
    || echo "[WARN] Some sequence sweep runs exited non-zero; continuing to next phase..."
fi

if run_phase "repro"; then
  echo "=================================================================="
  echo "[Phase 7/8] Output Reproducibility Suite (Greedy + Seeded)"
  echo "=================================================================="
  python3 scripts/suite.py repro --machine sapphire-rapids --suite "${SUITE}" \
    --cooldown-s 2 --dev --results "${AEB_DATA_ROOT}/results" --skip-existing \
    || echo "[WARN] Some repro runs exited non-zero; continuing to next phase..."
fi

if run_phase "report"; then
  echo "=================================================================="
  echo "[Phase 8/8] Generating Standardized Markdown & HTML Report & Charts"
  echo "=================================================================="
  make MACHINE=sapphire-rapids SUITE="${SUITE}" report
  python3 scripts/update_dashboard.py \
    --reports "${AEB_DATA_ROOT}/reports" \
    --results "${AEB_DATA_ROOT}/results"
  echo "==> Report generated in ${AEB_DATA_ROOT}/reports/ and dashboard updated in dashboards/index.html !"
  ls -la "${AEB_DATA_ROOT}/reports/"
fi
