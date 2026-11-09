#!/usr/bin/env python3
"""Cross-check the harness against each framework's own benchmark tool
(docs/benchmark_methodology.md, "Measurement boundaries"), natively on macOS.

  python3 scripts/native_tools_crosscheck.py --suite suites/gemma4-26b-a4b-gpu.json \\
      [--machine apple-m5-max] [--only llama.cpp,litert-lm,mlx,basert]

Runs, one process at a time, on the suite's model files and the suite
workload shape (prompt N, decode D):

  llama.cpp  llama-bench -ngl 999: pp N, and tg D at depth N (-d N)
  litert-lm  litert-lm benchmark --backend gpu -p N -d D (zero-filled prompt)
  mlx        mlx_lm.benchmark -p N -g D (random prompt tokens)
  basert     basert-bench -p N -n D --fill corpus:<harness passage> --tg-at N (decode at depth N)

The tools differ from the harness and from each other (synthetic vs real
prompts, decode depth, what is timed); their numbers explain differences, they
are not substitutes. Raw output goes to results/<machine>/native-tools/<fw>/<run>/
with the command, and parsed throughput to summary.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "build" / "native"
# Real-text prefill for basert-bench: the same public-domain passage as the
# harness prompt (the engine release ships no corpus of its own).
CORPUS = ROOT / "docker" / "harness" / "aeb" / "data" / "alice_ch1-3.txt"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import suite as suite_mod  # noqa: E402  (host_path mapping, model placeholders)


def run(cmd: list[str], out_dir: Path, env: dict, stdout_only: bool = False) -> str:
  print(f"[crosscheck] {' '.join(cmd)}", flush=True)
  p = subprocess.run(cmd, capture_output=True, text=True, env=env)
  (out_dir / "command.txt").open("a").write(" ".join(cmd) + f"\n# exit {p.returncode}\n")
  (out_dir / "stdout.txt").open("a").write(p.stdout)
  (out_dir / "stderr.txt").open("a").write(p.stderr)
  if p.returncode != 0:
    raise RuntimeError(f"{cmd[0]} exited {p.returncode}; see {out_dir}/stderr.txt")
  return p.stdout if stdout_only else p.stdout + "\n" + p.stderr


def floats(pattern: str, text: str) -> list[float]:
  return [float(x) for x in re.findall(pattern, text, re.I | re.M)]


def llama_cpp(model, n, d, threads, out, env):
  bench = [str(NATIVE / "llama.cpp/bin/llama-bench"), "-m", model, "-ngl", "999",
           "-t", str(threads), "-r", "5", "-o", "json"]
  pp = json.loads(run(bench + ["-p", str(n), "-n", "0"], out, env, stdout_only=True) or "[]")
  tg = json.loads(run(bench + ["-p", "0", "-n", str(d), "-d", str(n)], out, env,
                      stdout_only=True) or "[]")
  return {"prefill_tps": pp[0]["avg_ts"] if pp else None,
          "decode_tps": tg[0]["avg_ts"] if tg else None,
          "notes": f"llama-bench random token ids; tg at depth {n} (-d {n}); 5 reps; flash-attn auto"}


def litert_lm(model, n, d, threads, out, env):
  txt = run([str(NATIVE / "venv/bin/litert-lm"), "benchmark", model, "--backend", "gpu",
             "-p", str(n), "-d", str(d), "--runs", "5", "--max-num-tokens", str(n + d + 64)],
            out, env)
  pre = floats(r"prefill[^\n]*?([\d.]+)\s*(?:tokens?/s|tok/s)", txt)
  dec = floats(r"decode[^\n]*?([\d.]+)\s*(?:tokens?/s|tok/s)", txt)
  return {"prefill_tps": pre[-1] if pre else None, "decode_tps": dec[-1] if dec else None,
          "notes": "litert-lm benchmark: a fresh engine per iteration (warm-up included), so every "
                   "prefill timed is an engine's first prefill, which the harness treats as warm-up; "
                   "engine benchmark-mode counters; 5 runs"}


def mlx(model, n, d, threads, out, env):
  txt = run([str(NATIVE / "venv/bin/python"), "-m", "mlx_lm.benchmark", "--model", model,
             "-p", str(n), "-g", str(d), "-n", "5"], out, env)
  pre = floats(r"prompt_tps[=:\s]+([\d.]+)", txt)
  dec = floats(r"generation_tps[=:\s]+([\d.]+)", txt)
  return {"prefill_tps": pre[-1] if pre else None, "decode_tps": dec[-1] if dec else None,
          "notes": "mlx_lm.benchmark (random prompt tokens); averages over 5 trials"}


def basert(model, n, d, threads, out, env):
  txt = run([str(NATIVE / "basert/engine/basert-bench"), model, "-p", str(n), "-n", str(d),
             "--tg-at", str(n), "--fill", f"corpus:{CORPUS}", "-r", "5"], out, env)
  # Markdown table rows: "| model | size | pp1024:corpus | 3851.99 ± 7.55 |",
  # "| … | tg256@1024 | 80.55 (best of 5) |".
  pre = floats(rf"\|\s*pp{n}(?::\S+)?\s*\|\s*([\d.]+)", txt)
  dec = floats(rf"\|\s*tg{d}@{n}\s*\|\s*([\d.]+)", txt)
  return {"prefill_tps": pre[0] if pre else None, "decode_tps": dec[-1] if dec else None,
          "decode_tps_depth0": (floats(rf"\|\s*tg{d}\s*\|\s*([\d.]+)", txt) or [None])[0],
          "notes": f"basert-bench --fill corpus (the harness's passage); decode at depth {n} "
                   "(--tg-at, best of 5) and from an empty context; adaptive warm-up, 5 reps"}


TOOLS = {"llama.cpp": llama_cpp, "litert-lm": litert_lm, "mlx": mlx, "basert": basert}


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--suite", type=Path, required=True)
  ap.add_argument("--machine", default="apple-m5-max")
  ap.add_argument("--only", default=",".join(TOOLS))
  args = ap.parse_args(argv)
  suite, ctx = suite_mod.load(args.suite, args.machine)
  suite_mod.keep_awake()  # caffeinate -dis for this process's lifetime (native suite)
  w = suite["workload"]
  env = dict(os.environ, BASERT_LIB_PATH=str(NATIVE / "basert/engine/libbaseRT.dylib"),
             PATH=f"{NATIVE / 'venv/bin'}:{os.environ.get('PATH', '')}")
  stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  results = {}
  for c in suite["configs"]:
    fw = c["framework"]
    if fw not in args.only.split(","):
      continue
    model = suite_mod.host_path(suite_mod.resolve(c["args"][c["args"].index("--model") + 1], ctx))
    out = ROOT / "results" / args.machine / "native-tools" / fw / stamp
    out.mkdir(parents=True, exist_ok=True)
    try:
      res = TOOLS[fw](model, w["prompt_tokens"], w["gen_tokens"], ctx["threads"], out, env)
    except Exception as e:  # keep going; record the failure
      res = {"error": str(e)}
    res.update(config=c["id"], prompt_tokens=w["prompt_tokens"], gen_tokens=w["gen_tokens"],
               raw=str(out.relative_to(ROOT)))
    (out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    results[c["id"]] = res
    print(f"[crosscheck] {c['id']}: {res}", flush=True)
  print(json.dumps(results, indent=1))
  return 0


if __name__ == "__main__":
  sys.exit(main())
