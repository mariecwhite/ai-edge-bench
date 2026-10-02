"""Output reproducibility benchmark.

  python3 -m aeb.repro --framework llama.cpp --label matched --process 0 \\
      [--repeats 3] [--max-tokens 128] -- --model /models/... --threads 12

Question answered: given the same model file, framework build, prompt and
generation settings, does the framework produce the same output every time?
Each invocation is one driver process running ONE sampling configuration,
because some engines fix the sampler when the engine is created (LiteRT-LM
v0.17.1 builds its executor sampler from the first session's parameters and
reuses it for the engine's lifetime). For every prompt in
data/repro_prompts.json (12 prompts covering short answers, math, code,
structured output and open-ended text) the process generates `--repeats`
outputs with stop tokens honoured, each request on a fresh session/KV cache:

  --mode greedy   temperature 0
  --mode seeded   temperature 1.0, top_k 64, top_p 0.95 (Gemma's recommended
                  sampling) with --seed

scripts/suite.py runs, per configuration: greedy x3 processes (two at the
configured thread count, one at half), seeded seed=S x3 (same layout) and a
seed=S+1 control process. The control matters: if changing the seed never
changes the output, seeded "reproducibility" is vacuous (seed ignored, or
sampling collapsed to greedy). aeb.report compares outputs within a process,
across processes, across thread counts and (informational) across frameworks.

Each output is recorded as text plus the token ids the framework's tokenizer
assigns to it (the LiteRT-LM C API streams text, not ids), so divergence can
be located at token granularity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from importlib import resources

from . import common, workload
from .driver import FRAMEWORKS, Driver, driver_command

SAMPLING = {"temperature": 1.0, "top_k": 64, "top_p": 0.95}


def prompts() -> list[dict]:
  return json.loads(resources.files("aeb.data").joinpath("repro_prompts.json").read_text())


def first_divergence(a: list[int], b: list[int]) -> int | None:
  """Index of the first differing token, or None if identical."""
  for i, (x, y) in enumerate(zip(a, b)):
    if x != y:
      return i
  return None if len(a) == len(b) else min(len(a), len(b))


def main(argv=None) -> int:
  argv = list(sys.argv[1:] if argv is None else argv)
  driver_args: list[str] = []
  if "--" in argv:
    i = argv.index("--")
    argv, driver_args = argv[:i], argv[i + 1:]
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--framework", required=True, choices=FRAMEWORKS)
  ap.add_argument("--label", required=True)
  ap.add_argument("--process", type=int, default=0, help="process index within the config")
  ap.add_argument("--repeats", type=int, default=3)
  ap.add_argument("--max-tokens", type=int, default=128)
  ap.add_argument("--mode", choices=["greedy", "seeded"], default="greedy")
  ap.add_argument("--seed", type=int, default=1234)
  ap.add_argument("--notes", default="")
  args = ap.parse_args(argv)

  run_dir = common.new_run_dir("repro", args.framework, args.label)
  cmd = driver_command(args.framework, driver_args)
  model = common.model_arg(driver_args)
  threads = driver_args[driver_args.index("--threads") + 1] if "--threads" in driver_args else None
  meta = {
      "schema": "aeb.repro/1", "framework": args.framework, "label": args.label,
      "process": args.process, "threads": int(threads) if threads else None,
      "machine": os.environ.get("AEB_MACHINE", "unknown-machine"), "notes": args.notes,
      "started_at": common.utc_now(), "driver_cmd": cmd, "model_path": model,
      "model_sha256": common.model_sha256(model) if model else None,
      "build": common.build_info(), "repeats": args.repeats, "max_tokens": args.max_tokens,
      "mode": args.mode, "seed": args.seed if args.mode == "seeded" else None,
      "sampling": SAMPLING if args.mode == "seeded" else {"temperature": 0},
  }
  print(f"[aeb] repro {args.framework}/{args.label} {args.mode} p{args.process} -> {run_dir}",
        file=sys.stderr, flush=True)
  seed = args.seed if args.mode == "seeded" else None
  plan = [(p, args.mode, seed, r) for r in range(args.repeats) for p in prompts()]

  with Driver(cmd, run_dir / "driver.log") as drv, open(run_dir / "outputs.jsonl", "w") as out:
    meta["driver_ready"] = drv.wait_ready()
    for p, mode, seed, rep in plan:
      kwargs = {}
      if mode != "greedy":
        kwargs["sampling"] = dict(SAMPLING, seed=seed)
      r = drv.request("generate", prompt=workload.chat(p["text"]), max_tokens=args.max_tokens,
                      ignore_eos=False, return_text=True, stop_ids=workload.STOP_IDS, **kwargs)
      rec = {"prompt": p["id"], "mode": mode, "seed": seed, "repeat": rep, "ok": r.get("ok")}
      if r.get("ok"):
        text = r.get("text", "")
        ids = drv.request("tokenize", text=text, add_bos=False).get("ids", []) if text else []
        rec.update(text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                   ids=ids, gen_ids=r.get("gen_ids"), n_gen=r.get("n_gen"),
                   stopped_on_eos=r.get("stopped_on_eos"))
      else:
        rec["error"] = r.get("error")
      out.write(json.dumps(rec) + "\n")
      out.flush()

  # Within-process summary: does every repeat match repeat 0?
  recs = [json.loads(l) for l in open(run_dir / "outputs.jsonl")]
  same, total, div = 0, 0, []
  for p in prompts():
    reps = sorted((r for r in recs if r["prompt"] == p["id"] and r.get("ok")),
                  key=lambda r: r["repeat"])
    for r in reps[1:]:
      total += 1
      d = first_divergence(reps[0]["ids"], r["ids"])
      if d is None and r["text"] == reps[0]["text"]:
        same += 1
      else:
        div.append(d)
  summary = {"schema": "aeb.repro.summary/1", "framework": args.framework,
             "label": args.label, "process": args.process, "threads": meta["threads"],
             "mode": args.mode, "seed": meta["seed"],
             "within_process": {"identical": same, "compared": total, "divergence_tokens": div},
             "errors": sum(1 for r in recs if not r.get("ok"))}
  meta["ended_at"] = common.utc_now()
  common.dump(run_dir / "meta.json", meta)
  common.dump(run_dir / "summary.json", summary)
  w = summary["within_process"]
  print(f"[aeb] {args.framework}/{args.label} {args.mode} seed={meta['seed']} p{args.process}: "
        f"{w['identical']}/{w['compared']} repeats identical to the first, "
        f"errors {summary['errors']}", file=sys.stderr)
  return 0


if __name__ == "__main__":
  sys.exit(main())
