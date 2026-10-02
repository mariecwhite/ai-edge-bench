"""Generative accuracy checks through the same drivers used for timing.

  python3 -m aeb.accuracy --framework litert-lm --label matched --task mmlu \\
      [--limit N] [--shard 0/2] -- --model /models/... --threads 6

Tasks:
  mmlu   14,042 multiple-choice questions, short answers: exercises prefill
         and the first decoded tokens on a large, standard dataset.
  gsm8k  1,319 grade-school math problems, zero-shot chain of thought with up
         to 512 generated tokens: exercises long greedy decoding, where KV-cache
         or kernel errors accumulate and show up as wrong final answers.

MMLU: zero-shot, chat-templated, greedy decoding, thinking disabled. The prompt
follows lm-evaluation-harness `mmlu_generative` (question, lettered options,
"Answer:") with an explicit answer-format instruction, and the model turn is
pre-filled with "The answer is" so the generated continuation starts with the
option. The first standalone A-D letter is the prediction; anything else is
scored wrong and counted as a parse failure (never dropped).

The purpose is a validity gate (are both frameworks producing correct tokens
from the same weights?), not a leaderboard number: compare frameworks against
each other on identical prompts, and against the published BF16 quality only
loosely, since Google publishes MMLU-Pro rather than MMLU for Gemma 4 E2B.

GSM8K: zero-shot, chat-templated, greedy, thinking disabled; the model is asked
to end with "The final answer is N". The number after that phrase (else the
last number in the reply) is compared numerically with the reference.

Results stream to predictions.jsonl so an interrupted run can be resumed with
--resume <run_dir>. --limit takes a deterministic subset: subject-stratified
for MMLU, evenly spaced for GSM8K.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from pathlib import Path

from . import common, workload
from .driver import FRAMEWORKS, Driver, driver_command

INSTRUCTION = ("The following is a multiple choice question about {subject}. "
               "Reply with only the letter (A, B, C, or D) of the correct answer.\n\n")
# The model turn is pre-filled so the reply starts with the option letter
# instead of an unrequested derivation (zero-shot E2B otherwise reasons first
# on ~10% of math items and runs past any short token budget).
ANSWER_PREFIX = "The answer is"

_ANSWER_IS = re.compile(r"answer\s*(?:is|:)?\s*[\(\*\s]*([ABCD])\b", re.I)
_LETTER = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")


GSM8K_INSTRUCTION = ("Solve the following math problem step by step. At the end, write "
                     "the final answer on its own line in the form "
                     "\"The final answer is N\".\n\n")
_FINAL = re.compile(r"final answer is[:\s]*\**\$?\\?(?:boxed\{)?\s*(-?[\d,]*\.?\d+)", re.I)
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def render_gsm8k(item: dict) -> str:
  return workload.chat(GSM8K_INSTRUCTION + item["question"].strip())


def _num(s: str) -> float | None:
  try:
    return float(s.replace(",", "").rstrip("."))
  except ValueError:
    return None


def extract_gsm8k(text: str) -> str | None:
  m = _FINAL.search(text)
  if m:
    return m.group(1).replace(",", "")
  nums = _NUMBER.findall(text)
  return nums[-1].replace(",", "").rstrip(".") if nums else None


def score_gsm8k(pred: str | None, answer: str) -> bool:
  if pred is None:
    return False
  a, b = _num(pred), _num(answer)
  return a is not None and b is not None and abs(a - b) < 1e-6


def render(item: dict) -> str:
  subject = item["subject"].replace("_", " ")
  body = item["question"].strip() + "\n" + "\n".join(
      f"{l}. {c}" for l, c in zip("ABCD", item["choices"])) + "\nAnswer:"
  return workload.chat(INSTRUCTION.format(subject=subject) + body) + ANSWER_PREFIX


def extract(text: str) -> str | None:
  m = _ANSWER_IS.search(text)
  if m:
    return m.group(1).upper()
  m = _LETTER.search(text)
  return m.group(1) if m else None


def stratified(items: list[dict], limit: int) -> list[dict]:
  if len({it["subject"] for it in items}) == 1:
    step = len(items) / limit
    return [items[int(i * step)] for i in range(min(limit, len(items)))]
  by_subject = collections.defaultdict(list)
  for it in items:
    by_subject[it["subject"]].append(it)
  out, i = [], 0
  while len(out) < limit:
    added = False
    for subj in sorted(by_subject):
      if i < len(by_subject[subj]) and len(out) < limit:
        out.append(by_subject[subj][i])
        added = True
    if not added:
      break
    i += 1
  return sorted(out, key=lambda x: x["id"])


def summarize(preds: list[dict]) -> dict:
  per_subj = collections.defaultdict(lambda: [0, 0])
  correct = parse_fail = 0
  for p in preds:
    per_subj[p["subject"]][1] += 1
    if p["correct"]:
      per_subj[p["subject"]][0] += 1
      correct += 1
    if p["pred"] is None:
      parse_fail += 1
  n = len(preds)
  subj_acc = {s: c / t for s, (c, t) in sorted(per_subj.items())}
  return {
      "n": n,
      "correct": correct,
      "accuracy": correct / n if n else None,
      # Normal-approximation 95% CI half-width for a proportion.
      "accuracy_ci95": 1.96 * ((correct / n) * (1 - correct / n) / n) ** 0.5 if n else None,
      "macro_accuracy": sum(subj_acc.values()) / len(subj_acc) if subj_acc else None,
      "parse_failures": parse_fail,
      "per_subject": subj_acc,
  }


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
  ap.add_argument("--task", choices=["mmlu", "gsm8k"], default="mmlu")
  ap.add_argument("--dataset", type=Path, help="default: /datasets/<task>/test.jsonl")
  ap.add_argument("--limit", type=int, default=0, help="stratified subset size (0 = all)")
  ap.add_argument("--shard", default="0/1", help="k/n: evaluate every n-th item from k")
  ap.add_argument("--max-tokens", type=int, help="default: 8 (mmlu), 512 (gsm8k)")
  ap.add_argument("--resume", type=Path, help="existing run directory to continue")
  args = ap.parse_args(argv)
  args.dataset = args.dataset or Path(f"/datasets/{args.task}/test.jsonl")
  args.max_tokens = args.max_tokens or (8 if args.task == "mmlu" else 512)
  render_fn, extract_fn = (render, extract) if args.task == "mmlu" else (render_gsm8k, extract_gsm8k)
  score_fn = (lambda p, a: p == a) if args.task == "mmlu" else score_gsm8k

  items = [json.loads(l) for l in open(args.dataset)]
  if args.limit:
    items = stratified(items, args.limit)
  k, n = (int(x) for x in args.shard.split("/"))
  items = items[k::n]

  run_dir = args.resume or common.new_run_dir(f"accuracy-{args.task}", args.framework, args.label)
  pred_path = run_dir / "predictions.jsonl"
  done = {}
  if pred_path.exists():
    for line in open(pred_path):
      p = json.loads(line)
      done[p["id"]] = p
  cmd = driver_command(args.framework, driver_args)
  model = common.model_arg(driver_args)
  meta_path = run_dir / "meta.json"
  meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
      "schema": "aeb.accuracy/1",
      "framework": args.framework,
      "label": args.label,
      "machine": os.environ.get("AEB_MACHINE", "unknown-machine"),
      "started_at": common.utc_now(),
      "task": args.task,
      "task_desc": {"mmlu": "MMLU test, generative, 0-shot, chat template, greedy, answer-prefilled",
                    "gsm8k": "GSM8K test, 0-shot CoT, chat template, greedy"}[args.task],
      "dataset": str(args.dataset),
      "dataset_manifest": json.loads((args.dataset.parent / "manifest.json").read_text())
      if (args.dataset.parent / "manifest.json").exists() else None,
      "limit": args.limit, "shard": args.shard, "max_tokens": args.max_tokens,
      "instruction": INSTRUCTION if args.task == "mmlu" else GSM8K_INSTRUCTION,
      "answer_prefix": ANSWER_PREFIX if args.task == "mmlu" else "",
      "driver_cmd": cmd,
      "model_path": model,
      "model_sha256": common.model_sha256(model) if model else None,
      "build": common.build_info(),
  }
  common.dump(meta_path, meta)
  todo = [it for it in items if it["id"] not in done]
  print(f"[aeb] accuracy {args.framework}/{args.label}: {len(todo)} of {len(items)} items "
        f"to run -> {run_dir}", file=sys.stderr, flush=True)

  t0 = time.monotonic()
  with Driver(cmd, run_dir / "driver.log") as drv:
    drv.wait_ready()
    with open(pred_path, "a") as out:
      for i, it in enumerate(todo):
        r = drv.request("generate", prompt=render_fn(it), max_tokens=args.max_tokens,
                        ignore_eos=False, return_text=True, stop_ids=workload.STOP_IDS)
        if not r.get("ok"):
          rec = {"id": it["id"], "subject": it["subject"], "answer": it["answer"],
                 "pred": None, "correct": False, "error": r.get("error"), "text": None}
        else:
          pred = extract_fn(r.get("text", ""))
          rec = {"id": it["id"], "subject": it["subject"], "answer": it["answer"],
                 "pred": pred, "correct": score_fn(pred, it["answer"]), "text": r.get("text"),
                 "n_prompt": r.get("n_prompt"), "n_gen": r.get("n_gen"),
                 "e2e_ms": (r["t_end_ns"] - r["t_req_ns"]) / 1e6}
        out.write(json.dumps(rec) + "\n")
        done[it["id"]] = rec
        if (i + 1) % 200 == 0:
          out.flush()
          acc = sum(p["correct"] for p in done.values()) / len(done)
          rate = (i + 1) / (time.monotonic() - t0)
          print(f"[aeb] {len(done)}/{len(items)} acc {acc:.4f} ({rate:.2f} q/s)",
                file=sys.stderr, flush=True)

  preds = [done[it["id"]] for it in items if it["id"] in done]
  summary = summarize(preds)
  gens = [p.get("n_gen") for p in preds if p.get("n_gen") is not None]
  summary.update(mean_generated_tokens=(sum(gens) / len(gens)) if gens else None,
                 truncated=sum(1 for g in gens if g >= args.max_tokens))
  summary.update(schema="aeb.accuracy.summary/1", framework=args.framework, label=args.label,
                 task=args.task,
                 complete=len(preds) == len(items), n_expected=len(items),
                 wall_s_this_session=time.monotonic() - t0)
  meta["ended_at"] = common.utc_now()
  common.dump(meta_path, meta)
  common.dump(run_dir / "summary.json", summary)
  print(f"[aeb] {args.framework}/{args.label}: accuracy {summary['accuracy']:.4f} "
        f"± {summary['accuracy_ci95']:.4f} over {summary['n']} "
        f"(parse failures {summary['parse_failures']}) -> {run_dir}", file=sys.stderr)
  return 0


if __name__ == "__main__":
  sys.exit(main())
