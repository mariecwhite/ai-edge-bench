"""Diagnose accuracy differences between frameworks that share weights.

  python3 -m aeb.prep.hf_reference \\
      --qat-dir /models/google/gemma-4-E2B-it-qat-mobile-transformers \\
      --a /results/.../accuracy-mmlu/llama.cpp/matched --b /results/.../accuracy-mmlu/litert-lm/matched \\
      --n 120 --out /results/<machine>/diagnostics/hf-reference.json

Optional diagnostic (needs torch + transformers, not part of the timed
images). It reruns the upstream PyTorch reference implementation of the
mobile QAT checkpoint (transformers' `gemma` quantizer) on MMLU items where
two frameworks disagree, in two modes:

  srq-on    the checkpoint's static int8 activation rounding (SRQ) applied
            to every quantized linear layer's input and output, as trained
            and as LiteRT-LM executes it
  srq-off   the same integer weights with full-precision activations, which
            is what llama.cpp's kernels approximate (dynamic 8-bit
            activation blocks)

If srq-on tracks one framework and srq-off the other, the accuracy gap comes
from activation quantization, not from a weight-conversion or kernel bug.
Weights are dequantized once to fp32 so each mode runs at plain-matmul
speed; prompts and answer parsing are aeb.accuracy's.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .. import accuracy


def load_preds(run_root: Path) -> dict:
  """Merge predictions from every complete shard below run_root."""
  preds = {}
  for p in sorted(run_root.glob("*/predictions.jsonl")):
    for line in open(p):
      r = json.loads(line)
      preds[r["id"]] = r
  return preds


def main(argv=None) -> int:
  import torch
  import torch.nn.functional as F
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from transformers.integrations.gemma_quant import QuantizedLinear, apply_srq

  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--qat-dir", type=Path, required=True)
  ap.add_argument("--a", type=Path, required=True, help="accuracy run root of framework A")
  ap.add_argument("--b", type=Path, required=True, help="accuracy run root of framework B")
  ap.add_argument("--dataset", type=Path, default=Path("/datasets/mmlu/test.jsonl"))
  ap.add_argument("--n", type=int, default=120, help="discordant items to rerun")
  ap.add_argument("--n-concordant", type=int, default=40)
  ap.add_argument("--threads", type=int, default=0)
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args(argv)
  if args.threads:
    torch.set_num_threads(args.threads)

  pa, pb = load_preds(args.a), load_preds(args.b)
  items = {json.loads(l)["id"]: json.loads(l) for l in open(args.dataset)}
  common = sorted(set(pa) & set(pb))
  disc = [i for i in common if pa[i]["correct"] != pb[i]["correct"]]
  conc = [i for i in common if pa[i]["correct"] == pb[i]["correct"]]
  pick = lambda ids, n: [ids[int(k * len(ids) / n)] for k in range(min(n, len(ids)))]
  chosen = [(i, "discordant") for i in pick(disc, args.n)] + \
           [(i, "concordant") for i in pick(conc, args.n_concordant)]

  tok = AutoTokenizer.from_pretrained(args.qat_dir)
  model = AutoModelForCausalLM.from_pretrained(args.qat_dir, dtype=torch.float32)
  model.eval()
  linears = [m for m in model.modules() if isinstance(m, QuantizedLinear)]
  saved = {}
  for m in linears:
    w = m._dequantize_weights(torch.float32)
    m.register_buffer("w_deq", w, persistent=False)
    saved[m] = (m.input_activation_scale.detach().clone(), m.output_activation_scale.detach().clone())

    def fwd(x, m=m):
      x = apply_srq(x, m.input_activation_scale)
      return apply_srq(F.linear(x, m.w_deq.to(x.dtype), m.bias), m.output_activation_scale)
    m.forward = fwd

  results = {"a": str(args.a), "b": str(args.b), "n_common": len(common),
             "n_discordant_total": len(disc), "items": []}
  for mode in ("srq-on", "srq-off"):
    for m in linears:
      i_s, o_s = saved[m]
      m.input_activation_scale.data = i_s if mode == "srq-on" else torch.zeros_like(i_s)
      m.output_activation_scale.data = o_s if mode == "srq-on" else torch.zeros_like(o_s)
    t0 = time.monotonic()
    for k, (iid, kind) in enumerate(chosen):
      enc = tok(accuracy.render(items[iid]), return_tensors="pt", add_special_tokens=True)
      with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=8, do_sample=False)
      text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
      if mode == "srq-on":
        results["items"].append({"id": iid, "kind": kind, "answer": items[iid]["answer"],
                                 "a_pred": pa[iid]["pred"], "b_pred": pb[iid]["pred"]})
      rec = results["items"][k]
      rec[f"{mode}_pred"] = accuracy.extract(text)
      rec[f"{mode}_text"] = text
      if (k + 1) % 20 == 0:
        print(f"[aeb] {mode} {k + 1}/{len(chosen)} ({(time.monotonic() - t0) / (k + 1):.1f} s/item)",
              file=sys.stderr, flush=True)

  summ = {}
  for kind in ("discordant", "concordant"):
    rows = [r for r in results["items"] if r["kind"] == kind]
    if not rows:
      continue
    s = {"n": len(rows)}
    for mode in ("srq-on", "srq-off"):
      s[mode] = {
          "acc": sum(r[f"{mode}_pred"] == r["answer"] for r in rows) / len(rows),
          "agrees_with_a": sum(r[f"{mode}_pred"] == r["a_pred"] for r in rows) / len(rows),
          "agrees_with_b": sum(r[f"{mode}_pred"] == r["b_pred"] for r in rows) / len(rows),
      }
    s["a_acc"] = sum(r["a_pred"] == r["answer"] for r in rows) / len(rows)
    s["b_acc"] = sum(r["b_pred"] == r["answer"] for r in rows) / len(rows)
    summ[kind] = s
  results["summary"] = summ
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(results, indent=1) + "\n")
  print(json.dumps(summ, indent=1))
  return 0


if __name__ == "__main__":
  sys.exit(main())
