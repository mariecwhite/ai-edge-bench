"""In-process MLX (mlx-lm) driver for the AI Edge Bench harness.

Speaks the JSON-lines protocol (docs/harness.md, "Driver protocol") over
`mlx_lm.load` + `mlx_lm.stream_generate` on the Metal GPU.

* Tokenization uses the model's Hugging Face tokenizer without special
  tokens and prepends `bos_token_id` explicitly (the Gemma 4 MLX tokenizer's
  `add_special_tokens` does not add <bos>; its chat template writes it). The
  chat-template markers in the harness prompt are added tokens and parse to
  their single ids.
* Every request builds a fresh prompt cache inside `stream_generate` (empty
  KV cache, no prefix reuse).
* Greedy decoding is `make_sampler(temp=0)` (arg-max). Seeded sampling uses
  `make_sampler(temp, top_p, top_k)` after `mx.random.seed(seed)`, per request.
* `ignore_eos` clears the tokenizer's EOS set for the request, so exactly
  `max_tokens` tokens are produced (an EOS token is fed back like any other).
  Otherwise the request's `stop_ids` are added to the model's EOS set.
* `token_ns[i]` is taken when `stream_generate` yields token i to the caller.
  mlx-lm overlaps the next step's GPU work with that yield (async eval), which
  is the framework's normal streaming behaviour.
* The Metal wired-memory limit is set once at load to the device's
  recommended working-set size, as `mlx_lm.server` does, so the per-call
  `wired_limit` context in `stream_generate` does not change residency inside
  timed requests.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from ._serve import run


def _versions() -> dict:
  from importlib import metadata
  out = {}
  for p in ("mlx", "mlx-metal", "mlx-lm", "transformers"):
    try:
      out[p] = metadata.version(p)
    except metadata.PackageNotFoundError:
      out[p] = None
  return out


class Driver:

  def __init__(self, args):
    self.args = args
    t0 = time.monotonic_ns()
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    self.device = {k: info.get(k) for k in ("device_name", "architecture", "memory_size",
                                            "max_recommended_working_set_size")}
    mx.set_wired_limit(info["max_recommended_working_set_size"])
    self.model, self.tokenizer = load(args.model)
    # Materialize every weight before the ready event so load time is complete.
    mx.eval(self.model.parameters())
    self.load_ns = time.monotonic_ns() - t0
    self.base_eos = set(self.tokenizer.eos_token_ids)

  def tokenize(self, text: str, add_bos: bool) -> list[int]:
    # This tokenizer's add_special_tokens does not prepend <bos> (the chat
    # template writes it literally), so <bos> is added explicitly, once.
    ids = list(self.tokenizer.encode(text, add_special_tokens=False))
    bos = self.tokenizer.bos_token_id
    return ([bos] if add_bos and bos is not None else []) + ids

  def generate(self, req: dict) -> dict:
    max_tokens = int(req.get("max_tokens", 256))
    sampling = req.get("sampling") or {}
    temperature = float(sampling.get("temperature", 0.0))
    if temperature > 0:
      mx.random.seed(int(sampling.get("seed", 0)))
      sampler = make_sampler(temp=temperature, top_p=float(sampling.get("top_p", 1.0)),
                             top_k=int(sampling.get("top_k", 0)))
    else:
      sampler = make_sampler(temp=0.0)
    eos = set() if req.get("ignore_eos") else self.base_eos | set(req.get("stop_ids") or [])
    self.tokenizer.eos_token_ids = eos
    mx.reset_peak_memory()
    gen, token_ns, pieces = [], [], []
    stopped = False
    try:
      t_req = time.monotonic_ns()
      ids = self.tokenize(req["prompt"], True)
      if len(ids) + max_tokens > self.args.ctx:
        raise RuntimeError("prompt + max_tokens exceeds context")
      last = None
      for r in stream_generate(self.model, self.tokenizer, prompt=ids, max_tokens=max_tokens,
                               sampler=sampler, prefill_step_size=self.args.prefill_step_size):
        ts = time.monotonic_ns()
        last = r
        if r.finish_reason == "stop":  # the EOS token itself is not output
          stopped = True
          pieces.append(r.text)
          break
        gen.append(int(r.token))
        token_ns.append(ts)
        pieces.append(r.text)
      t_end = time.monotonic_ns()
    finally:
      self.tokenizer.eos_token_ids = self.base_eos
    resp = {
        "ok": True,
        "n_prompt": len(ids),
        "n_gen": len(gen),
        "stopped_on_eos": stopped,
        "t_req_ns": t_req,
        "t_first_ns": token_ns[0] if token_ns else 0,
        "t_end_ns": t_end,
        "token_ns": token_ns,
        "gen_ids": gen,
        "native": {"prompt_tps": last.prompt_tps if last else None,
                   "generation_tps": last.generation_tps if last else None,
                   "peak_memory_bytes": mx.get_peak_memory()},
    }
    if req.get("return_text", True):
      resp["text"] = "".join(pieces)
    if req.get("return_prompt_ids"):
      resp["prompt_ids"] = ids
    return resp


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--model", required=True, help="MLX model directory")
  ap.add_argument("--threads", type=int, default=0,
                  help="recorded only; MLX runs the model on the GPU")
  ap.add_argument("--ctx", type=int, default=4096,
                  help="request limit (prompt + generated); mlx-lm grows its KV cache on demand")
  ap.add_argument("--prefill-step-size", type=int, default=2048, help="mlx-lm default: 2048")
  args = ap.parse_args(argv)
  return run(lambda: Driver(args), lambda d: {
      "event": "ready", "framework": "mlx", "load_ns": d.load_ns,
      "info": {"versions": _versions(), "device": d.device, "ctx": args.ctx,
               "prefill_step_size": args.prefill_step_size,
               "default_device": str(mx.default_device()),
               "wired_limit_bytes": d.device["max_recommended_working_set_size"],
               "sampler": "make_sampler(temp=0) (arg-max)",
               "eos_token_ids": sorted(d.base_eos), "pid": os.getpid()}})


if __name__ == "__main__":
  sys.exit(main())
