"""In-process BaseRT driver for the AI Edge Bench harness.

Speaks the JSON-lines protocol (docs/harness.md, "Driver protocol") over the
BaseRT C API (`libbaseRT.dylib`, proprietary prebuilt engine with Metal
kernels) through the ctypes declarations of its Apache-2.0 Python binding.

* Tokenization is `baseRT_encode`, which parses the chat-template markers to
  their control ids and prepends <bos> when the model declares it.
* Each request calls `baseRT_reset` (empty KV cache) and then
  `baseRT_generate`, the engine's own generation loop; `token_ns[i]` is taken
  in the per-token callback, when token i is available to the caller.
* Greedy decoding is temperature 0. Seeded sampling passes temperature,
  top-k, top-p and seed in the sampling config (the engine reseeds per call).
* `baseRT_generate` stops on every id the loader registers as a stop token,
  without passing that token to the callback, and has no ignore-EOS option.
  For fixed-length timing (`ignore_eos`) the driver counts the sampled stop
  token as generated (as the other drivers do), feeds the engine's primary EOS
  id back with `baseRT_generate_continue` and decodes the remaining tokens.
  Each response records how often that happened (`eos_resumed`). A logit bias
  on the stop ids is deliberately not used: it moved BaseRT off its fast decode
  path (77.6 -> 42-58 tok/s on the timing workload).
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import os
import sys
import time

import baseRT
from baseRT import BASERT_TOKEN_CALLBACK, BaseRTSamplingConfig

from ._serve import run


class Driver:

  def __init__(self, args):
    self.args = args
    lib = self.lib = baseRT._get_lib()
    vp = ctypes.c_void_p
    for name, res, argt in (("baseRT_version_string", ctypes.c_char_p, []),
                            ("baseRT_eos_token_id", ctypes.c_uint32, [vp]),
                            ("baseRT_bos_id", ctypes.c_uint32, [vp]),
                            ("baseRT_add_bos", ctypes.c_int, [vp]),
                            ("baseRT_is_eos_token", ctypes.c_int32, [vp, ctypes.c_uint32]),
                            ("baseRT_reset", None, [vp]),
                            ("baseRT_has_speculator", ctypes.c_int, [vp]),
                            ("baseRT_get_speculation", ctypes.c_bool, [vp])):
      fn = getattr(lib, name)
      fn.restype, fn.argtypes = res, argt
    t0 = time.monotonic_ns()
    self.model = baseRT.Model(args.model, max_context=args.ctx)
    self.h = self.model._handle
    self.load_ns = time.monotonic_ns() - t0
    self.cfg = self.model.config  # property returning a dataclass
    vocab = getattr(self.cfg, "vocab_size", 0) or 262144
    self.bos = lib.baseRT_bos_id(self.h)
    self.adds_bos = bool(lib.baseRT_add_bos(self.h))
    self.eos_ids = sorted(t for t in range(vocab) if lib.baseRT_is_eos_token(self.h, t))
    self.eos = lib.baseRT_eos_token_id(self.h)
    self.version = lib.baseRT_version_string().decode()
    # Recorded, not changed: n-gram speculation is off by default and an
    # embedded MTP head is only used through the separate drafter API.
    self.speculation = bool(lib.baseRT_get_speculation(self.h))
    self.has_speculator = bool(lib.baseRT_has_speculator(self.h))

  def tokenize(self, text: str, add_bos: bool) -> list[int]:
    buf = (ctypes.c_uint32 * (len(text.encode()) + 16))()
    n = self.lib.baseRT_encode(self.h, text.encode("utf-8"), buf, len(buf))
    if n < 0:
      raise RuntimeError(baseRT.get_error() or "encode failed")
    ids = list(buf[:n])
    has_bos = bool(ids) and ids[0] == self.bos
    if add_bos and not has_bos:
      ids = [self.bos] + ids
    elif not add_bos and has_bos:
      ids = ids[1:]
    return ids

  def generate(self, req: dict) -> dict:
    max_tokens = int(req.get("max_tokens", 256))
    sampling = req.get("sampling") or {}
    temperature = float(sampling.get("temperature", 0.0))
    ignore_eos = bool(req.get("ignore_eos"))
    stop = set(req.get("stop_ids") or []) | set(self.eos_ids)
    cfg = BaseRTSamplingConfig(temperature=temperature if temperature > 0 else 0.0,
                               top_k=int(sampling.get("top_k", 0)) if temperature > 0 else 0,
                               top_p=float(sampling.get("top_p", 1.0)) if temperature > 0 else 1.0,
                               min_p=0.0, repeat_penalty=1.0,
                               seed=int(sampling.get("seed", 0)) if temperature > 0 else 0)
    gen, token_ns, pieces = [], [], []
    state = {"stopped": False}

    def on_token(tid, text, _ud):
      ts = time.monotonic_ns()
      if tid in stop and not ignore_eos:  # the engine's own stop ids never reach here
        state["stopped"] = True
        return False
      gen.append(int(tid))
      token_ns.append(ts)
      pieces.append(text.decode("utf-8", "replace") if text else "")
      return len(gen) < max_tokens

    cb = BASERT_TOKEN_CALLBACK(on_token)
    self.lib.baseRT_reset(self.h)
    t_req = time.monotonic_ns()
    ids = self.tokenize(req["prompt"], True)
    if len(ids) + max_tokens > self.args.ctx:
      raise RuntimeError("prompt + max_tokens exceeds context")
    arr = (ctypes.c_uint32 * len(ids))(*ids)
    st = first = self.lib.baseRT_generate(self.h, arr, len(ids), max_tokens, cfg, cb, None)
    reported = len(gen)
    resumed = 0
    # The engine stops on a sampled stop token without reporting it to the
    # callback (generated_tokens counts it). For fixed-length timing, count it
    # as generated, as the other drivers do, and feed the primary EOS id back.
    while ignore_eos and len(gen) < max_tokens and st.generated_tokens > reported:
      gen.append(self.eos)
      token_ns.append(time.monotonic_ns())
      pieces.append("")
      resumed += 1
      if len(gen) >= max_tokens:
        break
      before = len(gen)
      st = self.lib.baseRT_generate_continue(self.h, (ctypes.c_uint32 * 1)(self.eos), 1,
                                             max_tokens - len(gen), cfg, cb, None)
      reported = len(gen) - before
    t_end = time.monotonic_ns()
    err = baseRT.get_error()
    if not gen and err:
      raise RuntimeError(err)
    resp = {
        "ok": True,
        "n_prompt": len(ids),
        "n_gen": len(gen),
        "stopped_on_eos": not ignore_eos and (state["stopped"] or first.generated_tokens > len(gen)),
        "eos_resumed": resumed,
        "t_req_ns": t_req,
        "t_first_ns": token_ns[0] if token_ns else 0,
        "t_end_ns": t_end,
        "token_ns": token_ns,
        "gen_ids": gen,
        "native": {"prompt_tokens": first.prompt_tokens, "generated_tokens": first.generated_tokens,
                   "prefill_time_ms": first.prefill_time_ms, "decode_time_ms": first.decode_time_ms,
                   "prefill_tokens_per_sec": first.prefill_tokens_per_sec,
                   "decode_tokens_per_sec": first.decode_tokens_per_sec},
    }
    if req.get("return_text", True):
      resp["text"] = "".join(pieces)
    if req.get("return_prompt_ids"):
      resp["prompt_ids"] = ids
    return resp


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--model", required=True, help=".base bundle")
  ap.add_argument("--threads", type=int, default=0,
                  help="recorded only; BaseRT runs the model on the GPU")
  ap.add_argument("--ctx", type=int, default=4096, help="max_context passed to baseRT_load_model")
  args = ap.parse_args(argv)

  def ready(d):
    cfg = dataclasses.asdict(d.cfg) if dataclasses.is_dataclass(d.cfg) else {}
    return {"event": "ready", "framework": "basert", "load_ns": d.load_ns,
            "info": {"engine_version": d.version, "ctx": args.ctx, "model_config": cfg,
                     "bos_id": d.bos, "adds_bos": d.adds_bos, "eos_ids": d.eos_ids,
                     "primary_eos_id": d.eos,
                     "ngram_speculation": d.speculation, "bundle_has_speculator": d.has_speculator,
                     "lib": os.environ.get("BASERT_LIB_PATH"),
                     "sampler": "temperature 0 (greedy)", "pid": os.getpid()}}
  return run(lambda: Driver(args), ready)


if __name__ == "__main__":
  sys.exit(main())
