#!/usr/bin/env python3
"""Python ctypes driver for llama.cpp (`libllama.so`) with prefix KV caching.

Speaks the AI Edge Bench JSON-lines protocol on stdin/stdout and binds directly
to `/opt/llama.cpp/bin/libllama.so` and `/opt/llama.cpp/bin/libggml.so` inside
`aeb/llama-cpp:cpu`. Supports both stateless per-request execution and cached
prefix KV reuse (`--cache-prefix` / `req["cache_prefix"]` + `prefix_tokens`)
via `llama_memory_seq_rm(llama_get_memory(ctx), 0, prefix_tokens, -1)` without
requiring a C++ binary rebuild.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time


GGML_TYPE_MAP = {
    "f32": 0,
    "f16": 1,
    "q4_0": 2,
    "q8_0": 8,
    "bf16": 30,
}

FLASH_ATTN_MAP = {
    "auto": -1,
    "off": 0,
    "on": 1,
}


class _ModelParams(ctypes.Structure):
  _fields_ = [("raw", ctypes.c_uint64 * 10)]


class _ContextParams(ctypes.Structure):
  _fields_ = [("raw", ctypes.c_uint64 * 20)]


class _LlamaBatch(ctypes.Structure):
  _fields_ = [("raw", ctypes.c_uint64 * 7)]


class _PerfContextData(ctypes.Structure):
  _fields_ = [
      ("t_start_ms", ctypes.c_double),
      ("t_load_ms", ctypes.c_double),
      ("t_p_eval_ms", ctypes.c_double),
      ("t_eval_ms", ctypes.c_double),
      ("n_p_eval", ctypes.c_int32),
      ("n_eval", ctypes.c_int32),
      ("n_reused", ctypes.c_int32),
      ("_pad", ctypes.c_int32),
  ]


def _parse_args(argv=None):
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", required=True)
  ap.add_argument("--threads", type=int, default=4)
  ap.add_argument("--threads-batch", type=int, default=0)
  ap.add_argument("--ctx", type=int, default=4096)
  ap.add_argument("--batch", type=int, default=2048)
  ap.add_argument("--ubatch", type=int, default=512)
  ap.add_argument("--flash-attn", default="auto")
  ap.add_argument("--cache-type-k", default="f16")
  ap.add_argument("--cache-type-v", default="f16")
  ap.add_argument("--no-mmap", dest="mmap", action="store_false", default=True)
  ap.add_argument("--no-repack", dest="repack", action="store_false", default=True)
  ap.add_argument("--swa-full", action="store_true", default=False)
  ap.add_argument(
      "--cache-prefix",
      action="store_true",
      help="Keep shared prefix KV cache across requests and rewind suffix via llama_memory_seq_rm",
  )
  args = ap.parse_args(argv)
  if args.threads_batch <= 0:
    args.threads_batch = args.threads
  return args


class LlamaCppEngine:
  """In-process ctypes wrapper around `libllama.so`."""

  def __init__(self, args):
    self.args = args
    self._cache_prefix = bool(args.cache_prefix)
    self._cached_prefix_ids: list[int] | None = None

    lib_dir = os.environ.get("AEB_LLAMA_LIB_DIR", "/opt/llama.cpp/bin")
    self.libggml = ctypes.CDLL(
        os.path.join(lib_dir, "libggml.so"), mode=ctypes.RTLD_GLOBAL
    )
    self.lib = ctypes.CDLL(
        os.path.join(lib_dir, "libllama.so"), mode=ctypes.RTLD_GLOBAL
    )
    self._bind_symbols()

    self.libggml.ggml_backend_load_all_from_path(lib_dir.encode("utf-8"))
    self.lib.llama_backend_init()

    mp = self.lib.llama_model_default_params()
    ctypes.c_int32.from_buffer(mp.raw, 0x20).value = 0  # n_gpu_layers = 0
    ctypes.c_bool.from_buffer(mp.raw, 0x4D).value = bool(args.repack)
    if not args.mmap:
      ctypes.c_int32.from_buffer(mp.raw, 0x48).value = 0  # LLAMA_LOAD_MODE_NONE

    self.model = self.lib.llama_model_load_from_file(
        args.model.encode("utf-8"), mp
    )
    if not self.model:
      raise RuntimeError(f"failed to load model: {args.model}")

    cp = self.lib.llama_context_default_params()
    ctypes.c_uint32.from_buffer(cp.raw, 0x00).value = int(args.ctx)
    ctypes.c_uint32.from_buffer(cp.raw, 0x04).value = int(args.batch)
    ctypes.c_uint32.from_buffer(cp.raw, 0x08).value = int(args.ubatch)
    ctypes.c_uint32.from_buffer(cp.raw, 0x0C).value = 1  # n_seq_max
    ctypes.c_int32.from_buffer(cp.raw, 0x1C).value = int(args.threads)
    ctypes.c_int32.from_buffer(cp.raw, 0x20).value = int(args.threads_batch)
    ctypes.c_int32.from_buffer(cp.raw, 0x34).value = FLASH_ATTN_MAP.get(
        args.flash_attn, -1
    )
    ctypes.c_int32.from_buffer(cp.raw, 0x68).value = GGML_TYPE_MAP[
        args.cache_type_k
    ]
    ctypes.c_int32.from_buffer(cp.raw, 0x6C).value = GGML_TYPE_MAP[
        args.cache_type_v
    ]
    ctypes.c_bool.from_buffer(cp.raw, 0x82).value = False  # no_perf
    ctypes.c_bool.from_buffer(cp.raw, 0x84).value = bool(args.swa_full)

    self.ctx = self.lib.llama_init_from_model(self.model, cp)
    if not self.ctx:
      raise RuntimeError("failed to create llama_context")

    self.vocab = self.lib.llama_model_get_vocab(self.model)
    self.n_vocab = int(self.lib.llama_vocab_n_tokens(self.vocab))
    self.n_ctx = int(self.lib.llama_n_ctx(self.ctx))
    self.greedy_sampler = self.lib.llama_sampler_init_greedy()

  def _bind_symbols(self) -> None:
    self.libggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
    self.libggml.ggml_backend_load_all_from_path.restype = None

    lib = self.lib
    lib.llama_backend_init.argtypes = []
    lib.llama_backend_init.restype = None
    lib.llama_backend_free.argtypes = []
    lib.llama_backend_free.restype = None

    lib.llama_model_default_params.argtypes = []
    lib.llama_model_default_params.restype = _ModelParams
    lib.llama_context_default_params.argtypes = []
    lib.llama_context_default_params.restype = _ContextParams

    lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, _ModelParams]
    lib.llama_model_load_from_file.restype = ctypes.c_void_p
    lib.llama_model_free.argtypes = [ctypes.c_void_p]
    lib.llama_model_free.restype = None

    lib.llama_init_from_model.argtypes = [ctypes.c_void_p, _ContextParams]
    lib.llama_init_from_model.restype = ctypes.c_void_p
    lib.llama_free.argtypes = [ctypes.c_void_p]
    lib.llama_free.restype = None

    lib.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
    lib.llama_model_get_vocab.restype = ctypes.c_void_p
    lib.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
    lib.llama_vocab_n_tokens.restype = ctypes.c_int32
    lib.llama_vocab_is_eog.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.llama_vocab_is_eog.restype = ctypes.c_bool

    lib.llama_n_ctx.argtypes = [ctypes.c_void_p]
    lib.llama_n_ctx.restype = ctypes.c_uint32
    lib.llama_model_size.argtypes = [ctypes.c_void_p]
    lib.llama_model_size.restype = ctypes.c_uint64
    lib.llama_model_n_params.argtypes = [ctypes.c_void_p]
    lib.llama_model_n_params.restype = ctypes.c_uint64
    lib.llama_model_desc.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.llama_model_desc.restype = ctypes.c_int32
    lib.llama_print_system_info.argtypes = []
    lib.llama_print_system_info.restype = ctypes.c_char_p

    lib.llama_tokenize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_bool,
        ctypes.c_bool,
    ]
    lib.llama_tokenize.restype = ctypes.c_int32

    lib.llama_token_to_piece.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_bool,
    ]
    lib.llama_token_to_piece.restype = ctypes.c_int32

    lib.llama_batch_get_one.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
    ]
    lib.llama_batch_get_one.restype = _LlamaBatch
    lib.llama_decode.argtypes = [ctypes.c_void_p, _LlamaBatch]
    lib.llama_decode.restype = ctypes.c_int32

    lib.llama_sampler_init_greedy.argtypes = []
    lib.llama_sampler_init_greedy.restype = ctypes.c_void_p
    lib.llama_sampler_sample.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
    ]
    lib.llama_sampler_sample.restype = ctypes.c_int32
    lib.llama_sampler_free.argtypes = [ctypes.c_void_p]
    lib.llama_sampler_free.restype = None

    lib.llama_get_memory.argtypes = [ctypes.c_void_p]
    lib.llama_get_memory.restype = ctypes.c_void_p
    lib.llama_memory_clear.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    lib.llama_memory_clear.restype = None
    lib.llama_memory_seq_rm.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.llama_memory_seq_rm.restype = ctypes.c_bool

    lib.llama_perf_context_reset.argtypes = [ctypes.c_void_p]
    lib.llama_perf_context_reset.restype = None
    lib.llama_perf_context.argtypes = [ctypes.c_void_p]
    lib.llama_perf_context.restype = _PerfContextData

  def info_dict(self) -> dict:
    desc_buf = ctypes.create_string_buffer(256)
    self.lib.llama_model_desc(self.model, desc_buf, 256)
    sys_info = self.lib.llama_print_system_info()
    return {
        "model_desc": desc_buf.value.decode("utf-8", errors="replace"),
        "model_size_bytes": int(self.lib.llama_model_size(self.model)),
        "n_params": int(self.lib.llama_model_n_params(self.model)),
        "n_ctx": self.n_ctx,
        "n_vocab": self.n_vocab,
        "threads": self.args.threads,
        "threads_batch": self.args.threads_batch,
        "batch": self.args.batch,
        "ubatch": self.args.ubatch,
        "flash_attn": self.args.flash_attn,
        "cache_type_k": self.args.cache_type_k,
        "cache_type_v": self.args.cache_type_v,
        "mmap": self.args.mmap,
        "repack": self.args.repack,
        "swa_full": self.args.swa_full,
        "cache_prefix": self._cache_prefix,
        "system_info": (
            sys_info.decode("utf-8", errors="replace") if sys_info else ""
        ),
    }

  def tokenize(self, text: str, add_bos: bool = True) -> list[int]:
    raw = text.encode("utf-8")
    n = -int(
        self.lib.llama_tokenize(
            self.vocab, raw, len(raw), None, 0, add_bos, True
        )
    )
    if n <= 0:
      return []
    buf = (ctypes.c_int32 * n)()
    got = int(
        self.lib.llama_tokenize(
            self.vocab, raw, len(raw), buf, n, add_bos, True
        )
    )
    if got < 0:
      raise RuntimeError("tokenization failed")
    return [int(buf[i]) for i in range(got)]

  def _piece(self, token_id: int) -> str:
    buf = ctypes.create_string_buffer(256)
    n = int(
        self.lib.llama_token_to_piece(
            self.vocab, int(token_id), buf, 256, 0, False
        )
    )
    if n <= 0:
      return ""
    return buf.raw[:n].decode("utf-8", errors="replace")

  def _decode_chunked(self, token_ids: list[int]) -> None:
    batch_sz = int(self.args.batch)
    for i in range(0, len(token_ids), batch_sz):
      chunk = token_ids[i : i + batch_sz]
      arr = (ctypes.c_int32 * len(chunk))(*chunk)
      b = self.lib.llama_batch_get_one(arr, len(chunk))
      rc = int(self.lib.llama_decode(self.ctx, b))
      if rc != 0:
        raise RuntimeError(f"llama_decode failed with code {rc}")

  def _sample_next(self) -> int:
    return int(self.lib.llama_sampler_sample(self.greedy_sampler, self.ctx, -1))

  def generate(self, req: dict) -> dict:
    prompt = req.get("prompt", "")
    max_tokens = int(req.get("max_tokens", 256))
    ignore_eos = bool(req.get("ignore_eos", False))
    return_text = bool(req.get("return_text", True))
    stop_ids = set(int(t) for t in (req.get("stop_ids") or []))
    use_cache = bool(req.get("cache_prefix", self._cache_prefix))
    prefix_tokens = int(req.get("prefix_tokens", 0))

    mem = self.lib.llama_get_memory(self.ctx)
    n_cached = 0

    if use_cache and prefix_tokens > 0:
      prompt_ids = self.tokenize(prompt, add_bos=True)
      if len(prompt_ids) + max_tokens > self.n_ctx:
        raise RuntimeError("prompt + max_tokens exceeds context")
      if len(prompt_ids) > prefix_tokens:
        prefix_ids = prompt_ids[:prefix_tokens]
        if self._cached_prefix_ids != prefix_ids:
          self.lib.llama_memory_clear(mem, True)
          self._decode_chunked(prefix_ids)
          self._cached_prefix_ids = list(prefix_ids)
        else:
          self.lib.llama_memory_seq_rm(mem, 0, prefix_tokens, -1)
        n_cached = prefix_tokens
      else:
        self._cached_prefix_ids = None
        self.lib.llama_memory_clear(mem, True)

      self.lib.llama_perf_context_reset(self.ctx)
      t_req = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
      self._decode_chunked(prompt_ids[n_cached:])
    else:
      self._cached_prefix_ids = None
      self.lib.llama_memory_clear(mem, True)
      self.lib.llama_perf_context_reset(self.ctx)
      t_req = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
      prompt_ids = self.tokenize(prompt, add_bos=True)
      if len(prompt_ids) + max_tokens > self.n_ctx:
        raise RuntimeError("prompt + max_tokens exceeds context")
      self._decode_chunked(prompt_ids)

    t_prefill_done = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    gen: list[int] = []
    token_ns: list[int] = []
    stopped = False
    one_tok = (ctypes.c_int32 * 1)()

    while len(gen) < max_tokens:
      t = self._sample_next()
      ts = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
      is_stop = (t in stop_ids) or bool(
          self.lib.llama_vocab_is_eog(self.vocab, t)
      )
      if is_stop and not ignore_eos:
        stopped = True
        break
      gen.append(t)
      token_ns.append(ts)
      if len(gen) >= max_tokens:
        break
      one_tok[0] = t
      b = self.lib.llama_batch_get_one(one_tok, 1)
      if int(self.lib.llama_decode(self.ctx, b)) != 0:
        raise RuntimeError("decode failed")

    t_end = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    perf = self.lib.llama_perf_context(self.ctx)
    resp = {
        "ok": True,
        "n_prompt": len(prompt_ids),
        "n_cached": n_cached,
        "n_prefilled": len(prompt_ids) - n_cached,
        "n_gen": len(gen),
        "stopped_on_eos": stopped,
        "t_req_ns": t_req,
        "t_prefill_done_ns": t_prefill_done,
        "t_first_ns": token_ns[0] if token_ns else 0,
        "t_end_ns": t_end,
        "token_ns": token_ns,
        "gen_ids": gen,
        "native": {
            "t_p_eval_ms": float(perf.t_p_eval_ms),
            "t_eval_ms": float(perf.t_eval_ms),
            "n_p_eval": int(perf.n_p_eval),
            "n_eval": int(perf.n_eval),
        },
    }
    if req.get("return_prompt_ids", False):
      resp["prompt_ids"] = prompt_ids
    if return_text:
      resp["text"] = "".join(self._piece(t) for t in gen)
    return resp

  def close(self) -> None:
    if getattr(self, "greedy_sampler", None):
      self.lib.llama_sampler_free(self.greedy_sampler)
      self.greedy_sampler = None
    if getattr(self, "ctx", None):
      self.lib.llama_free(self.ctx)
      self.ctx = None
    if getattr(self, "model", None):
      self.lib.llama_model_free(self.model)
      self.model = None
    if getattr(self, "lib", None):
      self.lib.llama_backend_free()


def main(argv=None) -> int:
  args = _parse_args(argv)
  t0 = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
  try:
    engine = LlamaCppEngine(args)
  except Exception as e:  # pylint: disable=broad-except
    print(json.dumps({"event": "error", "error": str(e)}), flush=True)
    return 1
  t1 = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
  print(
      json.dumps({
          "event": "ready",
          "framework": "llama.cpp",
          "load_ns": t1 - t0,
          "info": engine.info_dict(),
      }),
      flush=True,
  )

  try:
    for line in sys.stdin:
      line = line.strip()
      if not line:
        continue
      try:
        req = json.loads(line)
      except Exception as e:  # pylint: disable=broad-except
        print(json.dumps({"ok": False, "error": f"bad json: {e}"}), flush=True)
        continue
      op = req.get("op", "")
      rid = req.get("id")
      try:
        if op == "quit":
          print(json.dumps({"id": rid, "ok": True}), flush=True)
          break
        elif op == "tokenize":
          ids = engine.tokenize(
              req["text"], add_bos=bool(req.get("add_bos", True))
          )
          print(json.dumps({"id": rid, "ok": True, "ids": ids}), flush=True)
        elif op == "generate":
          resp = engine.generate(req)
          resp["id"] = rid
          print(json.dumps(resp), flush=True)
        else:
          print(
              json.dumps({"id": rid, "ok": False, "error": f"unknown op: {op}"}),
              flush=True,
          )
      except Exception as e:  # pylint: disable=broad-except
        print(json.dumps({"id": rid, "ok": False, "error": str(e)}), flush=True)
  finally:
    engine.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
