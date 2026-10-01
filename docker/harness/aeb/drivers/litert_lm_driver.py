"""In-process LiteRT-LM driver for the AI Edge Bench harness.

Speaks the same JSON-lines protocol as docker/llama-cpp/aeb-llama-driver.cpp
(see harness/aeb/protocol.md) on stdin/stdout, over the LiteRT-LM C API
(liblitert-lm.so built from the pinned source, loaded via the upstream ctypes
bindings in `litert_lm._ffi`).

Requests run on a fresh session (empty KV cache) with the prompt template
disabled, so the harness sends exactly the same text to every framework. The
engine prepends <bos> itself; callers must not include it.

Greedy decoding is requested as the TOP_P sampler with top_k=1: at v0.17.1
the CPU sampler only implements TOP_P (GREEDY and TOP_K return UNIMPLEMENTED),
and keeping a single candidate selects the arg-max token, i.e. greedy.

Fixed-length decoding (--fixed-decode-tokens N) uses the engine's benchmark
parameter `num_decode_tokens`, which makes LiteRT-LM decode exactly N steps and
ignore stop tokens. This is an engine-creation setting, so a driver started
with it cannot also serve EOS-terminated requests; the harness starts a
separate driver process for accuracy runs.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import sys
import time

from litert_lm import _ffi
from litert_lm._ffi import InputDataType, STREAM_CALLBACK_TYPE, SamplerType


def emit(obj) -> None:
  sys.stdout.write(json.dumps(obj) + "\n")
  sys.stdout.flush()


class Driver:

  def __init__(self, args):
    self.args = args
    self.lib = _ffi._get_lib()
    _ffi.set_min_log_severity(_ffi.LogSeverity.ERROR)
    lib = self.lib
    t0 = time.monotonic_ns()
    settings = lib.litert_lm_engine_settings_create(args.model, "cpu", None, None)
    if not settings:
      raise RuntimeError("engine settings creation failed")
    lib.litert_lm_engine_settings_set_num_threads(settings, args.threads)
    lib.litert_lm_engine_settings_set_max_num_tokens(settings, args.ctx)
    if args.cache_dir:
      # LiteRT-LM silently skips the weight cache when the directory is
      # missing, which changes both load time and memory (packed weights stay
      # in anonymous memory instead of a mmap'd cache file).
      if args.cache_dir != ":memory":
        os.makedirs(args.cache_dir, exist_ok=True)
      lib.litert_lm_engine_settings_set_cache_dir(settings, args.cache_dir)
    if args.ynnpack:
      lib.litert_lm_engine_settings_set_enable_ynnpack(settings, True)
    # Benchmark mode only records per-turn timings unless num_*_tokens > 0.
    lib.litert_lm_engine_settings_enable_benchmark(settings)
    if args.fixed_decode_tokens > 0:
      lib.litert_lm_engine_settings_set_num_decode_tokens(settings, args.fixed_decode_tokens)
    self.engine = lib.litert_lm_engine_create(settings)
    lib.litert_lm_engine_settings_delete(settings)
    if not self.engine:
      raise RuntimeError("engine creation failed")
    self.load_ns = time.monotonic_ns() - t0

  # ---------------------------------------------------------------- helpers

  def tokenize(self, text: str, add_bos: bool) -> list[int]:
    lib = self.lib
    res = lib.litert_lm_engine_tokenize(self.engine, text)
    if not res:
      raise RuntimeError("tokenize failed")
    try:
      n = lib.litert_lm_tokenize_result_get_num_tokens(res)
      ptr = lib.litert_lm_tokenize_result_get_tokens(res)
      ids = [ptr[i] for i in range(n)]
    finally:
      lib.litert_lm_tokenize_result_delete(res)
    return ([2] if add_bos else []) + ids

  def _session(self, max_tokens: int, sampling: dict | None = None):
    lib = self.lib
    cfg = lib.litert_lm_session_config_create()
    lib.litert_lm_session_config_set_apply_prompt_template(cfg, False)
    lib.litert_lm_session_config_set_max_output_tokens(cfg, max_tokens)
    sampling = sampling or {}
    seeded = float(sampling.get("temperature", 0.0)) > 0.0
    sp = lib.litert_lm_sampler_params_create(SamplerType.TOP_P)
    # Greedy is top_k = 1; "no top-k" for seeded sampling is the full vocabulary.
    lib.litert_lm_sampler_params_set_top_k(sp, (int(sampling.get("top_k", 0)) or 262144) if seeded else 1)
    lib.litert_lm_sampler_params_set_top_p(sp, float(sampling.get("top_p", 1.0)) if seeded else 1.0)
    lib.litert_lm_sampler_params_set_temperature(sp, float(sampling.get("temperature", 1.0)) if seeded else 1.0)
    lib.litert_lm_sampler_params_set_seed(sp, int(sampling.get("seed", 0)))
    lib.litert_lm_session_config_set_sampler_params(cfg, sp)
    lib.litert_lm_sampler_params_delete(sp)
    sess = lib.litert_lm_engine_create_session(self.engine, cfg)
    lib.litert_lm_session_config_delete(cfg)
    if not sess:
      raise RuntimeError("session creation failed")
    return sess

  def _benchmark_info(self, sess) -> dict:
    lib = self.lib
    info = lib.litert_lm_session_get_benchmark_info(sess)
    if not info:
      return {}
    try:
      npt = lib.litert_lm_benchmark_info_get_num_prefill_turns(info)
      ndt = lib.litert_lm_benchmark_info_get_num_decode_turns(info)
      return {
          "ttft_s": lib.litert_lm_benchmark_info_get_time_to_first_token(info),
          "prefill_tokens": [lib.litert_lm_benchmark_info_get_prefill_token_count_at(info, i)
                             for i in range(npt)],
          "prefill_tok_s": [lib.litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(info, i)
                            for i in range(npt)],
          "decode_tokens": [lib.litert_lm_benchmark_info_get_decode_token_count_at(info, i)
                            for i in range(ndt)],
          "decode_tok_s": [lib.litert_lm_benchmark_info_get_decode_tokens_per_sec_at(info, i)
                           for i in range(ndt)],
      }
    finally:
      lib.litert_lm_benchmark_info_delete(info)

  # ------------------------------------------------------------------- ops

  def generate(self, req: dict) -> dict:
    lib = self.lib
    prompt = req["prompt"]
    max_tokens = int(req.get("max_tokens", 256))
    if req.get("ignore_eos") and self.args.fixed_decode_tokens != max_tokens:
      raise RuntimeError("ignore_eos requires a driver started with "
                         f"--fixed-decode-tokens {max_tokens}")
    sess = self._session(max_tokens, req.get("sampling"))
    events: queue.Queue = queue.Queue()

    def on_chunk(_unused, chunk):
      ts = time.monotonic_ns()
      err = lib.litert_lm_stream_chunk_get_error(chunk)
      if err:
        events.put(("error", ts, err.decode("utf-8", "replace")))
        return
      text = lib.litert_lm_stream_chunk_get_text(chunk)
      final = lib.litert_lm_stream_chunk_is_final(chunk)
      events.put(("final" if final else "chunk", ts,
                  text.decode("utf-8", "replace") if text else ""))

    cb = STREAM_CALLBACK_TYPE(on_chunk)
    encoded = prompt.encode("utf-8")
    try:
      inp = lib.litert_lm_input_data_create(InputDataType.TEXT, encoded, len(encoded))
      inputs = (ctypes.c_void_p * 1)(inp)
      t_req = time.monotonic_ns()
      rc = lib.litert_lm_session_run_prefill(sess, inputs, 1)
      t_prefill_done = time.monotonic_ns()
      lib.litert_lm_input_data_delete(inp)
      if rc != 0:
        raise RuntimeError("prefill failed")
      if lib.litert_lm_session_run_decode_async(sess, cb, None) != 0:
        raise RuntimeError("decode failed to start")
      chunk_ns, pieces, error = [], [], None
      while True:
        kind, ts, payload = events.get()
        if kind == "error":
          # Upstream reports hitting max_output_tokens / context as an error.
          if "Max number of tokens" not in payload and "CANCELLED" not in payload:
            error = payload
          break
        if kind == "chunk":
          chunk_ns.append(ts)
          pieces.append(payload)
        else:
          if payload:
            chunk_ns.append(ts)
            pieces.append(payload)
          break
      t_end = time.monotonic_ns()
      if error:
        raise RuntimeError(error)
      native = self._benchmark_info(sess)
    finally:
      lib.litert_lm_session_delete(sess)

    n_prompt = native.get("prefill_tokens", [None])[0] if native.get("prefill_tokens") else None
    n_gen = native.get("decode_tokens", [len(chunk_ns)])[0] if native.get("decode_tokens") else len(chunk_ns)
    text = "".join(pieces)
    resp = {
        "ok": True,
        "n_prompt": n_prompt,
        "n_gen": n_gen,
        "n_stream_chunks": len(chunk_ns),
        "stopped_on_eos": n_gen < max_tokens,
        "t_req_ns": t_req,
        "t_prefill_done_ns": t_prefill_done,
        "t_first_ns": chunk_ns[0] if chunk_ns else 0,
        "t_end_ns": t_end,
        "token_ns": chunk_ns,
        "native": native,
    }
    if req.get("return_text", True):
      resp["text"] = text
    if req.get("return_prompt_ids"):
      resp["prompt_ids"] = self.tokenize(prompt, add_bos=True)
    return resp

  def serve(self) -> None:
    emit({"event": "ready", "framework": "litert-lm", "load_ns": self.load_ns,
          "info": {"threads": self.args.threads, "ctx": self.args.ctx,
                   "ynnpack": self.args.ynnpack, "cache_dir": self.args.cache_dir,
                   "fixed_decode_tokens": self.args.fixed_decode_tokens,
                   "sampler": "top_p sampler with top_k=1 (greedy)", "pid": os.getpid()}})
    for line in sys.stdin:
      line = line.strip()
      if not line:
        continue
      try:
        req = json.loads(line)
      except json.JSONDecodeError as e:
        emit({"ok": False, "error": f"bad json: {e}"})
        continue
      rid = req.get("id")
      op = req.get("op")
      try:
        if op == "quit":
          emit({"id": rid, "ok": True})
          break
        if op == "tokenize":
          emit({"id": rid, "ok": True,
                "ids": self.tokenize(req["text"], req.get("add_bos", True))})
        elif op == "generate":
          out = self.generate(req)
          out["id"] = rid
          emit(out)
        else:
          emit({"id": rid, "ok": False, "error": f"unknown op: {op}"})
      except Exception as e:  # report and keep serving
        emit({"id": rid, "ok": False, "error": str(e)})
    self.lib.litert_lm_engine_delete(self.engine)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--model", required=True)
  ap.add_argument("--threads", type=int, default=4)
  ap.add_argument("--ctx", type=int, default=4096, help="max_num_tokens (KV cache length)")
  ap.add_argument("--cache-dir", default="", help="XNNPACK weight cache dir ('' = default)")
  ap.add_argument("--ynnpack", action="store_true")
  ap.add_argument("--fixed-decode-tokens", type=int, default=0)
  args = ap.parse_args(argv)
  try:
    driver = Driver(args)
  except Exception as e:
    emit({"event": "error", "error": str(e)})
    return 1
  driver.serve()
  return 0


if __name__ == "__main__":
  sys.exit(main())
