"""JSON-lines OpenVINO GenAI CPU driver for Gemma 4 E2B.

Uses openvino_genai.LLMPipeline on the CPU device with explicit thread count
and single-stream execution (NUM_STREAMS=1, INFERENCE_NUM_THREADS=threads).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

BOS_TOKEN_ID = 2
STOP_IDS = (1, 50, 106)


def emit(obj: dict) -> None:
  sys.stdout.write(json.dumps(obj) + "\n")
  sys.stdout.flush()


def _flatten_ids(ids) -> list[int]:
  values = ids.tolist() if hasattr(ids, "tolist") else list(ids)
  while values and isinstance(values[0], list):
    if len(values) != 1:
      raise ValueError("expected a single token sequence")
    values = values[0]
  return [int(token) for token in values]


def _normalize_bos(ids, add_bos: bool, bos_token_id: int = BOS_TOKEN_ID) -> list[int]:
  tokens = _flatten_ids(ids)
  if tokens and tokens[0] == bos_token_id:
    tokens = tokens[1:]
  if add_bos:
    tokens.insert(0, bos_token_id)
  return tokens


class Driver:

  def __init__(self, args):
    if args.threads < 1:
      raise ValueError("--threads must be positive")
    if args.ctx < 1:
      raise ValueError("--ctx must be positive")

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_THREAD_LIMIT"] = str(args.threads)

    self.args = args
    self.model_dir = Path(args.model).expanduser().resolve()
    if not self.model_dir.is_dir():
      raise FileNotFoundError(f"{self.model_dir} must be an OpenVINO model directory")

    config_path = self.model_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    text_cfg = config.get("text_config", config)
    self.bos_token_id = int(text_cfg.get("bos_token_id", BOS_TOKEN_ID))
    eos_ids = text_cfg.get("eos_token_id", [1])
    if isinstance(eos_ids, int):
      eos_ids = [eos_ids]
    self.eos_ids = {int(t) for t in eos_ids}

    import openvino as ov
    import openvino_genai as ov_genai

    self.ov = ov
    self.ov_genai = ov_genai

    ov_kwargs = {
        "INFERENCE_NUM_THREADS": args.threads,
        "NUM_STREAMS": "1",
    }
    if getattr(args, "dtype", None) == "fp32":
      ov_kwargs["INFERENCE_PRECISION_HINT"] = "f32"
    elif getattr(args, "dtype", None) == "bf16":
      ov_kwargs["INFERENCE_PRECISION_HINT"] = "bf16"
    elif getattr(args, "dtype", None) == "fp16":
      ov_kwargs["INFERENCE_PRECISION_HINT"] = "f16"

    started = time.monotonic_ns()
    if (self.model_dir / "openvino_vision_embeddings_model.xml").is_file() or (
        self.model_dir / "openvino_text_embeddings_per_layer_model.xml"
    ).is_file():
      self.pipe = ov_genai.VLMPipeline(
          str(self.model_dir),
          "CPU",
          **ov_kwargs,
      )
      self._is_vlm = True
    else:
      self.pipe = ov_genai.LLMPipeline(
          str(self.model_dir),
          "CPU",
          **ov_kwargs,
      )
      self._is_vlm = False
    self.tokenizer = self.pipe.get_tokenizer()
    self.load_ns = time.monotonic_ns() - started

  def tokenize(self, text: str, add_bos: bool = True) -> list[int]:
    encoded = self.tokenizer.encode(text, add_special_tokens=True)
    return _normalize_bos(encoded.input_ids.data, add_bos, self.bos_token_id)

  def generate(self, req: dict) -> dict:
    import numpy as np

    prompt = req["prompt"]
    max_tokens = int(req.get("max_tokens", req.get("n_predict", 256)))
    if max_tokens < 1:
      raise ValueError("max_tokens must be positive")
    ignore_eos = bool(req.get("ignore_eos", False))

    t_req = time.monotonic_ns()
    prompt_ids = self.tokenize(prompt, add_bos=True)
    if len(prompt_ids) + max_tokens > self.args.ctx:
      raise ValueError(
          f"prompt ({len(prompt_ids)}) + max_tokens ({max_tokens}) exceeds "
          f"context (--ctx {self.args.ctx})"
      )

    stop_ids = set(int(token) for token in req.get("stop_ids", STOP_IDS))
    stop_ids.update(self.eos_ids)

    sampling = req.get("sampling") or {}
    temperature = float(sampling.get("temperature", 0.0))
    do_sample = temperature > 0.0

    cfg = self.ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_tokens
    cfg.apply_chat_template = False
    cfg.ignore_eos = ignore_eos
    if ignore_eos:
      cfg.min_new_tokens = max_tokens
    else:
      cfg.stop_token_ids = stop_ids

    if do_sample:
      cfg.do_sample = True
      cfg.temperature = temperature
      cfg.top_k = int(sampling.get("top_k", 64))
      cfg.top_p = float(sampling.get("top_p", 1.0))
      if "seed" in sampling and sampling["seed"] is not None:
        cfg.rng_seed = int(sampling["seed"])
    else:
      cfg.do_sample = False

    ov_genai = self.ov_genai
    gen_ids: list[int] = []
    token_ns: list[int] = []
    stopped_on_eos = False

    class _Streamer(ov_genai.StreamerBase):

      def __init__(self):
        super().__init__()

      def write(self, token):
        nonlocal stopped_on_eos
        ts = time.monotonic_ns()
        tokens = [int(token)] if isinstance(token, int) else [int(t) for t in token]
        for tok in tokens:
          if not ignore_eos and tok in stop_ids:
            stopped_on_eos = True
            return ov_genai.StreamingStatus.STOP
          if len(gen_ids) < max_tokens:
            gen_ids.append(tok)
            token_ns.append(ts)
          if len(gen_ids) >= max_tokens:
            return ov_genai.StreamingStatus.STOP
        return ov_genai.StreamingStatus.RUNNING

      def end(self):
        pass

    streamer = _Streamer()
    if self._is_vlm:
      bos_str = self.tokenizer.get_bos_token() or "<bos>"
      vlm_prompt = prompt if prompt.startswith(bos_str) else (bos_str + prompt)
      res = self.pipe.generate(vlm_prompt, generation_config=cfg, streamer=streamer)
    else:
      input_tensor = self.ov.Tensor(np.asarray([prompt_ids], dtype=np.int64))
      res = self.pipe.generate(input_tensor, cfg, streamer)
    t_end = time.monotonic_ns()

    if not gen_ids and hasattr(res, "tokens") and res.tokens:
      raw_tokens = _flatten_ids(res.tokens[0])
      for tok in raw_tokens:
        if not ignore_eos and tok in stop_ids:
          stopped_on_eos = True
          break
        gen_ids.append(tok)
      if hasattr(res, "perf_metrics") and res.perf_metrics.raw_metrics.m_new_token_times:
        # Reconstruct monotonic timestamps relative to t_end if streamer was bypassed
        times_ms = list(res.perf_metrics.raw_metrics.m_new_token_times)
        if times_ms:
          last_ms = times_ms[-1]
          token_ns = [int(t_end - (last_ms - tm) * 1e6) for tm in times_ms[:len(gen_ids)]]

    t_first = token_ns[0] if token_ns else t_end
    text = (
        self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        if req.get("return_text", True)
        else None
    )
    native = {"decode_steps": len(token_ns), "ctx": self.args.ctx}
    if hasattr(res, "perf_metrics"):
      try:
        pm = res.perf_metrics
        native["ttft_ms"] = float(pm.get_ttft().mean)
        native["tpot_ms"] = float(pm.get_tpot().mean)
        native["throughput_tps"] = float(pm.get_throughput().mean)
      except Exception:
        pass

    resp = {
        "ok": True,
        "n_prompt": len(prompt_ids),
        "n_gen": len(gen_ids),
        "stopped_on_eos": stopped_on_eos,
        "t_req_ns": t_req,
        "t_prefill_done_ns": t_first,
        "t_first_ns": t_first if token_ns else 0,
        "t_end_ns": t_end,
        "token_ns": token_ns,
        "gen_ids": gen_ids,
        "native": native,
    }
    if text is not None:
      resp["text"] = text
    if req.get("return_prompt_ids"):
      resp["prompt_ids"] = prompt_ids
    return resp

  def serve(self) -> None:
    emit({
        "event": "ready",
        "framework": "openvino",
        "load_ns": self.load_ns,
        "info": {
            "threads": self.args.threads,
            "ctx": self.args.ctx,
            "dtype": getattr(self.args, "dtype", "auto"),
            "model": str(self.model_dir),
            "device": "CPU",
            "kv_cache": "fresh per request",
            "pid": os.getpid(),
        },
    })
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
          emit({
              "id": rid,
              "ok": True,
              "ids": self.tokenize(req["text"], req.get("add_bos", True)),
          })
        elif op == "generate":
          out = self.generate(req)
          out["id"] = rid
          emit(out)
        else:
          emit({"id": rid, "ok": False, "error": f"unknown op: {op}"})
      except Exception as e:
        emit({"id": rid, "ok": False, "error": str(e)})


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("--model", required=True, help="path to an OpenVINO model directory")
  ap.add_argument("--threads", type=int, default=4)
  ap.add_argument("--ctx", type=int, default=4096, help="maximum prompt + generated tokens")
  ap.add_argument(
      "--dtype",
      default="auto",
      choices=["auto", "fp32", "bf16", "fp16", "int8", "int4"],
      help="inference precision hint for OpenVINO CPU plugin",
  )
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
