"""JSON-lines ONNX Runtime GenAI CPU driver for Gemma 4 E2B.

The model is a Mobius/ONNX Runtime GenAI package (a model directory containing
genai_config.json). Each request creates a fresh Generator, so prompt state and
the KV cache are not reused. The staged model is treated as immutable: a
temporary directory with symlinked assets supplies a thread-configured
genai_config.json to ONNX Runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

BOS_TOKEN_ID = 2
STOP_IDS = (1, 50, 106)


def emit(obj: dict) -> None:
  sys.stdout.write(json.dumps(obj) + "\n")
  sys.stdout.flush()


def _flatten_ids(ids) -> list[int]:
  values = ids.tolist() if hasattr(ids, "tolist") else list(ids)
  if values and isinstance(values[0], list):
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


def _prepare_runtime_model(model_dir: Path, threads: int):
  """Create a disposable model directory with just its config customized."""
  source_config = model_dir / "genai_config.json"
  if not model_dir.is_dir() or not source_config.is_file():
    raise FileNotFoundError(
        f"{model_dir} must be a staged ONNX Runtime GenAI model directory "
        "containing genai_config.json"
    )

  config = json.loads(source_config.read_text())
  model_config = config.get("model")
  if not isinstance(model_config, dict) or model_config.get("type") != "gemma4":
    raise ValueError(f"{source_config} is not a Gemma 4 GenAI model config")
  if not isinstance(model_config.get("decoder"), dict):
    raise ValueError(f"{source_config} has no Gemma 4 decoder configuration")

  for component in ("decoder", "embedding", "vision", "speech"):
    component_config = model_config.get(component)
    if not isinstance(component_config, dict):
      continue
    session_options = component_config.setdefault("session_options", {})
    if not isinstance(session_options, dict):
      raise ValueError(f"invalid {component}.session_options in {source_config}")
    session_options["intra_op_num_threads"] = threads
    session_options["inter_op_num_threads"] = 1
    session_options["provider_options"] = []

  workspace = tempfile.TemporaryDirectory(prefix="aeb-onnxruntime-")
  runtime_dir = Path(workspace.name)
  try:
    for source in model_dir.iterdir():
      if source.name == "genai_config.json":
        continue
      (runtime_dir / source.name).symlink_to(source.resolve(), target_is_directory=source.is_dir())
    (runtime_dir / "genai_config.json").write_text(json.dumps(config, separators=(",", ":")))
  except Exception:
    workspace.cleanup()
    raise
  return workspace, runtime_dir, config


class Driver:

  def __init__(self, args, og_module=None):
    if args.threads < 1:
      raise ValueError("--threads must be positive")
    if args.ctx < 1:
      raise ValueError("--ctx must be positive")

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_THREAD_LIMIT"] = str(args.threads)

    self.args = args
    self.model_dir = Path(args.model).expanduser().resolve()
    started = time.monotonic_ns()
    self._workspace, runtime_dir, config = _prepare_runtime_model(self.model_dir, args.threads)
    try:
      model_config = config["model"]
      bos_token_id = model_config.get("bos_token_id")
      if isinstance(bos_token_id, bool) or not isinstance(bos_token_id, int):
        raise ValueError(f"{self.model_dir / 'genai_config.json'} has no valid bos_token_id")
      self.bos_token_id = bos_token_id
      eos_ids = model_config.get("eos_token_id", [])
      if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
      self.eos_ids = {int(token) for token in eos_ids}

      if og_module is None:
        import onnxruntime_genai as og_module
      self.og = og_module
      self.model = self.og.Model(str(runtime_dir))
      self.tokenizer = self.og.Tokenizer(self.model)
    except Exception:
      self._workspace.cleanup()
      raise
    self.load_ns = time.monotonic_ns() - started

  def tokenize(self, text: str, add_bos: bool = True) -> list[int]:
    return _normalize_bos(self.tokenizer.encode(text), add_bos, self.bos_token_id)

  def generate(self, req: dict) -> dict:
    import numpy as np

    prompt = req["prompt"]
    max_tokens = int(req.get("max_tokens", 256))
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

    max_length = len(prompt_ids) + max_tokens
    sampling = req.get("sampling") or {}
    temperature = float(sampling.get("temperature", 0.0))
    do_sample = temperature > 0.0
    params = self.og.GeneratorParams(self.model)
    search_options = {
        "do_sample": do_sample,
        "max_length": max_length,
        "min_length": max_length if ignore_eos else 0,
        "top_k": int(sampling.get("top_k", 64)) if do_sample else 1,
        "top_p": float(sampling.get("top_p", 1.0)) if do_sample else 1.0,
        "temperature": temperature if do_sample else 1.0,
    }
    if do_sample and "seed" in sampling:
      search_options["random_seed"] = int(sampling["seed"])
    params.set_search_options(**search_options)

    generator = self.og.Generator(self.model, params)
    generator.append_tokens(np.asarray(prompt_ids, dtype=np.int32))
    t_prefill_done = time.monotonic_ns()

    stop_ids = set(int(token) for token in req.get("stop_ids", STOP_IDS))
    stop_ids.update(self.eos_ids)
    gen_ids: list[int] = []
    token_ns: list[int] = []
    stopped_on_eos = False
    while not generator.is_done() and len(gen_ids) < max_tokens:
      generator.generate_next_token()
      ts = time.monotonic_ns()
      next_tokens = _flatten_ids(generator.get_next_tokens())
      if not next_tokens:
        raise RuntimeError("ONNX Runtime GenAI returned no token for a decode step")
      token = next_tokens[0]
      if token in stop_ids and not ignore_eos:
        stopped_on_eos = True
        break
      gen_ids.append(token)
      token_ns.append(ts)

    t_end = time.monotonic_ns()
    text = self.tokenizer.decode(np.asarray(gen_ids, dtype=np.int32)) \
        if req.get("return_text", True) else None
    resp = {
        "ok": True,
        "n_prompt": len(prompt_ids),
        "n_gen": len(gen_ids),
        "stopped_on_eos": stopped_on_eos,
        "t_req_ns": t_req,
        "t_prefill_done_ns": t_prefill_done,
        "t_first_ns": token_ns[0] if token_ns else 0,
        "t_end_ns": t_end,
        "token_ns": token_ns,
        "gen_ids": gen_ids,
        "native": {"decode_steps": len(token_ns), "ctx": self.args.ctx},
    }
    if text is not None:
      resp["text"] = text
    if req.get("return_prompt_ids"):
      resp["prompt_ids"] = prompt_ids
    return resp

  def serve(self) -> None:
    emit({"event": "ready", "framework": "onnxruntime", "load_ns": self.load_ns,
          "info": {"threads": self.args.threads, "ctx": self.args.ctx,
                   "model": str(self.model_dir), "device": "CPUExecutionProvider",
                   "kv_cache": "fresh per request", "pid": os.getpid()}})
    try:
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
        except Exception as e:
          emit({"id": rid, "ok": False, "error": str(e)})
    finally:
      del self.tokenizer
      del self.model
      self._workspace.cleanup()


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--model", required=True, help="path to a GenAI model directory")
  ap.add_argument("--threads", type=int, default=4)
  ap.add_argument("--ctx", type=int, default=4096, help="maximum prompt + generated tokens")
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
