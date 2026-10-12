#!/usr/bin/env python3
"""OpenXLA (JAX / XLA:CPU) JSON-lines driver for AI Edge Bench.

Executes Gemma 4 E2B on XLA:CPU using the official Google Tunix model
implementation (`tunix.models.gemma4.model.Gemma4` and
`tunix.models.gemma4.params_safetensors.create_model_from_safe_tensors`) and
Google Qwix (`qwix`) quantization on Flax NNX / JAX (`jax.jit`), supporting:
  - Default XLA:CPU backend (`--xla_cpu_experimental_ynn_fusion_type` active,
    using XLA's built-in YNNPACK `__ynn_fusion` kernels).
  - XLA:CPU with Intel oneDNN (`--onednn`, which sets
    `XLA_FLAGS="--xla_cpu_experimental_onednn_custom_call=true
    --xla_cpu_experimental_ynn_fusion_type=invalid"` so contractions are
    lowered to `__onednn$matmul` on AVX-512 / Intel AMX).
  - Data type variants (`--dtype`):
      * `bf16`               : native BFloat16 weights and activations
      * `fp16`               : Float16 weights and activations
      * `fp32`               : Float32 weights and activations
      * `int8`               : Qwix INT8 quantization (`weight_qtype=jnp.int8`)
      * `int4` / `qat-mobile`: Qwix INT4 quantization (`weight_qtype=jnp.int4`)
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
import numpy as np


PREFILL_BUCKETS = (16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
QUANTIZED_MODULE_REGEX = (
    r".*(q_einsum|kv_einsum|k_einsum|attn_vec_einsum|"
    r"gate_proj|up_proj|down_proj|"
    r"per_layer_input_gate|per_layer_projection)"
)


def _parse_args(argv=None):
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", required=True, help="Path to HF safetensors model directory")
  ap.add_argument("--threads", type=int, default=8, help="Intra-op CPU thread count")
  ap.add_argument(
      "--dtype",
      choices=["bf16", "fp16", "fp32", "int8", "int4", "qat-mobile"],
      default="bf16",
      help="Weight / compute data type variant",
  )
  ap.add_argument(
      "--onednn",
      action="store_true",
      help="Enable XLA oneDNN custom calls and disable YNNPACK fusion",
  )
  ap.add_argument("--ctx", type=int, default=4096, help="Maximum KV cache sequence length")
  ap.add_argument("--cache-dir", default="", help="Optional JAX compilation cache directory")
  return ap.parse_args(argv)


def _configure_xla_env(args) -> str:
  """Configure CPU affinity and XLA_FLAGS before importing JAX."""
  if args.threads > 0:
    try:
      avail = sorted(os.sched_getaffinity(0))
      if len(avail) > args.threads:
        os.sched_setaffinity(0, set(avail[:args.threads]))
    except OSError:
      pass
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["NPROC"] = str(args.threads)

  existing = os.environ.get("XLA_FLAGS", "").strip()
  flags = [existing] if existing else []
  if "--xla_cpu_multi_thread_eigen" not in existing:
    flags.append("--xla_cpu_multi_thread_eigen=true")
  if args.onednn:
    if "--xla_cpu_experimental_onednn_custom_call" not in existing:
      flags.append("--xla_cpu_experimental_onednn_custom_call=true")
    if "--xla_cpu_experimental_ynn_fusion_type" not in existing:
      flags.append("--xla_cpu_experimental_ynn_fusion_type=invalid")
  xla_flags = " ".join(flags).strip()
  os.environ["XLA_FLAGS"] = xla_flags
  return xla_flags


def _ensure_tunix_namespace() -> None:
  """Register top-level `tunix` package without importing RL trainer extras."""
  if "tunix" in sys.modules:
    return
  spec = importlib.util.find_spec("tunix")
  if spec is None or not spec.submodule_search_locations:
    raise ImportError("google-tunix is not installed")
  pkg = types.ModuleType("tunix")
  pkg.__path__ = list(spec.submodule_search_locations)
  sys.modules["tunix"] = pkg


def _resolve_safetensors_dir(model_dir: Path) -> Path:
  """Resolve a standard Hugging Face safetensors directory for Tunix loader."""
  cfg_path = model_dir / "config.json"
  if cfg_path.is_file():
    cfg_raw = json.loads(cfg_path.read_text())
    if "quantization_config" in cfg_raw:
      sibling = model_dir.parent / "gemma-4-E2B-it"
      if (sibling / "config.json").is_file():
        return sibling
  return model_dir


def _build_tunix_model_config(gemma4_model, tcfg: dict, compute_dtype):
  """Construct an unsharded Tunix Gemma 4 ModelConfig from HF config.json."""
  default_shd = gemma4_model.ShardingConfig.get_default_sharding()
  shd_cfg = gemma4_model.ShardingConfig(**{
      f.name: (
          tuple(None for _ in getattr(default_shd, f.name))
          if isinstance(getattr(default_shd, f.name), tuple)
          else None
      )
      for f in dataclasses.fields(gemma4_model.ShardingConfig)
  })
  base = gemma4_model.ModelConfig.gemma4_e2b(sharding_config=shd_cfg)
  num_layers = int(tcfg.get("num_hidden_layers", base.num_layers))
  num_kv_shared = int(
      tcfg.get("num_kv_shared_layers", round(base.frac_shared_layers * num_layers))
  )
  frac_shared = float(num_kv_shared) / float(num_layers) if num_layers > 0 else 0.0
  hidden_dim = int(tcfg.get("intermediate_size", base.hidden_dim))
  use_double_wide_mlp = bool(tcfg.get("use_double_wide_mlp", True))
  override_ffw = hidden_dim * 2 if (use_double_wide_mlp and num_kv_shared > 0) else None
  return dataclasses.replace(
      base,
      num_layers=num_layers,
      num_embed=int(tcfg.get("vocab_size", base.num_embed)),
      embed_dim=int(tcfg.get("hidden_size", base.embed_dim)),
      hidden_dim=hidden_dim,
      num_heads=int(tcfg.get("num_attention_heads", base.num_heads)),
      head_dim=int(tcfg.get("head_dim", base.head_dim)),
      global_key_size=int(tcfg.get("global_head_dim", base.global_key_size)),
      num_kv_heads=int(tcfg.get("num_key_value_heads", base.num_kv_heads)),
      sliding_window_size=int(tcfg.get("sliding_window", base.sliding_window_size)),
      per_layer_input_dim=int(
          tcfg.get("hidden_size_per_layer_input", base.per_layer_input_dim)
      ),
      frac_shared_layers=frac_shared,
      override_kv_shared_ffw_hidden=override_ffw,
      final_logit_softcap=float(
          tcfg.get("final_logit_softcapping", base.final_logit_softcap)
      ),
      param_dtype=compute_dtype,
      dtype=compute_dtype,
  )


class XLAGemma4Engine:
  """Compiled JAX / XLA:CPU engine for Gemma 4 using Tunix and Qwix."""

  def __init__(self, args):
    _ensure_tunix_namespace()
    from flax import nnx
    import jax
    import jax.numpy as jnp
    import qwix
    from tokenizers import Tokenizer
    from tunix.generate import utils as gen_utils
    from tunix.models.gemma4 import model as gemma4_model
    from tunix.models.gemma4 import params_safetensors

    self.jax = jax
    self.jnp = jnp
    self.nnx = nnx
    self.gen_utils = gen_utils
    self.args = args
    self.model_dir = _resolve_safetensors_dir(Path(args.model))
    if args.cache_dir:
      Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
      jax.config.update("jax_compilation_cache_dir", str(args.cache_dir))

    self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
    cfg_raw = json.loads((self.model_dir / "config.json").read_text())
    self.tcfg = cfg_raw.get("text_config", cfg_raw)
    self.max_seq_len = int(args.ctx)

    if args.dtype == "fp32":
      self.compute_dtype = jnp.float32
    elif args.dtype == "fp16":
      self.compute_dtype = jnp.float16
    else:
      self.compute_dtype = jnp.bfloat16

    self.model_cfg = _build_tunix_model_config(
        gemma4_model, self.tcfg, self.compute_dtype
    )
    model = params_safetensors.create_model_from_safe_tensors(
        str(self.model_dir),
        self.model_cfg,
        dtype=self.compute_dtype,
        mode="original",
    )

    # Clear unused KV modules on KV-shared layers (layers that reuse an earlier
    # layer's K/V projections and therefore have no k_proj/v_proj in the checkpoint).
    for i, layer in enumerate(model.layers):
      if model.kv_cache_sharing_patterns[i] != i:
        layer.attn.kv_einsum = None
        layer.attn._key_norm = None

    if args.dtype in ("int8", "int4", "qat-mobile"):
      weight_qtype = jnp.int8 if args.dtype == "int8" else jnp.int4
      act_qtype = jnp.int8 if not args.onednn else None

      class _TunixPtqProvider(qwix.PtqProvider):
        """PtqProvider that also intercepts `jnp.astype` used by Tunix Einsum."""

        def get_intercept_map(self):
          m = dict(super().get_intercept_map())
          m["jax.numpy.astype"] = (
              lambda a, dtype, **kw: self.asarray(a, dtype=dtype, **kw)
          )
          return m

      provider = _TunixPtqProvider([
          qwix.QuantizationRule(
              module_path=QUANTIZED_MODULE_REGEX,
              weight_qtype=weight_qtype,
              act_qtype=act_qtype,
          )
      ])
      sample_tokens = jnp.ones((1, 4), dtype=jnp.int32)
      sample_pos = jnp.arange(4, dtype=jnp.int32)[None, :]
      sample_mask = jnp.tril(jnp.ones((1, 4, 4), dtype=jnp.bool_))
      model = qwix.quantize_model(
          model,
          provider,
          sample_tokens,
          positions=sample_pos,
          cache=None,
          attention_mask=sample_mask,
          decode_only_last_token=True,
      )

    self.graphdef, self.state = nnx.split(model)
    self._build_jitted_fns()

  def _build_jitted_fns(self) -> None:
    jax = self.jax
    jnp = self.jnp
    nnx = self.nnx
    gen_utils = self.gen_utils
    max_seq_len = self.max_seq_len
    compute_dtype = self.compute_dtype

    def prefill_fn(graphdef, state, padded_tokens, in_mask):
      model = nnx.merge(graphdef, state)
      cache = model.init_cache(
          batch_size=1, max_seq_len=max_seq_len, dtype=compute_dtype
      )
      positions = gen_utils.build_positions_from_mask(in_mask)
      attn_mask = gen_utils.make_causal_attn_mask(in_mask, max_seq_len)
      logits, cache = model(
          padded_tokens,
          positions=positions,
          cache=cache,
          attention_mask=attn_mask,
          decode_only_last_token=True,
      )
      return logits[0, 0, :].astype(jnp.float32), cache

    def decode_step_fn(graphdef, state, cache, token_2d, pos_scalar, pad_len, cur_cache_idx):
      model = nnx.merge(graphdef, state)
      positions = jnp.reshape(pos_scalar, (1, 1))
      idx = jnp.arange(max_seq_len, dtype=jnp.int32)
      attn_mask = ((idx >= pad_len) & (idx <= cur_cache_idx))[None, None, :]
      logits, cache = model(
          token_2d,
          positions=positions,
          cache=cache,
          attention_mask=attn_mask,
          decode_only_last_token=True,
      )
      return logits[0, 0, :].astype(jnp.float32), cache

    self._prefill_jit = jax.jit(prefill_fn)
    self._decode_jit = jax.jit(decode_step_fn, donate_argnums=(2,))

  def tokenize(self, text: str, add_bos: bool = True) -> list[int]:
    ids = list(self.tokenizer.encode(text, add_special_tokens=False).ids)
    if add_bos and (not ids or ids[0] != 2):
      ids = [2] + ids
    return ids

  def detokenize(self, ids: list[int]) -> str:
    return self.tokenizer.decode(ids, skip_special_tokens=True)

  def _sample_token(
      self,
      logits_np: np.ndarray,
      sampling: dict | None,
      rng: np.random.Generator | None,
  ) -> int:
    if not sampling or sampling.get("temperature", 0.0) == 0.0:
      return int(np.argmax(logits_np))
    temp = float(sampling.get("temperature", 1.0))
    top_k = int(sampling.get("top_k", 64))
    top_p = float(sampling.get("top_p", 0.95))
    scaled = logits_np.astype(np.float64) / temp
    if 0 < top_k < scaled.shape[0]:
      top_idx = np.argpartition(scaled, -top_k)[-top_k:]
      top_vals = scaled[top_idx]
      order = np.argsort(-top_vals)
      top_idx = top_idx[order]
      top_vals = top_vals[order]
    else:
      top_idx = np.argsort(-scaled)
      top_vals = scaled[top_idx]
    probs = np.exp(top_vals - top_vals[0])
    probs /= probs.sum()
    if 0.0 < top_p < 1.0:
      csum = np.cumsum(probs)
      keep = np.searchsorted(csum, top_p, side="left") + 1
      top_idx = top_idx[:keep]
      probs = probs[:keep]
      probs /= probs.sum()
    return int(rng.choice(top_idx, p=probs))

  def generate(self, req: dict) -> dict:
    t_req_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    if "prompt_ids" in req and req["prompt_ids"] is not None:
      prompt_ids = [int(x) for x in req["prompt_ids"]]
    else:
      prompt_ids = self.tokenize(req.get("prompt", ""), add_bos=True)

    max_tokens = int(req.get("max_tokens", 256))
    ignore_eos = bool(req.get("ignore_eos", False))
    return_text = bool(req.get("return_text", True))
    stop_ids = set(int(t) for t in (req.get("stop_ids") or [1, 50, 106]))
    sampling = req.get("sampling")
    rng = None
    if sampling and sampling.get("temperature", 0.0) > 0.0:
      seed = int(sampling.get("seed", 1234))
      rng = np.random.default_rng(seed)

    L = len(prompt_ids)
    bucket = next((b for b in PREFILL_BUCKETS if b >= L), self.max_seq_len)
    pad_len = bucket - L
    padded = np.zeros((1, bucket), dtype=np.int32)
    padded[0, pad_len:] = prompt_ids
    in_mask = np.zeros((1, bucket), dtype=np.bool_)
    in_mask[0, pad_len:] = True

    first_logits_jax, kv_caches = self._prefill_jit(
        self.graphdef,
        self.state,
        self.jnp.asarray(padded),
        self.jnp.asarray(in_mask),
    )
    first_logits = np.asarray(first_logits_jax)
    next_id = self._sample_token(first_logits, sampling, rng)
    t_first_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    token_ns: list[int] = []
    gen_ids: list[int] = []
    stopped_on_eos = False

    if not ignore_eos and next_id in stop_ids:
      stopped_on_eos = True
    else:
      token_ns.append(t_first_ns)
      gen_ids.append(next_id)
      pad_len_jax = self.jnp.int32(pad_len)
      for step in range(max_tokens - 1):
        cur_arr = self.jnp.array([[next_id]], dtype=self.jnp.int32)
        step_logits, kv_caches = self._decode_jit(
            self.graphdef,
            self.state,
            kv_caches,
            cur_arr,
            self.jnp.int32(L + step),
            pad_len_jax,
            self.jnp.int32(bucket + step),
        )
        step_logits_np = np.asarray(step_logits)
        next_id = self._sample_token(step_logits_np, sampling, rng)
        t_tok_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        if not ignore_eos and next_id in stop_ids:
          stopped_on_eos = True
          break
        token_ns.append(t_tok_ns)
        gen_ids.append(next_id)

    t_end_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    resp = {
        "ok": True,
        "n_prompt": L,
        "n_gen": len(gen_ids),
        "t_req_ns": t_req_ns,
        "t_prefill_done_ns": t_first_ns,
        "t_first_ns": t_first_ns if token_ns else 0,
        "t_end_ns": t_end_ns,
        "token_ns": token_ns,
        "gen_ids": gen_ids,
        "stopped_on_eos": stopped_on_eos,
    }
    if return_text:
      resp["text"] = self.detokenize(gen_ids)
    if req.get("return_prompt_ids"):
      resp["prompt_ids"] = prompt_ids
    return resp


def main(argv=None) -> int:
  args = _parse_args(argv)
  xla_flags = _configure_xla_env(args)
  t0 = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
  try:
    engine = XLAGemma4Engine(args)
  except Exception as e:  # pylint: disable=broad-except
    print(json.dumps({"event": "error", "error": f"load failed: {e}"}), flush=True)
    return 1
  load_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC) - t0

  backend_name = "onednn" if args.onednn else "ynnpack"
  print(
      json.dumps({
          "event": "ready",
          "framework": "xla",
          "load_ns": load_ns,
          "load_time_s": load_ns / 1e9,
          "info": {
              "backend": backend_name,
              "dtype": args.dtype,
              "model_impl": "tunix.models.gemma4.model.Gemma4",
              "quantizer": "qwix" if args.dtype in ("int8", "int4", "qat-mobile") else "none",
              "xla_flags": xla_flags,
              "threads": args.threads,
              "ctx": args.ctx,
              "model": str(engine.model_dir),
              "pid": os.getpid(),
          },
      }),
      flush=True,
  )

  for line in sys.stdin:
    line = line.strip()
    if not line:
      continue
    try:
      req = json.loads(line)
    except json.JSONDecodeError as e:
      print(json.dumps({"ok": False, "error": f"bad json: {e}"}), flush=True)
      continue
    rid = req.get("id")
    op = req.get("op")
    if op == "quit":
      print(json.dumps({"id": rid, "ok": True}), flush=True)
      return 0
    if op == "tokenize":
      ids = engine.tokenize(req.get("text", ""), add_bos=bool(req.get("add_bos", True)))
      print(json.dumps({"id": rid, "ok": True, "ids": ids, "n": len(ids)}), flush=True)
      continue
    if op == "generate":
      try:
        resp = engine.generate(req)
        resp["id"] = rid
      except Exception as e:  # pylint: disable=broad-except
        resp = {"id": rid, "ok": False, "error": str(e)}
      print(json.dumps(resp), flush=True)
      continue
    print(json.dumps({"id": rid, "ok": False, "error": f"unknown op: {op}"}), flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
