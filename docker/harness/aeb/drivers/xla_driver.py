#!/usr/bin/env python3
"""OpenXLA (JAX / XLA:CPU) JSON-lines driver for AI Edge Bench.

Executes Gemma 4 E2B on XLA:CPU using the official Google Tunix model
implementation (`tunix.models.gemma4.model.Gemma4` and
`tunix.models.gemma4.params_safetensors.create_model_from_safe_tensors`),
official `safetensors` (`safetensors.safe_open`), and Google Qwix (`qwix`)
`QArray` dispatch on Flax NNX / JAX (`jax.jit`), supporting:
  - Default XLA:CPU backend (`--xla_cpu_experimental_ynn_fusion_type` active,
    using XLA's built-in YNNPACK `__ynn_fusion` kernels).
  - XLA:CPU with Intel oneDNN (`--onednn`, which sets
    `XLA_FLAGS="--xla_cpu_experimental_onednn_custom_call=true
    --xla_cpu_experimental_ynn_fusion_type=invalid
    --xla_disable_hlo_passes=onednn-ops-rewriter"` so contractions are
    lowered to `__onednn$matmul` on AVX-512 / Intel AMX while keeping softmax
    in native XLA fusions).
  - Data type variants (`--dtype`):
      * `bf16`               : native BFloat16 weights and activations
      * `fp16`               : Float16 weights and activations
      * `fp32`               : Float32 weights and activations
      * `int8` / `int4` / `qat-mobile`: pre-quantized checkpoint weights and
        scales (`GemmaQArray`), only when the checkpoint itself contains
        pre-quantized weights and scales (`quantization_config` in `config.json`).
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
  ap.add_argument(
      "--cache-prefix",
      action="store_true",
      help="Enable KV prefix caching across requests",
  )
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
    # NOTE (Why onednn-ops-rewriter is disabled): In jaxlib==0.11.2, XLA:CPU's
    # OneDnnOpsRewriter rewrites BF16 attention softmax into `__onednn$softmax`,
    # which produces corrupted BF16 outputs on AVX512F/AMX_BF16 (max error 0.72).
    # Disabling `onednn-ops-rewriter` keeps softmax in native XLA HLO while
    # `OneDnnContractionRewriter` still lowers all 227 matmuls to `__onednn$matmul`.
    if "--xla_disable_hlo_passes" not in existing:
      flags.append("--xla_disable_hlo_passes=onednn-ops-rewriter")
  xla_flags = " ".join(flags).strip()
  os.environ["XLA_FLAGS"] = xla_flags
  return xla_flags


def _ensure_tunix_namespace() -> None:
  """Register top-level `tunix` package without executing `tunix/__init__.py`.

  NOTE (Why custom code is needed): `tunix/__init__.py` eagerly imports all RL
  training submodules (`Grain`, ` Pathways`, `sglang`, `vllm`, `Pillow`,
  `sentencepiece`), which are not installed in the minimal CPU inference
  container. Stubbing the top-level package namespace lets Python import
  `tunix.models.gemma4` and `tunix.generate.utils` directly.
  """
  if "tunix" in sys.modules:
    return
  spec = importlib.util.find_spec("tunix")
  if spec is None or not spec.submodule_search_locations:
    raise ImportError("google-tunix is not installed")
  pkg = types.ModuleType("tunix")
  pkg.__path__ = list(spec.submodule_search_locations)
  sys.modules["tunix"] = pkg


def _unpack_gemma_int(packed: np.ndarray, width: int) -> np.ndarray:
  """Unpack Gemma offset-binary packed INT2/INT4 or INT8 weights to int8.

  NOTE (Why custom code is needed): Pre-quantized Hugging Face checkpoints with
  `quant_method: "gemma"` (such as `google/gemma-4-E2B-it-qat-mobile-transformers`)
  store sub-byte weights packed into `uint8` tensors using unsigned offset-binary
  encoding:
    - INT2 (ratio 4): 4 values per byte, 2-bit unsigned [0..3] with offset -2 -> [-2..1]
    - INT4 (ratio 2): 2 values per byte, 4-bit unsigned [0..15] with offset -8 -> [-8..7]
    - INT8 (ratio 1): 1 signed int8 value per byte
  Neither `google-tunix` (`params_safetensors`) nor `qwix` includes an unpacker
  for this checkpoint format (upstream only implements it in PyTorch
  `transformers.integrations.gemma_quant`).
  """
  ratio = width // packed.shape[-1]
  if ratio == 1:
    return (
        np.asarray(packed).view(np.int8).copy()
        if packed.dtype == np.uint8
        else packed.astype(np.int8, copy=True)
    )
  if ratio == 2:
    u16 = (packed.astype(np.uint16) & 0x0F) | (
        (packed.astype(np.uint16) & 0xF0) << 4
    )
    return (u16.view(np.int8).reshape(packed.shape[0], -1) - np.int8(8))[
        :, :width
    ]
  if ratio == 4:
    u32 = packed.astype(np.uint32)
    u32 = (
        (u32 & 0x03)
        | ((u32 & 0x0C) << 6)
        | ((u32 & 0x30) << 12)
        | ((u32 & 0xC0) << 18)
    )
    return (u32.view(np.int8).reshape(packed.shape[0], -1) - np.int8(2))[
        :, :width
    ]
  raise ValueError(
      f"Unsupported packed weight ratio {ratio} for shape {packed.shape} -> width {width}"
  )


def _build_tunix_model_config(
    gemma4_model,
    tcfg: dict,
    compute_dtype,
    use_sliding_window_kv_cache: bool = True,
):
  """Construct an unsharded Single-Host CPU `tunix.models.gemma4.model.ModelConfig`.

  NOTE (Why custom config override is needed): `ModelConfig.gemma4_e2b()`
  defaults to TPU multi-axis mesh sharding (`ShardingConfig.get_default_sharding()`)
  and `param_dtype=jnp.float32`. For single-socket CPU execution and `--dtype`
  selection (`bf16`/`fp32`), we zero out the TPU mesh axis names and bind the
  dimensions from the checkpoint's `config.json`.
  """
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
      use_sliding_window_kv_cache=use_sliding_window_kv_cache,
  )


def _patch_tunix_chunked_prefill(gemma4_model, jax, jnp) -> None:
  """Extend `Attention.block` to support multi-token suffix prefill into a warm KV cache.

  NOTE (Why custom code is needed): Upstream `tunix.models.gemma4.model.Attention.block`
  explicitly comments `# Does not support chunked prefill`:
    1. When `seq_len > 1`, upstream writes `cache['v'].at[:, :seq_len, ...].set(value_proj)`
       at index 0 (overwriting any cached prefix) instead of using
       `jax.lax.dynamic_update_slice` at `cache['end_index'][0]`, and does not
       bind `key_proj`/`value_proj` to the updated full cache.
    2. When `seq_len > 1`, upstream slices `attn_mask = attn_mask[..., :seq_len]`,
       preventing suffix queries from attending to cached prefix keys.
    3. Upstream `find_last_one_index` calls `.squeeze(axis=-1)`, which raises a
       `ValueError` when `seq_len > 1`.
  When `cache` contains `"chunked_prefill": True`, this wrapper uses Tunix's
  `dynamic_update_slice` cache update path for `seq_len > 1` and preserves the
  full `[B, seq_len, cache_len]` causal/sliding-window mask.
  """
  if getattr(gemma4_model, "_aeb_chunked_patched", False):
    return

  orig_attn_block = gemma4_model.Attention.block

  def _attn_block(
      self,
      x,
      segment_pos,
      cache,
      attn_mask,
      kv_shared_cache=None,
  ):
    is_chunked = (cache is not None and "chunked_prefill" in cache) or (
        kv_shared_cache is not None
        and kv_shared_cache["k"].shape[1] != x.shape[1]
        and x.shape[1] > 1
    )
    if not is_chunked:
      return orig_attn_block(
          self, x, segment_pos, cache, attn_mask, kv_shared_cache=kv_shared_cache
      )
    x = x.astype(self.config.dtype)
    seq_len = x.shape[1]
    query_proj = self.q_einsum(x)
    query_proj = gemma4_model.shard(query_proj, self.config.shd_config.act_btnh)
    query_proj = self._query_norm(query_proj)
    query_proj = gemma4_model.apply_rope(
        query_proj,
        segment_pos,
        base_frequency=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
        rope_proportion=self.rope_proportion,
    )
    if kv_shared_cache is not None:
      key_proj = kv_shared_cache["k"]
      value_proj = kv_shared_cache["v"]
      new_cache = None
    else:
      if hasattr(self, "k_einsum"):
        key_proj = self.k_einsum(x)
        value_proj = key_proj
      else:
        key_proj, value_proj = self.kv_einsum(x)
      key_proj = gemma4_model.shard(key_proj, self.config.shd_config.act_btnh)
      value_proj = gemma4_model.shard(
          value_proj, self.config.shd_config.act_btnh
      )
      value_var = jnp.mean(jnp.square(value_proj), axis=-1, keepdims=True)
      value_proj = value_proj * jax.lax.rsqrt(value_var + 1e-06)
      key_proj = self._key_norm(key_proj)
      key_proj = gemma4_model.apply_rope(
          key_proj,
          segment_pos,
          base_frequency=self.rope_base_frequency,
          scale_factor=self.rope_scale_factor,
          rope_proportion=self.rope_proportion,
      )
      end_index = cache["end_index"][0]
      slice_indices = (0, end_index, 0, 0)
      value_proj = jax.lax.dynamic_update_slice(
          cache["v"], value_proj, slice_indices
      )
      key_proj = jax.lax.dynamic_update_slice(
          cache["k"], key_proj, slice_indices
      )
      new_cache = {
          "v": value_proj,
          "k": key_proj,
          "end_index": cache["end_index"] + seq_len,
      }

    if self.use_gqa:
      b, t, kg, h = query_proj.shape
      n_groups = kg // self.num_kv_heads
      query_reshaped = query_proj.reshape(
          (b, t, self.num_kv_heads, n_groups, h)
      )
      logits = jnp.einsum("BTKGH,BSKH->BTKGS", query_reshaped, key_proj)
      b, t, k, g, s = logits.shape
      logits = logits.reshape((b, t, k * g, s))
    else:
      logits = jnp.einsum("BTNH,BSNH->BTNS", query_proj, key_proj)

    if self.attn_type == gemma4_model.AttentionType.LOCAL_SLIDING:
      k_idx = jnp.arange(key_proj.shape[1], dtype=jnp.int32)[None, None, :]
      q_idx = jnp.max(jnp.where(attn_mask, k_idx, 0), axis=-1, keepdims=True)
      sliding_mask = (q_idx - k_idx >= 0) & (
          q_idx - k_idx < self.config.sliding_window_size
      )
      attn_mask = sliding_mask & attn_mask

    attn = jnp.where(jnp.expand_dims(attn_mask, -2), logits, gemma4_model.K_MASK)
    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
        key_proj.dtype
    )
    if self.use_gqa:
      b, t, kg, s = attn.shape
      n_groups = kg // self.num_kv_heads
      probs_reshaped = attn.reshape((b, t, self.num_kv_heads, n_groups, s))
      encoded = jnp.einsum("BTKGS,BSKH->BTKGH", probs_reshaped, value_proj)
      b, t, k, g, h = encoded.shape
      encoded = encoded.reshape((b, t, k * g, h))
    else:
      encoded = jnp.einsum("BTNS,BSNH->BTNH", attn, value_proj)

    attn_output = self.attn_vec_einsum(encoded)
    attn_output = gemma4_model.shard(
        attn_output, self.config.shd_config.act_btd
    )
    return new_cache, attn_output, (key_proj, value_proj)

  gemma4_model.Attention.block = _attn_block

  # NOTE (Why M=2 padding is needed for single-token decode on XLA:CPU YNNPACK):
  # In `jaxlib==0.11.2`, XLA:CPU's `IsDotSupportedByYnn` (`xla/backends/cpu/ynn_support.cc:441`)
  # explicitly rejects any dot contraction with `dot_canonical_dims.m == 1` (`return false;`).
  # During single-token decode (`T=1`, `m=1`), rejecting YNNPACK causes XLA's
  # `FloatNormalization` pass to insert `convert(bf16 -> f32)` on all 9.25 GB of
  # model weights on every decode step and fall back to unfused Eigen F32 GEMV
  # (or forces `qwix` to dequantize all INT8 weights to F32), inflating RSS to
  # ~22 GB and slowing decode by ~3.5x–13x. Duplicating the `T=1` axis to `T=2`
  # around `Einsum`, `FeedForward`, and `Embedder.decode` keeps `m == 2` so all
  # 227 weight projections stay in YNNPACK `__ynn_fusion` custom calls.
  #
  # [COMMENTED OUT TO BENCHMARK UNPATCHED XLA/YNNPACK]
  # orig_einsum_call = gemma4_model.Einsum.__call__
  #
  # def _ynn_einsum_call(self, x):
  #   if x.ndim >= 2 and x.shape[1] == 1:
  #     x2 = jnp.concatenate([x, x], axis=1)
  #     out2 = orig_einsum_call(self, x2)
  #     return out2[:, :1, ...] if out2.shape[1] == 2 else out2[:, :, :1, ...]
  #   return orig_einsum_call(self, x)
  #
  # orig_ffw_call = gemma4_model.FeedForward.__call__
  #
  # def _ynn_ffw_call(self, x):
  #   if x.ndim >= 2 and x.shape[1] == 1:
  #     x2 = jnp.concatenate([x, x], axis=1)
  #     return orig_ffw_call(self, x2)[:, :1, ...]
  #   return orig_ffw_call(self, x)
  #
  # orig_embed_decode = gemma4_model.Embedder.decode
  #
  # def _ynn_embed_decode(self, x):
  #   if x.ndim >= 2 and x.shape[1] == 1:
  #     x2 = jnp.concatenate([x, x], axis=1)
  #     return orig_embed_decode(self, x2)[:, :1, ...]
  #   return orig_embed_decode(self, x)
  #
  # gemma4_model.Einsum.__call__ = _ynn_einsum_call
  # gemma4_model.FeedForward.__call__ = _ynn_ffw_call
  # gemma4_model.Embedder.decode = _ynn_embed_decode
  gemma4_model._aeb_chunked_patched = True


def _load_prequantized_tunix_model(
    model_dir: Path,
    model_cfg,
    compute_dtype,
    gemma4_model,
    params_safetensors,
    safetensors_loader,
    flax,
    nnx,
    jax,
    jnp,
    qwix,
    qarray,
):
  """Load pre-quantized integer weights, weight scales, and static activation scales into Tunix/Qwix.

  NOTE (Why custom code is needed):
    1. `tunix.models.gemma4.params_safetensors.create_model_from_safe_tensors`
       only supports unquantized floating-point checkpoints; it fails with shape
       mismatches on `google/gemma-4-E2B-it-qat-mobile-transformers` because the
       checkpoint stores packed `uint8` INT2/INT4 tensors alongside separate
       `.weight_scale`, `.embedding_scale`, `.input_activation_scale`, and
       `.output_activation_scale` tensors.
    2. `qwix.PtqProvider` only implements on-the-fly PTQ from floating-point
       weights (`create_quantized_param`) and dynamic `quantize_act`, and has no
       loader for pre-quantized checkpoint weights or static SRQ activation scales.
  This loader reuses `safetensors.safe_open` and Tunix's own
  `_get_key_and_transform_mapping`, `torch_key_to_jax_key`, `_make_preprocess_fn`,
  and `path_to_key` to populate pre-quantized `GemmaQArray` leaves without running
  any dynamic quantization.
  """
  from safetensors import safe_open

  @flax.struct.dataclass
  class GemmaQArray(qarray.QArray):
    input_scale: jax.Array | None = None
    output_scale: jax.Array | None = None

    def __getitem__(self, idx):
      # NOTE: `flax.nnx.Linear.__call__` accesses `self.kernel[...]`, passing
      # `Ellipsis` into `__getitem__`, which fails in `qarray.rewriting_take`
      # when 0-D `input_scale` / `output_scale` fields are present.
      if idx is Ellipsis:
        return self
      base = qarray.QArray(
          qvalue=self.qvalue,
          scale=self.scale,
          zero_point=self.zero_point,
          qtype=self.qtype,
      )[idx]
      return GemmaQArray(
          qvalue=base.qvalue,
          scale=base.scale,
          zero_point=base.zero_point,
          qtype=base.qtype,
          input_scale=self.input_scale,
          output_scale=self.output_scale,
      )

    def transpose(self, *args):
      return GemmaQArray(
          qvalue=self.qvalue.transpose(*args),
          scale=self.scale.transpose(*args),
          zero_point=(
              self.zero_point.transpose(*args)
              if self.zero_point is not None
              else None
          ),
          qtype=self.qtype,
          input_scale=self.input_scale,
          output_scale=self.output_scale,
      )

    def astype(self, dtype):
      return GemmaQArray(
          qvalue=self.qvalue,
          scale=self.scale.astype(dtype),
          zero_point=self.zero_point,
          qtype=self.qtype,
          input_scale=(
              self.input_scale.astype(dtype)
              if self.input_scale is not None
              else None
          ),
          output_scale=(
              self.output_scale.astype(dtype)
              if self.output_scale is not None
              else None
          ),
      )

  if not getattr(qarray, "_aeb_registered", False):
    nnx.register_data_type(GemmaQArray)
    nnx.register_data_type(qarray.QArray)
    qarray._aeb_registered = True

  def _apply_srq(x, scale, bits: int = 8):
    """Apply static per-tensor INT8 activation rounding (SRQ) from checkpoint scale."""
    s = scale.astype(x.dtype)
    max_val = float((1 << (bits - 1)) - 1)
    min_val = float(-(1 << (bits - 1)))
    return jnp.clip(jnp.round(x / s), min_val, max_val) * s

  if not getattr(gemma4_model, "_aeb_embedder_patched", False):
    # NOTE (Why Embedder patch is needed):
    # 1. Upstream `Embedder.encode` indexes `self.input_embedding.value[(x,)]`
    #    and multiplies by `sqrt(embed_dim)` directly on the array; when the
    #    embedding table is a `QArray`, `qarray.dequantize` must be called on
    #    the gathered rows before scaling.
    # 2. `embed_tokens_per_layer` has shape `[262144, 35, 256]` (2.35 billion
    #    elements). Unpacking all 262k rows upfront adds ~2.35 GB of RAM and ~8s
    #    of startup time; keeping it packed (`uint8` INT4) and unpacking only
    #    the looked-up token IDs `[B, T]` inside `encode_per_layer_input` avoids
    #    materializing the full table.
    orig_encode = gemma4_model.Embedder.encode
    orig_encode_ple = gemma4_model.Embedder.encode_per_layer_input

    def _encode(self, x):
      emb = self.input_embedding.get_value()
      if isinstance(emb, qarray.QArray):
        base_emb = qarray.QArray(qvalue=emb.qvalue, scale=emb.scale)
        out = qarray.dequantize(base_emb[(x,)])
        out *= jnp.sqrt(out.shape[-1]).astype(out.dtype)
        out = jnp.astype(out, self.config.dtype)
        return gemma4_model.shard(out, self.config.shd_config.act_btd)
      return orig_encode(self, x)

    def _encode_ple(self, x, t):
      ple = self.per_layer_input_embedding.get_value()
      if isinstance(ple, tuple):
        t_safe = jnp.where(
            jnp.logical_and(t >= 0, t < self.vocab_size), t, jnp.zeros_like(t)
        )
        x_proj = self.per_layer_model_projection(x)
        x_proj = self.per_layer_projection_norm(x_proj)
        packed_u8, scale = ple
        p = packed_u8[t_safe]
        lo = (p & 0x0F).astype(jnp.int8) - jnp.int8(8)
        hi = ((p >> 4) & 0x0F).astype(jnp.int8) - jnp.int8(8)
        unpacked = jnp.stack([lo, hi], axis=-1).reshape(
            *t_safe.shape,
            self.config.num_layers,
            self.config.per_layer_input_dim,
        )
        s = scale[t_safe]
        y = unpacked.astype(self.config.dtype) * s.astype(self.config.dtype)
        y *= jnp.sqrt(self.config.per_layer_input_dim).astype(y.dtype)
        return (x_proj + y) * jax.lax.rsqrt(2.0).astype(x_proj.dtype)
      return orig_encode_ple(self, x, t)

    gemma4_model.Embedder.encode = _encode
    gemma4_model.Embedder.encode_per_layer_input = _encode_ple
    gemma4_model._aeb_embedder_patched = True

  st_files = sorted(model_dir.glob("*.safetensors"))
  if not st_files:
    raise ValueError(f"No safetensors files found in {model_dir}")

  readers = [safe_open(str(p), framework="flax") for p in st_files]
  key_to_reader = {}
  for r in readers:
    for k in r.keys():
      key_to_reader[k] = r

  def _raw_tensor(name: str) -> np.ndarray:
    return np.asarray(key_to_reader[name].get_tensor(name))

  def _f32_tensor(name: str) -> np.ndarray:
    return np.asarray(key_to_reader[name].get_tensor(name).astype(jnp.float32))

  abs_model = nnx.eval_shape(
      lambda: gemma4_model.Gemma4(model_cfg, rngs=nnx.Rngs(params=0))
  )
  for i, layer in enumerate(abs_model.layers):
    if abs_model.kv_cache_sharing_patterns[i] != i:
      layer.attn.kv_einsum = None
      layer.attn._key_norm = None

  key_map = params_safetensors._get_key_and_transform_mapping(model_cfg)
  preprocess_fn = params_safetensors._make_preprocess_fn(model_cfg)

  qval_dict = {}
  scale_dict = {}
  in_act_dict = {}
  out_act_dict = {}
  fp_dict = {}

  for h_key in key_to_reader:
    if not h_key.startswith("model.language_model."):
      continue
    if h_key.endswith(".embedding_quantized"):
      s_key = h_key[: -len(".embedding_quantized")] + ".embedding_scale"
      if "embed_tokens_per_layer" in h_key:
        qval_dict["embedder.per_layer_input_embedding"] = np.asarray(
            _raw_tensor(h_key), dtype=np.uint8
        )
        scale_dict["embedder.per_layer_input_embedding"] = _f32_tensor(
            s_key
        ).reshape(model_cfg.num_embed, model_cfg.num_layers, 1)
      else:
        qval_dict["embedder.input_embedding"] = _unpack_gemma_int(
            _raw_tensor(h_key), model_cfg.embed_dim
        )
        scale_dict["embedder.input_embedding"] = _f32_tensor(s_key).reshape(
            model_cfg.num_embed, 1
        )
      continue
    if h_key.endswith((
        ".embedding_scale",
        ".weight_scale",
        ".input_activation_scale",
        ".output_activation_scale",
        ".k_cache_scale",
        ".v_cache_scale",
    )):
      continue
    try:
      jax_key, transform = safetensors_loader.torch_key_to_jax_key(
          key_map, h_key
      )
    except ValueError:
      continue
    prefix = h_key[: -len(".weight")] if h_key.endswith(".weight") else ""
    s_key = f"{prefix}.weight_scale" if prefix else ""
    if s_key and s_key in key_to_reader:
      raw_w = _raw_tensor(h_key)
      raw_s = _f32_tensor(s_key)
      width = raw_w.shape[1]
      if transform is not None:
        permute, reshape = transform
        if reshape is not None:
          width = int(np.prod(reshape)) // raw_w.shape[0]
        else:
          cur = abs_model
          for p in jax_key.split("."):
            cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
          width = cur.shape[0] if permute == (1, 0) else cur.shape[1]
      w_int8 = _unpack_gemma_int(raw_w, width)
      if transform is not None:
        permute, reshape = transform
        if permute:
          w_int8 = w_int8.transpose(permute)
          raw_s = raw_s.transpose(permute)
        if reshape:
          w_int8 = w_int8.reshape(reshape)
          s_reshape = list(reshape)
          if len(reshape) == 3 and reshape[0] == width:
            s_reshape[0] = 1
          elif len(reshape) == 3 and reshape[0] * reshape[1] == width:
            s_reshape[0] = 1
            s_reshape[1] = 1
          raw_s = raw_s.reshape(s_reshape)
      qval_dict[jax_key] = w_int8
      scale_dict[jax_key] = raw_s
      in_s_key = f"{prefix}.input_activation_scale"
      out_s_key = f"{prefix}.output_activation_scale"
      if in_s_key in key_to_reader:
        val = float(_f32_tensor(in_s_key).flat[0])
        if val != 0.0:
          in_act_dict[jax_key] = np.asarray(val, dtype=np.float32)
      if out_s_key in key_to_reader:
        val = float(_f32_tensor(out_s_key).flat[0])
        if val != 0.0:
          out_act_dict[jax_key] = np.asarray(val, dtype=np.float32)
    else:
      arr = _f32_tensor(h_key)
      if transform is not None:
        permute, reshape = transform
        if permute:
          arr = arr.transpose(permute)
        if reshape:
          arr = arr.reshape(reshape)
      fp_dict[jax_key] = arr

  qval_dict = preprocess_fn(qval_dict)
  for layer_id in range(model_cfg.num_layers):
    q_k = f"tmp.layers.{layer_id}.attn.q"
    if q_k in scale_dict:
      target_q = f"layers.{layer_id}.attn.q_einsum.w"
      scale_dict[target_q] = scale_dict.pop(q_k).transpose((1, 0, 2))
      if q_k in in_act_dict:
        in_act_dict[target_q] = in_act_dict.pop(q_k)
      if q_k in out_act_dict:
        out_act_dict[target_q] = out_act_dict.pop(q_k)
    k_k = f"tmp.layers.{layer_id}.attn.k"
    v_k = f"tmp.layers.{layer_id}.attn.v"
    if k_k in scale_dict and v_k in scale_dict:
      target_kv = f"layers.{layer_id}.attn.kv_einsum.w"
      ks = scale_dict.pop(k_k).transpose((1, 0, 2))
      vs = scale_dict.pop(v_k).transpose((1, 0, 2))
      scale_dict[target_kv] = np.stack([ks, vs], axis=0)
      if k_k in in_act_dict:
        in_act_dict[target_kv] = in_act_dict.pop(k_k)
        in_act_dict.pop(v_k, None)
      if k_k in out_act_dict:
        out_act_dict[target_kv] = out_act_dict.pop(k_k)
        out_act_dict.pop(v_k, None)

  graph_def, abs_state = nnx.split(abs_model)
  state_dict = abs_state.to_pure_dict()
  device = jax.devices()[0]

  def _populate(path, _):
    key = safetensors_loader.path_to_key(path)
    for k in (key, f"{key}.value"):
      if k == "embedder.per_layer_input_embedding":
        return (
            jax.device_put(qval_dict[k], device),
            jax.device_put(scale_dict[k].astype(compute_dtype), device),
        )
      if k in qval_dict:
        in_s = (
            jax.device_put(in_act_dict[k].astype(compute_dtype), device)
            if k in in_act_dict
            else None
        )
        out_s = (
            jax.device_put(out_act_dict[k].astype(compute_dtype), device)
            if k in out_act_dict
            else None
        )
        return GemmaQArray(
            qvalue=jax.device_put(qval_dict[k], device),
            scale=jax.device_put(scale_dict[k].astype(compute_dtype), device),
            input_scale=in_s,
            output_scale=out_s,
        )
      if k in fp_dict:
        return jax.device_put(fp_dict[k].astype(compute_dtype), device)
    raise KeyError(f"Missing parameter in pre-quantized checkpoint: {key}")

  state_dict = jax.tree.map_with_path(_populate, state_dict)
  model = nnx.merge(graph_def, state_dict)

  class _PreQuantizedProvider(qwix.PtqProvider):
    """Subclass of `qwix.PtqProvider` for already-quantized `GemmaQArray` weights.

    NOTE (Why custom provider subclass is needed): `qwix.PtqProvider` with empty
    rules `[]` leaves `dot_general` and `einsum` in full precision unless the
    RHS is a `qarray.QArray`, and does not apply the checkpoint's static
    `input_activation_scale` / `output_activation_scale`. This subclass delegates
    the actual quantized contraction to `PtqProvider._dot_general_fn` and
    `PtqProvider._einsum_fn` while applying static SRQ scales when present on
    the checkpoint's `GemmaQArray` leaf.
    """

    def get_intercept_map(self):
      m = dict(super().get_intercept_map())
      m["jax.numpy.astype"] = (
          lambda a, dtype, **kw: self.asarray(a, dtype=dtype, **kw)
      )
      return m

    def _unwrap_srq(self, lhs, rhs):
      in_scale = getattr(rhs, "input_scale", None)
      out_scale = getattr(rhs, "output_scale", None)
      if in_scale is not None and not isinstance(lhs, qarray.QArray):
        # NOTE (Why static SRQ is wrapped into QArray): When `lhs` is left as a
        # floating-point array, `qwix._src.core.dot_general` selects
        # `_slow_dot_general` (dequantizing the entire INT8 weight matrix to
        # BF16 on every forward pass because XLA:CPU YNNPACK does not support
        # mixed `{BF16, S8}` dots). Wrapping the statically quantized activation
        # into `QArray(int8)` with the checkpoint's `input_activation_scale`
        # enables `_fast_dot_general` (`s8 x s8 -> s32` in `__ynn_fusion`).
        #
        # [COMMENTED OUT TO BENCHMARK UNPATCHED XLA/YNNPACK]
        # s = in_scale.astype(lhs.dtype)
        # q_lhs = jnp.clip(jnp.round(lhs / s), -128.0, 127.0).astype(jnp.int8)
        # lhs = qarray.QArray(
        #     qvalue=q_lhs,
        #     scale=jnp.reshape(s, (1,) * lhs.ndim),
        #     zero_point=None,
        #     qtype=jnp.int8,
        # )
        lhs = _apply_srq(lhs, in_scale)
      if isinstance(rhs, GemmaQArray):
        rhs = qarray.QArray(
            qvalue=rhs.qvalue,
            scale=rhs.scale,
            zero_point=rhs.zero_point,
            qtype=rhs.qtype,
        )
      if isinstance(lhs, GemmaQArray):
        lhs = qarray.QArray(
            qvalue=lhs.qvalue,
            scale=lhs.scale,
            zero_point=lhs.zero_point,
            qtype=lhs.qtype,
        )
      return lhs, rhs, out_scale

    def dot_general(
        self,
        lhs,
        rhs,
        dimension_numbers,
        precision=None,
        preferred_element_type=None,
        *,
        out_sharding=None,
    ):
      lhs, rhs, out_scale = self._unwrap_srq(lhs, rhs)
      if isinstance(lhs, qarray.QArray) or isinstance(rhs, qarray.QArray):
        out = self._dot_general_fn(
            lhs, rhs, dimension_numbers, out_sharding=out_sharding
        )
      else:
        out = jax.lax.dot_general(
            lhs,
            rhs,
            dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
            out_sharding=out_sharding,
        )
      return _apply_srq(out, out_scale) if out_scale is not None else out

    def einsum(
        self,
        einsum_str,
        *operands,
        precision=None,
        preferred_element_type=None,
        _dot_general=jax.lax.dot_general,
        out_sharding=None,
    ):
      if len(operands) == 2:
        lhs, rhs, out_scale = self._unwrap_srq(operands[0], operands[1])
        if isinstance(lhs, qarray.QArray) or isinstance(rhs, qarray.QArray):
          out = self._einsum_fn(einsum_str, lhs, rhs)
          return _apply_srq(out, out_scale) if out_scale is not None else out
      return jax.numpy.einsum(
          einsum_str,
          *operands,
          precision=precision,
          preferred_element_type=preferred_element_type,
          _dot_general=_dot_general,
          out_sharding=out_sharding,
      )

  provider = _PreQuantizedProvider([])
  sample_tokens = jnp.ones((1, 4), dtype=jnp.int32)
  sample_pos = jnp.arange(4, dtype=jnp.int32)[None, :]
  sample_mask = jnp.tril(jnp.ones((1, 4, 4), dtype=jnp.bool_))
  return qwix.quantize_model(
      model,
      provider,
      sample_tokens,
      positions=sample_pos,
      cache=None,
      attention_mask=sample_mask,
      decode_only_last_token=True,
  )


class XLAGemma4Engine:
  """Compiled JAX / XLA:CPU engine for Gemma 4 using Tunix and Qwix."""

  def __init__(self, args):
    _ensure_tunix_namespace()
    import flax
    from flax import nnx
    import jax
    import jax.numpy as jnp
    import qwix
    from qwix._src.core import qarray
    from tokenizers import Tokenizer
    from tunix.generate import utils as gen_utils
    from tunix.models import safetensors_loader
    from tunix.models.gemma4 import model as gemma4_model
    from tunix.models.gemma4 import params_safetensors

    _patch_tunix_chunked_prefill(gemma4_model, jax, jnp)
    self.jax = jax
    self.jnp = jnp
    self.nnx = nnx
    self.gen_utils = gen_utils
    self.gemma4_model = gemma4_model
    self.args = args
    self._cache_prefix = bool(getattr(args, "cache_prefix", False))
    self._cached_prefix_key: tuple[int, ...] | None = None
    self._cached_prefix_kv = None
    self._cached_prefix_bucket: int = 0
    self._cached_prefix_pad_len: int = 0
    self.model_dir = Path(args.model)
    if args.cache_dir:
      Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
      jax.config.update("jax_compilation_cache_dir", str(args.cache_dir))

    self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
    cfg_raw = json.loads((self.model_dir / "config.json").read_text())
    self.quant_cfg = cfg_raw.get("quantization_config")
    self.tcfg = cfg_raw.get("text_config", cfg_raw)
    self.max_seq_len = int(args.ctx)

    if args.dtype == "fp32":
      self.compute_dtype = jnp.float32
    elif args.dtype == "fp16":
      self.compute_dtype = jnp.float16
    else:
      self.compute_dtype = jnp.bfloat16

    self.model_cfg = _build_tunix_model_config(
        gemma4_model,
        self.tcfg,
        self.compute_dtype,
        use_sliding_window_kv_cache=not self._cache_prefix,
    )

    if args.onednn and args.dtype != "fp32":
      raise ValueError(
          f"XLA oneDNN (--onednn) is only supported with --dtype=fp32 "
          f"(got --dtype={args.dtype}); jaxlib==0.11.2 oneDNN segfaults on "
          f"Intel AMX BF16 / QArray contractions."
      )

    if self.quant_cfg:
      self.quantizer_name = f"checkpoint-{self.quant_cfg.get('quant_method', 'quantized')}"
      model = _load_prequantized_tunix_model(
          self.model_dir,
          self.model_cfg,
          self.compute_dtype,
          gemma4_model,
          params_safetensors,
          safetensors_loader,
          flax,
          nnx,
          jax,
          jnp,
          qwix,
          qarray,
      )
    else:
      if args.dtype in ("int8", "int4", "qat-mobile"):
        raise ValueError(
            f"Checkpoint {self.model_dir} does not contain pre-quantized weights "
            f"(quantization_config is missing); refusing to run dynamic "
            f"quantization for --dtype={args.dtype}."
        )
      self.quantizer_name = "none"
      model = params_safetensors.create_model_from_safe_tensors(
          str(self.model_dir),
          self.model_cfg,
          dtype=self.compute_dtype,
          mode="original",
      )
      for i, layer in enumerate(model.layers):
        if model.kv_cache_sharing_patterns[i] != i:
          layer.attn.kv_einsum = None
          layer.attn._key_norm = None

    self.graphdef, self.state = nnx.split(model)
    del model
    import gc
    gc.collect()
    self._build_jitted_fns()

  def _build_jitted_fns(self) -> None:
    """Compile separate prefill, suffix-prefill, and per-token decode functions.

    NOTE (Why `tunix.generate.sampler.Sampler` is not used):
    `tunix.generate.sampler.Sampler` compiles the entire decode loop into a
    single `jax.lax.while_loop` (`_compiled_decode_fn`) that only returns to
    Python after all `max_generation_steps` tokens have finished. That makes it
    impossible to record per-token monotonic arrival timestamps (`token_ns[i]`),
    TTFT (`t_first_ns`), or inter-token latency (ITL) required by the benchmark
    protocol (`docs/benchmark_methodology.md`). Compiling `prefill_fn` and
    `decode_step_fn` separately with `jax.jit` (with `donate_argnums=(2,)` for
    in-place KV cache updates) runs the exact same Tunix `Gemma4` forward pass
    while exposing each generated token to the timing loop.
    """
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

    def chunked_prefill_fn(
        graphdef,
        state,
        cache,
        suffix_tokens,
        start_pos,
        pad_len,
        start_cache_idx,
    ):
      model = nnx.merge(graphdef, state)
      seq_len = suffix_tokens.shape[1]
      positions = (start_pos + jnp.arange(seq_len, dtype=jnp.int32))[None, :]
      q_idx = (start_cache_idx + jnp.arange(seq_len, dtype=jnp.int32))[
          None, :, None
      ]
      k_idx = jnp.arange(max_seq_len, dtype=jnp.int32)[None, None, :]
      causal_mask = (k_idx >= pad_len) & (k_idx <= q_idx)
      tagged_cache = {
          k: dict(v, chunked_prefill=True) for k, v in cache.items()
      }
      logits, new_cache = model(
          suffix_tokens,
          positions=positions,
          cache=tagged_cache,
          attention_mask=causal_mask,
          decode_only_last_token=True,
      )
      return logits[0, 0, :].astype(jnp.float32), new_cache

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
    self._suffix_prefill_jit = jax.jit(chunked_prefill_fn)
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

  def _left_pad_to_bucket(self, token_ids: list[int]) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Left-pad token IDs to a power-of-2 bucket using `tunix.generate.utils`."""
    bucket = next(
        (b for b in PREFILL_BUCKETS if b >= len(token_ids)), self.max_seq_len
    )
    pad_len = bucket - len(token_ids)
    padded = self.gen_utils.pad_to_length(
        np.asarray(token_ids, dtype=np.int32),
        target_length=bucket,
        pad_value=0,
        left=True,
    )[None, :]
    in_mask = (np.arange(bucket) >= pad_len)[None, :]
    return padded, in_mask, bucket, pad_len

  def generate(self, req: dict) -> dict:
    if "prompt_ids" in req and req["prompt_ids"] is not None:
      prompt_ids = [int(x) for x in req["prompt_ids"]]
    else:
      prompt_ids = self.tokenize(req.get("prompt", ""), add_bos=True)

    max_tokens = int(req.get("max_tokens", 256))
    ignore_eos = bool(req.get("ignore_eos", False))
    return_text = bool(req.get("return_text", True))
    stop_ids = set(int(t) for t in (req.get("stop_ids") or [1, 50, 106]))
    sampling = req.get("sampling")
    use_cache = bool(req.get("cache_prefix", self._cache_prefix))
    prefix_tokens = int(req.get("prefix_tokens", 0))
    rng = None
    if sampling and sampling.get("temperature", 0.0) > 0.0:
      seed = int(sampling.get("seed", 1234))
      rng = np.random.default_rng(seed)

    L = len(prompt_ids)
    n_cached = 0

    if use_cache and 0 < prefix_tokens < L:
      prefix_ids = prompt_ids[:prefix_tokens]
      prefix_key = tuple(prefix_ids)
      if self._cached_prefix_key != prefix_key or self._cached_prefix_kv is None:
        padded_prefix, in_mask, prefix_bucket, prefix_pad_len = (
            self._left_pad_to_bucket(prefix_ids)
        )
        pref_logits_jax, pref_kv = self._prefill_jit(
            self.graphdef,
            self.state,
            self.jnp.asarray(padded_prefix),
            self.jnp.asarray(in_mask),
        )
        _ = np.asarray(pref_logits_jax)
        self._cached_prefix_key = prefix_key
        self._cached_prefix_kv = pref_kv
        self._cached_prefix_bucket = prefix_bucket
        self._cached_prefix_pad_len = prefix_pad_len

      suffix_ids = prompt_ids[prefix_tokens:]
      suffix_len = len(suffix_ids)
      suffix_arr = np.asarray(suffix_ids, dtype=np.int32)[None, :]

      t_req_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
      first_logits_jax, kv_caches = self._suffix_prefill_jit(
          self.graphdef,
          self.state,
          self._cached_prefix_kv,
          self.jnp.asarray(suffix_arr),
          self.jnp.int32(prefix_tokens),
          self.jnp.int32(self._cached_prefix_pad_len),
          self.jnp.int32(self._cached_prefix_bucket),
      )
      first_logits = np.asarray(first_logits_jax)
      next_id = self._sample_token(first_logits, sampling, rng)
      t_first_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
      pad_len = self._cached_prefix_pad_len
      decode_base_cache_idx = self._cached_prefix_bucket + suffix_len
      n_cached = prefix_tokens
    else:
      t_req_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
      padded, in_mask, bucket, pad_len = self._left_pad_to_bucket(prompt_ids)

      first_logits_jax, kv_caches = self._prefill_jit(
          self.graphdef,
          self.state,
          self.jnp.asarray(padded),
          self.jnp.asarray(in_mask),
      )
      first_logits = np.asarray(first_logits_jax)
      next_id = self._sample_token(first_logits, sampling, rng)
      t_first_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
      decode_base_cache_idx = bucket

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
            self.jnp.int32(decode_base_cache_idx + step),
        )
        step_logits_np = np.asarray(step_logits)
        next_id = self._sample_token(step_logits_np, sampling, rng)
        t_tok_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        if not ignore_eos and next_id in stop_ids:
          stopped_on_eos = True
          break
        token_ns.append(t_tok_ns)
        gen_ids.append(next_id)

    del kv_caches
    t_end_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    resp = {
        "ok": True,
        "n_prompt": L,
        "n_cached": n_cached,
        "n_prefilled": L - n_cached,
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
              "quantizer": engine.quantizer_name,
              "xla_flags": xla_flags,
              "threads": args.threads,
              "ctx": args.ctx,
              "cache_prefix": engine._cache_prefix,
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
