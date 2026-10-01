"""Build a GGUF whose weights are bit-identical to the LiteRT-LM release.

The litert-community `gemma-4-E2B-it.litertlm` file is compiled from Google's
mobile QAT checkpoint (`google/gemma-4-E2B-it-qat-mobile-transformers`). That
checkpoint stores per-output-channel symmetric integer weights:

  * INT2: token embedding, lm_head and the MLPs of layers 15-34
  * INT4: attention projections, MLPs of layers 0-14, per-layer embeddings
  * INT8: per-layer input gate / projection

The stock `ggml-org` Q4_0 GGUF comes from a *different* QAT checkpoint (the
Q4_0 one), so comparing it with LiteRT-LM compares two quantization schemes.
This script re-encodes the mobile checkpoint's integers and scales into GGUF
block types that represent them exactly:

  * INT4 per-row scale s -> Q4_0, every 32-block uses d = s (nibble = q + 8)
  * INT8 per-row scale s -> Q8_0, every 32-block uses d = s
  * INT2 per-row scale s -> Q2_K, d = dmin = s, every 16-sub-block uses
    scale = 1 and min = 2, so x = s * v - 2 * s = s * q for v = q + 2

The only loss is storing each fp32 scale as fp16 (reported by --verify).

--int2-as q4_0 stores the INT2 tensors as Q4_0 instead (nibble = q + 8 with
q in [-2, 1]). The dequantized values are identical to the Q2_K encoding, but
llama.cpp's faster Q4_0 kernels (and Arm weight repacking) apply, at the cost
of ~1.9 extra bits per INT2 weight. This isolates kernel coverage from
numerics: both files compute with exactly the same weights.
Non-quantized tensors (norms, layer scalars) are copied from the checkpoint as
F32. Metadata, tokenizer and rope_freqs come from the stock GGUF so that the
architecture hyper-parameters stay exactly what llama.cpp expects.

`per_layer_model_projection` is BF16 in the checkpoint but INT8 per-channel in
the .litertlm (the LiteRT converter quantizes it post-training). With
--litert-tflite the exact INT8 values and scales are lifted from the
.litertlm's prefill/decode TFLite section; otherwise it is quantized here with
symmetric per-row INT8, which may differ from LiteRT by 1 LSB on ties.

usage:
  python3 -m aeb.prep.convert_qat_mobile_gguf \\
      --qat-dir /models/google/gemma-4-E2B-it-qat-mobile-transformers \\
      --template /models/ggml-org/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_0.gguf \\
      --out /models/aeb/gemma-4-E2B-it-qat-mobile-matched.gguf \\
      [--litert-tflite <unpacked>/Section10_TFLiteModel_tf_lite_prefill_decode.tflite] \\
      [--verify] [--report report.json]
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
import gguf
from gguf import GGMLQuantizationType as QT

HF = "model.language_model."


class SafeTensors:
  """Minimal zero-copy safetensors reader (avoids a torch dependency)."""

  _DT = {"F32": np.float32, "F16": np.float16, "I8": np.int8, "U8": np.uint8,
         "I32": np.int32, "BF16": np.uint16}

  def __init__(self, path: Path):
    with open(path, "rb") as f:
      n = struct.unpack("<Q", f.read(8))[0]
      self.header = json.loads(f.read(n))
    self.base = 8 + n
    self.mm = np.memmap(path, dtype=np.uint8, mode="r")

  def raw(self, name: str) -> np.ndarray:
    h = self.header[name]
    a, b = h["data_offsets"]
    return self.mm[self.base + a:self.base + b].view(self._DT[h["dtype"]]).reshape(h["shape"])

  def f32(self, name: str) -> np.ndarray:
    arr = self.raw(name)
    if self.header[name]["dtype"] == "BF16":
      return (arr.astype(np.uint32) << 16).view(np.float32)
    return arr.astype(np.float32)


def unpack(packed: np.ndarray, bits: int, width: int) -> np.ndarray:
  """Unpack the checkpoint's offset-binary layout (transformers
  integrations/gemma_quant.py): low bits first, value - 2**(bits-1)."""
  if bits == 8:
    return np.asarray(packed).view(np.int8) if packed.dtype == np.uint8 else packed.astype(np.int8)
  p = np.asarray(packed, dtype=np.uint8)
  mask = (1 << bits) - 1
  parts = [((p >> (bits * i)) & mask).astype(np.int8) - (1 << (bits - 1))
           for i in range(8 // bits)]
  return np.stack(parts, axis=-1).reshape(*p.shape[:-1], -1)[..., :width]


def _fp16(s: np.ndarray) -> np.ndarray:
  return np.asarray(s, dtype=np.float32).astype(np.float16)


def encode_q4_0(q: np.ndarray, d: np.ndarray) -> np.ndarray:
  """q: int8 [R, K] in [-8, 7]; d: fp32 [R, K/32] block scales."""
  r, k = q.shape
  nib = (q.reshape(r, k // 32, 32) + 8).astype(np.uint8)
  out = np.empty((r, k // 32, 18), dtype=np.uint8)
  out[..., :2] = _fp16(d).reshape(r, k // 32, 1).view(np.uint8)
  out[..., 2:] = nib[..., :16] | (nib[..., 16:] << 4)
  return out.reshape(r, -1)


def encode_q8_0(q: np.ndarray, d: np.ndarray) -> np.ndarray:
  r, k = q.shape
  out = np.empty((r, k // 32, 34), dtype=np.uint8)
  out[..., :2] = _fp16(d).reshape(r, k // 32, 1).view(np.uint8)
  out[..., 2:] = q.reshape(r, k // 32, 32).astype(np.int8).view(np.uint8)
  return out.reshape(r, -1)


def encode_q2_k(q: np.ndarray, s: np.ndarray) -> np.ndarray:
  """q: int8 [R, K] in [-2, 1]; s: fp32 [R] row scale; K % 256 == 0.

  block_q2_K = { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin } (84 bytes).
  Element n*128 + j*32 + l (l < 32) lives in qs[n*32 + l] at bit 2*j.
  """
  r, k = q.shape
  nb = k // 256
  v = (q.reshape(r, nb, 2, 4, 32) + 2).astype(np.uint8)
  qs = (v[:, :, :, 0] | (v[:, :, :, 1] << 2) | (v[:, :, :, 2] << 4) |
        (v[:, :, :, 3] << 6)).reshape(r, nb, 64)
  out = np.empty((r, nb, 84), dtype=np.uint8)
  out[..., :16] = 0x21  # sub-block scale = 1 (low nibble), min = 2 (high nibble)
  out[..., 16:80] = qs
  dd = np.repeat(_fp16(s).reshape(r, 1, 1), 2, axis=2)  # d == dmin == s
  out[..., 80:84] = np.ascontiguousarray(np.broadcast_to(dd, (r, nb, 2))).view(np.uint8)
  return out.reshape(r, -1)


def tflite_int8_tensor(path: Path, name_regex: str):
  """(int8 values, per-row fp32 scales, name) of the first INT8 constant whose
  name matches name_regex in a .tflite file (supports >2 GB offset buffers)."""
  import tflite  # optional dependency, only needed for --litert-tflite

  data = path.read_bytes()
  m = tflite.Model.GetRootAsModel(data, 0)
  pat = re.compile(name_regex)
  for si in range(m.SubgraphsLength()):
    sg = m.Subgraphs(si)
    for ti in range(sg.TensorsLength()):
      t = sg.Tensors(ti)
      if t.Type() != tflite.TensorType.INT8 or not pat.search(t.Name().decode()):
        continue
      b = m.Buffers(t.Buffer())
      if b.Offset() > 1:
        raw = np.frombuffer(data, dtype=np.int8, count=b.Size(), offset=b.Offset())
      elif b.DataLength() > 0:
        raw = b.DataAsNumpy().view(np.int8)
      else:
        continue
      q = raw.reshape(tuple(t.ShapeAsNumpy()))
      qp = t.Quantization()
      assert np.all(qp.ZeroPointAsNumpy() == 0), "expected symmetric quantization"
      return q.copy(), qp.ScaleAsNumpy().astype(np.float32), t.Name().decode()
  raise KeyError(name_regex)


def build_plan(n_layer: int, n_kv_shared: int) -> dict:
  """GGUF tensor name -> (kind, source[, bits])."""
  plan = {
      "token_embd.weight": ("emb", HF + "embed_tokens", 2),
      "output.weight": ("lin", "lm_head", 2),
      "per_layer_token_embd.weight": ("emb", HF + "embed_tokens_per_layer", 4),
      "per_layer_model_proj.weight": ("ptq8", HF + "per_layer_model_projection.weight"),
      "per_layer_proj_norm.weight": ("f32", HF + "per_layer_projection_norm.weight"),
      "output_norm.weight": ("f32", HF + "norm.weight"),
  }
  first_shared = n_layer - n_kv_shared
  for i in range(n_layer):
    L, b = f"{HF}layers.{i}.", f"blk.{i}."
    mlp_bits = 4 if i < 15 else 2
    plan[b + "attn_norm.weight"] = ("f32", L + "input_layernorm.weight")
    plan[b + "attn_q.weight"] = ("lin", L + "self_attn.q_proj", 4)
    plan[b + "attn_q_norm.weight"] = ("f32", L + "self_attn.q_norm.weight")
    if i < first_shared:
      plan[b + "attn_k.weight"] = ("lin", L + "self_attn.k_proj", 4)
      plan[b + "attn_v.weight"] = ("lin", L + "self_attn.v_proj", 4)
      plan[b + "attn_k_norm.weight"] = ("f32", L + "self_attn.k_norm.weight")
    plan[b + "attn_output.weight"] = ("lin", L + "self_attn.o_proj", 4)
    plan[b + "post_attention_norm.weight"] = ("f32", L + "post_attention_layernorm.weight")
    plan[b + "ffn_norm.weight"] = ("f32", L + "pre_feedforward_layernorm.weight")
    plan[b + "ffn_gate.weight"] = ("lin", L + "mlp.gate_proj", mlp_bits)
    plan[b + "ffn_up.weight"] = ("lin", L + "mlp.up_proj", mlp_bits)
    plan[b + "ffn_down.weight"] = ("lin", L + "mlp.down_proj", mlp_bits)
    plan[b + "post_ffw_norm.weight"] = ("f32", L + "post_feedforward_layernorm.weight")
    plan[b + "inp_gate.weight"] = ("lin", L + "per_layer_input_gate", 8)
    plan[b + "proj.weight"] = ("lin", L + "per_layer_projection", 8)
    plan[b + "post_norm.weight"] = ("f32", L + "post_per_layer_input_norm.weight")
    plan[b + "layer_output_scale.weight"] = ("f32", L + "layer_scalar")
  return plan


class LazyInts:
  """Row-sliceable view that unpacks packed integers on demand."""

  def __init__(self, packed: np.ndarray, bits: int, width: int):
    self.packed, self.bits, self.width = packed, bits, width
    self.shape = (packed.shape[0], width)

  def rows(self, r0: int, r1: int) -> np.ndarray:
    return unpack(self.packed[r0:r1], self.bits, self.width)

  def __getitem__(self, sl: slice) -> np.ndarray:
    r0, r1, _ = sl.indices(self.shape[0])
    return self.rows(r0, r1)


class EagerInts(LazyInts):
  def __init__(self, q: np.ndarray):
    self.q, self.shape = q, q.shape

  def rows(self, r0: int, r1: int) -> np.ndarray:
    return self.q[r0:r1]


def lazy_ints_and_scales(st: SafeTensors, kind: str, src: str, bits: int, width: int):
  if kind == "lin":
    return LazyInts(st.raw(src + ".weight"), bits, width), st.f32(src + ".weight_scale").reshape(-1, 1)
  return (LazyInts(st.raw(src + ".embedding_quantized"), bits, width),
          st.f32(src + ".embedding_scale"))


INT2_AS = "q2_k"


def encode(q: np.ndarray, s: np.ndarray, bits: int):
  chunk = q.shape[1] // s.shape[1]
  if bits == 2 and INT2_AS == "q2_k":
    assert s.shape[1] == 1
    return encode_q2_k(q, s[:, 0]), QT.Q2_K
  d = np.repeat(s, chunk // 32, axis=1)
  return (encode_q4_0(q, d), QT.Q4_0) if bits in (2, 4) else (encode_q8_0(q, d), QT.Q8_0)


ROWS_PER_CHUNK = 8192  # bounds peak memory for the 262144-row embedding tables


def verify(name, data, qt, q, s, report):
  chunk = q.shape[1] // s.shape[1]
  ints_ok, err = True, 0.0
  for r0 in range(0, q.shape[0], ROWS_PER_CHUNK):
    qq, ss = q[r0:r0 + ROWS_PER_CHUNK], s[r0:r0 + ROWS_PER_CHUNK]
    deq = gguf.quants.dequantize(data[r0:r0 + ROWS_PER_CHUNK], qt).reshape(qq.shape)
    s16 = np.repeat(_fp16(ss).astype(np.float32), chunk, axis=1)
    ints_ok &= bool(np.array_equal(deq, qq.astype(np.float32) * s16))
    exact = qq.astype(np.float32) * np.repeat(ss, chunk, axis=1)
    nz = exact != 0
    if nz.any():
      err = max(err, float((np.abs(deq - exact)[nz] / np.abs(exact[nz])).max()))
  report.update(max_rel_err_vs_fp32_scale=err, ints_roundtrip_exact=ints_ok)
  print(f"  verify {name:36s} {qt.name:5s} ints_exact={ints_ok} max_rel_err={err:.2e}", flush=True)
  if not ints_ok:
    raise SystemExit(f"integer round-trip failed for {name}")


def encode_chunked(load_rows, n_rows: int, s: np.ndarray, bits: int):
  """Encode row chunks; load_rows(r0, r1) returns the int8 rows."""
  parts, qt = [], None
  for r0 in range(0, n_rows, ROWS_PER_CHUNK):
    r1 = min(n_rows, r0 + ROWS_PER_CHUNK)
    data, qt = encode(load_rows(r0, r1), s[r0:r1], bits)
    parts.append(data)
  return np.concatenate(parts, axis=0), qt


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--qat-dir", type=Path, required=True)
  ap.add_argument("--template", type=Path, required=True,
                  help="stock gemma-4-E2B-it GGUF supplying metadata/tokenizer/rope_freqs")
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--litert-tflite", type=Path,
                  help="prefill/decode .tflite unpacked from the .litertlm; lifts "
                       "per_layer_model_projection INT8 exactly")
  ap.add_argument("--verify", action="store_true",
                  help="dequantize every quantized tensor and check the integers round-trip")
  ap.add_argument("--report", type=Path, help="write a JSON conversion report here")
  ap.add_argument("--int2-as", choices=["q2_k", "q4_0"], default="q2_k",
                  help="GGUF block type for INT2 tensors (both are lossless)")
  args = ap.parse_args(argv)
  global INT2_AS
  INT2_AS = args.int2_as

  st = SafeTensors(args.qat_dir / "model.safetensors")
  reader = gguf.GGUFReader(args.template)
  assert reader.get_field("general.architecture").contents() == "gemma4"
  n_layer = int(reader.get_field("gemma4.block_count").contents())
  n_shared = int(reader.get_field("gemma4.attention.shared_kv_layers").contents())
  plan = build_plan(n_layer, n_shared)

  tmpl = {t.name: t for t in reader.tensors}
  missing = [n for n in tmpl if n not in plan and n != "rope_freqs.weight"]
  if missing:
    raise SystemExit(f"template tensors without a source: {missing}")

  writer = gguf.GGUFWriter(args.out, "gemma4")
  for field in reader.fields.values():
    if field.name == "general.architecture" or field.name.startswith("GGUF."):
      continue
    vtype = field.types[0]
    sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
    val = field.contents()
    if field.name == "general.name":
      val = "gemma-4-E2B-it-qat-mobile-matched"
    writer.add_key_value(field.name, val, vtype, sub_type=sub)
  writer.add_key_value("aeb.source_checkpoint",
                       "google/gemma-4-E2B-it-qat-mobile-transformers", gguf.GGUFValueType.STRING)
  writer.add_key_value("aeb.encoding",
                       f"per-channel int2/int4/int8 -> {args.int2_as.upper()}/Q4_0/Q8_0 "
                       "with per-row fp16 scales", gguf.GGUFValueType.STRING)

  order = list(tmpl)
  order.insert(order.index("token_embd.weight") + 1, "output.weight")
  report: dict = {"tensors": {}}
  worst_scale_rel = 0.0

  for name in order:
    if name == "rope_freqs.weight":
      writer.add_tensor(name, np.array(tmpl[name].data, dtype=np.float32))
      report["tensors"][name] = {"type": "F32", "source": "template (architecture constant)"}
      continue
    kind, src, *rest = plan[name]
    ref = tmpl.get(name)
    entry = {"source": src}
    if kind == "f32":
      arr = st.f32(src).reshape(-1).astype(np.float32)
      if ref is not None:
        assert arr.size == int(np.prod(ref.shape)), (name, arr.shape, ref.shape)
        entry["template_type"] = ref.tensor_type.name
      writer.add_tensor(name, arr)
      entry["type"] = "F32"
      report["tensors"][name] = entry
      continue

    if kind == "ptq8":
      w = st.f32(src)
      if args.litert_tflite:
        q, s, tname = tflite_int8_tensor(args.litert_tflite,
                                         r"per_layer_model_projection/.*dot_general")
        s = s.reshape(-1, 1)
        assert q.shape == w.shape, (q.shape, w.shape)
        entry.update(int8_source=f"litert tflite: {tname}",
                     rel_diff_vs_bf16=float(np.abs(q * s - w).max() / np.abs(w).max()))
      else:
        s = np.abs(w).max(axis=1, keepdims=True) / 127.0
        q = np.clip(np.rint(w / s), -127, 127).astype(np.int8)
        entry["int8_source"] = "per-row symmetric int8 PTQ (no --litert-tflite)"
      q = EagerInts(q)
      bits = 8
    else:
      bits = rest[0]
      width = int((ref if ref is not None else tmpl["token_embd.weight"]).shape[0])
      q, s = lazy_ints_and_scales(st, kind, src, bits, width)
    if ref is not None:
      assert q.shape == (int(ref.shape[1]), int(ref.shape[0])), (name, q.shape, ref.shape)
      entry["template_type"] = ref.tensor_type.name
    data, qt = encode_chunked(q.rows, q.shape[0], s, bits)
    rel = np.abs(_fp16(s).astype(np.float32) - s) / np.maximum(np.abs(s), 1e-30)
    worst_scale_rel = max(worst_scale_rel, float(rel.max()))
    entry.update(type=qt.name, bits=bits, scale_fp16_max_rel_err=float(rel.max()))
    if args.verify:
      verify(name, data, qt, q, s, entry)
    writer.add_tensor(name, data, raw_dtype=qt)
    report["tensors"][name] = entry
    print(f"{name:40s} int{bits} -> {qt.name}", flush=True)

  report["max_scale_fp16_rel_err"] = worst_scale_rel
  args.out.parent.mkdir(parents=True, exist_ok=True)
  writer.write_header_to_file()
  writer.write_kv_data_to_file()
  writer.write_tensors_to_file(progress=False)
  writer.close()
  if args.report:
    args.report.write_text(json.dumps(report, indent=1))
  print(f"wrote {args.out} ({args.out.stat().st_size / 2**20:.1f} MiB); "
        f"worst fp16 scale rel err {worst_scale_rel:.2e}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
