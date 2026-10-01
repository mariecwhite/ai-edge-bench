"""Prove that the .litertlm and the mobile QAT checkpoint hold the same weights.

  python3 -m aeb.prep.verify_litert_parity \\
      --qat-dir /models/google/gemma-4-E2B-it-qat-mobile-transformers \\
      --litert-dir /tmp/unpacked-litertlm --report parity.json

`--litert-dir` is the output of `litert-lm unpack <model>.litertlm`. For every
quantized text-decoder tensor (attention, MLP, per-layer gate/projection,
lm_head, token embedding, per-layer embeddings) the integer values and
per-channel scales in the TFLite sections are compared element by element with
the checkpoint. Together with the integer round-trip check in
convert_qat_mobile_gguf --verify, this closes the chain
.litertlm == checkpoint == GGUF.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from .convert_qat_mobile_gguf import HF, SafeTensors, unpack

OPS = {  # tflite op path fragment -> (checkpoint module, bits)
    "q_einsum": ("self_attn.q_proj", 4),
    "k_einsum": ("self_attn.k_proj", 4),
    "v_einsum": ("self_attn.v_proj", 4),
    "attn_vec_einsum": ("self_attn.o_proj", 4),
    "gating_einsum1": ("mlp.gate_proj", None),
    "gating_einsum2": ("mlp.up_proj", None),
    "mlp/linear": ("mlp.down_proj", None),
    "per_layer_embedding_gate": ("per_layer_input_gate", 8),
    "per_layer_embedding_projection": ("per_layer_projection", 8),
}


class TFLiteFile:

  def __init__(self, path: Path):
    import tflite
    self.tflite = tflite
    self.data = path.read_bytes()
    self.model = tflite.Model.GetRootAsModel(self.data, 0)
    self.tensors = []
    seen = set()
    for si in range(self.model.SubgraphsLength()):
      sg = self.model.Subgraphs(si)
      for ti in range(sg.TensorsLength()):
        t = sg.Tensors(ti)
        if t.Type() not in (tflite.TensorType.INT4, tflite.TensorType.INT8, 19) or t.Buffer() in seen:
          continue
        b = self.model.Buffers(t.Buffer())
        if not (b.Offset() > 1 or b.DataLength() > 0):
          continue
        seen.add(t.Buffer())
        self.tensors.append(t)

  def find(self, pattern: str):
    rx = re.compile(pattern)
    return [t for t in self.tensors if rx.search(t.Name().decode())]

  def values(self, t):
    b = self.model.Buffers(t.Buffer())
    raw = (np.frombuffer(self.data, dtype=np.uint8, count=b.Size(), offset=b.Offset())
           if b.Offset() > 1 else b.DataAsNumpy())
    shape = tuple(int(x) for x in t.ShapeAsNumpy())
    bits = {self.tflite.TensorType.INT4: 4, self.tflite.TensorType.INT8: 8, 19: 2}[t.Type()]
    if bits == 8:
      q = raw.view(np.int8).reshape(shape)
    else:
      per, mask = 8 // bits, (1 << bits) - 1
      v = np.stack([((raw >> (bits * i)) & mask).astype(np.int16) for i in range(per)], -1)
      v = v.reshape(-1)[:int(np.prod(shape))]
      q = np.where(v >= (1 << (bits - 1)), v - (1 << bits), v).reshape(shape).astype(np.int8)
    scales = t.Quantization().ScaleAsNumpy().astype(np.float32)
    return q, scales, bits


def compare(name, q_t, s_t, q_h, s_h) -> dict:
  ints = bool(np.array_equal(q_t, q_h))
  sc = bool(np.array_equal(s_t.reshape(-1), s_h.reshape(-1)))
  return {"tensor": name, "shape": list(q_h.shape), "ints_equal": ints, "scales_equal": sc,
          "int_mismatch_frac": float((q_t != q_h).mean()) if q_t.shape == q_h.shape else None}


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--qat-dir", type=Path, required=True)
  ap.add_argument("--litert-dir", type=Path, required=True)
  ap.add_argument("--report", type=Path)
  args = ap.parse_args(argv)

  st = SafeTensors(args.qat_dir / "model.safetensors")
  dec = TFLiteFile(next(args.litert_dir.glob("*prefill_decode.tflite")))
  results = []
  n_layer = 35
  for i in range(n_layer):
    for frag, (module, bits) in OPS.items():
      key = f"{HF}layers.{i}.{module}"
      cands = dec.find(rf"decode_graph/.*/layer_{i}/.*{frag}/.*dot_general\d*$")
      if not cands:
        continue  # e.g. k/v projections of KV-shared layers are not exported
      q_t, s_t, b = dec.values(cands[0])
      q_h = unpack(st.raw(key + ".weight"), b, q_t.shape[1])
      results.append(compare(key, q_t, s_t, q_h, st.f32(key + ".weight_scale")))
  head = dec.find(r"decode_softmax/.*composite\d*$")
  q_t, s_t, b = dec.values(head[0])
  results.append(compare("lm_head", q_t, s_t, unpack(st.raw("lm_head.weight"), b, q_t.shape[1]),
                         st.f32("lm_head.weight_scale")))

  emb = TFLiteFile(next(args.litert_dir.glob("*_embedder.tflite")))
  q_t, s_t, b = emb.values(emb.tensors[0])
  results.append(compare(HF + "embed_tokens", q_t, s_t,
                         unpack(st.raw(HF + "embed_tokens.embedding_quantized"), b, q_t.shape[1]),
                         st.f32(HF + "embed_tokens.embedding_scale")))

  ple = TFLiteFile(next(args.litert_dir.glob("*per_layer_embedder.tflite")))
  q_full = unpack(st.raw(HF + "embed_tokens_per_layer.embedding_quantized"), 4, 35 * 256)
  s_full = st.f32(HF + "embed_tokens_per_layer.embedding_scale")
  ple_ok = []
  for t in ple.tensors:
    q_t, s_t, b = ple.values(t)
    if q_t.shape != (q_full.shape[0], 256):
      continue
    # Match each exported per-layer table to its 256-wide slice of the checkpoint.
    hit = None
    for layer in range(35):
      if np.array_equal(s_t, s_full[:, layer]) and np.array_equal(
          q_t[:4096], q_full[:4096, layer * 256:(layer + 1) * 256]):
        hit = layer
        break
    ok = hit is not None and np.array_equal(q_t, q_full[:, hit * 256:(hit + 1) * 256])
    ple_ok.append(ok)
  results.append({"tensor": HF + "embed_tokens_per_layer", "tables_checked": len(ple_ok),
                  "ints_equal": bool(ple_ok) and all(ple_ok), "scales_equal": bool(ple_ok) and all(ple_ok)})

  bad = [r for r in results if not (r["ints_equal"] and r["scales_equal"])]
  summary = {"tensors_compared": len(results), "all_identical": not bad, "mismatches": bad,
             "results": results}
  if args.report:
    args.report.write_text(json.dumps(summary, indent=1) + "\n")
  print(f"compared {len(results)} quantized tensors: "
        f"{'ALL IDENTICAL (integers and scales)' if not bad else f'{len(bad)} MISMATCHES'}")
  for r in bad:
    print("  mismatch:", r)
  return 0 if not bad else 1


if __name__ == "__main__":
  sys.exit(main())
