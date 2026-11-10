#!/usr/bin/env python3
"""Extract exact operator & delegate parameter signatures from TFLite FlatBuffers in .litertlm.

This utility parses TFLite v3 FlatBuffers embedded inside `.litertlm` containers,
recursively inlining `STABLEHLO_COMPOSITE` (`OP_206`) decomposition subgraphs
exactly as LiteRT-LM does prior to delegate partitioning, and extracts every
distinct operator/delegate parameter signature (`M, N, K`, input/output tensor
shapes and data types, per-channel/per-tensor quantization parameters, and
builtin option values such as `adj_x`, `adj_y`, `axes`, `perm`, `new_shape`,
`stride`, `padding`, `depth_multiplier`, `beta`, `keep_dims`, etc.).

Usage:
  python3 kernel/scripts/extract_tflite_params.py <path_to.litertlm> <offset> <size> [max_tokens_override]
"""

from __future__ import annotations

from collections import Counter
import json
import mmap
import struct
import sys
from typing import Any

TENSOR_TYPES = {
    0: "f32",
    1: "f16",
    2: "i32",
    3: "u8",
    4: "i64",
    5: "str",
    6: "bool",
    7: "i16",
    8: "c64",
    9: "i8",
    10: "f64",
    11: "c128",
    12: "u64",
    13: "resource",
    14: "variant",
    15: "u32",
    16: "u16",
    17: "i4",
    18: "bf16",
    19: "i2",
}

BUILTIN_OPS = {
    0: "ADD",
    1: "AVERAGE_POOL_2D",
    2: "CONCATENATION",
    3: "CONV_2D",
    4: "DEPTHWISE_CONV_2D",
    5: "DEPTH_TO_SPACE",
    6: "DEQUANTIZE",
    7: "EMBEDDING_LOOKUP",
    8: "FLOOR",
    9: "FULLY_CONNECTED",
    10: "HASHTABLE_LOOKUP",
    11: "L2_NORMALIZATION",
    12: "L2_POOL_2D",
    13: "LOCAL_RESPONSE_NORMALIZATION",
    14: "LOGISTIC",
    15: "LSH_PROJECTION",
    16: "LSTM",
    17: "MAX_POOL_2D",
    18: "MUL",
    19: "RELU",
    20: "RELU_N1_TO_1",
    21: "RELU6",
    22: "RESHAPE",
    23: "RESIZE_BILINEAR",
    24: "RNN",
    25: "SOFTMAX",
    26: "SPACE_TO_DEPTH",
    27: "SVDF",
    28: "TANH",
    29: "CONCAT_EMBEDDINGS",
    30: "SKIP_GRAM",
    31: "CALL",
    32: "CUSTOM",
    34: "PAD",
    36: "GATHER",
    37: "BATCH_TO_SPACE_ND",
    38: "SPACE_TO_BATCH_ND",
    39: "TRANSPOSE",
    40: "MEAN",
    41: "SUB",
    42: "DIV",
    43: "SQUEEZE",
    45: "STRIDED_SLICE",
    47: "EXP",
    49: "SPLIT",
    53: "CAST",
    54: "PRELU",
    55: "MAXIMUM",
    56: "ARG_MAX",
    57: "MINIMUM",
    58: "LESS",
    59: "NEG",
    60: "PADV2",
    61: "GREATER",
    62: "GREATER_EQUAL",
    63: "LESS_EQUAL",
    64: "SELECT",
    65: "SLICE",
    66: "SIN",
    67: "TRANSPOSE_CONV",
    68: "SPARSE_TO_DENSE",
    69: "TILE",
    70: "EXPAND_DIMS",
    71: "EQUAL",
    72: "NOT_EQUAL",
    73: "LOG",
    74: "SUM",
    75: "SQRT",
    76: "RSQRT",
    77: "SHAPE",
    78: "POW",
    79: "ARG_MIN",
    80: "FAKE_QUANT",
    81: "REDUCE_PROD",
    82: "REDUCE_MAX",
    83: "PACK",
    84: "LOGICAL_OR",
    85: "ONE_HOT",
    86: "LOGICAL_AND",
    87: "LOGICAL_NOT",
    88: "UNPACK",
    89: "REDUCE_MIN",
    90: "FLOOR_DIV",
    91: "REDUCE_ANY",
    92: "SQUARE",
    93: "ZEROS_LIKE",
    94: "FILL",
    95: "FLOOR_MOD",
    96: "RANGE",
    97: "RESIZE_NEAREST_NEIGHBOR",
    98: "LEAKY_RELU",
    99: "SQUARED_DIFFERENCE",
    100: "MIRROR_PAD",
    101: "ABS",
    102: "SPLIT_V",
    103: "UNIQUE",
    104: "CEIL",
    105: "REVERSE_V2",
    106: "ADD_N",
    107: "GATHER_ND",
    108: "COS",
    109: "WHERE",
    110: "RANK",
    111: "ELU",
    112: "REVERSE_SEQUENCE",
    113: "MATRIX_DIAG",
    114: "QUANTIZE",
    115: "MATRIX_SET_DIAG",
    116: "ROUND",
    117: "HARD_SWISH",
    118: "IF",
    119: "WHILE",
    120: "NON_MAX_SUPPRESSION_V4",
    121: "NON_MAX_SUPPRESSION_V5",
    122: "SCATTER_ND",
    123: "SELECT_V2",
    124: "DENSIFY",
    125: "SEGMENT_SUM",
    126: "BATCH_MATMUL",
    127: "PLACEHOLDER_FOR_GREATER_OP_CODES",
    128: "CUMSUM",
    129: "CALL_ONCE",
    130: "BROADCAST_TO",
    131: "RFFT2D",
    132: "CONV_3D",
    133: "IMAG",
    134: "REAL",
    135: "COMPLEX_ABS",
    136: "HASHTABLE",
    137: "HASHTABLE_FIND",
    138: "HASHTABLE_IMPORT",
    139: "HASHTABLE_SIZE",
    140: "REDUCE_ALL",
    141: "CONV_3D_TRANSPOSE",
    142: "VAR_HANDLE",
    143: "READ_VARIABLE",
    144: "ASSIGN_VARIABLE",
    145: "BROADCAST_ARGS",
    146: "RANDOM_STANDARD_NORMAL",
    147: "BUCKETIZE",
    148: "RANDOM_UNIFORM",
    149: "MULTINOMIAL",
    150: "GELU",
    151: "DYNAMIC_UPDATE_SLICE",
    152: "RELU_0_TO_1",
    153: "UNSORTED_SEGMENT_PROD",
    154: "UNSORTED_SEGMENT_MAX",
    155: "UNSORTED_SEGMENT_SUM",
    156: "ATAN2",
    157: "UNSORTED_SEGMENT_MIN",
    158: "SIGN",
    159: "BITCAST",
    160: "BITWISE_XOR",
    161: "RIGHT_SHIFT",
    206: "STABLEHLO_COMPOSITE",
}

ACT_NAMES = {0: "NONE", 1: "RELU", 2: "RELU_N1_TO_1", 3: "RELU6", 4: "TANH", 5: "SIGN_BIT"}
PAD_NAMES = {0: "SAME", 1: "VALID"}


class FB:
  """Minimal zero-dependency FlatBuffer reader for TFLite v3 schemas."""

  def __init__(self, buf: memoryview):
    self.buf = buf

  def u32(self, pos: int) -> int:
    return struct.unpack_from("<I", self.buf, pos)[0]

  def i32(self, pos: int) -> int:
    return struct.unpack_from("<i", self.buf, pos)[0]

  def u16(self, pos: int) -> int:
    return struct.unpack_from("<H", self.buf, pos)[0]

  def i16(self, pos: int) -> int:
    return struct.unpack_from("<h", self.buf, pos)[0]

  def u8(self, pos: int) -> int:
    return self.buf[pos]

  def i8(self, pos: int) -> int:
    return struct.unpack_from("<b", self.buf, pos)[0]

  def f32(self, pos: int) -> float:
    return struct.unpack_from("<f", self.buf, pos)[0]

  def u64(self, pos: int) -> int:
    return struct.unpack_from("<Q", self.buf, pos)[0]

  def field(self, table: int, voffset: int) -> int:
    vtable = table - self.i32(table)
    vlen = self.u16(vtable)
    if voffset >= vlen:
      return 0
    off = self.u16(vtable + voffset)
    return (table + off) if off else 0

  def indirect(self, pos: int) -> int:
    return pos + self.u32(pos)

  def string(self, pos: int) -> str:
    p = self.indirect(pos)
    n = self.u32(p)
    return bytes(self.buf[p + 4 : p + 4 + n]).decode("utf-8", errors="replace")

  def vec_len(self, pos: int) -> int:
    p = self.indirect(pos)
    return self.u32(p)

  def vec_elem_table(self, pos: int, idx: int) -> int:
    p = self.indirect(pos)
    elem_pos = p + 4 + idx * 4
    return elem_pos + self.u32(elem_pos)

  def vec_i32(self, pos: int) -> list[int]:
    p = self.indirect(pos)
    n = self.u32(p)
    return list(struct.unpack_from(f"<{n}i", self.buf, p + 4))


def inspect_tflite(buf: memoryview, max_tokens_override: int | None = None) -> dict[str, Any]:
  """Parse all primary subgraphs in a TFLite v3 buffer and aggregate operator parameter signatures."""
  fb = FB(buf)
  root = fb.u32(0)

  opcodes: list[tuple[int, str]] = []
  oc_f = fb.field(root, 6)
  if oc_f:
    for i in range(fb.vec_len(oc_f)):
      oc_t = fb.vec_elem_table(oc_f, i)
      b_f = fb.field(oc_t, 10)
      b_code = fb.i32(b_f) if b_f else (fb.i8(fb.field(oc_t, 4)) if fb.field(oc_t, 4) else 0)
      c_f = fb.field(oc_t, 6)
      cname = fb.string(c_f) if c_f else ""
      op_name = BUILTIN_OPS.get(b_code, f"OP_{b_code}")
      if b_code == 32 and cname:
        op_name = f"CUSTOM({cname})"
      opcodes.append((b_code, op_name))

  buffers_f = fb.field(root, 12)

  def get_buffer_bytes(buf_idx: int) -> memoryview | None:
    if not buffers_f or buf_idx <= 0 or buf_idx >= fb.vec_len(buffers_f):
      return None
    bt = fb.vec_elem_table(buffers_f, buf_idx)
    df = fb.field(bt, 4)
    if df:
      p = fb.indirect(df)
      n = fb.u32(p)
      if n > 0:
        return buf[p + 4 : p + 4 + n]
    of = fb.field(bt, 6)
    sf = fb.field(bt, 8)
    if of and sf:
      offset = fb.u64(of)
      size = fb.u64(sf)
      if size > 0 and offset + size <= len(buf):
        return buf[offset : offset + size]
    return None

  subgraphs_f = fb.field(root, 8)
  num_sg = fb.vec_len(subgraphs_f) if subgraphs_f else 0

  def collect_subgraph_ops(sg_idx: int, sig_counter: Counter[tuple[str, str]]) -> int:
    sg_t = fb.vec_elem_table(subgraphs_f, sg_idx)
    tensors_f = fb.field(sg_t, 4)
    ntensors = fb.vec_len(tensors_f) if tensors_f else 0

    def get_tensor_info(t_idx: int) -> dict[str, Any]:
      if t_idx < 0 or t_idx >= ntensors:
        return {"dtype": "none", "shape": []}
      tt = fb.vec_elem_table(tensors_f, t_idx)
      sf = fb.field(tt, 4)
      shape = fb.vec_i32(sf) if sf else []
      if max_tokens_override is not None:
        shape = [
            max_tokens_override if d == 32771 else (max_tokens_override - 1024 if d == 31747 else d)
            for d in shape
        ]
      tf = fb.field(tt, 6)
      dtype_code = fb.u8(tf) if tf else 0
      dtype = TENSOR_TYPES.get(dtype_code, f"dt{dtype_code}")
      bf = fb.field(tt, 8)
      buf_idx = fb.u32(bf) if bf else 0
      qf = fb.field(tt, 12)
      q_str = ""
      if qf:
        qt = fb.indirect(qf)
        sc_f = fb.field(qt, 8)
        qd_f = fb.field(qt, 16)
        n_scales = fb.vec_len(sc_f) if sc_f else 0
        qdim = fb.i32(qd_f) if qd_f else 0
        if n_scales > 1:
          q_str = f"per_channel(axis={qdim}, n={n_scales})"
        elif n_scales == 1:
          sc = fb.f32(fb.indirect(sc_f) + 4)
          q_str = f"per_tensor(scale={sc:.4g})"
      const_vals = None
      if buf_idx > 0 and len(shape) <= 1 and (not shape or shape[0] <= 16):
        raw = get_buffer_bytes(buf_idx)
        if raw is not None:
          if dtype == "i32" and len(raw) % 4 == 0:
            const_vals = list(struct.unpack(f"<{len(raw)//4}i", raw))
          elif dtype == "i64" and len(raw) % 8 == 0:
            const_vals = list(struct.unpack(f"<{len(raw)//8}q", raw))
          elif dtype == "f32" and len(raw) % 4 == 0 and len(raw) <= 16:
            const_vals = [round(x, 6) for x in struct.unpack(f"<{len(raw)//4}f", raw)]
      return {
          "dtype": dtype,
          "shape": shape,
          "quant": q_str,
          "const": const_vals,
      }

    ops_f = fb.field(sg_t, 10)
    nops = fb.vec_len(ops_f) if ops_f else 0
    total_expanded = 0

    for op_idx in range(nops):
      op_t = fb.vec_elem_table(ops_f, op_idx)
      oci_f = fb.field(op_t, 4)
      oci = fb.u32(oci_f) if oci_f else 0
      _, op_name = opcodes[oci] if oci < len(opcodes) else (-1, "UNK")

      opt_f = fb.field(op_t, 12)
      opt_t = fb.indirect(opt_f) if opt_f else 0
      opt2_f = fb.field(op_t, 28)
      opt2_t = fb.indirect(opt2_f) if opt2_f else 0

      if op_name == "STABLEHLO_COMPOSITE" and opt2_t:
        decomp_f = fb.field(opt2_t, 6)
        if decomp_f:
          decomp_sg = fb.i32(decomp_f)
          if 0 <= decomp_sg < num_sg and decomp_sg != sg_idx:
            total_expanded += collect_subgraph_ops(decomp_sg, sig_counter)
            continue

      total_expanded += 1
      in_f = fb.field(op_t, 6)
      out_f = fb.field(op_t, 8)
      ins = [get_tensor_info(i) for i in (fb.vec_i32(in_f) if in_f else [])]
      outs = [get_tensor_info(i) for i in (fb.vec_i32(out_f) if out_f else [])]
      params: dict[str, Any] = {}

      if op_name == "FULLY_CONNECTED" and len(ins) >= 2 and outs:
        in0, w, out0 = ins[0], ins[1], outs[0]
        K = w["shape"][-1] if w["shape"] else 0
        N = w["shape"][0] if w["shape"] else 0
        M = 1
        for d in in0["shape"][:-1]:
          M *= d
        act = ACT_NAMES.get(fb.u8(fb.field(opt_t, 4)) if (opt_t and fb.field(opt_t, 4)) else 0, "NONE")
        keep_dims = bool(fb.u8(fb.field(opt_t, 8))) if (opt_t and fb.field(opt_t, 8)) else False
        asym_q = bool(fb.u8(fb.field(opt_t, 10))) if (opt_t and fb.field(opt_t, 10)) else False
        params = {
            "M": M,
            "N": N,
            "K": K,
            "input": f"{in0['dtype']}{in0['shape']}",
            "weights": f"{w['dtype']}{w['shape']} {w['quant']}".strip(),
            "bias": f"{ins[2]['dtype']}{ins[2]['shape']}" if len(ins) > 2 and ins[2]["dtype"] != "none" else "none",
            "output": f"{out0['dtype']}{out0['shape']}",
            "fused_act": act,
            "keep_num_dims": keep_dims,
            "asymmetric_quantize_input": asym_q,
        }
      elif op_name == "BATCH_MATMUL" and len(ins) >= 2 and outs:
        x, y, out0 = ins[0], ins[1], outs[0]
        adj_x = bool(fb.u8(fb.field(opt_t, 4))) if (opt_t and fb.field(opt_t, 4)) else False
        adj_y = bool(fb.u8(fb.field(opt_t, 6))) if (opt_t and fb.field(opt_t, 6)) else False
        asym_q = bool(fb.u8(fb.field(opt_t, 8))) if (opt_t and fb.field(opt_t, 8)) else False
        M = x["shape"][-1] if adj_x else (x["shape"][-2] if len(x["shape"]) >= 2 else 1)
        K = x["shape"][-2] if adj_x else (x["shape"][-1] if x["shape"] else 1)
        N = y["shape"][-2] if adj_y else (y["shape"][-1] if y["shape"] else 1)
        batch_dims = out0["shape"][:-2]
        params = {
            "batch": batch_dims,
            "M": M,
            "N": N,
            "K": K,
            "x": f"{x['dtype']}{x['shape']}",
            "y": f"{y['dtype']}{y['shape']}",
            "output": f"{out0['dtype']}{out0['shape']}",
            "adj_x": adj_x,
            "adj_y": adj_y,
            "asymmetric_quantize_input": asym_q,
        }
      elif op_name == "SOFTMAX" and ins and outs:
        beta = fb.f32(fb.field(opt_t, 4)) if (opt_t and fb.field(opt_t, 4)) else 1.0
        params = {
            "input": f"{ins[0]['dtype']}{ins[0]['shape']}",
            "output": f"{outs[0]['dtype']}{outs[0]['shape']}",
            "axis_dim": ins[0]["shape"][-1] if ins[0]["shape"] else 0,
            "beta": round(beta, 4),
        }
      elif op_name in ("MEAN", "SUM", "REDUCE_MAX", "REDUCE_MIN", "REDUCE_ANY") and len(ins) >= 2 and outs:
        keep_dims = bool(fb.u8(fb.field(opt_t, 4))) if (opt_t and fb.field(opt_t, 4)) else False
        params = {
            "input": f"{ins[0]['dtype']}{ins[0]['shape']}",
            "axes": ins[1]["const"],
            "keep_dims": keep_dims,
            "output": f"{outs[0]['dtype']}{outs[0]['shape']}",
        }
      elif op_name == "TRANSPOSE" and len(ins) >= 2 and outs:
        params = {
            "input": f"{ins[0]['dtype']}{ins[0]['shape']}",
            "perm": ins[1]["const"],
            "output": f"{outs[0]['dtype']}{outs[0]['shape']}",
        }
      elif op_name == "RESHAPE" and ins and outs:
        params = {
            "input": f"{ins[0]['dtype']}{ins[0]['shape']}",
            "new_shape": ins[1]["const"] if len(ins) > 1 else outs[0]["shape"],
            "output": f"{outs[0]['dtype']}{outs[0]['shape']}",
        }
      elif op_name in ("SLICE", "STRIDED_SLICE", "DYNAMIC_UPDATE_SLICE") and ins and outs:
        params = {
            "inputs": [f"{x['dtype']}{x['shape']}" + (f"={x['const']}" if x["const"] else "") for x in ins],
            "output": f"{outs[0]['dtype']}{outs[0]['shape']}",
        }
      elif op_name in ("CONV_2D", "DEPTHWISE_CONV_2D") and len(ins) >= 2 and outs:
        pad = PAD_NAMES.get(fb.u8(fb.field(opt_t, 4)) if (opt_t and fb.field(opt_t, 4)) else 0, "SAME")
        sw = fb.i32(fb.field(opt_t, 6)) if (opt_t and fb.field(opt_t, 6)) else 1
        sh = fb.i32(fb.field(opt_t, 8)) if (opt_t and fb.field(opt_t, 8)) else 1
        params = {
            "input": f"{ins[0]['dtype']}{ins[0]['shape']}",
            "filter": f"{ins[1]['dtype']}{ins[1]['shape']} {ins[1]['quant']}".strip(),
            "output": f"{outs[0]['dtype']}{outs[0]['shape']}",
            "padding": pad,
            "stride": [sh, sw],
        }
        if op_name == "DEPTHWISE_CONV_2D":
          dm = fb.i32(fb.field(opt_t, 10)) if (opt_t and fb.field(opt_t, 10)) else 1
          params["depth_multiplier"] = dm
      elif op_name == "AVERAGE_POOL_2D" and ins and outs:
        pad = PAD_NAMES.get(fb.u8(fb.field(opt_t, 4)) if (opt_t and fb.field(opt_t, 4)) else 0, "SAME")
        sw = fb.i32(fb.field(opt_t, 6)) if (opt_t and fb.field(opt_t, 6)) else 1
        sh = fb.i32(fb.field(opt_t, 8)) if (opt_t and fb.field(opt_t, 8)) else 1
        fw = fb.i32(fb.field(opt_t, 10)) if (opt_t and fb.field(opt_t, 10)) else 1
        fh = fb.i32(fb.field(opt_t, 12)) if (opt_t and fb.field(opt_t, 12)) else 1
        params = {
            "input": f"{ins[0]['dtype']}{ins[0]['shape']}",
            "output": f"{outs[0]['dtype']}{outs[0]['shape']}",
            "filter": [fh, fw],
            "stride": [sh, sw],
            "padding": pad,
        }
      else:
        params = {
            "inputs": [f"{x['dtype']}{x['shape']}" + (f" {x['quant']}" if x["quant"] else "") + (f"={x['const']}" if x["const"] else "") for x in ins],
            "outputs": [f"{x['dtype']}{x['shape']}" for x in outs],
        }

      key = (op_name, json.dumps(params, sort_keys=True))
      sig_counter[key] += 1

    return total_expanded

  out_subgraphs: dict[str, Any] = {}
  for sg_idx in range(num_sg):
    sg_t = fb.vec_elem_table(subgraphs_f, sg_idx)
    name_f = fb.field(sg_t, 12)
    sg_name = fb.string(name_f) if name_f else f"subgraph_{sg_idx}"
    if sg_idx != 0 and sg_name not in ("decode", "prefill_128", "prefill_1024", "main", "subgraph_0"):
      continue
    sig_counter: Counter[tuple[str, str]] = Counter()
    total_ops = collect_subgraph_ops(sg_idx, sig_counter)
    out_subgraphs[sg_name] = {
        "total_inlined_ops": total_ops,
        "distinct_signatures": len(sig_counter),
        "operators": [
            {"op": op_name, "count": cnt, "params": json.loads(p_json)}
            for (op_name, p_json), cnt in sig_counter.most_common()
        ],
    }
  return out_subgraphs


if __name__ == "__main__":
  path = sys.argv[1]
  offset = int(sys.argv[2])
  size = int(sys.argv[3])
  max_tok = int(sys.argv[4]) if len(sys.argv) > 4 else None
  with open(path, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    res = inspect_tflite(memoryview(mm)[offset : offset + size], max_tok)
    print(json.dumps(res, indent=2))
