#!/usr/bin/env python3
"""
janus_to_gguf.py — Convert Janus v4 raw fp32 .bin → GGUF (Q8_0 or Q4_K).

Janus v4 176M weight layout (matches infer_v4.c assign() in dario):
    resid_lambdas[B] + x0_lambdas[B] + smear_lambda[1] + backout_lambda[1]
    transformer.wte.weight[V, E]
    for layer i in 0..B-1:
        attn.wr_a[H, E, R]
        attn.wr_b[H, R, T]
        attn.gate[H, 3]
        attn.c_q.weight[E, E]
        attn.c_k.weight[E, E]
        attn.c_v.weight[E, E]
        attn.wvr.weight[E, E]
        attn.wj.weight[E, E]
        attn.c_proj.weight[E, E]
        mlp.w_gate.weight[M, E]
        mlp.w_up.weight[M, E]
        mlp.w_down.weight[E, M]
    lm_head.weight[V, E]
    smear_gate.weight[1, 24]

Default config (Yent SFT 176M):
    V=32768  E=640  H=10  D=64  B=20  M=1664  T=1024  R=64

Output is GGUF v3, readable by libnotorch's gguf_open() / gguf_dequant().

Usage:
    python3 janus_to_gguf.py input.bin output.gguf --quant q8_0
    python3 janus_to_gguf.py input.bin output.gguf --quant q4_k
"""
import argparse
import struct
import sys
import numpy as np


# ── GGUF type IDs ──────────────────────────────────────────────────────────
GGML_TYPE_F32   = 0
GGML_TYPE_F16   = 1
GGML_TYPE_Q4_0  = 2
GGML_TYPE_Q8_0  = 8
GGML_TYPE_Q4_K  = 12

# GGUF metadata value types
GGUF_VT_UINT32   = 4
GGUF_VT_INT32    = 5
GGUF_VT_FLOAT32  = 6
GGUF_VT_BOOL     = 7
GGUF_VT_STRING   = 8
GGUF_VT_ARRAY    = 9
GGUF_VT_UINT64   = 10

GGUF_MAGIC   = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGN   = 32  # tensor data alignment


# ── Config (Janus v4 176M) ─────────────────────────────────────────────────
DEFAULT_CFG = dict(V=32768, E=640, H=10, D=64, B=20, M=1664, T=1024, R=64)


def fp32_to_fp16_bits(x: np.ndarray) -> np.ndarray:
    """Convert fp32 array to uint16 bit pattern of IEEE-754 binary16."""
    return x.astype(np.float16).view(np.uint16)


# ── Q8_0 quantizer ─────────────────────────────────────────────────────────
# Block: 32 fp32 values → fp16 scale (2 bytes) + 32 int8 values (32 bytes)
# Storage: 34 bytes per 32 elements → 1.0625 bytes/param.
def quantize_q8_0(weights: np.ndarray) -> bytes:
    """Quantize 1-D fp32 array to Q8_0. Returns packed bytes."""
    n = weights.size
    assert n % 32 == 0, f"Q8_0 requires multiple of 32 elements, got {n}"
    w = weights.reshape(-1, 32).astype(np.float32)
    # per-block max-abs
    amax = np.maximum(np.abs(w).max(axis=1), 1e-30)
    scale = amax / 127.0                          # fp32 scale
    qs = np.round(w / scale[:, None]).clip(-128, 127).astype(np.int8)
    scale_h = fp32_to_fp16_bits(scale)            # uint16
    # pack: for each block: scale (2 bytes LE) + 32 int8
    out = np.empty((w.shape[0], 34), dtype=np.uint8)
    out[:, 0] = scale_h & 0xFF
    out[:, 1] = (scale_h >> 8) & 0xFF
    out[:, 2:] = qs.view(np.uint8)
    return out.tobytes()


# ── Q4_K quantizer ─────────────────────────────────────────────────────────
# Super-block: 256 fp32 values → 144 bytes:
#   fp16 d                (2 bytes)              — super scale
#   fp16 dmin             (2 bytes)              — super min scale
#   uint8 sc[12]          (12 bytes)             — 8× 6-bit scales + 8× 6-bit mins
#   uint8 qs[128]         (128 bytes)            — 256 nibbles, paired sub-blocks
#
# 8 sub-blocks of 32 elements each. dequant formula (notorch/gguf.c):
#   out[j+l]    = (d * sc6[is])   * (qs[qi+l] & 0x0F) - (dmin * mn6[is])     l=0..31
#   out[j+32+l] = (d * sc6[is+1]) * (qs[qi+l]  >> 4 ) - (dmin * mn6[is+1])   l=0..31
# So sub-blocks are paired (is, is+1); each pair shares 32 nibble bytes:
# low nibble → first sub-block of pair, high nibble → second sub-block.
#
# 6-bit scales/mins layout (matches get_scale_min_k4 in notorch/gguf.c):
#   sc[0..3]  : bits0-5 = scale[0..3] low6,  bits6-7 = scale[4..7] high2
#   sc[4..7]  : bits0-5 =   min[0..3] low6,  bits6-7 =   min[4..7] high2
#   sc[8..11] : bits0-3 = scale[4..7] low4,  bits4-7 =   min[4..7] low4

def _pack_scales_mins_q4k(sc6: np.ndarray, mn6: np.ndarray) -> np.ndarray:
    """sc6, mn6: shape [n_super, 8] uint8 with 6-bit values. Returns [n_super, 12]."""
    n = sc6.shape[0]
    out = np.zeros((n, 12), dtype=np.uint8)
    for j in range(4):
        out[:, j]     = (sc6[:, j]     & 0x3F) | (((sc6[:, j + 4] >> 4) & 0x03) << 6)
        out[:, j + 4] = (mn6[:, j]     & 0x3F) | (((mn6[:, j + 4] >> 4) & 0x03) << 6)
        out[:, j + 8] = (sc6[:, j + 4] & 0x0F) |  ((mn6[:, j + 4]       & 0x0F) << 4)
    return out


def quantize_q4_k(weights: np.ndarray) -> bytes:
    """Quantize 1-D fp32 array to Q4_K. Returns packed bytes."""
    n = weights.size
    assert n % 256 == 0, f"Q4_K requires multiple of 256 elements, got {n}"
    w = weights.reshape(-1, 8, 32).astype(np.float32)  # [n_super, 8 sub, 32 elem]
    n_super = w.shape[0]

    # Per-sub-block min and max
    mins = w.min(axis=2)            # [n_super, 8]
    maxs = w.max(axis=2)            # [n_super, 8]

    # Reconstruction: dq = effscale * q4 - effmin   with q4 ∈ [0,15], effmin ≥ 0.
    # Naive choice: effscale = (max-min)/15, effmin = -min  (correct when min ≤ 0).
    # When min > 0 (rare, mostly biases) we clip effmin to 0 so q=0 doesn't go
    # below w_min — the resulting precision loss is bounded by the sub-block
    # range, which is small for those tensors anyway.
    sub_scale = np.maximum((maxs - mins) / 15.0, 1e-30)   # [n_super, 8]
    sub_min   = np.maximum(-mins, 0.0)                    # [n_super, 8]

    # Compress 8 sub-block scales/mins into 6-bit unsigned ints by dividing by
    # super-block d, dmin (themselves stored as fp16).
    d_super_scale = np.maximum(sub_scale.max(axis=1) / 63.0, 1e-30)   # [n_super]
    d_super_min   = np.maximum(sub_min.max(axis=1)   / 63.0, 1e-30)   # [n_super]

    sc6 = np.round(sub_scale / d_super_scale[:, None]).clip(0, 63).astype(np.uint8)
    mn6 = np.round(sub_min   / d_super_min[:, None]).clip(0, 63).astype(np.uint8)

    # Round-trip d/dmin through fp16 so quantization sees the same numbers as
    # dequant will reconstruct.
    d_super_scale_f16 = fp32_to_fp16_bits(d_super_scale).astype(np.uint16)
    d_super_min_f16   = fp32_to_fp16_bits(d_super_min  ).astype(np.uint16)
    d_super_scale = np.frombuffer(d_super_scale_f16.tobytes(),
                                  dtype=np.float16).astype(np.float32)
    d_super_min   = np.frombuffer(d_super_min_f16.tobytes(),
                                  dtype=np.float16).astype(np.float32)

    eff_sub_scale = np.maximum(sc6.astype(np.float32) * d_super_scale[:, None], 1e-30)
    eff_sub_min   = mn6.astype(np.float32) * d_super_min[:, None]

    # q4 = round((w + effmin) / effscale), clipped to [0,15]
    qs = np.round((w + eff_sub_min[:, :, None]) / eff_sub_scale[:, :, None])
    qs = qs.clip(0, 15).astype(np.uint8)             # [n_super, 8, 32]

    # Nibble packing: pair sub-blocks (0,1), (2,3), (4,5), (6,7). Each pair
    # contributes 32 bytes: low nibble = first sub, high nibble = second sub.
    pair_packed = np.empty((n_super, 4, 32), dtype=np.uint8)
    for p in range(4):
        lo = qs[:, 2 * p,     :]                     # [n_super, 32]
        hi = qs[:, 2 * p + 1, :]                     # [n_super, 32]
        pair_packed[:, p, :] = (lo & 0x0F) | ((hi & 0x0F) << 4)
    qs_bytes = pair_packed.reshape(n_super, 128)

    scales_bytes = _pack_scales_mins_q4k(sc6, mn6)   # [n_super, 12]

    out = np.empty((n_super, 144), dtype=np.uint8)
    out[:, 0] = d_super_scale_f16 & 0xFF
    out[:, 1] = (d_super_scale_f16 >> 8) & 0xFF
    out[:, 2] = d_super_min_f16 & 0xFF
    out[:, 3] = (d_super_min_f16 >> 8) & 0xFF
    out[:, 4:16]   = scales_bytes
    out[:, 16:144] = qs_bytes
    return out.tobytes()


# ── GGUF writer ────────────────────────────────────────────────────────────
def _write_str(buf: bytearray, s: str) -> None:
    b = s.encode("utf-8")
    buf += struct.pack("<Q", len(b))
    buf += b


def _write_kv_string(buf: bytearray, key: str, value: str) -> None:
    _write_str(buf, key)
    buf += struct.pack("<I", GGUF_VT_STRING)
    _write_str(buf, value)


def _write_kv_uint32(buf: bytearray, key: str, value: int) -> None:
    _write_str(buf, key)
    buf += struct.pack("<I", GGUF_VT_UINT32)
    buf += struct.pack("<I", value)


def _write_kv_uint64(buf: bytearray, key: str, value: int) -> None:
    _write_str(buf, key)
    buf += struct.pack("<I", GGUF_VT_UINT64)
    buf += struct.pack("<Q", value)


def _align(off: int, align: int = GGUF_ALIGN) -> int:
    return (off + align - 1) & ~(align - 1)


def write_gguf(path: str, kvs: list, tensors: list) -> None:
    """tensors: list of (name, shape, ggml_type, data_bytes)."""
    # --- header + KVs + tensor metadata ---
    head = bytearray()
    head += GGUF_MAGIC
    head += struct.pack("<I", GGUF_VERSION)
    head += struct.pack("<Q", len(tensors))
    head += struct.pack("<Q", len(kvs))

    for kv in kvs:
        kind = kv[0]
        if kind == "str":
            _write_kv_string(head, kv[1], kv[2])
        elif kind == "u32":
            _write_kv_uint32(head, kv[1], kv[2])
        elif kind == "u64":
            _write_kv_uint64(head, kv[1], kv[2])
        else:
            raise ValueError(f"unsupported kv kind: {kind}")

    # First pass: compute offsets
    tmeta = bytearray()
    for name, shape, gtype, _ in tensors:
        _write_str(tmeta, name)
        tmeta += struct.pack("<I", len(shape))
        for d in shape:
            tmeta += struct.pack("<Q", d)
        tmeta += struct.pack("<I", gtype)
        tmeta += struct.pack("<Q", 0)  # placeholder offset

    head_len = len(head) + len(tmeta)
    data_start = _align(head_len)

    # Second pass: fill offsets
    tmeta = bytearray()
    cur_off = 0
    offsets = []
    for name, shape, gtype, data in tensors:
        cur_off = _align(cur_off)
        offsets.append(cur_off)
        _write_str(tmeta, name)
        tmeta += struct.pack("<I", len(shape))
        for d in shape:
            tmeta += struct.pack("<Q", d)
        tmeta += struct.pack("<I", gtype)
        tmeta += struct.pack("<Q", cur_off)
        cur_off += len(data)

    with open(path, "wb") as f:
        f.write(head)
        f.write(tmeta)
        # pad to data_start
        pad = data_start - (len(head) + len(tmeta))
        if pad > 0:
            f.write(b"\x00" * pad)
        # write tensor data with alignment
        write_off = 0
        for (name, shape, gtype, data), off in zip(tensors, offsets):
            pad = off - write_off
            if pad > 0:
                f.write(b"\x00" * pad)
                write_off += pad
            f.write(data)
            write_off += len(data)


# ── Janus weight reader ────────────────────────────────────────────────────
def read_janus_bin(path: str, cfg: dict):
    """Stream raw fp32 .bin in infer_v4.c order, yield (name, shape, fp32 array)."""
    V, E, H, D, B, M, T, R = (cfg[k] for k in "VEHDBMTR")

    with open(path, "rb") as f:
        def take(n_floats):
            buf = f.read(n_floats * 4)
            if len(buf) != n_floats * 4:
                raise IOError(f"short read at offset {f.tell()}, wanted {n_floats * 4} bytes")
            return np.frombuffer(buf, dtype=np.float32).copy()

        yield "resid_lambdas",  (B,),    take(B)
        yield "x0_lambdas",     (B,),    take(B)
        yield "smear_lambda",   (1,),    take(1)
        yield "backout_lambda", (1,),    take(1)
        yield "transformer.wte.weight", (V, E), take(V * E)

        for i in range(B):
            yield f"transformer.h.{i}.attn.wr_a",        (H, E, R), take(H * E * R)
            yield f"transformer.h.{i}.attn.wr_b",        (H, R, T), take(H * R * T)
            yield f"transformer.h.{i}.attn.gate",        (H, 3),    take(H * 3)
            yield f"transformer.h.{i}.attn.c_q.weight",  (E, E),    take(E * E)
            yield f"transformer.h.{i}.attn.c_k.weight",  (E, E),    take(E * E)
            yield f"transformer.h.{i}.attn.c_v.weight",  (E, E),    take(E * E)
            yield f"transformer.h.{i}.attn.wvr.weight",  (E, E),    take(E * E)
            yield f"transformer.h.{i}.attn.wj.weight",   (E, E),    take(E * E)
            yield f"transformer.h.{i}.attn.c_proj.weight", (E, E),  take(E * E)
            yield f"transformer.h.{i}.mlp.w_gate.weight", (M, E),   take(M * E)
            yield f"transformer.h.{i}.mlp.w_up.weight",   (M, E),   take(M * E)
            yield f"transformer.h.{i}.mlp.w_down.weight", (E, M),   take(E * M)

        yield "lm_head.weight",      (V, E), take(V * E)
        yield "smear_gate.weight",   (1, 24), take(24)


# ── Quantization picker ────────────────────────────────────────────────────
QUANTIZERS = {
    "q8_0": (GGML_TYPE_Q8_0, quantize_q8_0, 32),
    "q4_k": (GGML_TYPE_Q4_K, quantize_q4_k, 256),
    "f16":  (GGML_TYPE_F16,  lambda w: w.astype(np.float16).tobytes(), 1),
    "f32":  (GGML_TYPE_F32,  lambda w: w.astype(np.float32).tobytes(), 1),
}


# Tensors that should stay in fp32 regardless of --quant choice:
# small per-layer scalars and the gate matrix where 4-bit/8-bit lossiness
# would visibly hurt the 3-way blend.
KEEP_F32 = {
    "resid_lambdas",
    "x0_lambdas",
    "smear_lambda",
    "backout_lambda",
    "smear_gate.weight",
}
def _is_kept_f32(name: str) -> bool:
    if name in KEEP_F32: return True
    if name.endswith(".attn.gate"): return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="raw fp32 .bin (Janus v4 layout)")
    ap.add_argument("output", help="output GGUF path")
    ap.add_argument("--quant", choices=list(QUANTIZERS), default="q8_0")
    ap.add_argument("--name", default="janus_v4_sft_yent_176m",
                    help="general.name metadata")
    for k in "VEHDBMTR":
        ap.add_argument(f"--{k}", type=int, default=DEFAULT_CFG[k])
    args = ap.parse_args()

    cfg = {k: getattr(args, k) for k in "VEHDBMTR"}
    gtype, quantize, block = QUANTIZERS[args.quant]
    print(f"[janus_to_gguf] cfg = {cfg}")
    print(f"[janus_to_gguf] quant = {args.quant} (block={block}, type={gtype})")

    tensors = []
    total_params = 0
    for name, shape, w in read_janus_bin(args.input, cfg):
        n = w.size
        total_params += n

        if _is_kept_f32(name) or n % block != 0:
            data = w.astype(np.float32).tobytes()
            tensor_type = GGML_TYPE_F32
        else:
            data = quantize(w)
            tensor_type = gtype

        tensors.append((name, shape, tensor_type, data))
        suffix = " [f32]" if tensor_type == GGML_TYPE_F32 else f" [{args.quant}]"
        print(f"  {name:50s} {str(shape):20s} {n:>11d} elems{suffix}")

    print(f"[janus_to_gguf] total {total_params/1e6:.2f}M params from {args.input}")

    kvs = [
        ("str", "general.architecture", "janus"),
        ("str", "general.name",          args.name),
        ("str", "general.quantization",  args.quant),
        ("u32", "janus.context_length",      cfg["T"]),
        ("u32", "janus.embedding_length",    cfg["E"]),
        ("u32", "janus.feed_forward_length", cfg["M"]),
        ("u32", "janus.attention.head_count", cfg["H"]),
        ("u32", "janus.attention.head_dim",   cfg["D"]),
        ("u32", "janus.block_count",          cfg["B"]),
        ("u32", "janus.vocab_size",           cfg["V"]),
        ("u32", "janus.rrpram.rank",          cfg["R"]),
        ("u32", "janus.rrpram.context",       cfg["T"]),
    ]

    write_gguf(args.output, kvs, tensors)
    import os
    print(f"[janus_to_gguf] wrote {args.output} "
          f"({os.path.getsize(args.output)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
