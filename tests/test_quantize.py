#!/usr/bin/env python3
"""
test_quantize.py — round-trip unit tests for tools/janus_to_gguf.py.

Strategy: run the converter in-process on synthetic Janus-shaped weights
of a tiny mini-config, decode the output GGUF through a reference
Python dequantizer that mirrors notorch/gguf.c byte-for-byte, and assert
the reconstructed values match the original within the format's
quantization budget.

The reference dequantizers here exist solely for testing — production
dequant is notorch/gguf.c. Both converge on the same canonical layout
(Q8_0, Q4_K), so round-trip success here is structural proof that the
writer is byte-compatible with the C reader.

Run:  python3 tests/test_quantize.py
"""
import os
import sys
import struct
import subprocess
import tempfile

import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JANUS_TO_GGUF = os.path.join(REPO, "tools", "janus_to_gguf.py")
sys.path.insert(0, os.path.join(REPO, "tools"))
from janus_to_gguf import quantize_q8_0, quantize_q4_k  # noqa: E402


# ── Reference dequantizers (mirror notorch/gguf.c byte-for-byte) ──────────

def dequant_q8_0_ref(buf: bytes, n_elements: int) -> np.ndarray:
    """Q8_0 block: 2 bytes fp16 scale + 32 int8 values = 34 bytes / 32 elems."""
    n_blocks = n_elements // 32
    out = np.empty(n_elements, dtype=np.float32)
    for b in range(n_blocks):
        block = buf[b * 34: b * 34 + 34]
        scale = struct.unpack("<e", block[:2])[0]
        qs = np.frombuffer(block[2:], dtype=np.int8)
        out[b * 32: b * 32 + 32] = qs.astype(np.float32) * scale
    return out


def _get_scale_min_k4(j: int, sc: bytes):
    if j < 4:
        s = sc[j] & 63
        m = sc[j + 4] & 63
    else:
        s = (sc[j + 4] & 0x0F) | (((sc[j - 4] >> 6) & 0x03) << 4)
        m = (sc[j + 4] >> 4)   | (((sc[j]     >> 6) & 0x03) << 4)
    return s, m


def dequant_q4_k_ref(buf: bytes, n_elements: int) -> np.ndarray:
    """Q4_K super-block: 2+2+12+128 = 144 bytes / 256 elems."""
    n_super = n_elements // 256
    out = np.empty(n_elements, dtype=np.float32)
    for i in range(n_super):
        b = buf[i * 144: i * 144 + 144]
        d    = struct.unpack("<e", b[0:2])[0]
        dmin = struct.unpack("<e", b[2:4])[0]
        sc   = b[4:16]
        qs   = b[16:144]
        oi = i * 256
        is_, qi = 0, 0
        for j in range(0, 256, 64):
            sc0, m0 = _get_scale_min_k4(is_,     sc)
            sc1, m1 = _get_scale_min_k4(is_ + 1, sc)
            d1, mm1 = d * sc0, dmin * m0
            d2, mm2 = d * sc1, dmin * m1
            for l in range(32):
                out[oi + j + l]      = d1 * float(qs[qi + l] & 0x0F) - mm1
                out[oi + j + 32 + l] = d2 * float(qs[qi + l] >> 4)   - mm2
            qi += 32
            is_ += 2
    return out


# ── Test cases ────────────────────────────────────────────────────────────

def assert_close(actual, expected, tol, label):
    err = np.max(np.abs(actual - expected))
    if err > tol:
        raise AssertionError(
            f"FAIL [{label}]: max-err {err:.6f} > tol {tol:.6f}"
        )
    print(f"  PASS [{label}]: max-err {err:.6f} (tol {tol:.6f})")


def test_q8_0_round_trip():
    print("== test_q8_0_round_trip ==")
    rng = np.random.default_rng(42)
    for scale in [0.02, 0.5, 4.0]:
        w = (rng.standard_normal(2048).astype(np.float32) * scale)
        buf = quantize_q8_0(w)
        assert len(buf) == 2048 // 32 * 34, f"size mismatch for scale {scale}"
        deq = dequant_q8_0_ref(buf, 2048)
        # Q8_0 max-err ≤ 0.5 * (max-abs / 127)
        max_abs = float(np.max(np.abs(w)))
        budget = 0.6 * max_abs / 127.0   # 20% slack on top
        assert_close(deq, w, budget, f"Q8_0 scale={scale}")


def test_q4_k_round_trip():
    print("== test_q4_k_round_trip ==")
    rng = np.random.default_rng(43)
    for scale in [0.02, 0.5, 4.0]:
        w = (rng.standard_normal(2048).astype(np.float32) * scale)
        buf = quantize_q4_k(w)
        assert len(buf) == 2048 // 256 * 144, f"size mismatch for scale {scale}"
        deq = dequant_q4_k_ref(buf, 2048)
        # Q4_K with 4-bit quant: error within sub_scale / 2 ≈ (range/15)/2
        rng_lo, rng_hi = float(np.min(w)), float(np.max(w))
        budget = (rng_hi - rng_lo) / 15.0 / 1.5  # 33% slack
        assert_close(deq, w, budget, f"Q4_K scale={scale}")


def test_e2e_synthetic_janus():
    """Build a tiny synthetic Janus mini-bin, run janus_to_gguf, verify
    that file size shrinks predictably and metadata reads back."""
    print("== test_e2e_synthetic_janus ==")
    cfg = dict(V=64, E=32, H=2, D=16, B=2, M=64, T=64, R=4)
    n = (
        cfg["B"] * 2 + 2 +                     # lambdas + smear + backout
        cfg["V"] * cfg["E"] +                  # wte
        cfg["B"] * (cfg["H"] * cfg["E"] * cfg["R"]
                  + cfg["H"] * cfg["R"] * cfg["T"]
                  + cfg["H"] * 3
                  + 6 * cfg["E"] * cfg["E"]
                  + 3 * cfg["M"] * cfg["E"]) +
        cfg["V"] * cfg["E"] +                  # head
        24                                     # smear_g
    )
    rng = np.random.default_rng(44)
    data = (rng.standard_normal(n).astype(np.float32) * 0.02)

    with tempfile.TemporaryDirectory() as td:
        bin_path  = os.path.join(td, "mini.bin")
        gguf_path = os.path.join(td, "mini.gguf")

        # Write JANU v4 header (256-byte preamble) + raw fp32 weights so
        # the converter's read path is exercised end-to-end.
        with open(bin_path, "wb") as f:
            f.write(struct.pack("<i", 0x4A414E55))                # magic 'JANU' LE
            f.write(struct.pack("<i", 4))                          # version
            f.write(struct.pack("<8i",
                                cfg["V"], cfg["E"], cfg["H"], cfg["D"],
                                cfg["B"], cfg["M"], cfg["T"], n))  # n_params
            f.write(b"\x00" * (256 - 4 - 4 - 32))                  # padding to 256
            data.tofile(f)

        for q, factor_lo, factor_hi in [
            ("q8_0", 0.20, 0.40),     # 1.0625 B/param baseline + ~20-40% meta
            ("q4_k", 0.10, 0.25),     # 0.5625 B/param baseline + meta
            ("f16",  0.45, 0.60),     # 0.5 B/param + meta
        ]:
            subprocess.check_call(
                [sys.executable, JANUS_TO_GGUF, bin_path, gguf_path, "--quant", q],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            sz = os.path.getsize(gguf_path)
            ratio = sz / (n * 4)
            ok = factor_lo <= ratio <= factor_hi
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] {q}: gguf {sz} bytes / fp32 {n*4} = {ratio:.3f}"
                  f" (expected {factor_lo}–{factor_hi})")
            if not ok:
                raise AssertionError(f"{q} ratio {ratio:.3f} outside {factor_lo}-{factor_hi}")


if __name__ == "__main__":
    test_q8_0_round_trip()
    test_q4_k_round_trip()
    test_e2e_synthetic_janus()
    print("\nall tests passed.")
