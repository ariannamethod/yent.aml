"""Generate a tiny GGUF + JSON expectations file for the JS test.

Writes:
  tests/mini.gguf         — Janus mini-config (V=64 E=32 H=2 D=16 B=2 M=64 T=64 R=4)
                            quantised Q4_K with Q8_0 embeddings (default mixed
                            mode). Tensors are deterministic random fp32.
  tests/mini_expect.json  — for each tensor: name, dtype label, n_elements,
                            first 8 reference fp32 values (the original input,
                            before quantisation).

The JS side reads mini.gguf, dequantises each named tensor, and asserts the
first 8 values are within the format-specific tolerance from the originals.
"""
import os
import sys
import json
import struct

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'tools'))
from janus_to_gguf import (read_janus_bin, QUANTIZERS,  # noqa: E402
                            GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K,
                            write_gguf, _is_kept_f32, _is_embedding)

DTYPE_LABEL = {
    GGML_TYPE_F32: 'F32', GGML_TYPE_F16: 'F16',
    GGML_TYPE_Q8_0: 'Q8_0', GGML_TYPE_Q4_K: 'Q4_K',
}

cfg = dict(V=64, E=32, H=2, D=16, B=2, M=64, T=64, R=4)
B = cfg['B']
n = (B + B + 1 + 1 +
     cfg['V'] * cfg['E'] +
     B * (cfg['H'] * cfg['E'] * cfg['R']
          + cfg['H'] * cfg['R'] * cfg['T']
          + cfg['H'] * 3
          + 6 * cfg['E'] * cfg['E']
          + 3 * cfg['M'] * cfg['E']) +
     cfg['V'] * cfg['E'] +
     24)
rng = np.random.default_rng(2026)
data = (rng.standard_normal(n).astype(np.float32) * 0.05)

here = os.path.dirname(os.path.abspath(__file__))
bin_path  = os.path.join(here, 'mini.bin')
gguf_path = os.path.join(here, 'mini.gguf')
exp_path  = os.path.join(here, 'mini_expect.json')

# Write JANU header (256 byte preamble) + raw fp32, so read_janus_bin can
# parse it like a real Janus checkpoint would.
with open(bin_path, 'wb') as f:
    f.write(struct.pack('<I', 0x4A414E55))                  # magic
    f.write(struct.pack('<i', 4))                            # version
    f.write(struct.pack('<8i',
                        cfg['V'], cfg['E'], cfg['H'], cfg['D'],
                        cfg['B'], cfg['M'], cfg['T'], n))
    f.write(b'\x00' * (256 - 4 - 4 - 32))
    data.tofile(f)

# Read tensors in named_parameters order, build expectation table from
# the *original* fp32 values (we know what they were before quantisation).
quantizers = QUANTIZERS
q4k_gtype, q4k_quantize, q4k_block = QUANTIZERS['q4_k']
q8_gtype,  q8_quantize,  q8_block  = QUANTIZERS['q8_0']

tensors = []
expectations = []
for name, shape, w in read_janus_bin(bin_path, dict(cfg)):
    nelt = w.size
    if _is_kept_f32(name) or nelt % q8_block != 0:
        data_bytes = w.astype(np.float32).tobytes()
        gtype = GGML_TYPE_F32
    elif _is_embedding(name):
        data_bytes = q8_quantize(w)
        gtype = q8_gtype
    elif nelt % q4k_block != 0:
        data_bytes = w.astype(np.float32).tobytes()
        gtype = GGML_TYPE_F32
    else:
        data_bytes = q4k_quantize(w)
        gtype = q4k_gtype
    tensors.append((name, shape, gtype, data_bytes))
    expectations.append({
        'name': name,
        'dtype': DTYPE_LABEL[gtype],
        'n_elements': int(nelt),
        'first8': [float(x) for x in w[:8]],
    })

kvs = [
    ('str', 'general.architecture', 'janus'),
    ('str', 'general.name',          'mini-fixture'),
    ('u32', 'janus.context_length',  cfg['T']),
    ('u32', 'janus.embedding_length', cfg['E']),
    ('u32', 'janus.feed_forward_length', cfg['M']),
    ('u32', 'janus.attention.head_count', cfg['H']),
    ('u32', 'janus.attention.head_dim',   cfg['D']),
    ('u32', 'janus.block_count',          cfg['B']),
    ('u32', 'janus.vocab_size',           cfg['V']),
    ('u32', 'janus.rrpram.rank',          cfg['R']),
    ('u32', 'janus.rrpram.context',       cfg['T']),
]
write_gguf(gguf_path, kvs, tensors)

with open(exp_path, 'w') as f:
    json.dump({'cfg': cfg, 'tensors': expectations}, f, indent=2)

print(f"wrote {gguf_path} ({os.path.getsize(gguf_path):,} bytes), "
      f"{exp_path} ({len(expectations)} tensors)")
os.unlink(bin_path)
