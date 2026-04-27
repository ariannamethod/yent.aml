/* web/lib/gguf.js — GGUF reader for the JS engine.
 *
 * Bit-for-bit port of notorch/gguf.c. Same magic ("GGUF"), same header
 * layout (version + n_tensors + n_kv + KV pairs + tensor metas + 32-byte
 * aligned data section), same dequant kernels (Q8_0, Q4_K, F16, F32).
 *
 * Usage:
 *   const gf = await loadGGUF(arrayBuffer);
 *   const wte = gf.dequant('transformer.wte.weight');   // Float32Array
 *
 * Q4_K layout matches our writer (yent.aml/tools/janus_to_gguf.py) which
 * we verified bit-correct against the C reader. Embeddings (wte, lm_head)
 * are usually Q8_0 baseline even in a Q4_K file — the dequantizer here
 * handles either type per tensor.
 */

const GGUF_MAGIC = 0x46554747;  // "GGUF" little-endian

const GGML_TYPE_F32   = 0;
const GGML_TYPE_F16   = 1;
const GGML_TYPE_Q4_0  = 2;
const GGML_TYPE_Q8_0  = 8;
const GGML_TYPE_Q4_K  = 12;

const GGUF_VT = {
    UINT8: 0, INT8: 1, UINT16: 2, INT16: 3,
    UINT32: 4, INT32: 5, FLOAT32: 6, BOOL: 7,
    STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11, FLOAT64: 12,
};


/* ── F16 → F32 (port of notorch/gguf.c f16_to_f32) ───────────────────── */

const _f16_buf  = new ArrayBuffer(4);
const _f16_u32  = new Uint32Array(_f16_buf);
const _f16_f32  = new Float32Array(_f16_buf);

function f16ToF32(h) {
    const sign = (h >> 15) & 1;
    let   exp  = (h >> 10) & 0x1F;
    let   mant = h & 0x3FF;
    if (exp === 0) {
        if (mant === 0) {
            _f16_u32[0] = sign << 31;
            return _f16_f32[0];
        }
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++;
        mant &= ~0x400;
    } else if (exp === 31) {
        _f16_u32[0] = (sign << 31) | 0x7F800000 | (mant << 13);
        return _f16_f32[0];
    }
    exp = exp + 127 - 15;
    _f16_u32[0] = (sign << 31) | (exp << 23) | (mant << 13);
    return _f16_f32[0];
}


/* ── Reader primitives ───────────────────────────────────────────────── */

class _Reader {
    constructor(buf) {
        this.dv     = new DataView(buf);
        this.bytes  = new Uint8Array(buf);
        this.pos    = 0;
    }
    u32() { const v = this.dv.getUint32(this.pos, true); this.pos += 4; return v; }
    i32() { const v = this.dv.getInt32 (this.pos, true); this.pos += 4; return v; }
    f32() { const v = this.dv.getFloat32(this.pos, true); this.pos += 4; return v; }
    u64() {
        const lo = this.dv.getUint32(this.pos, true);
        const hi = this.dv.getUint32(this.pos + 4, true);
        this.pos += 8;
        // GGUF tensor counts < 2^32 in our use, so this is safe.
        return hi * 0x100000000 + lo;
    }
    u8 () { return this.bytes[this.pos++]; }
    str() {
        const len = this.u64();
        const end = this.pos + len;
        const s = new TextDecoder('utf-8').decode(this.bytes.subarray(this.pos, end));
        this.pos = end;
        return s;
    }
    skip(n) { this.pos += n; }
}


/* ── KV value parser (mirrors notorch's skip_value layout) ───────────── */

function _readKVValue(r, type) {
    switch (type) {
        case GGUF_VT.UINT32:  return r.u32();
        case GGUF_VT.INT32:   return r.i32();
        case GGUF_VT.FLOAT32: return r.f32();
        case GGUF_VT.BOOL:    return r.u8() !== 0;
        case GGUF_VT.STRING:  return r.str();
        case GGUF_VT.UINT64:
        case GGUF_VT.INT64:
        case GGUF_VT.FLOAT64: return r.u64();
        case GGUF_VT.UINT8:
        case GGUF_VT.INT8:    return r.u8();
        case GGUF_VT.UINT16:
        case GGUF_VT.INT16:   { const v = r.dv.getInt16(r.pos, true); r.pos += 2; return v; }
        case GGUF_VT.ARRAY: {
            const atype = r.u32();
            const alen  = r.u64();
            const arr   = new Array(alen);
            for (let i = 0; i < alen; i++) arr[i] = _readKVValue(r, atype);
            return arr;
        }
        default:
            throw new Error(`unsupported KV type ${type}`);
    }
}


/* ── Dequantizers (port of notorch/gguf.c) ────────────────────────────── */

function dequantF32(src, n) {
    return new Float32Array(src.buffer, src.byteOffset, n).slice();
}

function dequantF16(src, n) {
    const out = new Float32Array(n);
    const dv  = new DataView(src.buffer, src.byteOffset, n * 2);
    for (let i = 0; i < n; i++) out[i] = f16ToF32(dv.getUint16(i * 2, true));
    return out;
}

function dequantQ8_0(src, n) {
    const nblk = n / 32;
    const out  = new Float32Array(n);
    const dv   = new DataView(src.buffer, src.byteOffset);
    for (let b = 0; b < nblk; b++) {
        const off  = b * 34;
        const scale = f16ToF32(dv.getUint16(off, true));
        for (let i = 0; i < 32; i++) {
            const q = dv.getInt8(off + 2 + i);
            out[b * 32 + i] = q * scale;
        }
    }
    return out;
}

/* Q4_K — same get_scale_min_k4 layout as notorch dequant_q4_k. */
function _scaleMinK4(j, sc) {
    let s, m;
    if (j < 4) {
        s = sc[j]     & 63;
        m = sc[j + 4] & 63;
    } else {
        s = (sc[j + 4] & 0x0F) | (((sc[j - 4] >> 6) & 0x03) << 4);
        m = (sc[j + 4] >> 4)   | (((sc[j]     >> 6) & 0x03) << 4);
    }
    return [s, m];
}

function dequantQ4_K(src, n) {
    const nblk = n / 256;
    const out  = new Float32Array(n);
    const dv   = new DataView(src.buffer, src.byteOffset);
    const u8   = new Uint8Array(src.buffer, src.byteOffset);
    for (let i = 0; i < nblk; i++) {
        const off  = i * 144;
        const d    = f16ToF32(dv.getUint16(off,     true));
        const dmin = f16ToF32(dv.getUint16(off + 2, true));
        const sc   = u8.subarray(off + 4,  off + 16);
        const qs   = u8.subarray(off + 16, off + 144);
        const oi   = i * 256;
        let is = 0, qi = 0;
        for (let j = 0; j < 256; j += 64) {
            const [sc0, m0] = _scaleMinK4(is,     sc);
            const [sc1, m1] = _scaleMinK4(is + 1, sc);
            const d1  = d * sc0,  mm1 = dmin * m0;
            const d2  = d * sc1,  mm2 = dmin * m1;
            for (let l = 0; l < 32; l++) {
                out[oi + j + l]      = d1 * (qs[qi + l] & 0x0F) - mm1;
                out[oi + j + 32 + l] = d2 * (qs[qi + l] >> 4)   - mm2;
            }
            qi += 32;
            is += 2;
        }
    }
    return out;
}


/* ── Public API ──────────────────────────────────────────────────────── */

const GGUF_TYPE_BLOCK = {
    [GGML_TYPE_F32]:  { bytes: 4,   block: 1   },
    [GGML_TYPE_F16]:  { bytes: 2,   block: 1   },
    [GGML_TYPE_Q8_0]: { bytes: 34,  block: 32  },
    [GGML_TYPE_Q4_K]: { bytes: 144, block: 256 },
};

export class GGUF {
    constructor(arrayBuffer) {
        this.buffer  = arrayBuffer;
        this.kv      = {};
        this.tensors = {};
        this._parse();
    }

    _parse() {
        const r = new _Reader(this.buffer);
        const magic = r.u32();
        if (magic !== GGUF_MAGIC) {
            throw new Error(`bad GGUF magic 0x${magic.toString(16)}`);
        }
        this.version  = r.u32();
        this.nTensors = r.u64();
        this.nKv      = r.u64();

        for (let i = 0; i < this.nKv; i++) {
            const key  = r.str();
            const type = r.u32();
            this.kv[key] = _readKVValue(r, type);
        }

        const tensorList = [];
        for (let i = 0; i < this.nTensors; i++) {
            const name = r.str();
            const ndim = r.u32();
            const shape = [];
            let n_elements = 1;
            for (let d = 0; d < ndim; d++) {
                const s = r.u64();
                shape.push(s);
                n_elements *= s;
            }
            const dtype  = r.u32();
            const offset = r.u64();
            const ti = { name, ndim, shape, dtype, offset, n_elements };
            this.tensors[name] = ti;
            tensorList.push(ti);
        }

        // Data section is 32-byte aligned from start of file.
        this.dataOffset = (r.pos + 31) & ~31;
    }

    dequant(name) {
        const ti = this.tensors[name];
        if (!ti) throw new Error(`tensor '${name}' not found`);
        const tb = GGUF_TYPE_BLOCK[ti.dtype];
        if (!tb) throw new Error(`tensor '${name}' has unsupported dtype ${ti.dtype}`);
        const nbytes = (ti.n_elements / tb.block) * tb.bytes;
        const src = new Uint8Array(this.buffer, this.dataOffset + ti.offset, nbytes);
        switch (ti.dtype) {
            case GGML_TYPE_F32:  return dequantF32(src, ti.n_elements);
            case GGML_TYPE_F16:  return dequantF16(src, ti.n_elements);
            case GGML_TYPE_Q8_0: return dequantQ8_0(src, ti.n_elements);
            case GGML_TYPE_Q4_K: return dequantQ4_K(src, ti.n_elements);
            default:
                throw new Error(`unsupported dtype ${ti.dtype} for '${name}'`);
        }
    }

    has(name) { return !!this.tensors[name]; }

    listTensors() { return Object.keys(this.tensors); }
}

export async function loadGGUFFromURL(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);
    const buf  = await resp.arrayBuffer();
    return new GGUF(buf);
}

/* Exported for tests / WebGPU upload paths. */
export { f16ToF32, dequantF32, dequantF16, dequantQ8_0, dequantQ4_K,
         GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K };
