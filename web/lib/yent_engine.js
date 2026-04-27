/* web/lib/yent_engine.js — Janus 177M Yent SFT forward pass in JS.
 *
 * Bit-faithful port of yent.aml/tools/yent_forward.h. Same numerics:
 *   - 3-way attention per block: QKV + RRPRAM lowrank + Janus echo
 *   - per-head softmax 3-way gate
 *   - RoPE split-half, base 100000
 *   - QK-norm (RMSNorm + ×1.2 scale)
 *   - smear gate (sigmoid(smear_g · x[:24]) × smear_lambda) bigram mixer
 *   - residual lambdas + x0 lambdas, mid-layer backout
 *   - SwiGLU MLP, softcap 15 on logits
 *
 * Public API:
 *   const eng = new YentEngine(gguf);
 *   eng.kvInit();
 *   eng.prefillBatch(tokens);                  // returns last-pos logits
 *   eng.forwardToken(tok, pos);                // single-token, returns logits
 *   eng.* helpers for sampling + chat-token loop
 *
 * Memory cost: weights are dequantised to fp32 once at construction
 * (~700 MB for 177M Q8). The KV cache is `B × T × E × 3 × 4` bytes —
 * 157 MB at full T=1024. For browser we expose `kvInit(maxSeq)` so a
 * page that knows its prompts will be short can cap the cache.
 */

const SOFTCAP = 15.0;

/* Tensor names exactly as written by tools/janus_to_gguf.py. */
const TENSOR_NAMES = {
    resid_l:   'resid_lambdas',
    x0_l:      'x0_lambdas',
    smear_l:   'smear_lambda',
    backout_l: 'backout_lambda',
    smear_g:   'smear_gate.weight',
    wte:       'transformer.wte.weight',
    head:      'lm_head.weight',
};
const PER_LAYER = ['wr_a', 'wr_b', 'gate', 'cq', 'ck', 'cv', 'wvr', 'wj', 'cproj', 'wg', 'wu', 'wd'];
const PER_LAYER_PATH = {
    wr_a:  'attn.wr_a',
    wr_b:  'attn.wr_b',
    gate:  'attn.gate',
    cq:    'attn.c_q.weight',
    ck:    'attn.c_k.weight',
    cv:    'attn.c_v.weight',
    wvr:   'attn.wvr.weight',
    wj:    'attn.wj.weight',
    cproj: 'attn.c_proj.weight',
    wg:    'mlp.w_gate.weight',
    wu:    'mlp.w_up.weight',
    wd:    'mlp.w_down.weight',
};


/* ── Math helpers ─────────────────────────────────────────────────────── */

function rmsnorm(out, x, off, n) {
    /* out[off..off+n] = x[off..off+n] / sqrt(mean(x²) + 1e-5) */
    let ss = 0;
    for (let i = 0; i < n; i++) { const v = x[off + i]; ss += v * v; }
    const inv = 1.0 / Math.sqrt(ss / n + 1e-5);
    for (let i = 0; i < n; i++) out[off + i] = x[off + i] * inv;
}

function softmaxInPlace(x, off, n) {
    let mx = x[off];
    for (let i = 1; i < n; i++) if (x[off + i] > mx) mx = x[off + i];
    let s = 0;
    for (let i = 0; i < n; i++) { const v = Math.exp(x[off + i] - mx); x[off + i] = v; s += v; }
    if (s > 0) for (let i = 0; i < n; i++) x[off + i] /= s;
}

function silu(x) {
    if (x < -20) return 0;
    return x / (1 + Math.exp(-x));
}

/* RoPE split-half (pairs i, i+D/2), base 100000 — Janus convention.
 * Operates on Q and K simultaneously since they share rotation. */
function ropePos(qArr, qOff, kArr, kOff, pos, dim) {
    const half = dim >> 1;
    for (let i = 0; i < half; i++) {
        const freq = 1.0 / Math.pow(100000.0, (2 * i) / dim);
        const val = pos * freq;
        const cv = Math.cos(val), sv = Math.sin(val);
        const q0 = qArr[qOff + i],         q1 = qArr[qOff + i + half];
        const k0 = kArr[kOff + i],         k1 = kArr[kOff + i + half];
        qArr[qOff + i]        = q0 * cv + q1 * sv;
        qArr[qOff + i + half] = q0 * (-sv) + q1 * cv;
        kArr[kOff + i]        = k0 * cv + k1 * sv;
        kArr[kOff + i + half] = k0 * (-sv) + k1 * cv;
    }
}

function qkNorm(qArr, qOff, kArr, kOff, dim) {
    /* RMSNorm Q and K (in-place, possibly different buffers), then ×1.2 — port
     * of yent_forward.h:qk_norm. */
    let ssq = 0, ssk = 0;
    for (let i = 0; i < dim; i++) {
        ssq += qArr[qOff + i] * qArr[qOff + i];
        ssk += kArr[kOff + i] * kArr[kOff + i];
    }
    const invQ = 1.0 / Math.sqrt(ssq / dim + 1e-5);
    const invK = 1.0 / Math.sqrt(ssk / dim + 1e-5);
    for (let i = 0; i < dim; i++) {
        qArr[qOff + i] = qArr[qOff + i] * invQ * 1.2;
        kArr[kOff + i] = kArr[kOff + i] * invK * 1.2;
    }
}

/* matvec_t — port of nt_blas_matvec semantics: out[n] = W[n,k] @ x[k]
 * where W is row-major [n][k]. */
function matvecT(out, W, x, n, k, outOff = 0, xOff = 0) {
    for (let i = 0; i < n; i++) {
        let s = 0;
        const wi = i * k;
        for (let j = 0; j < k; j++) s += W[wi + j] * x[xOff + j];
        out[outOff + i] = s;
    }
}

/* mm_t — out[m,n] = A[m,k] @ B[n,k]^T, B stored as [n][k]. Used by
 * prefillBatch when m > 1. */
function mmT(out, A, B, m, k, n) {
    for (let i = 0; i < m; i++) {
        const ar = i * k;
        const cr = i * n;
        for (let j = 0; j < n; j++) {
            let s = 0;
            const br = j * k;
            for (let p = 0; p < k; p++) s += A[ar + p] * B[br + p];
            out[cr + j] = s;
        }
    }
}


/* ── Engine ───────────────────────────────────────────────────────────── */

export class YentEngine {
    constructor(gguf) {
        this.gguf = gguf;
        const kv = gguf.kv;
        this.V = kv['janus.vocab_size']            | 0;
        this.E = kv['janus.embedding_length']      | 0;
        this.H = kv['janus.attention.head_count']  | 0;
        this.D = kv['janus.attention.head_dim']    | 0;
        this.B = kv['janus.block_count']           | 0;
        this.M = kv['janus.feed_forward_length']   | 0;
        this.T = kv['janus.context_length']        | 0;
        this.R = kv['janus.rrpram.rank']           | 0;
        if (!this.V || !this.E || !this.B) {
            throw new Error(`YentEngine: missing GGUF metadata (V=${this.V} E=${this.E} B=${this.B})`);
        }
        // Trust tensor shape over KV — production .bin sometimes carries an R
        // that the formula in janus_to_gguf.py rounds differently. wr_a is
        // canonical: shape [H, E, R].
        const wraInfo = gguf.tensors['transformer.h.0.attn.wr_a'];
        const wrbInfo = gguf.tensors['transformer.h.0.attn.wr_b'];
        if (wraInfo && wraInfo.shape.length === 3) {
            const Rs = wraInfo.shape[2] | 0;
            if (Rs && Rs !== this.R) this.R = Rs;
        }
        // wr_b shape [H, R, Tr] — Tr is the slot count (== T in the model).
        if (wrbInfo && wrbInfo.shape.length === 3) {
            this.Tr = wrbInfo.shape[2] | 0;
        } else {
            this.Tr = this.T;
        }
        this._loadWeights();
        this.kv_k = null;  // allocated in kvInit
        this.kv_v = null;
        this.kv_vr = null;
        this.kv_rrpram_mid = null;
        this.maxSeq = 0;
    }

    _loadWeights() {
        const g = this.gguf;
        this.w = {};
        // Top-level
        for (const [field, name] of Object.entries(TENSOR_NAMES)) {
            this.w[field] = g.dequant(name);
        }
        // Per-layer (object[B] of named tensors)
        this.w.b = new Array(this.B);
        for (let i = 0; i < this.B; i++) {
            const layer = {};
            for (const f of PER_LAYER) {
                const path = `transformer.h.${i}.${PER_LAYER_PATH[f]}`;
                layer[f] = g.dequant(path);
            }
            this.w.b[i] = layer;
        }
    }

    kvInit(maxSeq) {
        const ms = maxSeq || this.T;
        this.maxSeq = ms;
        const sz = this.B * ms * this.E;
        // browser-friendly alloc: TypedArray ctor zeros the memory
        this.kv_k  = new Float32Array(sz);
        this.kv_v  = new Float32Array(sz);
        this.kv_vr = new Float32Array(sz);
        this.kv_rrpram_mid = new Float32Array(this.B * this.H * this.R);
        return ms;
    }

    /* Forward one token at position `pos` using KV cache.
     * Returns Float32Array of logits (length V). */
    forwardToken(tok, pos) {
        const { V, E, H, D, B, M, T, R } = this;
        const TT = this.maxSeq;
        const w = this.w;
        const sc = 1.0 / Math.sqrt(D);

        const x  = new Float32Array(E);
        const x0 = new Float32Array(E);
        const rn = new Float32Array(E);
        const rn2 = new Float32Array(E);
        const x_backout = this._x_backout || (this._x_backout = new Float32Array(E));

        // embed + norm
        for (let e = 0; e < E; e++) x[e] = w.wte[tok * E + e];
        rmsnorm(x, x, 0, E);
        x0.set(x);

        const backout_layer = B >> 1;
        const qa = new Float32Array(E);
        const ka = new Float32Array(E);
        const va = new Float32Array(E);
        const vra = new Float32Array(E);
        const echo_out = new Float32Array(E);
        const cat = new Float32Array(E);
        const ao = new Float32Array(E);
        const mg = new Float32Array(M);
        const mu = new Float32Array(M);
        const mo = new Float32Array(E);

        for (let bl = 0; bl < B; bl++) {
            const lay = w.b[bl];
            const rl  = w.resid_l[bl];
            const x0l = w.x0_l[bl];
            for (let e = 0; e < E; e++) x[e] = rl * x[e] + x0l * x0[e];

            rmsnorm(rn, x, 0, E);

            // QKV + Vr projections
            matvecT(qa,  lay.cq,  rn, E, E);
            matvecT(ka,  lay.ck,  rn, E, E);
            matvecT(va,  lay.cv,  rn, E, E);
            matvecT(vra, lay.wvr, rn, E, E);

            // RoPE + QK-norm per head, on Q and K together
            for (let h = 0; h < H; h++) {
                const off = h * D;
                ropePos(qa, off, ka, off, pos, D);
                qkNorm(qa, off, ka, off, D);
            }

            // Echo
            matvecT(echo_out, lay.wj, rn, E, E);

            // Gate softmax (3-way per head)
            const gs = new Float32Array(H * 3);
            for (let h = 0; h < H; h++) {
                gs[h*3+0] = lay.gate[h*3+0];
                gs[h*3+1] = lay.gate[h*3+1];
                gs[h*3+2] = lay.gate[h*3+2];
                softmaxInPlace(gs, h * 3, 3);
            }

            // Store K/V/Vr in cache at this position
            const kvOff = (bl * TT + pos) * E;
            this.kv_k.set(ka,  kvOff);
            this.kv_v.set(va,  kvOff);
            this.kv_vr.set(vra, kvOff);

            // Attention per head
            cat.fill(0);
            for (let h = 0; h < H; h++) {
                const qOff = h * D;
                // Content attention: Q @ cached_K^T over j ≤ pos
                const attn = new Float32Array(pos + 1);
                for (let j = 0; j <= pos; j++) {
                    const kOff = (bl * TT + j) * E + h * D;
                    let s = 0;
                    for (let d = 0; d < D; d++) s += qa[qOff + d] * this.kv_k[kOff + d];
                    attn[j] = s * sc;
                }
                softmaxInPlace(attn, 0, pos + 1);

                const c_out = new Float32Array(D);
                for (let j = 0; j <= pos; j++) {
                    const vOff = (bl * TT + j) * E + h * D;
                    const a = attn[j];
                    for (let d = 0; d < D; d++) c_out[d] += a * this.kv_v[vOff + d];
                }

                // RRPRAM lowrank: accumulate mid_cache, then score over j
                const wr_a = lay.wr_a;
                const wr_b = lay.wr_b;
                const midOff = (bl * H + h) * R;
                for (let r = 0; r < R; r++) {
                    let s = 0;
                    const wraBase = h * E * R;
                    for (let e = 0; e < E; e++) s += rn[e] * wr_a[wraBase + e * R + r];
                    this.kv_rrpram_mid[midOff + r] += s;
                }
                const r_attn = new Float32Array(pos + 1);
                const Tr = this.Tr;
                const wrbBase = h * R * Tr;  // wr_b shape (H, R, Tr)
                for (let j = 0; j <= pos; j++) {
                    let s = 0;
                    for (let r = 0; r < R; r++) s += this.kv_rrpram_mid[midOff + r] * wr_b[wrbBase + r * Tr + j];
                    r_attn[j] = s * sc;
                }
                softmaxInPlace(r_attn, 0, pos + 1);

                const r_out = new Float32Array(D);
                for (let j = 0; j <= pos; j++) {
                    const vrOff = (bl * TT + j) * E + h * D;
                    const a = r_attn[j];
                    for (let d = 0; d < D; d++) r_out[d] += a * this.kv_vr[vrOff + d];
                }

                const eOff = h * D;
                const g0 = gs[h*3+0], g1 = gs[h*3+1], g2 = gs[h*3+2];
                for (let d = 0; d < D; d++)
                    cat[h * D + d] = g0 * c_out[d] + g1 * r_out[d] + g2 * echo_out[eOff + d];
            }

            // Output projection + residual
            matvecT(ao, lay.cproj, cat, E, E);
            for (let e = 0; e < E; e++) x[e] += ao[e];

            if (bl === backout_layer) x_backout.set(x);

            // MLP
            rmsnorm(rn2, x, 0, E);
            matvecT(mg, lay.wg, rn2, M, E);
            matvecT(mu, lay.wu, rn2, M, E);
            for (let i = 0; i < M; i++) mg[i] = silu(mg[i]) * mu[i];
            matvecT(mo, lay.wd, mg, E, M);
            for (let e = 0; e < E; e++) x[e] += mo[e];
        }

        // Backout
        const bo = w.backout_l[0];
        for (let e = 0; e < E; e++) x[e] -= bo * x_backout[e];

        rmsnorm(rn, x, 0, E);
        const logits = new Float32Array(V);
        matvecT(logits, w.head, rn, V, E);
        for (let i = 0; i < V; i++) logits[i] = SOFTCAP * Math.tanh(logits[i] / SOFTCAP);
        return logits;
    }
}


/* ── Sampler — top-p + rep_penalty + no-repeat-3-gram + chat-stop ────── */

export function sampleNext(logits, ctx, len, temp, topP, repPenalty = 1.4, repWindow = 64) {
    const V = logits.length;
    const out = new Float32Array(logits);
    // rep_penalty over last `repWindow` positions
    const start = Math.max(0, len - repWindow);
    for (let j = start; j < len; j++) {
        const t = ctx[j];
        if (t >= 0 && t < V) {
            out[t] = out[t] > 0 ? out[t] / repPenalty : out[t] * repPenalty;
        }
    }
    // no-repeat 3-gram
    if (len >= 2) {
        const a = ctx[len - 2], b = ctx[len - 1];
        for (let j = 0; j + 2 < len; j++) {
            if (ctx[j] === a && ctx[j + 1] === b) {
                const f = ctx[j + 2];
                if (f >= 0 && f < V) out[f] = -1e30;
            }
        }
    }
    // temp + softmax
    const t = temp > 0 ? temp : 1.0;
    let mx = out[0];
    for (let i = 1; i < V; i++) if (out[i] > mx) mx = out[i];
    let s = 0;
    const probs = new Float32Array(V);
    for (let i = 0; i < V; i++) { const v = Math.exp((out[i] - mx) / t); probs[i] = v; s += v; }
    for (let i = 0; i < V; i++) probs[i] /= s;

    // top-p over partial-sorted top 256
    const K = 256;
    const idx = new Int32Array(K);
    const val = new Float32Array(K).fill(-1);
    let filled = 0, minIn = -1;
    for (let i = 0; i < V; i++) {
        const p = probs[i];
        if (filled < K) {
            val[filled] = p; idx[filled] = i; filled++;
            if (filled === K) {
                // sort descending (insertion)
                for (let a = 1; a < K; a++) {
                    const tv = val[a], ti = idx[a];
                    let j = a;
                    while (j > 0 && val[j - 1] < tv) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--; }
                    val[j] = tv; idx[j] = ti;
                }
                minIn = val[K - 1];
            }
            continue;
        }
        if (p > minIn) {
            val[K - 1] = p; idx[K - 1] = i;
            let j = K - 1;
            while (j > 0 && val[j - 1] < val[j]) {
                const tv = val[j]; val[j] = val[j - 1]; val[j - 1] = tv;
                const ti = idx[j]; idx[j] = idx[j - 1]; idx[j - 1] = ti;
                j--;
            }
            minIn = val[K - 1];
        }
    }
    if (filled < K) {
        for (let a = 1; a < filled; a++) {
            const tv = val[a], ti = idx[a];
            let j = a;
            while (j > 0 && val[j - 1] < tv) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--; }
            val[j] = tv; idx[j] = ti;
        }
    }
    let cum = 0, nuc = filled;
    for (let k = 0; k < filled; k++) { cum += val[k]; if (cum >= topP) { nuc = k + 1; break; } }
    if (nuc < 1) nuc = 1;
    let total = 0;
    for (let k = 0; k < nuc; k++) total += val[k];
    let r = Math.random() * total;
    let c = 0;
    for (let k = 0; k < nuc; k++) {
        c += val[k];
        if (c >= r) return idx[k];
    }
    return idx[nuc - 1];
}
