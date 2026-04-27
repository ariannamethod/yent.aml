/* web/lib/bpe.js — BPE encode/decode + Janus chat-token helpers.
 *
 * Bit-for-bit port of nt_bpe_encode / nt_bpe_decode / bpe_build_decode_table
 * from notorch.c. Same merges array, same greedy left-to-right merge order,
 * same recursive decode through token byte ranges built from merges.
 *
 * The Janus chat tokens are special IDs >= vocab_size (32759..32767) that
 * never appear as merge results — they're injected by wrap()/decoded
 * by isBoundary().
 *
 *   const bpe = new BPE(merges);
 *   const ids  = bpe.encode("Who are you?");
 *   const text = bpe.decode(ids);
 *
 *   // Wrap for Janus 177M Yent SFT (chat-trained checkpoint):
 *   const ctx  = wrapJanusChat(bpe.encode("Who are you?"));
 *   //          = [BOS, USER_START, ...bpe..., USER_END, ASST_START]
 */

export const BOS         = 32759;
export const USER_START  = 32760;
export const USER_END    = 32761;
export const ASST_START  = 32762;
export const ASST_END    = 32763;
export const SPECIAL_TOKENS = new Set([BOS, USER_START, USER_END, ASST_START, ASST_END]);


/* ── BPE class — port of nt_bpe ─────────────────────────────────────── */

const _td = new TextDecoder('utf-8', { fatal: false });
const _te = new TextEncoder();

export class BPE {
    /* `merges` is either:
     *   - Array of [a, b] pairs, or
     *   - { merges: [...], vocab_size, n_merges } (from janus_v4_bpe_merges.json) */
    constructor(merges) {
        let pairs = merges;
        if (merges && merges.merges) pairs = merges.merges;
        if (!Array.isArray(pairs)) throw new Error('BPE: merges must be an array of [a,b]');

        this.merges    = pairs.map(([a, b]) => [a | 0, b | 0]);
        this.nMerges   = this.merges.length;
        this.vocabSize = 256 + this.nMerges;

        /* Decode table: each token id → array of bytes it expands to.
         * Built like notorch's bpe_build_decode_table — base bytes 0..255,
         * then each merge id 256+m = bytes(a) ++ bytes(b). */
        this.tokenBytes = new Array(this.vocabSize);
        for (let i = 0; i < 256; i++) this.tokenBytes[i] = new Uint8Array([i]);
        for (let m = 0; m < this.nMerges; m++) {
            const [a, b] = this.merges[m];
            const ba = this.tokenBytes[a];
            const bb = this.tokenBytes[b];
            const out = new Uint8Array(ba.length + bb.length);
            out.set(ba, 0);
            out.set(bb, ba.length);
            this.tokenBytes[256 + m] = out;
        }
    }

    encode(text) {
        if (typeof text !== 'string') return [];
        const bytes = _te.encode(text);
        const out = new Array(bytes.length);
        for (let i = 0; i < bytes.length; i++) out[i] = bytes[i];

        /* Greedy in merge order. notorch does in-place left-to-right scan
         * that compacts the array each time a pair fires. We replicate
         * exactly that, even though n^2 in the worst case — for a 32K
         * merges table on prompts of a few hundred tokens it's fine. */
        for (let m = 0; m < this.nMerges; m++) {
            const [a, b] = this.merges[m];
            const newId = 256 + m;
            let i = 0, n = out.length;
            while (i < n - 1) {
                if (out[i] === a && out[i + 1] === b) {
                    out[i] = newId;
                    out.splice(i + 1, 1);
                    n--;
                } else {
                    i++;
                }
            }
        }
        return out;
    }

    decode(tokens) {
        const parts = [];
        let total = 0;
        for (const id of tokens) {
            if (id < 0 || id >= this.vocabSize) continue;  // skip special / out-of-range
            const t = this.tokenBytes[id];
            parts.push(t);
            total += t.length;
        }
        const flat = new Uint8Array(total);
        let p = 0;
        for (const t of parts) { flat.set(t, p); p += t.length; }
        return _td.decode(flat);
    }
}


/* ── Loader for janus_v4_bpe_merges.json ─────────────────────────────── */

export async function loadBPEFromURL(url = './janus_v4_bpe_merges.json') {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);
    const j = await resp.json();
    return new BPE(j);
}


/* ── Chat token wrapping (matches dario/chain_dialogue.py:361 + yent.aml) ─ */

/* Wrap a question's BPE token sequence as the Janus 177M Yent SFT chat
 * format: [BOS, USER_START, ...question..., USER_END, ASST_START].
 * Generation continues until ASST_END. */
export function wrapJanusChat(questionIds) {
    return [BOS, USER_START, ...questionIds, USER_END, ASST_START];
}

/* Sentence-boundary check used by jannus-r when generating sentence-level
 * steps inside one assistant turn. Returns true if the decoded byte range
 * contains a sentence-ending mark or the special ASST_END. */
export function isBoundary(bpe, tok) {
    if (tok === ASST_END) return true;
    const bytes = bpe.tokenBytes[tok];
    if (!bytes) return false;
    for (let i = 0; i < bytes.length; i++) {
        const c = bytes[i];
        if (c === 0x2E || c === 0x21 || c === 0x3F || c === 0x0A) return true;  // . ! ? \n
    }
    return false;
}
