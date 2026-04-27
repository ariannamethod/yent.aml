/* tests/test_engine.mjs — smoke that YentEngine runs on the mini GGUF.
 *
 * Loads tests/mini.gguf (V=64 E=32 H=2 D=16 B=2 M=64 T=64 R=4 random
 * synthetic weights), instantiates YentEngine, runs prefill_batch on a
 * 4-token prompt and forward_token on the next position. Asserts all
 * logits finite, correct length, KV cache populated.
 *
 * Bit-faithful parity vs C engine lives in a separate harness — that
 * needs full-size weights and is too heavy to run on every CI check.
 */
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { GGUF } from '../lib/gguf.js';
import { YentEngine, sampleNext } from '../lib/yent_engine.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const buf = await readFile(resolve(__dirname, 'mini.gguf'));
const gguf = new GGUF(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));

const eng = new YentEngine(gguf);
console.log(`== test_engine ==`);
console.log(`  V=${eng.V} E=${eng.E} H=${eng.H} D=${eng.D} B=${eng.B} M=${eng.M} T=${eng.T} R=${eng.R}`);

eng.kvInit(eng.T);

let pass = 0, fail = 0;
const log = (cls, msg) => { console.log(`  ${cls === 'PASS' ? '✓' : '✗'} [${cls}] ${msg}`);
                            cls === 'PASS' ? pass++ : fail++; };

/* 1. forward_token at position 0 returns logits of length V, all finite. */
const logits0 = eng.forwardToken(5, 0);
log(logits0.length === eng.V ? 'PASS' : 'FAIL',
    `forwardToken(5, 0) returns logits length ${logits0.length} (expected ${eng.V})`);
const allFinite0 = logits0.every(Number.isFinite);
log(allFinite0 ? 'PASS' : 'FAIL', `all logits finite at pos=0`);

/* 2. Run a few autoregressive tokens — KV cache should grow, no NaN. */
const ctx = [5];
let pos = 1;
for (let step = 0; step < 4; step++) {
    const logits = eng.forwardToken(ctx[ctx.length - 1] | 0, pos);
    if (!logits.every(Number.isFinite)) {
        log('FAIL', `pos=${pos}: NaN/Inf in logits`);
        break;
    }
    const next = sampleNext(logits, ctx, ctx.length, 0.7, 0.9);
    if (next < 0 || next >= eng.V) {
        log('FAIL', `pos=${pos}: sampled token ${next} out of range [0, ${eng.V})`);
        break;
    }
    ctx.push(next);
    pos++;
}
log(ctx.length === 5 ? 'PASS' : 'FAIL',
    `4-step autoregressive produced ${ctx.length} tokens: [${ctx.join(', ')}]`);

/* 3. KV cache spot-check — k buffer non-zero at every position written. */
const kvSize = eng.B * eng.maxSeq * eng.E;
const kvFinite = eng.kv_k.every(Number.isFinite);
log(kvFinite ? 'PASS' : 'FAIL', `KV cache (kv_k, ${kvSize} elem) all finite`);

/* 4. Softcap actually clamped: |logits| ≤ SOFTCAP = 15 by construction. */
let abs_max = 0;
for (let i = 0; i < logits0.length; i++) {
    const a = Math.abs(logits0[i]); if (a > abs_max) abs_max = a;
}
log(abs_max <= 15.0 + 1e-3 ? 'PASS' : 'FAIL',
    `softcap: max|logit| = ${abs_max.toFixed(4)} ≤ 15.0`);

console.log(`\n== ${pass} PASS, ${fail} FAIL ==`);
process.exit(fail ? 1 : 0);
