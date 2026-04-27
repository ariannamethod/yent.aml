/* tests/test_bpe.mjs — Node-side parity check.
 *
 * Same checks as test_bpe.html but runs through `node tests/test_bpe.mjs`
 * so CI / Makefile / smoke scripts can verify without a browser. The
 * browser test stays as a visible artefact for hand-debugging.
 */
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

import { BPE, BOS, USER_START, USER_END, ASST_START, wrapJanusChat }
    from '../lib/bpe.js';

let pass = 0, fail = 0;
const log = (cls, msg) => {
    console.log(`  ${cls === 'PASS' ? '✓' : '✗'} [${cls}] ${msg}`);
    cls === 'PASS' ? pass++ : fail++;
};

const merges  = JSON.parse(await readFile(resolve(__dirname, '../janus_v4_bpe_merges.json'), 'utf8'));
const fixtures= JSON.parse(await readFile(resolve(__dirname, 'bpe_fixtures.json'),         'utf8'));
const bpe     = new BPE(merges);

console.log(`== test_bpe ==`);
log('PASS', `loaded BPE: vocab_size=${bpe.vocabSize} n_merges=${bpe.nMerges}`);
log('PASS', `loaded ${fixtures.length} C fixtures`);

for (const fx of fixtures) {
    const got = bpe.encode(fx.prompt);
    const eq  = got.length === fx.ids.length && got.every((v, i) => v === fx.ids[i]);
    log(eq ? 'PASS' : 'FAIL',
        `encode ${JSON.stringify(fx.prompt).slice(0, 50)} → ${got.length} ids` +
        (eq ? '' : ` (expected ${JSON.stringify(fx.ids)} got ${JSON.stringify(got)})`));
}

for (const fx of fixtures) {
    const got = bpe.decode(fx.ids);
    log(got === fx.decoded ? 'PASS' : 'FAIL',
        `decode ${fx.ids.length} ids → ${JSON.stringify(got).slice(0, 50)}` +
        (got === fx.decoded ? '' : ` (expected ${JSON.stringify(fx.decoded)})`));
}

{
    const ids  = bpe.encode("Who are you?");
    const ctx  = wrapJanusChat(ids);
    const want = [BOS, USER_START, ...ids, USER_END, ASST_START];
    const eq   = ctx.length === want.length && ctx.every((v, i) => v === want[i]);
    log(eq ? 'PASS' : 'FAIL', `wrapJanusChat layout (len=${ctx.length})`);
}

console.log(`\n== ${pass} PASS, ${fail} FAIL ==`);
process.exit(fail ? 1 : 0);
