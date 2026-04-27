/* tests/test_soma.mjs — round-trip + cap behaviour for browser soma.
 *
 * Soma uses localStorage in the browser; under Node we install a small
 * in-memory shim before importing the module so the exact code path
 * fires. Tests:
 *
 *   1. load() on empty slot returns null
 *   2. save() then load() round-trips messages + tail
 *   3. messages over MSG_MAX (64) are trimmed to last 64
 *   4. tail over TAIL_MAX (256) is trimmed to last 256
 *   5. clear() removes the slot
 *   6. listSlots() returns names of saved slots
 *   7. corrupt JSON in storage → load() returns null (no throw)
 *   8. wrong schema version → load() returns null
 */

const _ls = {};
globalThis.localStorage = {
    getItem(k)        { return Object.prototype.hasOwnProperty.call(_ls, k) ? _ls[k] : null; },
    setItem(k, v)     { _ls[k] = String(v); },
    removeItem(k)     { delete _ls[k]; },
    key(i)            { return Object.keys(_ls)[i] ?? null; },
    get length()      { return Object.keys(_ls).length; },
};

const { Soma } = await import('../lib/soma.js');

let pass = 0, fail = 0;
const log = (cls, msg) => {
    console.log(`  ${cls === 'PASS' ? '✓' : '✗'} [${cls}] ${msg}`);
    cls === 'PASS' ? pass++ : fail++;
};

function reset() { for (const k of Object.keys(_ls)) delete _ls[k]; }

console.log(`== test_soma ==`);

/* 1. empty slot → null */
{
    reset();
    const s = new Soma('default');
    log(s.load() === null ? 'PASS' : 'FAIL', `empty slot → load() === null`);
}

/* 2. round-trip */
{
    reset();
    const s = new Soma('default');
    const msgs = [
        { role: 'user', text: 'who are you?' },
        { role: 'asst', text: 'I am Yent.' },
    ];
    const tail = [32759, 32760, 1234, 5678, 32761, 32762];
    s.save({ messages: msgs, tail });
    const got = s.load();
    const ok = got
        && got.messages.length === 2
        && got.messages[0].text === 'who are you?'
        && got.messages[1].text === 'I am Yent.'
        && got.tail.length === 6
        && got.tail[0] === 32759
        && got.tail[5] === 32762
        && Number.isFinite(got.savedAt);
    log(ok ? 'PASS' : 'FAIL', `save/load round-trip (msgs=${got?.messages.length} tail=${got?.tail.length})`);
}

/* 3. messages cap = 64 */
{
    reset();
    const s = new Soma('default');
    const huge = Array.from({ length: 100 }, (_, i) => ({ role: 'user', text: `m${i}` }));
    s.save({ messages: huge, tail: [] });
    const got = s.load();
    const ok = got
        && got.messages.length === 64
        && got.messages[0].text === 'm36'         // 100-64 = 36
        && got.messages[63].text === 'm99';
    log(ok ? 'PASS' : 'FAIL',
        `messages > 64 trimmed (got len=${got?.messages.length}, first='${got?.messages[0].text}')`);
}

/* 4. tail cap = 256 */
{
    reset();
    const s = new Soma('default');
    const huge = Array.from({ length: 1000 }, (_, i) => i);
    s.save({ messages: [], tail: huge });
    const got = s.load();
    const ok = got
        && got.tail.length === 256
        && got.tail[0]   === 744       // 1000-256
        && got.tail[255] === 999;
    log(ok ? 'PASS' : 'FAIL',
        `tail > 256 trimmed (got len=${got?.tail.length}, first=${got?.tail[0]})`);
}

/* 5. clear() removes slot */
{
    reset();
    const s = new Soma('default');
    s.save({ messages: [{ role: 'user', text: 'x' }], tail: [1] });
    s.clear();
    log(s.load() === null ? 'PASS' : 'FAIL', `clear() removes slot`);
}

/* 6. listSlots() */
{
    reset();
    new Soma('alpha').save({ messages: [], tail: [] });
    new Soma('beta').save({ messages: [], tail: [] });
    new Soma('gamma').save({ messages: [], tail: [] });
    const names = Soma.listSlots().sort();
    const ok = names.length === 3
        && names[0] === 'alpha' && names[1] === 'beta' && names[2] === 'gamma';
    log(ok ? 'PASS' : 'FAIL', `listSlots() = [${names.join(', ')}]`);
}

/* 7. corrupt JSON in storage → null, no throw */
{
    reset();
    localStorage.setItem('yent.aml.soma::default', '{not valid json');
    const s = new Soma('default');
    let got = undefined;
    try { got = s.load(); } catch (e) { got = '__threw__'; }
    log(got === null ? 'PASS' : 'FAIL', `corrupt JSON → load() returns null`);
}

/* 8. wrong schema version → null */
{
    reset();
    localStorage.setItem('yent.aml.soma::default',
        JSON.stringify({ v: 999, savedAt: 0, name: 'default', messages: [], tail: [] }));
    const s = new Soma('default');
    log(s.load() === null ? 'PASS' : 'FAIL', `unknown schema version → load() returns null`);
}

console.log(`\n== ${pass} PASS, ${fail} FAIL ==`);
process.exit(fail ? 1 : 0);
