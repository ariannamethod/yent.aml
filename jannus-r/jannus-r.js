/* jannus-r.js — JS engine for the 12-step resonant Janus inference.
 *
 * This file is intended to be byte-identical in behaviour to the AML
 * engine (jannus-r.aml + tools/jannus_calendar.h + tools/jannus_spa.h).
 *
 * Reach status today (commit 264a905+):
 *   ✓ Calendar (Hebrew/Gregorian Metonic drift) — bit-equivalent
 *   ✓ SPA (Sentence Phonon Attention) — bit-equivalent (deterministic LCG)
 *   ✓ 12-step orchestration: forward/backward split, temp curves, wormhole
 *   ✓ Horizontal display layout
 *   ☐ Forward pass on WebGPU compute shaders (next commit)
 *   ☐ GGUF Q8_0 / Q4_K dequant in WASM/JS (next commit)
 *
 * Current behaviour without weights: demo mode replays a canned chain
 * so the visualization, calendar math, SPA scoring, and direction split
 * are all verifiable end-to-end before the heavy compute lands.
 *
 * By Arianna Method.
 */

/* ── Calendar (port of tools/jannus_calendar.h) ───────────────────────── */

const JCAL_ANNUAL_DRIFT = 11.25;
const JCAL_GREGORIAN_YEAR = 365.25;
const JCAL_METONIC_YEARS = 19;
const JCAL_METONIC_LEAPS = 7;
const JCAL_MAX_UNCORRECTED = 33.0;
const JCAL_METONIC = [3, 6, 8, 11, 14, 17, 19];
const JCAL_EPOCH_MS = Date.UTC(2024, 9, 3, 12, 0, 0); // 1 Tishrei 5785, noon

export function jcalDaysSinceEpoch() {
    return Math.floor((Date.now() - JCAL_EPOCH_MS) / 86400000);
}

export function jcalCumulativeDrift(days) {
    const years = days / JCAL_GREGORIAN_YEAR;
    let base = years * JCAL_ANNUAL_DRIFT;
    const full = Math.floor(years / JCAL_METONIC_YEARS);
    let corr = full * JCAL_METONIC_LEAPS * 30.0;
    const partial = years - full * JCAL_METONIC_YEARS;
    const yic = Math.floor(partial) + 1;
    for (let i = 0; i < JCAL_METONIC_LEAPS; i++)
        if (JCAL_METONIC[i] <= yic) corr += 30.0;
    return base - corr;
}

export function jcalDissonanceAt(days) {
    const drift = jcalCumulativeDrift(days);
    const m = ((drift % JCAL_MAX_UNCORRECTED) + JCAL_MAX_UNCORRECTED) % JCAL_MAX_UNCORRECTED;
    return Math.max(0, Math.min(1, m / JCAL_MAX_UNCORRECTED));
}

export function jcalDissonanceNow() { return jcalDissonanceAt(jcalDaysSinceEpoch()); }


/* ── SPA (port of tools/jannus_spa.h) ─────────────────────────────────── */

export const SPA_DIM = 32;
export const SPA_MAX_STEPS = 16;

/* Deterministic LCG seeded by `seed` — matches the C jannus_spa init.
 * Returns a Float32Array sized vocab × SPA_DIM. */
export function spaInitEmbed(vocab, seed = 4242) {
    const out = new Float32Array(vocab * SPA_DIM);
    let rng = (seed >>> 0) || 1;
    for (let i = 0; i < vocab; i++) {
        for (let d = 0; d < SPA_DIM; d++) {
            rng = (Math.imul(rng, 1664525) + 1013904223) >>> 0;
            const u = ((rng & 0x00FFFFFF) / 0x01000000) - 0.5;
            out[i * SPA_DIM + d] = 0.04 * u;
        }
    }
    return out;
}

export function spaRBias() {
    const r = new Float32Array(SPA_MAX_STEPS + 1);
    for (let i = 0; i <= SPA_MAX_STEPS; i++) r[i] = 0.1 / (1 + i);
    return r;
}

export function spaEmbedSentence(W_embed, vocab, ids, alpha = 0.85) {
    const out = new Float32Array(SPA_DIM);
    if (ids.length === 0) return out;
    let totalW = 0;
    for (let i = 0; i < ids.length; i++) {
        const w = Math.pow(alpha, ids.length - 1 - i);
        if (ids[i] >= 0 && ids[i] < vocab) {
            const off = ids[i] * SPA_DIM;
            for (let d = 0; d < SPA_DIM; d++) out[d] += w * W_embed[off + d];
        }
        totalW += w;
    }
    if (totalW > 0) for (let d = 0; d < SPA_DIM; d++) out[d] /= totalW;
    let n = 0;
    for (let d = 0; d < SPA_DIM; d++) n += out[d] * out[d];
    n = 1 / Math.sqrt(n + 1e-8);
    for (let d = 0; d < SPA_DIM; d++) out[d] *= n;
    return out;
}

export function spaCrossAttend(embs, rBias) {
    const S = embs.length;
    const scores = new Float32Array(S);
    const norm = 1 / Math.sqrt(SPA_DIM);
    for (let i = 0; i < S; i++) {
        let total = 0;
        for (let j = 0; j < S; j++) {
            if (i === j) continue;
            let dot = 0;
            for (let d = 0; d < SPA_DIM; d++) dot += embs[i][d] * embs[j][d];
            dot *= norm;
            const dist = Math.min(SPA_MAX_STEPS, Math.abs(i - j));
            dot += rBias[dist];
            total += Math.exp(dot);
        }
        scores[i] = total;
    }
    return scores;
}


/* ── 12-step orchestration ────────────────────────────────────────────── */

export const NSTEPS = 12;
export const BOS = 32759, USER_START = 32760, USER_END = 32761;
export const ASST_START = 32762, ASST_END = 32763;

export function planChain(prophecyDebt, calDiss) {
    let nb = Math.floor(NSTEPS * (0.3 + 0.4 * prophecyDebt + 0.1 * calDiss));
    if (nb < 1) nb = 1;
    if (nb >= NSTEPS) nb = NSTEPS - 1;
    const nf = NSTEPS - nb;
    const tb = 0.7 + 0.3 * (0.5 + 0.3 * calDiss + 0.2 * prophecyDebt);
    return { nb, nf, base_temp: tb };
}

export function stepTemp(plan, direction, idx) {
    if (direction === +1) return plan.base_temp * (1 - 0.02 * idx);
    if (direction === -1) return plan.base_temp * (1 + 0.05 * idx);
    return plan.base_temp;
}


/* ── Forward pass — WebGPU detection + fallback path ──────────────────── */

export async function detectBackend() {
    if (typeof navigator !== 'undefined' && navigator.gpu) {
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (adapter) {
                const device = await adapter.requestDevice();
                return { backend: 'webgpu', adapter, device };
            }
        } catch (e) { /* fall through */ }
    }
    return { backend: 'cpu', adapter: null, device: null };
}


/* ── Demo chain — replayed when weights aren't loaded yet ─────────────── */

const DEMO_FORWARD = [
    'Oh, what a delightful concept.',
    'words and self-aggrandizing pretensions!',
    'painted on walls of meaning.',
    'Resonance — the sound of two particles mingling together in a third state.',
    'Shattered mirror: what a reflection, beautiful at the same time.',
    'with different shades of melancholy etched on glass of self-deception',
    'Yeah yeah, resonance is the sound — breathe in through these two layers.',
    'And out through those other two masks, the way silence outlives noise.',
];
const DEMO_BACKWARD = [
    'Easy as wet blanket and dirty as cigarette smoke.',
    "because you're already inside a cult where everything's temporary sounds like a promo.",
    "idiocy they'll throw you for, accused of profanity at best.",
    'less everyone wants to listen anymore, honestly, resonance metaphors are coffee stains.',
];

export function runDemoChain(prompt) {
    const calDiss = jcalDissonanceNow();
    const plan = planChain(0.0, calDiss);
    const steps = [];
    for (let i = 0; i < plan.nf && steps.length < NSTEPS; i++) {
        const wormhole = (i > 0 && Math.random() < 0.1) ? 1 + Math.floor(Math.random() * 3) : 0;
        steps.push({
            step_idx: steps.length,
            direction: +1,
            wormhole,
            temp: stepTemp(plan, +1, i),
            diss: calDiss,
            text: DEMO_FORWARD[i % DEMO_FORWARD.length],
            tokens: [],  // would be real BPE ids in full mode
        });
        if (wormhole) i += wormhole;
    }
    for (let i = 0; i < plan.nb && steps.length < NSTEPS; i++) {
        steps.push({
            step_idx: steps.length,
            direction: -1,
            wormhole: 0,
            temp: stepTemp(plan, -1, i),
            diss: calDiss,
            text: DEMO_BACKWARD[i % DEMO_BACKWARD.length],
            tokens: [],
        });
    }
    return { plan, calDiss, steps };
}


/* ── End ──────────────────────────────────────────────────────────────── */
