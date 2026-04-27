/* web/lib/soma.js — browser-side persistence for yent.aml chat.
 *
 * The C side of yent.aml uses LOAD/SAVE directives in libaml that dump an
 * AM_State struct to a .soma file. The browser doesn't run libaml, so we
 * keep the *spirit* of cross-session memory without binding to C ABI: a
 * compact JSON soma in localStorage holding the chat history and a tail
 * of the most recent token sequence, so a page reload picks up where the
 * conversation left off.
 *
 * Format (JSON, schemaVersion 1):
 *   { v: 1, savedAt: <unix>, name: <slot>,
 *     messages: [ { role: 'user'|'asst', text: '...' }, ... ],
 *     tail:     [ tokenIds... ]   // last <= 256 ids, for prompt warm-start
 *   }
 *
 * Public API:
 *   const soma = new Soma('yent.default');     // slot name
 *   soma.load();                                // returns parsed obj or null
 *   soma.save({ messages, tail });
 *   soma.clear();
 *   Soma.listSlots();                           // browser-stored slot names
 */

const STORAGE_PREFIX = 'yent.aml.soma::';
const SCHEMA_VERSION = 1;
const TAIL_MAX       = 256;
const MSG_MAX        = 64;

function _store() {
    if (typeof localStorage !== 'undefined') return localStorage;
    return null;  // Node/test contexts — Soma is then a no-op.
}

export class Soma {
    constructor(name = 'default') {
        this.name = name;
        this.key  = STORAGE_PREFIX + name;
    }

    load() {
        const ls = _store();
        if (!ls) return null;
        const raw = ls.getItem(this.key);
        if (!raw) return null;
        try {
            const obj = JSON.parse(raw);
            if (obj.v !== SCHEMA_VERSION) return null;
            return {
                name:     obj.name || this.name,
                savedAt:  obj.savedAt | 0,
                messages: Array.isArray(obj.messages) ? obj.messages : [],
                tail:     Array.isArray(obj.tail)     ? obj.tail     : [],
            };
        } catch (e) {
            return null;
        }
    }

    save({ messages = [], tail = [] } = {}) {
        const ls = _store();
        if (!ls) return false;
        const trimmedMessages = messages.length > MSG_MAX
            ? messages.slice(messages.length - MSG_MAX) : messages.slice();
        const trimmedTail = tail.length > TAIL_MAX
            ? Array.from(tail.slice(tail.length - TAIL_MAX)) : Array.from(tail);
        const obj = {
            v:        SCHEMA_VERSION,
            savedAt:  Math.floor(Date.now() / 1000),
            name:     this.name,
            messages: trimmedMessages,
            tail:     trimmedTail,
        };
        try {
            ls.setItem(this.key, JSON.stringify(obj));
            return true;
        } catch (e) {
            return false;
        }
    }

    clear() {
        const ls = _store();
        if (ls) ls.removeItem(this.key);
    }

    static listSlots() {
        const ls = _store();
        if (!ls) return [];
        const out = [];
        for (let i = 0; i < ls.length; i++) {
            const k = ls.key(i);
            if (k && k.startsWith(STORAGE_PREFIX)) out.push(k.slice(STORAGE_PREFIX.length));
        }
        return out;
    }
}
