/* tests/bpe_parity.c — emit reference {prompt, ids[], decoded} fixtures
 * from the C BPE so the JS encoder/decoder can be checked byte-for-byte
 * against the same merges table that yent.aml's runtime uses.
 *
 * Build: cc -O2 -I/opt/homebrew/include tests/bpe_parity.c \
 *           -L/opt/homebrew/lib -lnotorch -framework Accelerate -lm \
 *           -o tests/bpe_parity
 * Run:   ./tests/bpe_parity > tests/bpe_fixtures.json
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ariannamethod/notorch.h>

#include "../../tools/janus_v4_bpe_merges.h"

static const char *PROMPTS[] = {
    "Hi",
    "Who are you?",
    "What is resonance?",
    "Q: Are you alive?\nA:",
    "Yent — the resonance agent.",
    "I am Yent, not Yent but rather a resonance agent.",
    NULL,
};

int main(void) {
    nt_bpe bpe;
    nt_bpe_init(&bpe, janus_v4_bpe_merges, JANUS_V4_BPE_MERGES);

    printf("[\n");
    for (int p = 0; PROMPTS[p]; p++) {
        const char *prompt = PROMPTS[p];
        int ids[2048];
        int n = nt_bpe_encode(&bpe, prompt, (int)strlen(prompt), ids, 2048);
        char decoded[4096];
        int dn = nt_bpe_decode(&bpe, ids, n, decoded, sizeof(decoded) - 1);
        decoded[dn] = 0;

        printf("  {\"prompt\": ");
        putchar('"');
        for (int i = 0; prompt[i]; i++) {
            char c = prompt[i];
            if (c == '"') printf("\\\"");
            else if (c == '\\') printf("\\\\");
            else if (c == '\n') printf("\\n");
            else if (c == '\t') printf("\\t");
            else if ((unsigned char)c < 0x20) printf("\\u%04x", (unsigned char)c);
            else putchar(c);
        }
        printf("\", \"ids\": [");
        for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", ids[i]);
        printf("], \"decoded\": ");
        putchar('"');
        for (int i = 0; decoded[i]; i++) {
            char c = decoded[i];
            if (c == '"') printf("\\\"");
            else if (c == '\\') printf("\\\\");
            else if (c == '\n') printf("\\n");
            else if (c == '\t') printf("\\t");
            else if ((unsigned char)c < 0x20) printf("\\u%04x", (unsigned char)c);
            else putchar(c);
        }
        printf("\"}%s\n", PROMPTS[p + 1] ? "," : "");
    }
    printf("]\n");
    return 0;
}
