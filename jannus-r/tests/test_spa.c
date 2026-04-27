/* tests/test_spa.c — verify Sentence Phonon Attention.
 *
 *   1. spa_init produces deterministic embeddings (same seed → same table)
 *   2. spa_embed_sentence returns L2-normalised vectors
 *   3. spa_cross_attend gives higher scores to sentences with more
 *      semantic overlap (on a synthetic corpus)
 *   4. JS-side reproduces the same byte values for the seeded LCG (the
 *      JS test runs separately in the browser, see tests/test_jannus.html)
 *
 * Build:  cc -O2 tests/test_spa.c -o tests/test_spa -lm
 * Run:    ./tests/test_spa
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../tools/jannus_spa.h"

static int passes = 0, failures = 0;

static void check(const char *name, int cond) {
    if (cond) { printf("  PASS [%s]\n", name); passes++; }
    else      { printf("  FAIL [%s]\n", name); failures++; }
}

int main(void) {
    printf("== test_spa ==\n");

    SPACtx s1, s2;
    spa_init(&s1, 64, 4242u);
    spa_init(&s2, 64, 4242u);

    /* Same seed → identical embedding tables */
    int identical = !memcmp(s1.W_embed, s2.W_embed, 64 * SPA_DIM * sizeof(float));
    check("seed=4242 deterministic", identical);

    /* Different seed → different tables */
    SPACtx s3;
    spa_init(&s3, 64, 999u);
    int different = memcmp(s1.W_embed, s3.W_embed, 64 * SPA_DIM * sizeof(float)) != 0;
    check("seed=999 differs from 4242", different);

    /* spa_embed_sentence — L2 normalised */
    int ids[] = {1, 5, 12, 7, 3};
    float emb[SPA_DIM];
    spa_embed_sentence(&s1, ids, 5, emb);
    float norm = 0;
    for (int d = 0; d < SPA_DIM; d++) norm += emb[d] * emb[d];
    norm = sqrtf(norm);
    check("|emb| ≈ 1", fabsf(norm - 1.0f) < 1e-4f);

    /* Empty sentence → all zeros (no normalisation undefined) */
    float emb0[SPA_DIM];
    spa_embed_sentence(&s1, ids, 0, emb0);
    int all_zero = 1;
    for (int d = 0; d < SPA_DIM; d++) if (emb0[d] != 0) { all_zero = 0; break; }
    check("empty sentence → zero embedding", all_zero);

    /* spa_cross_attend — score is sum of exp(dot_ij) terms, must be > 0 */
    float embs[5][SPA_DIM];
    int ts[5][3] = {
        {1, 2, 3},
        {1, 2, 4},   /* close to 0 */
        {10, 11, 12}, /* unrelated */
        {1, 2, 3},   /* identical to 0 */
        {20, 21, 22},
    };
    for (int i = 0; i < 5; i++) spa_embed_sentence(&s1, ts[i], 3, embs[i]);
    float scores[5];
    spa_cross_attend(&s1, embs, 5, scores);

    int positive = 1;
    for (int i = 0; i < 5; i++) if (scores[i] <= 0) { positive = 0; break; }
    check("all SPA scores > 0", positive);
    printf("    scores: %.3f %.3f %.3f %.3f %.3f\n",
           scores[0], scores[1], scores[2], scores[3], scores[4]);

    /* The pair (0, 3) is identical token-wise — they should attend to
     * each other strongly, so score(0) and score(3) > score of an
     * isolated unrelated sentence. */
    int isolated_lower = (scores[2] < scores[0]) && (scores[2] < scores[3]);
    check("isolated sentence has lower score than identical pair", isolated_lower);

    spa_free(&s1);
    spa_free(&s2);
    spa_free(&s3);

    printf("\n== %d PASS, %d FAIL ==\n", passes, failures);
    return failures ? 1 : 0;
}
