/* tests/test_split.c — chain split + temperature ladder.
 *
 * The 12-step jannus-r chain is its heart: split between forward and
 * backward steps governed by prophecy debt and calendar dissonance,
 * with monotone temperature ladders in each direction. This file tests
 * the helper jannus_compute_split / jannus_temp_forward / jannus_temp_backward
 * directly, no model needed — they are the source of truth used by
 * jannus_run_chain in jannus-r.aml.
 */
#include "../tools/jannus_split.h"

#include <stdio.h>
#include <math.h>

#define NSTEPS 12

static int pass = 0, fail = 0;

static int feq(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

#define CHECK(cond, msg, ...) do { \
    if (cond) { printf("  PASS [%-46s] " msg "\n", __func__, ##__VA_ARGS__); pass++; } \
    else      { printf("  FAIL [%-46s] " msg "\n", __func__, ##__VA_ARGS__); fail++; } \
} while (0)

/* 1. nb + nf == NSTEPS for all reasonable (debt, cal_diss) inputs. */
static void test_split_sums(void) {
    for (int di = 0; di <= 10; di++) {
        for (int ci = 0; ci <= 10; ci++) {
            float debt = di / 10.0f;
            float cd   = ci / 10.0f;
            JannusSplit s = jannus_compute_split(debt, cd, NSTEPS);
            int ok = (s.nb + s.nf == NSTEPS);
            if (!ok) {
                printf("    di=%d ci=%d nb=%d nf=%d sum=%d\n",
                       di, ci, s.nb, s.nf, s.nb + s.nf);
                fail++; return;
            }
        }
    }
    pass++;
    printf("  PASS [%-46s] 121 (debt, cal_diss) → nb + nf == 12\n", __func__);
}

/* 2. nb is clamped into [1, NSTEPS-1]. */
static void test_split_clamp(void) {
    JannusSplit lo = jannus_compute_split(0.0f, 0.0f, NSTEPS);
    JannusSplit hi = jannus_compute_split(1.0f, 1.0f, NSTEPS);
    CHECK(lo.nb >= 1 && lo.nb <= NSTEPS - 1,
          "(debt=0, cd=0)  nb=%d ∈ [1, %d]",  lo.nb, NSTEPS - 1);
    CHECK(hi.nb >= 1 && hi.nb <= NSTEPS - 1,
          "(debt=1, cd=1)  nb=%d ∈ [1, %d]",  hi.nb, NSTEPS - 1);
}

/* 3. nb monotone non-decreasing in debt and cal_diss. */
static void test_split_monotone(void) {
    int ok_debt = 1, ok_cd = 1;
    JannusSplit prev = jannus_compute_split(0.0f, 0.5f, NSTEPS);
    for (int i = 1; i <= 10; i++) {
        JannusSplit s = jannus_compute_split(i / 10.0f, 0.5f, NSTEPS);
        if (s.nb < prev.nb) { ok_debt = 0; break; }
        prev = s;
    }
    CHECK(ok_debt, "nb non-decreasing in debt at cd=0.5");

    prev = jannus_compute_split(0.5f, 0.0f, NSTEPS);
    for (int i = 1; i <= 10; i++) {
        JannusSplit s = jannus_compute_split(0.5f, i / 10.0f, NSTEPS);
        if (s.nb < prev.nb) { ok_cd = 0; break; }
        prev = s;
    }
    CHECK(ok_cd, "nb non-decreasing in cal_diss at debt=0.5");
}

/* 4. tb = 0.7 + 0.3*(0.5 + 0.3*cal_diss + 0.2*debt) — exact at corners. */
static void test_tb_formula(void) {
    JannusSplit s00 = jannus_compute_split(0.0f, 0.0f, NSTEPS);
    JannusSplit s11 = jannus_compute_split(1.0f, 1.0f, NSTEPS);
    JannusSplit shp = jannus_compute_split(0.5f, 0.5f, NSTEPS);
    CHECK(feq(s00.tb, 0.85f, 1e-5f),
          "tb(debt=0, cd=0) = %.4f (expect 0.8500)", s00.tb);
    CHECK(feq(s11.tb, 1.00f, 1e-5f),
          "tb(debt=1, cd=1) = %.4f (expect 1.0000)", s11.tb);
    CHECK(feq(shp.tb, 0.925f, 1e-5f),
          "tb(debt=.5, cd=.5) = %.4f (expect 0.9250)", shp.tb);
}

/* 5. forward temps strictly decreasing in step index — focus rises. */
static void test_temp_forward_decreasing(void) {
    float tb = 0.85f;
    int ok = 1;
    float prev = jannus_temp_forward(tb, 0);
    for (int i = 1; i < NSTEPS; i++) {
        float t = jannus_temp_forward(tb, i);
        if (!(t < prev)) { ok = 0; break; }
        prev = t;
    }
    CHECK(ok, "tb=0.85: temp_forward(0..11) strictly decreasing");
}

/* 6. backward temps strictly increasing in step index — exploration rises. */
static void test_temp_backward_increasing(void) {
    float tb = 0.85f;
    int ok = 1;
    float prev = jannus_temp_backward(tb, 0);
    for (int i = 1; i < NSTEPS; i++) {
        float t = jannus_temp_backward(tb, i);
        if (!(t > prev)) { ok = 0; break; }
        prev = t;
    }
    CHECK(ok, "tb=0.85: temp_backward(0..11) strictly increasing");
}

/* 7. Step 0 baseline: forward(0)==tb, backward(0)==tb. */
static void test_step0_baseline(void) {
    float tb = 0.85f;
    CHECK(feq(jannus_temp_forward(tb, 0),  tb, 1e-6f),
          "temp_forward(tb, 0) == tb (got %.6f)",  jannus_temp_forward(tb, 0));
    CHECK(feq(jannus_temp_backward(tb, 0), tb, 1e-6f),
          "temp_backward(tb, 0) == tb (got %.6f)", jannus_temp_backward(tb, 0));
}

/* 8. Concrete scenarios from the milestone documentation:
 *      cal_diss = 0.532 (today's drift in milestone),  debt = 0
 *      → nb = floor(12 * (0.3 + 0 + 0.0532)) = floor(4.2384) = 4
 *      → nf = 8
 */
static void test_documented_scenario(void) {
    JannusSplit s = jannus_compute_split(0.0f, 0.532f, NSTEPS);
    CHECK(s.nb == 4 && s.nf == 8,
          "cal_diss=0.532, debt=0 → nb=%d nf=%d (expect 4 + 8)", s.nb, s.nf);
}

int main(void) {
    printf("== test_split ==\n");
    test_split_sums();
    test_split_clamp();
    test_split_monotone();
    test_tb_formula();
    test_temp_forward_decreasing();
    test_temp_backward_increasing();
    test_step0_baseline();
    test_documented_scenario();
    printf("\n== %d PASS, %d FAIL ==\n", pass, fail);
    return fail ? 1 : 0;
}
