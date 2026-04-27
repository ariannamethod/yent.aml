/* tests/test_calendar.c — verify Hebrew/Gregorian calendar drift formulas.
 *
 * Tests the canonical numbers from resonance-janus-bpe.c — the leap
 * correction kicks in at year-in-cycle ≥ leap_year[i], so the drift
 * is monotonic only inside an "uncorrected" stretch. Reference values:
 *
 *   epoch     : 1 Tishrei 5785 = 3 Oct 2024 noon
 *   year 0    : drift = 0  (no time, no correction)
 *   year ~1   : drift ≈ 11.27  (yic=2 < 3, no leap applied)
 *   year ~2   : drift ≈ -7.48  (yic=3 == 3, first leap applied: 22.5 − 30)
 *   year ~3   : drift ≈ +3.76  (yic=4, still 1 leap: 33.76 − 30)
 *   year ~19  : drift ≈ +3.75  (yic=19, all 7 leaps: 213.75 − 210)
 *
 * dissonance = |drift mod 33| / 33 ∈ [0, 1].
 *
 * Build:  cc -O2 -Wall -Wextra tests/test_calendar.c -o tests/test_calendar -lm
 * Run:    ./tests/test_calendar
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../tools/jannus_calendar.h"

static int passes = 0, failures = 0;

static void check_close(const char *name, float got, float want, float tol) {
    float err = fabsf(got - want);
    if (err <= tol) {
        printf("  PASS [%-40s] got=%.4f want=%.4f (tol=%g)\n", name, got, want, tol);
        passes++;
    } else {
        printf("  FAIL [%-40s] got=%.4f want=%.4f (err=%.4f)\n", name, got, want, err);
        failures++;
    }
}

int main(void) {
    printf("== test_calendar ==\n");

    jcal_init();

    /* Drift is purely a function of `days`, no time(NULL) involved here. */
    float d_year0  = jcal_cumulative_drift(0);
    float d_year1  = jcal_cumulative_drift(366);     /* ~1 year */
    float d_year2  = jcal_cumulative_drift(731);     /* ~2 years */
    float d_year3  = jcal_cumulative_drift(1096);    /* ~3 years (leap year 3 → -30) */
    float d_year19 = jcal_cumulative_drift(6940);    /* ~19 years (full Metonic) */

    check_close("drift(0d)",     d_year0,   0.00f,  0.05f);
    check_close("drift(~1y)",    d_year1,  11.27f,  0.20f);
    /* yic=3 ≥ leap[0]=3 — first leap month subtracted at the year-2 boundary */
    check_close("drift(~2y)",    d_year2,  -7.48f,  0.20f);
    /* yic=4, still only 1 leap applied */
    check_close("drift(~3y)",    d_year3,   3.76f,  0.20f);
    /* yic=19, all 7 leaps applied: 213.75 − 210 */
    check_close("drift(~19y)",   d_year19,  3.75f,  0.20f);

    /* Dissonance ∈ [0, 1] — check it's well-bounded for many years */
    for (int y = 0; y < 50; y++) {
        int days = (int)(y * 365.25f);
        float diss = jcal_dissonance_at(days);
        if (diss < 0 || diss > 1) {
            printf("  FAIL [dissonance bounds y=%d] got=%.4f\n", y, diss);
            failures++;
            return 1;
        }
    }
    printf("  PASS [dissonance ∈ [0,1] for y=0..49]\n");
    passes++;

    /* MetaJanus snapshot: birth dissonance equals current dissonance,
     * personal dissonance == 0 immediately after init. */
    JcalMetaJanus mj = {0};
    jcal_metajanus_init(&mj);
    check_close("MJ.alive after init",       (float)mj.alive,                  1.0f, 0.0f);
    check_close("MJ.birth_dissonance == now", mj.birth_dissonance,             jcal_dissonance_now(), 1e-5f);
    check_close("personal_dissonance(t=0)",   jcal_personal_dissonance(&mj),   0.0f, 1e-5f);

    /* dissonance value at "now" should agree with the formula on today's day. */
    int now_days = jcal_days_since_epoch();
    float diss_now      = jcal_dissonance_now();
    float diss_formula  = jcal_dissonance_at(now_days);
    check_close("dissonance_now == dissonance_at(today)", diss_now, diss_formula, 1e-6f);

    printf("\n== %d PASS, %d FAIL ==\n", passes, failures);
    return failures ? 1 : 0;
}
