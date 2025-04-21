#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "params.h"
#include "xoroshiro128p.h"

#include <immintrin.h>

#define TRUE -1

inline float random(){ return -logf(xoroshiro128p_next()); }
inline float random2(){ return 2.0f * xoroshiro128p_next() - 1.0f; }

/* Ahora se realizara el calculo de a 8 */
void photon(float* heats, float* heats_squared)
{
    __m256 albedo = _mm256_set1_ps(MU_S / (MU_S + MU_A));
    __m256 shells_per_mfp = _mm256_set1_ps(1e4 / MICRONS_PER_SHELL / (MU_A + MU_S));
    __m256 live = _mm256_set1_ps(TRUE);

    /* launch */
    __m256 x = _mm256_setzero_ps();
    __m256 y = _mm256_setzero_ps();
    __m256 z = _mm256_setzero_ps();
    __m256 u = _mm256_setzero_ps();
    __m256 v = _mm256_setzero_ps();
    __m256 w = _mm256_set1_ps(1.0f);
    __m256 weight = _mm256_set1_ps(1.0f);

    for (;_mm256_movemask_ps(live) != 0;) {
        __m256 t = _mm256_set_ps(random(),random(),random(),random(),random(),random(),random(),random()); /* move */

        x = _mm256_add_ps(x, _mm256_mul_ps(t, u));
        y = _mm256_add_ps(y, _mm256_mul_ps(t, v));
        z = _mm256_add_ps(z, _mm256_mul_ps(t, w));

        __m256 sq = _mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y)), _mm256_mul_ps(z, z)));
        __m256 sq_s = _mm256_mul_ps(sq, shells_per_mfp);

        __m256i shell  = _mm256_cvttps_epi32(sq_s); /* absorb */

        shell = _mm256_min_epi32(shell, _mm256_set1_epi32(SHELLS - 1));

        __m256 res = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), albedo), weight);
        __m256 res2 = _mm256_mul_ps(res,res);

        for (int i = 0; i < 8; ++i) {
            if (((float*)&live)[i]) {
                int idx = ((int*)&shell)[i];
                if(idx < 0 || idx > SHELLS -1 ){
                    //printf("fucked");
                }
                float r = ((float*)&res)[i];
                float r2 = ((float*)&res2)[i];
                heats_squared[idx] += r2;
                heats[idx] += r;
            }
        }

        weight = _mm256_mul_ps(weight, albedo);

        __m256 xi1 = _mm256_set_ps(random2(), random2(), random2(), random2(), random2(), random2(), random2(), random2());
        __m256 xi2 = _mm256_set_ps(random2(), random2(), random2(), random2(), random2(), random2(), random2(), random2());
        t = _mm256_add_ps(_mm256_mul_ps(xi1, xi1), _mm256_mul_ps(xi2, xi2));
        __m256 mask = _mm256_cmp_ps(t, _mm256_set1_ps(1.0f), _CMP_GT_OQ);
        while (_mm256_movemask_ps(mask) != 0) {
            __m256 new_xi1 = _mm256_set_ps(random2(), random2(), random2(), random2(), random2(), random2(), random2(), random2());
            __m256 new_xi2 = _mm256_set_ps(random2(), random2(), random2(), random2(), random2(), random2(), random2(), random2());
            xi1 = _mm256_blendv_ps(xi1, new_xi1, mask);
            xi2 = _mm256_blendv_ps(xi2, new_xi2, mask);
            t = _mm256_add_ps(_mm256_mul_ps(xi1, xi1), _mm256_mul_ps(xi2, xi2));
            mask = _mm256_cmp_ps(t, _mm256_set1_ps(1.0f), _CMP_GT_OQ);
        }

        u = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), t), _mm256_set1_ps(1.0f));
        __m256 sqrt_val = _mm256_sqrt_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), t));
        v = _mm256_mul_ps(_mm256_mul_ps(xi1, _mm256_set1_ps(2.0f)), sqrt_val);
        w = _mm256_mul_ps(_mm256_mul_ps(xi2, _mm256_set1_ps(2.0f)), sqrt_val);

        mask = _mm256_cmp_ps(weight, _mm256_set1_ps(0.001f), _CMP_LT_OQ);
        mask = _mm256_and_ps(mask, live);
        xi1  = _mm256_set_ps(xoroshiro128p_next(),xoroshiro128p_next(),xoroshiro128p_next(),xoroshiro128p_next(),xoroshiro128p_next(),xoroshiro128p_next(),xoroshiro128p_next(),xoroshiro128p_next());
        xi1  = _mm256_cmp_ps(xi1, _mm256_set1_ps(0.1f), _CMP_LE_OQ); 
        live = _mm256_and_ps(live, xi1);
        weight = _mm256_div_ps(weight, _mm256_set1_ps(0.1f));
    }
}
