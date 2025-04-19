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
        unsigned int shell_tmp[8];
        
        weight = _mm256_blendv_ps(_mm256_set1_ps(0.0f), weight, live);

        x = _mm256_add_ps(x, _mm256_mul_ps(t, u));
        y = _mm256_add_ps(y, _mm256_mul_ps(t, v));
        z = _mm256_add_ps(z, _mm256_mul_ps(t, w));

        __m256i shell  = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y)), _mm256_mul_ps(z, z))), shells_per_mfp)); /* absorb */
        __m256i mask_i = _mm256_and_si256(_mm256_cmpgt_epi32(shell, _mm256_set1_epi32(SHELLS - 1)), _mm256_cmpgt_epi32(_mm256_set1_epi32(0),shell));
        // __m256i mask_i = _mm256_cmpgt_epi32(shell, _mm256_set1_epi32(SHELLS - 1));


        // shell /\ not( mask_i) + (SHELLS - 1) /\ mask_i
        shell = _mm256_add_epi32(_mm256_andnot_si256(mask_i, shell), _mm256_and_si256(mask_i, _mm256_set1_epi32(SHELLS - 1)));

        __m256 res = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), albedo), weight);

        /* --------------------------------- */
        float res_tmp[8];

        _mm256_storeu_si256((__m256i*)shell_tmp, shell);
        _mm256_storeu_ps(res_tmp, res);
        
        
        for (int i = 0; i < 8; i++){
            if (shell_tmp[i] > SHELLS - 1) {
                shell_tmp[i] = SHELLS - 1;
            }
            heats[shell_tmp[i]] += res_tmp[i];
        }
            
            
        res = _mm256_mul_ps(res,res);
        _mm256_storeu_ps(res_tmp, res);

        for (int i = 0; i < 8; i++)
            heats_squared[shell_tmp[i]] += res_tmp[i];

        /* --------------------------------- */

        // weight *= albedo;
        weight = _mm256_mul_ps(weight, albedo);


        __m256 xi1 = _mm256_set_ps(random2(), random2(), random2(), random2(), random2(), random2(), random2(), random2());
        __m256 xi2 = _mm256_set_ps(random2(), random2(), random2(), random2(), random2(), random2(), random2(), random2());
        t = _mm256_add_ps(_mm256_mul_ps(xi1, xi1), _mm256_mul_ps(xi2, xi2));
        __m256 mask = _mm256_cmp_ps(t, _mm256_set1_ps(1.0f), _CMP_GT_OQ);
        while (_mm256_movemask_ps(mask) != 0) {
            xi1 = _mm256_set_ps(random2(), random2(), random2(), random2(), random2(), random2(), random2(), random2());
            xi2 = _mm256_set_ps(random2(), random2(), random2(), random2(), random2(), random2(), random2(), random2());
            xi1 = _mm256_and_ps(xi1, mask);
            xi2 = _mm256_and_ps(xi2, mask);
            t = _mm256_add_ps(_mm256_mul_ps(xi1, xi1), _mm256_mul_ps(xi2, xi2));
            mask = _mm256_cmp_ps(t, _mm256_set1_ps(1.0f), _CMP_GT_OQ);
        }
        u = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), t), _mm256_set1_ps(1.0f));
        v = _mm256_mul_ps(xi1, _mm256_sqrt_ps(_mm256_div_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(u, u)), t)));
        w = _mm256_mul_ps(xi2, _mm256_sqrt_ps(_mm256_div_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(u, u)), t)));


        mask = _mm256_cmp_ps(weight, _mm256_set1_ps(0.001f), _CMP_LT_OQ);
        mask = _mm256_and_ps(mask, live);
        xi1  = _mm256_set_ps(random(),random(),random(),random(),random(),random(),random(),random());
        xi1  = _mm256_cmp_ps(xi1, _mm256_set1_ps(0.1f), _CMP_LE_OQ); 
        live = _mm256_and_ps(live, xi1);
        weight = _mm256_div_ps(weight, _mm256_set1_ps(0.1f));

    }
}
