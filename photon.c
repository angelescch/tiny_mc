#include "params.h"
#include "xoroshiro128p.h"
#include "log_256.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

void photon(int target, float* heats, float* heats_squared)
{
    __m256 albedo = _mm256_set1_ps(MU_S / (MU_S + MU_A));
    __m256 shells_per_mfp = _mm256_set1_ps(1e4 / MICRONS_PER_SHELL / (MU_A + MU_S));

    /* launch */
    __m256 x = _mm256_setzero_ps();
    __m256 y = _mm256_setzero_ps();
    __m256 z = _mm256_setzero_ps();
    __m256 u = _mm256_setzero_ps();
    __m256 v = _mm256_setzero_ps();
    __m256 w = _mm256_set1_ps(1.0f);
    __m256 weight = _mm256_set1_ps(1.0f);

    int counter = 0;
    while(counter < target) {
        __m256 t = _mm256_sub_ps(_mm256_set1_ps(0.0f), calculate_log_simd(next_random())); /* move */

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
            int idx = ((int*)&shell)[i];
            float r = ((float*)&res)[i];
            float r2 = ((float*)&res2)[i];
            heats_squared[idx] += r2;
            heats[idx] += r;
        }

        weight = _mm256_mul_ps(weight, albedo);

        __m256 if_mask = _mm256_cmp_ps(weight, _mm256_set1_ps(0.001f), _CMP_LT_OQ);
        __m256 rand_mask  = _mm256_cmp_ps(_mm256_set1_ps(0.1f), next_random(), _CMP_LT_OQ); 

        __m256 reset_mask = _mm256_and_ps(if_mask, rand_mask);
        counter += __builtin_popcount(_mm256_movemask_ps(reset_mask));

        weight = _mm256_blendv_ps(weight, _mm256_mul_ps(weight, _mm256_set1_ps(10.0f)), if_mask);
        weight = _mm256_blendv_ps(weight, _mm256_set1_ps(1.0f), reset_mask);
        x = _mm256_blendv_ps(x, _mm256_set1_ps(0), reset_mask);
        y = _mm256_blendv_ps(y, _mm256_set1_ps(0), reset_mask);
        z = _mm256_blendv_ps(z, _mm256_set1_ps(0), reset_mask);

        /*
        __m256 xi1 = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), next_random()), _mm256_set1_ps(1.0f));
        __m256 xi2 = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), next_random()), _mm256_set1_ps(1.0f));
        t = _mm256_add_ps(_mm256_mul_ps(xi1, xi1), _mm256_mul_ps(xi2, xi2));
        __m256 mask = _mm256_cmp_ps(t, _mm256_set1_ps(1.0f), _CMP_GT_OQ);
        while (_mm256_movemask_ps(mask) != 0) {
            __m256 new_xi1 = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), next_random()), _mm256_set1_ps(1.0f));
            __m256 new_xi2 = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), next_random()), _mm256_set1_ps(1.0f));
            xi1 = _mm256_blendv_ps(xi1, new_xi1, mask);
            xi2 = _mm256_blendv_ps(xi2, new_xi2, mask);
            t = _mm256_add_ps(_mm256_mul_ps(xi1, xi1), _mm256_mul_ps(xi2, xi2));
            mask = _mm256_cmp_ps(t, _mm256_set1_ps(1.0f), _CMP_GT_OQ);
        }

        __m256 new_u = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), t), _mm256_set1_ps(1.0f));
        __m256 sqrt_val = _mm256_sqrt_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), t));
        __m256 new_v = _mm256_mul_ps(_mm256_mul_ps(xi1, _mm256_set1_ps(2.0f)), sqrt_val);
        __m256 new_w = _mm256_mul_ps(_mm256_mul_ps(xi2, _mm256_set1_ps(2.0f)), sqrt_val);
    */




        __m256 new_u = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), next_random()), _mm256_set1_ps(1.0f));
        __m256 theta = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), _mm256_set1_ps(M_PI)), next_random());
        
        __m256 r = _mm256_sqrt_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f),_mm256_mul_ps(new_u,new_u)));
        __m256 new_v = _mm256_mul_ps(r,_mm256_cos_ps(theta));
        __m256 new_w = _mm256_mul_ps(r,_mm256_sin_ps(theta));





        
        u = _mm256_blendv_ps(u, new_u, reset_mask);
        v = _mm256_blendv_ps(v, new_v, reset_mask);
        w = _mm256_blendv_ps(w, new_w, reset_mask);

        // mask = _mm256_cmp_ps(weight, _mm256_set1_ps(0.001f), _CMP_LT_OQ);
        // mask = _mm256_and_ps(mask, live); //entran al if y estan vivos
        // xi1  = next_random();
        // xi1  = _mm256_cmp_ps(_mm256_set1_ps(0.1f), xi1, _CMP_LT_OQ); 
        // __m256 death_mask = _mm256_and_ps(mask, xi1); // entraron al if vivos y murieron
        // live = _mm256_andnot_ps(death_mask, live); // vivos de antes y vivos del rulette
        // __m256 survival_mask = _mm256_andnot_ps(xi1, mask); // entraron al if vivos y sobrevivieron
        // weight = _mm256_blendv_ps(weight, _mm256_div_ps(weight, _mm256_set1_ps(0.1f)), survival_mask);
    }
}
