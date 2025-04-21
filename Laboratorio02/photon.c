#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "params.h"
#include "xoroshiro128p.h"

#include <immintrin.h>

#define TRUE -1

inline float random(){ return -logf(xoroshiro128p_next()); }
inline float random2(){ return 2.0f * xoroshiro128p_next() - 1.0f; }
inline float random3(){ return xoroshiro128p_next(); }


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


        // unsigned int debug_shell[8];
        // unsigned int debug_mask_i[8];
        // float debug_live[8];
        // float debug_weight[8];
        // _mm256_store_ps(debug_weight, weight);
        // _mm256_store_ps(debug_live, live);



        // for (int i = 0; i < 8; i++){
        //     printf("weight %d = %f\n", i, debug_weight[i]);
        //     printf("live   %d = %f\n", i, debug_live[i]);
        // }

        __m256i shell  = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y)), _mm256_mul_ps(z, z))), shells_per_mfp)); /* absorb */

        // _mm256_storeu_si256((__m256i*)debug_shell, shell);



        // for (int i = 0; i < 8; i++){
        //     printf("shell[%d] = %d\n", i, debug_shell[i]);
        // }

        __m256i mask_i = _mm256_or_si256(_mm256_cmpgt_epi32(shell, _mm256_set1_epi32(SHELLS - 1)),
                                          _mm256_cmpgt_epi32(_mm256_set1_epi32(0),shell));
        // __m256i mask_i = _mm256_cmpgt_epi32(shell, _mm256_set1_epi32(SHELLS - 1));


        // shell /\ not( mask_i) + (SHELLS - 1) /\ mask_i
        shell = _mm256_add_epi32(_mm256_andnot_si256(mask_i, shell), _mm256_and_si256(mask_i, _mm256_set1_epi32(SHELLS - 1)));

        // _mm256_storeu_si256((__m256i*)debug_mask_i, mask_i);

        // for (int i = 0; i < 8; i++){
        //     printf("mask_i %d = %d\n", i, debug_mask_i[i]);
        // }


        __m256 res = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), albedo), weight);

        /* --------------------------------- */
        float res_tmp[8];

        _mm256_storeu_si256((__m256i*)shell_tmp, shell);
        _mm256_storeu_ps(res_tmp, res);
        
        
        for (int i = 0; i < 8; i++){
            // if (shell_tmp[i] > SHELLS - 1) {
            //     shell_tmp[i] = SHELLS - 1;
            // }
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


        
        /* Si estoy en el primer if */
        // if (weight < 0.001f)
        mask = _mm256_cmp_ps(weight, _mm256_set1_ps(0.001f), _CMP_LT_OQ); // Condicion
        __m256 weight_temp = _mm256_div_ps(weight, _mm256_set1_ps(0.1f)); 
        weight = _mm256_blendv_ps(weight, weight_temp, mask); // Todos los que estan 

        
        /* Estoy en el if anidado */
        // if (xoroshiro128p_next() > 0.1f)
        // Uso (>=) para ahorrarme la negacion
        __m256 if_rand  = _mm256_set_ps(random3(),random3(),random3(),random3(),random3(),random3(),random3(),random3());
        if_rand  = _mm256_cmp_ps(if_rand, _mm256_set1_ps(0.1f), _CMP_LE_OQ); 

        // Pasan a estar muertos (FALSE) los que sacaron un random inferior a 0.1f
        // Esto se actualiza solo para aquellos que estan dentro de la mascara
        live = _mm256_blendv_ps(live,_mm256_and_ps(live, if_rand) ,mask);
    }
}
