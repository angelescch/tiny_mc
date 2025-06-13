#include <immintrin.h>
#include <stdio.h>

// Función rápida para calcular logaritmo en base 2 utilizando AVX2 (SIMD)
static __m256 calculate_log2_simd(__m256 x) {
    // Conjunto de constantes para la aproximación
    const __m256 const1 = _mm256_set1_ps(1.1920928955078125e-7f);
    const __m256 const2 = _mm256_set1_ps(-124.22551499f);
    const __m256 const3 = _mm256_set1_ps(-1.498030302f);
    const __m256 const4 = _mm256_set1_ps(-1.72587999f);
    const __m256 const5 = _mm256_set1_ps(0.3520887068f);
    
    // Extract the mantissa and exponent
    union {
        __m256 f;
        __m256i i;
    } vx = {x};

    __m256i mask1 = _mm256_set1_epi32(0x007FFFFF);  // Parte fraccionaria
    __m256i mask2 = _mm256_set1_epi32(0x3F000000);  // Parte del exponente

    __m256i extracted_part = _mm256_and_si256(vx.i, mask1);  // Extrae la fracción
    __m256i adjusted_part = _mm256_or_si256(extracted_part, mask2);  // Ajusta el exponente

    union {
        __m256i i;
        __m256 f;
    } mx = {adjusted_part};

    // Cálculos para el logaritmo
    __m256 y = _mm256_cvtepi32_ps(vx.i);
    y = _mm256_mul_ps(y, const1);  // Multiplicación con factor

    __m256 add1 = _mm256_add_ps(y, const2);  // Primer término
    __m256 add2 = _mm256_add_ps(const5, mx.f);  // Segundo término
    __m256 add3 = _mm256_add_ps(_mm256_mul_ps(const3, mx.f), _mm256_div_ps(const4, add2));  // Tercer término

    return _mm256_add_ps(add1, add3);  // Resultado final del log2
}

// Función para calcular el logaritmo natural utilizando AVX2
__m256 calculate_log_simd(__m256 x) {
    // Constante ln(2)
    __m256 ln2 = _mm256_set1_ps(0.69314718f);
    // Calcular log2(x) y luego multiplicar por ln(2) para obtener ln(x)
    return _mm256_mul_ps(ln2, calculate_log2_simd(x));
}

__m256 fast_sin_ps(__m256 x) {
    const __m256 _ps256_pi       = _mm256_set1_ps(3.14159265359f);       // π
    const __m256 _ps256_2pi      = _mm256_set1_ps(6.28318530718f);       // 2π
    const __m256 _ps256_halfpi   = _mm256_set1_ps(1.57079632679f);       // π/2
    const __m256 _ps256_inv2pi   = _mm256_set1_ps(0.15915494309f);       // 1 / (2π)

    // Reducimos x a [-π, π]
    x = _mm256_sub_ps(
        x,
        _mm256_mul_ps(
            _mm256_round_ps(_mm256_mul_ps(x, _ps256_inv2pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
            _ps256_2pi
        )
    );

    // Reflejamos a [-π/2, π/2] para mejorar precisión
    __m256 sign    = _mm256_and_ps(x, _mm256_set1_ps(-0.0f));             // signo
    __m256 absx    = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);          // |x|
    __m256 reflected = _mm256_sub_ps(_ps256_pi, absx);                    // π - |x|
    __m256 mask    = _mm256_cmp_ps(absx, _ps256_halfpi, _CMP_GT_OS);      // si |x| > π/2
    __m256 x_final = _mm256_blendv_ps(x, _mm256_xor_ps(reflected, sign), mask); // reflejado con signo

    // Polinomio de Taylor: x - x^3/6 + x^5/120
    __m256 x2 = _mm256_mul_ps(x_final, x_final);
    __m256 x3 = _mm256_mul_ps(x2, x_final);
    __m256 x5 = _mm256_mul_ps(x3, x2);

    return _mm256_add_ps(
        x_final,
        _mm256_add_ps(
            _mm256_mul_ps(x3, _mm256_set1_ps(-1.0f / 6.0f)),
            _mm256_mul_ps(x5, _mm256_set1_ps(1.0f / 120.0f))
        )
    );
}

__m256 fast_cos_ps(__m256 x) {
    // cos(x) = sin(x + π/2)
    return fast_sin_ps(_mm256_add_ps(x, _mm256_set1_ps(1.57079632679f)));
}

// int main() {
//         printf("sfsdf");    
//     // Valores: 0, π/2, π, -π/2 y el resto relleno con ceros
//     float test_vals[8] __attribute__((aligned(32))) = {
//         (float)(3.14159265359f * 2.0f),
//         (float)(3.14159265359f / 4.0f),
//         (float)(3.14159265359f / 2.0f),
//         (float)(3.14159265359f * 0.75f),
//         (float) 3.14159265359f,
//         (float)(3.14159265359f * 1.25f),
//         (float)(3.14159265359f * 1.5f),
//         (float)(3.14159265359f * 1.75f)
//     };

//     __m256 x = _mm256_load_ps(test_vals);
//     __m256 sines = fast_sin_ps(x);
//     __m256 cosines = fast_cos_ps(x);
//     printf("sfsdf");
//     float sin_res[8] __attribute__((aligned(32)));
//     float cos_res[8] __attribute__((aligned(32)));
//     //_mm256_store_ps(sin_res, sines);
//     //_mm256_store_ps(cos_res, cosines);

//     printf(" x         | fast_sin_ps(x) | fast_cos_ps(x)\n");
//     printf("-------------------------------------------\n");
//     for (int i = 0; i < 8; ++i) {
//         printf("%10.6f | %14.6f | %14.6f\n", test_vals[i], sines[i], cosines[i]);
//     }

//     return 0;
// }