#include <immintrin.h>

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
