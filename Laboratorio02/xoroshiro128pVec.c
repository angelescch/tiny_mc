#include "xoroshiro128pVec.h"
#include <stdint.h>
#include <immintrin.h>


#define UINT32_MAX_INV (1.0f / (float)UINT32_MAX)

/* This is xoshiro128+ 1.0, our best and fastest 32-bit generator for 32-bit
   floating-point numbers. We suggest to use its upper bits for
   floating-point generation, as it is slightly faster than xoshiro128**.
   It passes all tests we are aware of except for
   linearity tests, as the lowest four bits have low linear complexity, so
   if low linear complexity is not considered an issue (as it is usually
   the case) it can be used to generate 32-bit outputs, too.

   We suggest to use a sign test to extract a random Boolean value, and
   right shifts to extract subsets of bits.

   The state must be seeded so that it is not everywhere zero. */

// static uint32_t s[4];
static __m256i s0;
static __m256i s1;
static __m256i s2;
static __m256i s3;


void xoroshiro128pVec_seed(uint32_t seed[8]) {

    unsigned int s[8];

    for (int i = 0; i < 8; i++)
        s[i] = seed[i];

    s0 = _mm256_loadu_si256((__m256i*)s);

    for (int i = 0; i < 8; i++)
        s[i] = seed[i] ^ 0x9e3779b9;

    s1 = _mm256_loadu_si256((__m256i*)s);
    
    for (int i = 0; i < 8; i++)
        s[i] = seed[i] ^ 0xabcdef01;
    
    s2 = _mm256_loadu_si256((__m256i*)s);

    for (int i = 0; i < 8; i++)
        s[i] = seed[i] ^ 0x12345678;

    s3 = _mm256_loadu_si256((__m256i*)s);
}

// static inline uint32_t rotl(const uint32_t x, int k) {
//     return (x << k) | (x >> (32 - k));
// }

static inline __m256i rotl(const __m256i x, int k) {
    __m256i l = _mm256_slli_epi32(x, k);
    __m256i r = _mm256_srli_epi32(x, 32 - k);
    return _mm256_or_si256(l, r);
}

__m256 xoroshiro128pVec_next(void) {
    const __m256i res = _mm256_add_epi32(s0, s3);
    // const uint32_t res = s[0] + s[3];

    // const uint32_t t = s[1] << 9;
    const __m256i t = _mm256_slli_epi32(s1, 9);

    // s2 ^= s0;
    s2 = _mm256_xor_si256(s2, s0);
    // s3 ^= s1;
    s3 = _mm256_xor_si256(s3, s1);
    // s1 ^= s2;
    s1 = _mm256_xor_si256(s1, s2);
    // s0 ^= s3;
    s0 = _mm256_xor_si256(s0, s3);

    // s[2] ^= t;
    s2 = _mm256_xor_si256(s2, t);

    s3 = rotl(s3, 11);

    // float result = (float)res * UINT32_MAX_INV;
    __m256 result = _mm256_mul_ps(_mm256_cvtepi32_ps(res) , _mm256_set1_ps(UINT32_MAX_INV));

    return result;
}