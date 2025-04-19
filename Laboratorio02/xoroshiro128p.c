#include "xoroshiro128p.h"
#include <stdint.h>

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

static uint32_t s[4];

void xoroshiro128p_seed(uint32_t seed) {
    s[0] = seed;
    s[1] = seed ^ 0x9e3779b9;
    s[2] = seed ^ 0xabcdef01;
    s[3] = seed ^ 0x12345678;
}

static inline uint32_t rotl(const uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

float xoroshiro128p_next(void) {
    const uint32_t res = s[0] + s[3];

    const uint32_t t = s[1] << 9;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = rotl(s[3], 11);

    float result = (float)res * UINT32_MAX_INV;

    return result;
}

float xoroshiro128plog_next(void) {
    const uint32_t res = s[0] + s[3];

    const uint32_t t = s[1] << 9;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = rotl(s[3], 11);

    uint32_t result = res * UINT32_MAX_INV;

    return ((result >> 23) - 127) * 144269 / 100000;
}