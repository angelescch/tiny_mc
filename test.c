#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>

#define UINT32_MAX_INV (1.0f / (float)UINT32_MAX)

static __m256i s0, s1, s2, s3;

static inline __m256i rotl32_avx(__m256i x, int k) {
    return _mm256_or_si256(_mm256_slli_epi32(x, k),
                           _mm256_srli_epi32(x, 32 - k));
}

void init_random(uint32_t seed_base) {
uint32_t base = seed_base;
__m256i seed = _mm256_set_epi32(
    base ^ 0x243F6A88,
    base ^ 0x85A308D3,
    base ^ 0x13198A2E,
    base ^ 0x03707344,
    base ^ 0xA4093822,
    base ^ 0x299F31D0,
    base ^ 0x082EFA98,
    base ^ 0xEC4E6C89
);
    s0 = seed;
    s1 = _mm256_xor_si256(seed, _mm256_set1_epi32(0x9e3779b9));
    s2 = _mm256_xor_si256(seed, _mm256_set1_epi32(0xabcdef01));
    s3 = _mm256_xor_si256(seed, _mm256_set1_epi32(0x12345678));
}

__m256 next_random() {
    __m256i result = _mm256_add_epi32(s0, s3);

    __m256i t = _mm256_slli_epi32(s1, 9);

    s2 = _mm256_xor_si256(s2, s0);
    s3 = _mm256_xor_si256(s3, s1);
    s1 = _mm256_xor_si256(s1, s2);
    s0 = _mm256_xor_si256(s0, s3);

    s2 = _mm256_xor_si256(s2, t);
    s3 = rotl32_avx(s3, 11);

    unsigned int* res = (unsigned int*)&result;

    __m256 r = _mm256_setr_ps(
        res[0] * UINT32_MAX_INV, 
        res[1] * UINT32_MAX_INV, 
        res[2] * UINT32_MAX_INV,
        res[3] * UINT32_MAX_INV, 
        res[4] * UINT32_MAX_INV,
        res[5] * UINT32_MAX_INV,
        res[6] * UINT32_MAX_INV,
        res[7] * UINT32_MAX_INV);

    return r;
}
int main() {
    FILE *f = fopen("resultados.csv", "w");
    if (f == NULL) {
        printf("error al abrir el archivo\n");
        return 1;
    }
    init_random(19);

    for (int i = 0; i < 1e6; i++) { // 1e6 * 8 valores = 8 millones
        __m256 re = next_random();
        for (int j = 0; j < 8; j++) {
            float r = ((float*)&re)[j];
            fprintf(f, "%f\n", r);
        }
    }

    fclose(f);
    return 0;
}
