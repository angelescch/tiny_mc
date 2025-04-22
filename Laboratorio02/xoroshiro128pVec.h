#pragma once

#include <stdint.h>
#include <immintrin.h>


#define XOSHIRO_MAX UINT32_MAX

void xoroshiro128pVec_seed(uint32_t seed[8]);
__m256 xoroshiro128pVec_next(void);
