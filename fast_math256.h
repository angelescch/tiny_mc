#ifndef FAST_MATH256_H
#define FAST_MATH256_H

#include <immintrin.h>

__m256 fast_log_ps(__m256 x);
__m256 fast_sin_ps(__m256 x);
__m256 fast_cos_ps(__m256 x);

#endif
