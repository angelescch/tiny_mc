#ifndef LOG_256_H
#define LOG_256_H

#include <immintrin.h>

__m256 calculate_log_simd(__m256 x);
__m256 fast_sin_ps(__m256 x);
__m256 fast_cos_ps(__m256 x);

#endif
