#pragma once

#ifndef XOROSHIRO128P_H
#define XOROSHIRO128P_H

#include <immintrin.h>
#include <stdint.h>

void init_random(uint32_t seed);
__m256 next_random();

#endif
