#pragma once

#include <stdint.h>

#define XOSHIRO_MAX UINT32_MAX

void xoroshiro128p_seed(uint32_t seed);
float xoroshiro128p_next(void);
