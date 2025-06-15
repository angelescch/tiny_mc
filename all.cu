#include <cuda_runtime.h>
#include "params.h"
// #include "xoroshiro128p.h"

#define UINT32_MAX  (0xffffffff)

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

__device__ unsigned* xoroshiro128p_seed(unsigned seed) {
    unsigned* state;
    cudaMalloc((void **) &state, sizeof(unsigned) * 4);
    state[0] = seed;
    state[1] = seed ^ 0x9e3779b9;
    state[2] = seed ^ 0xabcdef01;
    state[3] = seed ^ 0x12345678;
    return state;
}

__device__ static inline unsigned rotl(const unsigned x, int k) {
    return (x << k) | (x >> (32 - k));
}

__device__ float xoroshiro128p_next(unsigned* state) {
    const unsigned res = state[0] + state[3];

    const unsigned t = state[1] << 9;

    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];

    state[2] ^= t;

    state[3] = rotl(state[3], 11);

    float result = (float)res * UINT32_MAX_INV;

    return result;
}

__device__ void init_heats(float* heats_, float* heats_squared_){
    size_t tid = threadIdx.x;
    size_t dim = blockDim.x;
    for (size_t i = tid; i < SHELLS; i += dim){
        heats_[i]         = 0.0f;
        heats_squared_[i] = 0.0f;
    }
    __syncthreads();
}

__device__ void reduce_heats(float* heats_, float* heats_squared_, float* heats, float* heats_squared){
    size_t tid = threadIdx.x;
    size_t dim = blockDim.x;
    __syncthreads();
    for (size_t i = tid; i < SHELLS; i += dim){
        atomicAdd(&heats[i],         heats_[i]);
        atomicAdd(&heats_squared[i], heats_squared[i]);
    }
    __syncthreads();
}

__global__ void photon(int target, float* heats, float* heats_squared)
{
    size_t gtid      = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned * state = xoroshiro128p_seed(gtid);
    float albedo     = MU_S / (MU_S + MU_A);
    float shells_per_mfp = 1e4 / MICRONS_PER_SHELL / (MU_A + MU_S);

    __shared__ float heats_[SHELLS];
    __shared__ float heats_squared_[SHELLS];

    init_heats(heats_, heats_squared_);

    /* launch */
    float x      = 0.0f;
    float y      = 0.0f;
    float z      = 0.0f;
    float u      = 0.0f;
    float v      = 0.0f;
    float w      = 0.0f;
    float weight = 1.0f;

    int counter = 0;
    while(counter < target) {
        float t = - logf(xoroshiro128p_next(state)); /* move */

        x = x + t * u;
        y = y + t * v;
        z = z + t * w;

        unsigned int shell = rsqrtf(x * x + y * y + z * z) * shells_per_mfp; /* absorb */
        if (shell > SHELLS - 1) {
            shell = SHELLS - 1;
        }
        float hs  = (1.0f - albedo) * weight;
        float hss = (1.0f - albedo) * (1.0f - albedo) * weight * weight; /* add up squares */
        weight *= albedo;


        /* New direction, rejection method */
        float xi1, xi2;
        do {
            xi1 = 2.0f * xoroshiro128p_next(state) - 1.0f;
            xi2 = 2.0f * xoroshiro128p_next(state) - 1.0f;
            t = xi1 * xi1 + xi2 * xi2;
        } while (1.0f < t);
        u = 2.0f * t - 1.0f;
        v = xi1 * rsqrtf((1.0f - u * u) / t);
        w = xi2 * rsqrtf((1.0f - u * u) / t);

        if (weight < 0.001f) { /* roulette */
            if (xoroshiro128p_next(state) > 0.1f){
                x      = 0.0f;
                y      = 0.0f;
                z      = 0.0f;
                u      = 0.0f;
                v      = 0.0f;
                w      = 0.0f;
                weight = 1.0f;
                target++;
            } else
                weight /= 0.1f;
        }

        // heats_[shell]         += hs;
        atomicAdd(&heats_[shell], hs);
        // heats_squared_[shell] += hss;
        atomicAdd(&heats_squared_[shell], hss);
    }
    
    reduce_heats(heats_, heats_squared_, heats, heats_squared);
}

__host__ void run(int total, float* heats, float* heats_squared){
    float* heats_d;
    float* heats_squared_d;
    cudaMalloc((void **)&heats_d,         sizeof(float) * SHELLS);
    cudaMalloc((void **)&heats_squared_d, sizeof(float) * SHELLS);

    // Configuramos los bloques y hilos para la ejecución del kernel

    int target        = total / (CANT_BLOQUES * HILOS_X_BLOQUE);

    photon<<<CANT_BLOQUES, HILOS_X_BLOQUE>>>(target, heats_d, heats_squared_d);

    // Esperamos a que el kernel termine
    cudaDeviceSynchronize();

    cudaMemcpy(heats,         heats_d,         sizeof(float) * SHELLS, cudaMemcpyDeviceToHost);
    cudaMemcpy(heats_squared, heats_squared_d, sizeof(float) * SHELLS, cudaMemcpyDeviceToHost);
}

/* Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)"
 * 1 W Point Source Heating in Infinite Isotropic Scattering Medium
 * http://omlc.ogi.edu/software/mc/tiny_mc.c
 *
 * Adaptado para CP2014, Nicolas Wolovick
 */


// #define _POSIX_C_SOURCE 199309L
#include <time.h>

double wtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);

    return 1e-9 * ts.tv_nsec + (double)ts.tv_sec;
}


#include "wtime.h"

#include <assert.h>
#include <stdio.h>

char t1[] = "Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)";
char t2[] = "1 W Point Source Heating in Infinite Isotropic Scattering Medium";
char t3[] = "CPU version, adapted for PEAGPGPU by Gustavo Castellano"
            " and Nicolas Wolovick";

/***
 * Main matter
 ***/

int main(void)
{
    // heading
    printf("# %s\n# %s\n# %s\n", t1, t2, t3);
    printf("# Scattering = %8.3f/cm\n", MU_S);
    printf("# Absorption = %8.3f/cm\n", MU_A);
    printf("# Photons    = %8d\n#\n", PHOTONS);


    float heat_reduction[SHELLS] = {0.0f};
    float heat2_reduction[SHELLS] = {0.0f};

    // start timer
    double start = wtime();

    run(PHOTONS, heat_reduction, heat2_reduction);

    // stop timer
    double end = wtime();
    assert(start <= end);
    double elapsed = end - start;

    printf("# %lf seconds\n", elapsed);
    printf("# %lf K photons per second\n", 1e-3 * PHOTONS / elapsed);

    printf("# Radius\tHeat\n");
    printf("# [microns]\t[W/cm^3]\tError\n");
    float t = 4.0f * M_PI * powf(MICRONS_PER_SHELL, 3.0f) * PHOTONS / 1e12;
    for (unsigned int i = 0; i < SHELLS - 1; ++i) {
        printf("%6.0f\t%12.5f\t%12.5f\n", i * (float)MICRONS_PER_SHELL,
               heat_reduction[i] / t / (i * i + i + 1.0 / 3.0),
               sqrt(heat2_reduction[i] - heat_reduction[i] * heat_reduction[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
    }
    printf("# extra\t%12.5f\n", heat_reduction[SHELLS - 1] / PHOTONS);

    return 0;
}

