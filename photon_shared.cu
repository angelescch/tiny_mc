#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
#define PHOTONS_PER_THREAD 128
#define THREADS_PER_BLOCK 512
#define UINT32_MAX_INV (1.0f / (float)UINT32_MAX)

#include "params.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "helper_cuda.h"


char t1[] = "Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)";
char t2[] = "1 W Point Source Heating in Infinite Isotropic Scattering Medium";
char t3[] = "CPU version, adapted for PEAGPGPU by Gustavo Castellano"
            " and Nicolas Wolovick";

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

__device__ inline uint32_t rotl(const uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val += other;
    }
    return val;
}

__device__ inline float xoroshiro128p_next(uint32_t& s0, uint32_t& s1, uint32_t& s2, uint32_t& s3) {
    const uint32_t res = s0 + s3;
    const uint32_t t = s1 << 9;

    s2 ^= s0; 
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;

    s2 ^= t;
    s3 = rotl(s3, 11);

     return (float)res * UINT32_MAX_INV;
}

__device__ inline void xoroshiro128p_seed(uint32_t seed, uint32_t tid, uint32_t& s0, uint32_t& s1, uint32_t& s2, uint32_t& s3) {
    s0 = seed ^ tid;
    s1 = seed ^ 0x9e3779b9 ^ tid;
    s2 = seed ^ 0xabcdef01 ^ tid;
    s3 = seed ^ 0x12345678 ^ tid;
}


__global__ void photon_kernel(float* d_heats, float* d_heats_2, int p, uint32_t seed) {
    __shared__ float shared_heats[SHELLS][32];
    __shared__ float shared_heats_2[SHELLS][32];

    int tid = threadIdx.x;
    int block_threads = blockDim.x;
    int lane_id = tid % warpSize;

    for (int i = tid; i < SHELLS; i += blockDim.x) {
        for (int j = 0; j<32;j++){
            shared_heats[i][j] = 0.0f;
            shared_heats_2[i][j] = 0.0f;
        }
    }

    __syncthreads();

    uint32_t s0, s1, s2, s3;
    xoroshiro128p_seed(seed, tid + blockIdx.x * block_threads, s0, s1, s2, s3);

    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / (MU_A + MU_S);

    float x = 0.0f, y = 0.0f, z = 0.0f;
    float u = 0.0f, v = 0.0f, w = 1.0f;
    float weight = 1.0f;

    while (p > 0) {
        float t = -logf(xoroshiro128p_next(s0, s1, s2, s3));
        x += t * u;
        y += t * v;
        z += t * w;

        unsigned int shell = (unsigned int)(sqrtf(x * x + y * y + z * z) * shells_per_mfp);
        if (shell >= SHELLS) shell = SHELLS - 1;

        float absorb = (1.0f - albedo) * weight;
        //local_heats[shell][0] += absorb;
        //local_heats[shell][1] += absorb * absorb;
        atomicAdd(&shared_heats[shell][lane_id], absorb);
        atomicAdd(&shared_heats_2[shell][lane_id], absorb * absorb);

        weight *= albedo;

        u = 2.0f * xoroshiro128p_next(s0, s1, s2, s3) - 1.0f;
        float theta = 2.0f * M_PI * xoroshiro128p_next(s0, s1, s2, s3);
        float r = sqrtf(1.0f - u * u);
        v = r * cosf(theta);
        w = r * sinf(theta);

        if (weight < 0.001f) {
            weight /= 0.1f;
            if (xoroshiro128p_next(s0, s1, s2, s3) > 0.1f) {
                x = y = z = 0.0f;
                u = v = 0.0f;
                w = 1.0f;
                weight = 1.0f; p--;
            }
        }
    }

    __syncthreads();

    for (int i = tid; i < SHELLS; i += block_threads) {
        float sum = 0.0f;
        float sum2 = 0.0f;
        for (int k = 0; k < 32; ++k) {
            sum += shared_heats[i][k];
            sum2 += shared_heats_2[i][k];
        }
        shared_heats[i][0] = sum;
        shared_heats_2[i][0] = sum;
    }
    __syncthreads();

    for (int i = tid; i < SHELLS; i += block_threads) {
        atomicAdd(&d_heats[i], shared_heats[i][0]);
        atomicAdd(&d_heats_2[i], shared_heats_2[i][0]);
    }
}

int main(void)
{
    printf("# %s\n# %s\n# %s\n", t1, t2, t3);
    printf("# Scattering = %8.3f/cm\n", MU_S);
    printf("# Absorption = %8.3f/cm\n", MU_A);
    printf("# Photons    = %8d\n#\n", PHOTONS);

    float *d_heats, *d_heats_squared;
    checkCudaErrors(cudaMalloc(&d_heats, SHELLS * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_heats_squared, SHELLS * sizeof(float)));
    checkCudaErrors(cudaMemset(d_heats, 0, SHELLS * sizeof(float)));
    checkCudaErrors(cudaMemset(d_heats_squared, 0, SHELLS * sizeof(float)));

    dim3 threadsPerBlock(THREADS_PER_BLOCK);

    // total fotones procesados por bloque
    int photonsPerBlock = threadsPerBlock.x * PHOTONS_PER_THREAD;

    // calcular número de bloques para cubrir PHOTONS
    int numBlocks = (PHOTONS + photonsPerBlock - 1) / photonsPerBlock;

    uint32_t seed = 0xA511E9B3;

    cudaEvent_t start, stop;
    float elapsed = 0.0f;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    photon_kernel<<<numBlocks, threadsPerBlock>>>(d_heats, d_heats_squared, PHOTONS_PER_THREAD, seed);

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));

    float h_heats[SHELLS], h_heats_squared[SHELLS];
    checkCudaErrors(cudaMemcpy(h_heats, d_heats, SHELLS * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_heats_squared, d_heats_squared, SHELLS * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_heats));
    checkCudaErrors(cudaFree(d_heats_squared));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Shared memory per SM: %d bytes\n", prop.sharedMemPerMultiprocessor);


    float t = 4.0f * M_PI * powf(MICRONS_PER_SHELL, 3.0f) * PHOTONS / 1e12;

    float elapsed_seconds = elapsed / 1000.0f;

    printf("# %lf seconds\n", elapsed_seconds);
    printf("# %lf K photons per second\n", 1e-3 * PHOTONS / elapsed_seconds);

    for (unsigned int i = 0; i < SHELLS - 1; ++i) {
        printf("%6.0f\t%12.5f\t%12.5f\n", i * (float)MICRONS_PER_SHELL,
               h_heats[i] / t / (i * i + i + 1.0 / 3.0),
               sqrt(h_heats_squared[i] - h_heats[i] * h_heats[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
    }
    printf("# extra\t%12.5f\n", h_heats[SHELLS - 1] / PHOTONS);

    return 0;
}
