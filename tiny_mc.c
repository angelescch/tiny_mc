/* Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)"
 * 1 W Point Source Heating in Infinite Isotropic Scattering Medium
 * http://omlc.ogi.edu/software/mc/tiny_mc.c
 *
 * Adaptado para CP2014, Nicolas Wolovick
 */

#define _XOPEN_SOURCE 500 // M_PI

#include "xoroshiro128p.h"
#include "params.h"
#include "photon.h"
#include "wtime.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>

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

    int nthreads = omp_get_max_threads();
    printf("maxxxxxx threads: %d\n",nthreads);
    int base = (PHOTONS + nthreads - 1) / nthreads;

    float heat_reduction[SHELLS] = {0.0f};
    float heat2_reduction[SHELLS] = {0.0f};

    // start timer
    double start = wtime();

    // configure RNG
    #pragma omp parallel reduction(+ : heat_reduction[:SHELLS], heat2_reduction[:SHELLS])
    {
        int tid = omp_get_thread_num();
        init_random(0xA511E9B3 ^ (tid * 0x45D9F3B));

        photon(base, heat_reduction, heat2_reduction);
    }

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
