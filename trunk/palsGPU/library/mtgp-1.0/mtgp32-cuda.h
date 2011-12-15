#include <stdint.h>

#define N 726

#ifndef MTGP32_CUDA_H_ 
#define MTGP32_CUDA_H_ 

/**
 * kernel I/O
 * This structure must be initialized before first use.
 */
struct mtgp32_kernel_status_t {
    uint32_t status[N];
};

mtgp32_kernel_status_t* mtgp32_init();

void mtgp32_dispose(mtgp32_kernel_status_t* d_status);

void mtgp32_uint32_random(mtgp32_kernel_status_t* d_status, int rnd_numbers_size, uint32_t *rnd_numbers);

#endif
