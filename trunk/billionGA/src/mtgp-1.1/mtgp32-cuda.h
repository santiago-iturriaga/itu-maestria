#include "mtgp32-fast.h"

#ifndef MTGP32_CUDA__H
#define MTGP32_CUDA__H

#define MTGPDC_N 351

/**
 * kernel I/O
 * This structure must be initialized before first use.
 */
struct mtgp32_kernel_status_t {
    uint32_t status[MTGPDC_N];
};

struct mtgp32_status {
    int numbers_per_gen;
    
    int block_num;
    int num_unit;
    struct mtgp32_kernel_status_t *d_status;
    
    // Números generados.
    uint32_t *d_data;
    int num_data;
};
    
void mtgp32_initialize(struct mtgp32_status *status, int numbers_per_gen, unsigned int seed);
void mtgp32_free(struct mtgp32_status *status);
int mtgp32_get_suitable_block_num();

void mtgp32_generate_float(struct mtgp32_status *status);
void mtgp32_generate_uint32(struct mtgp32_status *status);

void mtgp32_print_generated_floats(struct mtgp32_status *status);
void mtgp32_print_generated_uint32(struct mtgp32_status *status);

#endif
