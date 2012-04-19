#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda.h>

#include "config.h"
#include "cuda-util.h"
#include "mtgp-1.1/mtgp32-cuda.h"
#include "billionga.h"

inline int termination_criteria_met(struct bga_state *problem_state) {
    // ...
    return true;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stdout, "Wrong! RFM!\n\nUsage: %s <problem size>\n(where 1 <= problem size <= %ld and problem_size can be divided by 8)\n\n", argv[0], LONG_MAX);
        return EXIT_FAILURE;
    }
    
    long problem_size;
    problem_size = atol(argv[1]);
    
    #ifdef INFO
    fprintf(stdout, "[INFO] Problem size: %ld\n", problem_size);
    #endif

    ccudaSetDevice(0);

    // === Inicialización del Mersenne Twister.
    mtgp32_status mt_status;    
    mtgp32_initialize(&mt_status, RNUMBERS_PER_GEN);
        
    //mtgp32_generate_float(&mt_status);
    //mtgp32_print_generated_floats(&mt_status);

    //mtgp32_generate_uint32(&mt_status);
    //mtgp32_print_generated_uint32(&mt_status);
        
    // === Inicialización del cGA
    struct bga_state problem_state;   
    bga_initialization(&problem_state, problem_size, NUMBER_OF_SAMPLES);
    
    while (!termination_criteria_met(&problem_state)) {
        bga_model_sampling_mt(&problem_state);
        bga_evaluation(&problem_state);
        bga_model_update(&problem_state);
    }
    
    // === Libero la memoria del cGA y del Mersenne Twister.
    bga_free(&problem_state);
    mtgp32_free(&mt_status);
    
    return EXIT_SUCCESS;
}
