#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda.h>

#include "config.h"
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

    // === Inicialización del Mersenne Twister.
    /*int seed = 325498732;
    int prng_numbers_per_iteration = 80;
    char *data_path = "mt/data/";
    
    mersenne_twister_init_data mt_data;
    mersenne_twister_init(data_path, prng_numbers_per_iteration, mt_data);*/
    
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
    
    //mersenne_twister_free(mt_data);
    
    return EXIT_SUCCESS;
}
