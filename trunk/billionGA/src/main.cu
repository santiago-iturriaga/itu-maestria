#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda.h>

#include "config.h"
#include "cuda-util.h"
#include "mtgp-1.1/mtgp32-cuda.h"
#include "billionga.h"

// Debe ser divisible entre 32 (8 y 4)... y 512, 128???
//#define TEST_PROBLEM_SIZE 32
//#define TEST_PROBLEM_SIZE 128
//#define TEST_PROBLEM_SIZE 524288
//#define TEST_PROBLEM_SIZE 1048576
//#define TEST_PROBLEM_SIZE 2097152
//#define TEST_PROBLEM_SIZE 899999744

struct termination_criteria {
    int max_iteration_count;
};

inline void termination_criteria_init(struct termination_criteria *term_state, 
    int max_iteration_count) {
        
    term_state->max_iteration_count = max_iteration_count;
}

inline int termination_criteria_eval(struct termination_criteria *term_state, 
    struct bga_state *problem_state, int iteration_count) {
        
    return (iteration_count == term_state->max_iteration_count);
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stdout, "Wrong! RFM!\n\nUsage: %s <problem size> <max iteration> <prng vector size> <gpu device>\n(where 1 <= problem size <= %ld and problem_size can be divided by 8)\n\n", argv[0], LONG_MAX);
        return EXIT_FAILURE;
    }
    
    long problem_size;
    //problem_size = TEST_PROBLEM_SIZE;
    problem_size = atol(argv[1]);
    
    int max_iteration_count = atoi(argv[2]);
    struct termination_criteria term_state;
    termination_criteria_init(&term_state, max_iteration_count);
    
    int gpu_device = atoi(argv[4]);
    ccudaSetDevice(gpu_device);

    // === Inicialización del Mersenne Twister.
    int prng_vector_size = atoi(argv[3]);
    
    mtgp32_status mt_status;    
    mtgp32_initialize(&mt_status, prng_vector_size);

    // === Inicialización del cGA
    struct bga_state problem_state;   
    bga_initialization(&problem_state, problem_size, NUMBER_OF_SAMPLES);
    
    #if defined(DEBUG)
    bga_show_prob_vector_state(&problem_state);
    #endif
    
    int current_iteration = 0;
    
    while (!termination_criteria_eval(&term_state, &problem_state, current_iteration)) {
        current_iteration++;
        
        #if defined(DEBUG)
        fprintf(stdout, "*** ITERACION %d *********************************************\n", current_iteration);
        #endif
        
        bga_model_sampling_mt(&problem_state, &mt_status);
        bga_evaluation(&problem_state);
        bga_model_update(&problem_state);
        
        #if defined(DEBUG)
        bga_show_prob_vector_state(&problem_state);
        #endif
    }
    
    // === Libero la memoria del cGA y del Mersenne Twister.
    bga_free(&problem_state);
    mtgp32_free(&mt_status);
    
    return EXIT_SUCCESS;
}
