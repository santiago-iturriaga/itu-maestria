#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "config.h"
#include "billionga.h"

// Paso 1 del algoritmo.
void bga_initialization(struct bga_state *state, long number_of_bits, int number_of_samples) {
    state->number_of_bits = number_of_bits;
    state->number_of_samples = number_of_samples;
    
    cudaError_t error;
    
    #ifdef INFO
    fprintf(stdout, "[INFO] Requesting prob_vector_size memory\n");
    #endif
    
    size_t prob_vector_size = sizeof(float) * state->number_of_bits;
    error = cudaMalloc((void**)&(state->gpu_prob_vector), prob_vector_size);
    
    if (error != cudaSuccess) {
        fprintf(stderr, "[ERROR] Requesting memory for prob_vector\n");
        exit(EXIT_FAILURE);
    }
   
    #ifdef INFO
    fprintf(stdout, "[INFO] Requesting samples memory\n");
    #endif
    
    size_t samples_array_size = sizeof(float*) * state->number_of_samples;
    state->gpu_samples = (float**)malloc(samples_array_size);
    if (!state->gpu_samples) {
        fprintf(stderr, "[ERROR] Requesting samples_fitness memory\n");
        exit(EXIT_FAILURE);
    }
   
    size_t sample_size = sizeof(char) * (state->number_of_bits / 8);
    
    for (int sample_number = 0; sample_number < state->number_of_samples; sample_number++) {
        #ifdef INFO
        fprintf(stdout, "[INFO] Requesting memory for sample %d\n", sample_number);
        #endif

        error = cudaMalloc((void**)&(state->gpu_samples[sample_number]), sample_size);
        
        if (error != cudaSuccess) {
            fprintf(stderr, "[ERROR] Requesting memory for sample_number[%d]\n", sample_number);
            exit(EXIT_FAILURE);
        }
    }
    
    #ifdef INFO
    fprintf(stdout, "[INFO] Requesting samples_fitness memory\n");
    #endif
    
    size_t samples_fitness_size = sizeof(long) * state->number_of_samples;
    error = cudaMalloc((void**)&(state->gpu_samples_fitness), samples_fitness_size);
    if (error != cudaSuccess) {
        fprintf(stderr, "[ERROR] Requesting memory for samples_fitness\n");
        exit(EXIT_FAILURE);
    }
}

// Paso 2 del algoritmo.
void bga_model_sampling(struct bga_state *state) {
}

// Paso 3 del algoritmo.
void bga_evaluation(struct bga_state *state) {
}

// Paso 4 y 5 del algoritmo.
void bga_model_update(struct bga_state *state) {
}

// Libera la memoria pedida para de estado.
void bga_free(struct bga_state *state) {
}
