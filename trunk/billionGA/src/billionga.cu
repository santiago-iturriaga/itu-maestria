#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

#include "config.h"
#include "billionga.h"

#define INIT_PROB_VECTOR_VALUE 0.5

/*
 * Inicializa el vector de probabilidad (p.ej. a 0.5).
 */
__global__ void kern_init_prob_vector(float *gpu_prob_vector, int max_size, int starting_position) {
    const int current_position = starting_position + (blockIdx.x * blockDim.x + threadIdx.x);
    
    if (current_position < max_size) {
        gpu_prob_vector[current_position] = INIT_PROB_VECTOR_VALUE;
    }
}

// Paso 1 del algoritmo.
void bga_initialization(struct bga_state *state, long number_of_bits, int number_of_samples) {
    state->number_of_bits = number_of_bits;
    state->number_of_samples = number_of_samples;
    
    // === Pido la memoria =============================================================
    cudaError_t error;
    
    if (state->number_of_bits > MAX_PROB_VECTOR_BITS) {
        state->number_of_prob_vectors = state->number_of_bits / MAX_PROB_VECTOR_BITS;
        state->last_prob_vector_bit_count = state->number_of_bits % MAX_PROB_VECTOR_BITS;
        
        if (state->last_prob_vector_bit_count == 0) {
            state->last_prob_vector_bit_count = MAX_PROB_VECTOR_BITS;
        } else {
            state->number_of_prob_vectors++;
        }
    } else {
        state->number_of_prob_vectors = 1;
        state->last_prob_vector_bit_count = state->number_of_bits;
    }
      
    #ifdef INFO
    fprintf(stdout, "[INFO] Requesting a size %d prob_vector_size CPU memory\n", state->number_of_prob_vectors);
    #endif
      
    size_t prob_vector_array_size = sizeof(float*) * state->number_of_prob_vectors;
    state->gpu_prob_vectors = (float**)malloc(prob_vector_array_size);
    if (!state->gpu_prob_vectors) {
        fprintf(stderr, "[ERROR] Requesting CPU memory for the prob_vector\n");
        exit(EXIT_FAILURE);
    }

    for (int prob_vector_number = 0; prob_vector_number < state->number_of_prob_vectors; prob_vector_number++) {
        int current_prob_vector_number_of_bits = MAX_PROB_VECTOR_BITS;
        if (prob_vector_number + 1 == state->number_of_prob_vectors) {
            current_prob_vector_number_of_bits = state->last_prob_vector_bit_count;
        }

        #ifdef INFO
        fprintf(stdout, "[INFO] > Requesting %d bits GPU memory for prob_vector %d\n", current_prob_vector_number_of_bits, prob_vector_number);
        #endif

        size_t prob_vector_size = sizeof(float) * current_prob_vector_number_of_bits;
        error = cudaMalloc((void**)&(state->gpu_prob_vectors[prob_vector_number]), prob_vector_size);
        
        if (error != cudaSuccess) {
            fprintf(stderr, "[ERROR] Requesting GPU memory for prob_vector_number[%d]\n", prob_vector_number);
            exit(EXIT_FAILURE);
        }
    }
       
    #ifdef INFO
    fprintf(stdout, "[INFO] Requesting a size %d samples CPU memory\n", state->number_of_samples);
    #endif
    
    size_t samples_array_size = sizeof(float*) * state->number_of_samples;
    state->gpu_samples = (float***)malloc(samples_array_size);
    if (!state->gpu_samples) {
        fprintf(stderr, "[ERROR] Requesting samples_fitness CPU memory\n");
        exit(EXIT_FAILURE);
    }
   
    for (int sample_number = 0; sample_number < state->number_of_samples; sample_number++) {
        #ifdef INFO
        fprintf(stdout, "[INFO] > Requesting CPU memory for sample %d vectors array\n", sample_number);
        #endif

        size_t samples_vector_array_size = sizeof(float*) * state->number_of_prob_vectors;
        state->gpu_samples[sample_number] = (float**)malloc(samples_vector_array_size);
        if (!state->gpu_samples) {
            fprintf(stderr, "[ERROR] > Requesting CPU memory for sample_vector_array[%d]\n", sample_number);
            exit(EXIT_FAILURE);
        }
    
        for (int prob_vector_number = 0; prob_vector_number < state->number_of_prob_vectors; prob_vector_number++) {
            int current_prob_vector_number_of_bits = MAX_PROB_VECTOR_BITS;
            if (prob_vector_number + 1 == state->number_of_prob_vectors) {
                current_prob_vector_number_of_bits = state->last_prob_vector_bit_count;
            }
            size_t sample_vector_size = sizeof(char) * (current_prob_vector_number_of_bits / 8);
            assert(current_prob_vector_number_of_bits % 8 == 0);

            #ifdef INFO
            fprintf(stdout, "[INFO] > Requesting sample %d GPU memory for vector %d\n", sample_number, prob_vector_number);
            #endif

            error = cudaMalloc((void**)&(state->gpu_samples[sample_number][prob_vector_number]), sample_vector_size);
            if (error != cudaSuccess) {
                fprintf(stderr, "[ERROR] > Requesting GPU memory for sample_number[%d]\n", sample_number);
                exit(EXIT_FAILURE);
            }
        }
    }
    
    #ifdef INFO
    fprintf(stdout, "[INFO] Requesting samples_fitness CPU memory\n");
    #endif
    
    size_t samples_fitness_size = sizeof(long*) * state->number_of_samples;
    state->gpu_samples_fitness = (long**)malloc(samples_fitness_size);
    if (!state->gpu_samples_fitness) {
        fprintf(stderr, "[ERROR] > Requesting CPU memory for samples_fitness_size\n");
        exit(EXIT_FAILURE);
    }
        
    for (int sample_number = 0; sample_number < state->number_of_samples; sample_number++) {
        #ifdef INFO
        fprintf(stdout, "[INFO] > Requesting GPU memory for sample %d fitness vector array\n", sample_number);
        #endif

        size_t samples_fitness_vector_size = sizeof(long) * state->number_of_prob_vectors;
        error = cudaMalloc((void**)&(state->gpu_samples_fitness[sample_number]), samples_fitness_vector_size);
        if (error != cudaSuccess) {
            fprintf(stderr, "[ERROR] Requesting memory for samples_fitness_vector_size[%d]\n", sample_number);
            exit(EXIT_FAILURE);
        }
    }
    
    // === Inicializo el vector de probabilidades ============================================
    for (int prob_vector_number = 0; prob_vector_number < state->number_of_prob_vectors; prob_vector_number++) {
        int current_prob_vector_number_of_bits = MAX_PROB_VECTOR_BITS;
        if (prob_vector_number + 1 == state->number_of_prob_vectors) {
            current_prob_vector_number_of_bits = state->last_prob_vector_bit_count;
        }

        #ifdef INFO
        fprintf(stdout, "[INFO] Inicializando GPU memory of prob_vector %d (%d bits)\n", prob_vector_number, current_prob_vector_number_of_bits);
        #endif

        const int max_blocks = 128;
        const int max_threads = 768;
        int starting_position = 0;
        
        int total_loops = current_prob_vector_number_of_bits / (max_blocks * max_threads);
        if (current_prob_vector_number_of_bits % (max_blocks * max_threads) > 0) total_loops++;

        #if defined(DEBUG)
        fprintf(stdout, "[DEBUG] Total de loops: %d\n", total_loops);
        #endif

        for (int loop = 0; loop < total_loops; loop++) {
            starting_position = loop * (max_blocks * max_threads);
            
            #if defined(DEBUG)
            fprintf(stdout, "[DEBUG] Loops: %d, Starting posititon: %d, To: %d\n", total_loops, starting_position, starting_position+(max_blocks * max_threads));
            #endif
            
            kern_init_prob_vector<<< max_blocks, max_threads >>>(state->gpu_prob_vectors[prob_vector_number], 
                current_prob_vector_number_of_bits, starting_position);
        }
    }
}

// Paso 2 del algoritmo.
void bga_model_sampling_mt(struct bga_state *state) {

}

// Paso 3 del algoritmo.
void bga_evaluation(struct bga_state *state) {
}

// Paso 4 y 5 del algoritmo.
void bga_model_update(struct bga_state *state) {
}

// Libera la memoria pedida para de estado.
void bga_free(struct bga_state *state) {
    #ifdef INFO
    fprintf(stdout, "[INFO] Freeing memory\n");
    #endif

    for (int vector_number = 0; vector_number < state->number_of_prob_vectors; vector_number++) {
        cudaFree(state->gpu_prob_vectors[vector_number]);
    }
    
    free(state->gpu_prob_vectors);
    
    for (int vector_number = 0; vector_number < state->number_of_prob_vectors; vector_number++) {
        cudaFree(state->gpu_prob_vectors[vector_number]);
    }

    for (int sample_number = 0; sample_number < state->number_of_samples; sample_number++) {
        for (int vector_number = 0; vector_number < state->number_of_prob_vectors; vector_number++) {
            cudaFree(state->gpu_samples[sample_number][vector_number]);
        }
        free(state->gpu_samples[sample_number]);
    }
    free(state->gpu_samples);

    for (int sample_number = 0; sample_number < state->number_of_samples; sample_number++) {
        cudaFree(state->gpu_samples_fitness[sample_number]);
    }
    free(state->gpu_samples_fitness);   
}
