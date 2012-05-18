#include "mtgp-1.1/mtgp32-cuda.h"

#ifndef BILLION_GA__H
#define BILLION_GA__H

struct bga_state {
    long number_of_bits;
    int number_of_samples;
    
    double population_size; 
    double update_value; 
       
    // Probabilities =====
    int **gpu_prob_vectors;           // [VECTOR][BIT_PROBABILITY]
    int number_of_prob_vectors;
    long prob_vector_bit_count;
    long last_prob_vector_bit_count;
    float *prob_vectors_acc_prob;       // [VECTOR]
    
    // Samples ============
    int ***gpu_samples;             // [SAMPLE][VECTOR][32 BIT]
    int **samples_vector_fitness;   // [SAMPLE][VECTOR]
    int *samples_fitness;           // [SAMPLE]
    
    // Auxiliares =========
    long **gpu_int32_vector_sum;      // [VECTOR][INTERNAL]
    long **cpu_int32_vector_sum;      // [VECTOR][INTERNAL]
    int **gpu_bit_vector_sum;       // [VECTOR][INTERNAL]
    int **cpu_bit_vector_sum;       // [VECTOR][INTERNAL]
};

// Paso 1 del algortimo.
// number_of_prob_vectors debe ser potencia de 2 (1, 2, 4, 8,...)
void bga_initialization(struct bga_state *state, long number_of_bits, int number_of_prob_vectors, int number_of_samples);
void bga_initialize_thread(struct bga_state *state, int prob_vector_number);

// Paso 2 del algoritmo.
void bga_model_sampling_mt(struct bga_state *state, mtgp32_status *mt_status, int prob_vector_number);

// Paso 3 del algoritmo.
void bga_compute_sample_part_fitness(struct bga_state *state, int prob_vector_number);
void bga_compute_sample_full_fitness(struct bga_state *state);

// Paso 4 y 5 del algoritmo.
void bga_model_update(struct bga_state *state, int prob_vector_number);

// Libera la memoria pedida para de estado.
void bga_free(struct bga_state *state);

// Mostrar resultados.
float bga_get_part_accumulated_prob(struct bga_state *state, int prob_vector_number);
float bga_get_full_accumulated_prob(struct bga_state *state);

// DEBUG
void bga_show_prob_vector_state(struct bga_state *state);
void bga_show_samples(struct bga_state *state);

#endif
