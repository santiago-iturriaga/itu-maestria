#include "mtgp-1.1/mtgp32-cuda.h"

#ifndef BILLION_GA__H
#define BILLION_GA__H

struct bga_state {
    long number_of_bits;
    int number_of_samples;
    
    float **gpu_prob_vectors; // gpu_prob_vectors[VECTOR][BIT_PROBABILITY]
    int number_of_prob_vectors;
    long last_prob_vector_bit_count;
    
    int ***gpu_samples; // gpu_samples[SAMPLE][VECTOR][32 BIT]
    int *samples_fitness; // gpu_samples_fitness[SAMPLE]
};

// Paso 1 del algortimo.
void bga_initialization(struct bga_state *state, long number_of_bits, int number_of_samples);

void bga_show_prob_vector_state(struct bga_state *state);

// Paso 2 del algoritmo.
void bga_model_sampling_mt(struct bga_state *state, mtgp32_status *mt_status);

// Paso 3 del algoritmo.
void bga_evaluation(struct bga_state *state);

// Paso 4 y 5 del algoritmo.
void bga_model_update(struct bga_state *state);

// Libera la memoria pedida para de estado.
void bga_free(struct bga_state *state);

#endif
