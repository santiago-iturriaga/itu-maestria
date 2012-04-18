#ifndef BILLION_GA__H
#define BILLION_GA__H

struct bga_state {
    long number_of_bits;
    int number_of_samples;
    
    float **gpu_prob_vectors;
    int number_of_prob_vectors;
    long last_prob_vector_bit_count;
    
    float **gpu_samples;
    long *gpu_samples_fitness;
};

// Paso 1 del algortimo.
void bga_initialization(struct bga_state *state, long number_of_bits, int number_of_samples);

// Paso 2 del algoritmo.
void bga_model_sampling(struct bga_state *state);

// Paso 3 del algoritmo.
void bga_evaluation(struct bga_state *state);

// Paso 4 y 5 del algoritmo.
void bga_model_update(struct bga_state *state);

// Libera la memoria pedida para de estado.
void bga_free(struct bga_state *state);

#endif
