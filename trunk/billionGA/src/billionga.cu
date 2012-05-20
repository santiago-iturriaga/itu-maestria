#include <assert.h>
#include <cuda.h>
#include <math.h>

#include "config.h"
#include "util.h"
#include "cuda-util.h"
#include "mtgp-1.1/mtgp32-cuda.h"
#include "billionga.h"

//#define SHOW_PROB_VECTOR_BITS   16
#define SHOW_PROB_VECTOR_BITS   128
#define SHOW_SAMPLE_BITS        128

#define SAMPLE_PROB_VECTOR_BLOCKS    128
#define SAMPLE_PROB_VECTOR_THREADS   256
#define SAMPLE_PROB_VECTOR_SHMEM     (SAMPLE_PROB_VECTOR_THREADS >> 5)

#define UPDATE_PROB_VECTOR_BLOCKS    128
#define UPDATE_PROB_VECTOR_THREADS   256
#define UPDATE_PROB_VECTOR_SHMEM     (UPDATE_PROB_VECTOR_THREADS >> 5)

// Paso 1 del algoritmo.
void bga_initialization(struct bga_state *state, long number_of_bits, int number_of_prob_vectors, int number_of_samples) {
    state->number_of_bits = number_of_bits;
    state->number_of_samples = number_of_samples;
    state->number_of_prob_vectors = number_of_prob_vectors;

    state->population_size = POPULATION_SIZE; //sqrt(3.1416) * sqrt(number_of_bits) * log10(number_of_bits) / 2;
    state->update_value = 1; // / state->population_size;
    state->max_prob_sum = (number_of_bits * POPULATION_SIZE);

    //#if defined(INFO) || defined(DEBUG)
        fprintf(stdout, "[INFO] === Initializing Billion GA ====================\n");
        fprintf(stdout, "[INFO] Problem size   : %ld\n", number_of_bits);
        fprintf(stdout, "[INFO] Population size: %d\n", state->population_size);
        fprintf(stdout, "[INFO] Update value   : %d\n", state->update_value);
        fprintf(stdout, "[INFO] Num. of vectors: %d\n", state->number_of_prob_vectors);
    //#endif

    // === Pido la memoria =============================================================
    #if defined(INFO) || defined(DEBUG)
        fprintf(stdout, "[INFO] === Solicitando memoria =======================\n");
    #endif

    // === Pido la memoria para el vector de probabilidades ==================================

    state->prob_vector_bit_count = state->number_of_bits / number_of_prob_vectors;

    int bits_left = state->number_of_bits % number_of_prob_vectors;
    if (bits_left == 0) {
        state->last_prob_vector_bit_count = state->prob_vector_bit_count;
    } else {
        state->last_prob_vector_bit_count = bits_left;
    }

    #ifdef INFO
        fprintf(stdout, "[INFO] Requesting a size %d prob_vector_size CPU memory\n", state->number_of_prob_vectors);
    #endif

    size_t prob_vectors_acc_prob_array_size = sizeof(float) * state->number_of_prob_vectors;
    state->prob_vectors_acc_prob = (float*)malloc(prob_vectors_acc_prob_array_size);
    if (!state->prob_vectors_acc_prob) {
        fprintf(stderr, "[ERROR] Requesting CPU memory for the prob_vectors_acc_prob\n");
        exit(EXIT_FAILURE);
    }

    size_t prob_vector_array_size = sizeof(int*) * state->number_of_prob_vectors;
    state->gpu_prob_vectors = (int**)malloc(prob_vector_array_size);
    if (!state->gpu_prob_vectors) {
        fprintf(stderr, "[ERROR] Requesting CPU memory for the prob_vector\n");
        exit(EXIT_FAILURE);
    }

    // === Pido la memoria para los samples ==================================================

    #ifdef INFO
        fprintf(stdout, "[INFO] Requesting a size %d samples CPU memory\n", state->number_of_samples);
    #endif

    size_t samples_vector_fitness_array_size = sizeof(int*) * state->number_of_samples;
    state->samples_vector_fitness = (int**)malloc(samples_vector_fitness_array_size);
    if (!state->samples_vector_fitness) {
        fprintf(stderr, "[ERROR] Requesting samples_vector_fitness CPU memory\n");
        exit(EXIT_FAILURE);
    }

    size_t samples_array_size = sizeof(int*) * state->number_of_samples;
    state->gpu_samples = (int***)malloc(samples_array_size);
    if (!state->gpu_samples) {
        fprintf(stderr, "[ERROR] Requesting samples_fitness CPU memory\n");
        exit(EXIT_FAILURE);
    }

    for (int sample_number = 0; sample_number < state->number_of_samples; sample_number++) {
        #ifdef INFO
            fprintf(stdout, "[INFO] > Requesting CPU memory for sample %d vectors array\n", sample_number);
        #endif

        size_t samples_vector_array_size = sizeof(int*) * state->number_of_prob_vectors;
        state->gpu_samples[sample_number] = (int**)malloc(samples_vector_array_size);
        if (!state->gpu_samples) {
            fprintf(stderr, "[ERROR] > Requesting CPU memory for sample_vector_array[%d]\n", sample_number);
            exit(EXIT_FAILURE);
        }

        size_t samples_vector_fitness_size = sizeof(int) * state->number_of_prob_vectors;
        state->samples_vector_fitness[sample_number] = (int*)malloc(samples_vector_fitness_size);
        if (!state->samples_vector_fitness[sample_number]) {
            fprintf(stderr, "[ERROR] Requesting samples_fitness CPU memory\n");
            exit(EXIT_FAILURE);
        }
    }

    size_t samples_fitness_size = sizeof(int*) * state->number_of_samples;

    #ifdef INFO
        fprintf(stdout, "[INFO] Requesting samples_fitness CPU memory (size: %i)\n", state->number_of_samples);
    #endif

    state->samples_fitness = (int*)malloc(samples_fitness_size);
    if (!state->samples_fitness) {
        fprintf(stderr, "[ERROR] > Requesting CPU memory for samples_fitness_size\n");
        exit(EXIT_FAILURE);
    }

    // === Memoria auxiliar ==================================================================
    size_t gpu_int32_vector_sum_size = sizeof(long*) * state->number_of_prob_vectors;
    state->gpu_int32_vector_sum = (long**)malloc(gpu_int32_vector_sum_size);

    size_t cpu_int32_vector_sum_size = sizeof(long*) * state->number_of_prob_vectors;
    state->cpu_int32_vector_sum = (long**)malloc(cpu_int32_vector_sum_size);

    size_t gpu_bit_vector_sum_size = sizeof(int*) * state->number_of_prob_vectors;
    state->gpu_bit_vector_sum = (int**)malloc(gpu_bit_vector_sum_size);

    size_t cpu_bit_vector_sum_size = sizeof(int*) * state->number_of_prob_vectors;
    state->cpu_bit_vector_sum = (int**)malloc(cpu_bit_vector_sum_size);
}

void bga_initialize_thread(struct bga_state *state, int prob_vector_number) {
    cudaError_t error;

    #if defined(TIMMING)
        float gputime;
        cudaEvent_t start;
        cudaEvent_t end;

        ccudaEventCreate(&start);
        ccudaEventCreate(&end);

        ccudaEventRecord(start, 0);
    #endif

    #if defined(INFO) || defined(DEBUG)
        fprintf(stdout, "[INFO] === Solicitando memoria para el thread %d =====\n", prob_vector_number);
    #endif

    // === Pido la memoria para el vector de probabilidades ==================================
    {
        int current_prob_vector_number_of_bits = state->prob_vector_bit_count;
        if (prob_vector_number + 1 == state->number_of_prob_vectors) {
            current_prob_vector_number_of_bits = state->last_prob_vector_bit_count;
        }

        size_t prob_vector_size = sizeof(int) * current_prob_vector_number_of_bits;
        #ifdef INFO
            fprintf(stdout, "[INFO] > Requesting %d bits GPU memory for prob_vector %d (size: %i / %lu Mb)\n",
                current_prob_vector_number_of_bits, prob_vector_number, current_prob_vector_number_of_bits,
                prob_vector_size >> 20);
        #endif
        error = cudaMalloc((void**)&(state->gpu_prob_vectors[prob_vector_number]), prob_vector_size);

        if (error != cudaSuccess) {
            fprintf(stderr, "[ERROR] Requesting GPU memory for prob_vector_number[%d]\n", prob_vector_number);
            exit(EXIT_FAILURE);
        }
    }
    
    #if defined(TIMMING)
        ccudaEventRecord(end, 0);
        ccudaEventSynchronize(end);
        ccudaEventElapsedTime(&gputime, start, end);
        fprintf(stdout, "TIME] Processing time: %f (ms)\n", gputime);
        
        ccudaEventRecord(start, 0);
    #endif

    // === Pido la memoria para los samples ==================================================
    #ifdef INFO
        fprintf(stdout, "[INFO] Requesting a size %d samples CPU memory\n", state->number_of_samples);
    #endif

    {
        for (int sample_number = 0; sample_number < state->number_of_samples; sample_number++) {
            int current_prob_vector_number_of_bits = state->prob_vector_bit_count;
            if (prob_vector_number + 1 == state->number_of_prob_vectors) {
                current_prob_vector_number_of_bits = state->last_prob_vector_bit_count;
            }
            size_t sample_vector_size = sizeof(int) * (current_prob_vector_number_of_bits >> 5);

            int right_size = current_prob_vector_number_of_bits & ((1<<5)-1);
            assert(right_size == 0);

            #ifdef INFO
                fprintf(stdout, "[INFO] > Requesting sample %d GPU memory for vector %d (size: %i / %lu Mb)\n",
                    sample_number, prob_vector_number, current_prob_vector_number_of_bits >> 5, sample_vector_size >> 20);
            #endif

            error = cudaMalloc((void**)&(state->gpu_samples[sample_number][prob_vector_number]), sample_vector_size);
            if (error != cudaSuccess) {
                fprintf(stderr, "[ERROR] > Requesting GPU memory for sample_number[%d]\n", sample_number);
                exit(EXIT_FAILURE);
            }
        }
    }

    #if defined(TIMMING)
        ccudaEventRecord(end, 0);
        ccudaEventSynchronize(end);
        ccudaEventElapsedTime(&gputime, start, end);
        fprintf(stdout, "TIME] Processing time: %f (ms)\n", gputime);
        
        ccudaEventRecord(start, 0);
    #endif

    // === Inicializo el vector de probabilidades ============================================
    #if defined(INFO) || defined(DEBUG)
        fprintf(stdout, "[INFO] === Inicializando memoria =======================\n");
    #endif

    #if defined(TIMMING)
        ccudaEventCreate(&start);
        ccudaEventCreate(&end);

        ccudaEventRecord(start, 0);
    #endif

    {
        int current_prob_vector_number_of_bits = state->prob_vector_bit_count;
        if (prob_vector_number + 1 == state->number_of_prob_vectors) {
            current_prob_vector_number_of_bits = state->last_prob_vector_bit_count;
        }

        #ifdef INFO
            fprintf(stdout, "[INFO] Inicializando GPU memory of prob_vector %d (%d bits)\n",
                prob_vector_number, current_prob_vector_number_of_bits);
        #endif

        vector_set_int(state->gpu_prob_vectors[prob_vector_number],
            current_prob_vector_number_of_bits, INIT_PROB_VECTOR_VALUE);
    }

    #if defined(TIMMING)
    ccudaEventRecord(end, 0);
    ccudaEventSynchronize(end);
    ccudaEventElapsedTime(&gputime, start, end);
    fprintf(stdout, "[TIME] Processing time: %f (ms)\n", gputime);

    ccudaEventDestroy(start);
    ccudaEventDestroy(end);
    #endif

    // === Memoria auxiliar ==================================================================
    vector_sum_int_alloc(&(state->gpu_int32_vector_sum[prob_vector_number]),
        &(state->cpu_int32_vector_sum[prob_vector_number]));

    vector_sum_bit_alloc(&(state->gpu_bit_vector_sum[prob_vector_number]),
        &(state->cpu_bit_vector_sum[prob_vector_number]));
}

float bga_get_part_accumulated_prob(struct bga_state *state, int prob_vector_number) {
    vector_sum_int_init(state->gpu_int32_vector_sum[prob_vector_number]);

    int current_prob_vector_number_of_bits = state->prob_vector_bit_count;
    if (prob_vector_number + 1 == state->number_of_prob_vectors) {
        current_prob_vector_number_of_bits = state->last_prob_vector_bit_count;
    }

    vector_sum_int(state->gpu_prob_vectors[prob_vector_number],
        state->gpu_int32_vector_sum[prob_vector_number],
        current_prob_vector_number_of_bits);

    state->prob_vectors_acc_prob[prob_vector_number] = (vector_sum_int_get(
        state->gpu_int32_vector_sum[prob_vector_number],
        state->cpu_int32_vector_sum[prob_vector_number]) / state->population_size);

    return state->prob_vectors_acc_prob[prob_vector_number];
}

float bga_get_full_accumulated_prob(struct bga_state *state) {
    float result = 0.0;

    for (int prob_vector_number = 0; prob_vector_number < state->number_of_prob_vectors; prob_vector_number++) {
        result += state->prob_vectors_acc_prob[prob_vector_number];
    }

    return result;
}

void bga_show_prob_vector_state(struct bga_state *state) {
    #if defined(TIMMING)
    float gputime;
    cudaEvent_t start;
    cudaEvent_t end;

    ccudaEventCreate(&start);
    ccudaEventCreate(&end);

    ccudaEventRecord(start, 0);
    #endif

    fprintf(stdout, "[INFO] === Probability vector status =======================\n");

    vector_sum_int_init(state->gpu_int32_vector_sum[0]);
    
    vector_sum_int_show(state->gpu_int32_vector_sum[0], state->cpu_int32_vector_sum[0]);

    fprintf(stdout, "[INFO] Prob. vector sample:");

    for (int prob_vector_number = 0; prob_vector_number < state->number_of_prob_vectors; prob_vector_number++) {
        int current_prob_vector_number_of_bits = state->prob_vector_bit_count;
        if (prob_vector_number + 1 == state->number_of_prob_vectors) {
            current_prob_vector_number_of_bits = state->last_prob_vector_bit_count;
        }

        if (prob_vector_number == 0) {
            int probs_to_show_count = SHOW_PROB_VECTOR_BITS;
            if (current_prob_vector_number_of_bits < SHOW_PROB_VECTOR_BITS)
                probs_to_show_count = state->prob_vector_bit_count;

            int *probs_to_show = (int*)malloc(sizeof(int) * probs_to_show_count);
            
            ccudaMemcpy(probs_to_show, state->gpu_prob_vectors[prob_vector_number],
                sizeof(uint32_t) * probs_to_show_count, cudaMemcpyDeviceToHost);

            long sum = 0;

            for (int i = 0; i < probs_to_show_count; i++) {               
                fprintf(stdout, " %d (%.4f)", probs_to_show[i], (float)probs_to_show[i] / (float)state->population_size);
                sum += probs_to_show[i];
            }
                      
            fprintf(stdout, "... Total [%d]: %ld ( %f )\n", probs_to_show_count, sum, (float)sum / (float)(probs_to_show_count * state->population_size));
            
            free(probs_to_show);
        }

        vector_sum_int(state->gpu_prob_vectors[prob_vector_number],
            state->gpu_int32_vector_sum[0], current_prob_vector_number_of_bits);
    }

    long accumulated_probability = 0;
    accumulated_probability = vector_sum_int_get(
        state->gpu_int32_vector_sum[0], 
        state->cpu_int32_vector_sum[0]);
    fprintf(stdout, "[INFO] Prob. vector accumulated probability (%ld / %ld): %f\n", 
        accumulated_probability, state->max_prob_sum, 
        (float)accumulated_probability / (float)state->max_prob_sum);

    #if defined(TIMMING)
    ccudaEventRecord(end, 0);
    ccudaEventSynchronize(end);
    ccudaEventElapsedTime(&gputime, start, end);
    fprintf(stdout, "[TIME] Processing time: %f (ms)\n", gputime);

    ccudaEventDestroy(start);
    ccudaEventDestroy(end);
    #endif
}

void bga_compute_sample_part_fitness(struct bga_state *state, int prob_vector_number) {
    #if defined(TIMMING)
        float gputime;
        cudaEvent_t start;
        cudaEvent_t end;

        ccudaEventCreate(&start);
        ccudaEventCreate(&end);

        ccudaEventRecord(start, 0);
    #endif
    
    #if defined(INFO)
        fprintf(stdout, "[INFO] === Sample vectors fitness =============================\n");
    #endif

    for (int sample_number = 0; sample_number < state->number_of_samples; sample_number++) {
        vector_sum_bit_init(state->gpu_bit_vector_sum[prob_vector_number]);

        #if defined(DEBUG)
            fprintf(stdout, "[INFO] Computing sample vector %d fitness: ", sample_number);
        #endif

        int current_prob_vector_number_of_bits = state->prob_vector_bit_count;
        if (prob_vector_number + 1 == state->number_of_prob_vectors) {
            current_prob_vector_number_of_bits = state->last_prob_vector_bit_count;
        }

        vector_sum_bit(state->gpu_samples[sample_number][prob_vector_number],
            state->gpu_bit_vector_sum[prob_vector_number], current_prob_vector_number_of_bits);

        state->samples_vector_fitness[sample_number][prob_vector_number] = vector_sum_bit_get(
            state->gpu_bit_vector_sum[prob_vector_number], 
            state->cpu_bit_vector_sum[prob_vector_number]);

        #if defined(DEBUG)
            fprintf(stdout, "%d\n", state->samples_vector_fitness[sample_number][prob_vector_number]);
        #endif
    }

    #if defined(TIMMING)
        ccudaEventRecord(end, 0);
        ccudaEventSynchronize(end);
        ccudaEventElapsedTime(&gputime, start, end);
        fprintf(stdout, "[TIME] Processing time: %f (ms)\n", gputime);

        ccudaEventDestroy(start);
        ccudaEventDestroy(end);
    #endif
}

void bga_compute_sample_full_fitness(struct bga_state *state) {
    int result;

    for (int sample_number = 0; sample_number < state->number_of_samples; sample_number++) {
        result = 0;

        for (int prob_vector_number = 0; prob_vector_number < state->number_of_prob_vectors; prob_vector_number++) {
            result += state->samples_vector_fitness[sample_number][prob_vector_number];
        }

        state->samples_fitness[sample_number] = result;
    }
}

void bga_show_samples(struct bga_state *state) {
    #if defined(TIMMING)
        float gputime;
        cudaEvent_t start;
        cudaEvent_t end;

        ccudaEventCreate(&start);
        ccudaEventCreate(&end);

        ccudaEventRecord(start, 0);
    #endif

    fprintf(stdout, "[INFO] === Sample vectors =====================================\n");

    for (int sample_number = 0; sample_number < state->number_of_samples; sample_number++) {
        fprintf(stdout, "[INFO] Sample vector sample (%d):", sample_number);

        for (int prob_vector_number = 0; prob_vector_number < state->number_of_prob_vectors; prob_vector_number++) {
            int current_prob_vector_number_of_bits = state->prob_vector_bit_count;
            if (prob_vector_number + 1 == state->number_of_prob_vectors) {
                current_prob_vector_number_of_bits = state->last_prob_vector_bit_count;
            }

            if (prob_vector_number == 0) {
                int bits_to_show_count = SHOW_SAMPLE_BITS;
                if (current_prob_vector_number_of_bits < SHOW_SAMPLE_BITS)
                    bits_to_show_count = state->prob_vector_bit_count;

                int bytes_to_show_count = bits_to_show_count >> 5;
                int *bytes_to_show = (int*)malloc(sizeof(int) * bytes_to_show_count);

                ccudaMemcpy(bytes_to_show, state->gpu_samples[sample_number][prob_vector_number],
                    sizeof(uint32_t) * bytes_to_show_count, cudaMemcpyDeviceToHost);

                for (int i = 0; i < bytes_to_show_count; i++) {
                    fprintf(stdout, " %s", int_to_binary(bytes_to_show[i]));
                }

                free(bytes_to_show);

                fprintf(stdout, "...\n");
            }
        }

        fprintf(stdout, "[INFO] Sample %d fitness: %d\n", sample_number, state->samples_fitness[sample_number]);
    }

    #if defined(TIMMING)
        ccudaEventRecord(end, 0);
        ccudaEventSynchronize(end);
        ccudaEventElapsedTime(&gputime, start, end);
        fprintf(stdout, "[TIME] Processing time: %f (ms)\n", gputime);

        ccudaEventDestroy(start);
        ccudaEventDestroy(end);
    #endif
}

__global__ void kern_sample_prob_vector(int *gpu_prob_vector, int prob_vector_size,
    int prob_vector_starting_pos, float *prng_vector, int prng_vector_size, int *gpu_sample,
    int population_size) {

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int samples_per_loop = gridDim.x * blockDim.x;

    int max_samples_doable = prob_vector_size - prob_vector_starting_pos;
    if (max_samples_doable > prng_vector_size) max_samples_doable = prng_vector_size;

    int loops_count = max_samples_doable / samples_per_loop;
    if (max_samples_doable % samples_per_loop > 0) loops_count++;

    __shared__ int current_block_sample[SAMPLE_PROB_VECTOR_THREADS];

    int prob_vector_position;
    int prng_position;
    int block_starting_pos;

    for (int loop = 0; loop < loops_count; loop++) {
        // 0 por defecto.
        current_block_sample[tid] = 0;

        // Cada loop genera blockDim.x bits y los guarda en el array de __shared__ memory.
        block_starting_pos = (samples_per_loop * loop) + (bid * blockDim.x);
        prng_position = block_starting_pos + tid;
        prob_vector_position = prob_vector_starting_pos + prng_position;

        if (prng_position < max_samples_doable) {
            if ((gpu_prob_vector[prob_vector_position] + population_size) >= (prng_vector[prng_position] * population_size)) {
                // 1
                current_block_sample[tid] = 1 << (tid & ((1 << 5)-1));
            }
        }

        __syncthreads();

        if ((tid << 5) < SAMPLE_PROB_VECTOR_THREADS) {
            int aux = current_block_sample[tid << 5];

            #pragma unroll
            for (int i = 1; i < 32; i++) {
                aux = aux | current_block_sample[(tid << 5)+i];
            }

            int sample_pos = prob_vector_starting_pos + block_starting_pos;

            if ((sample_pos + (tid << 5)) < prob_vector_size) {
                gpu_sample[(sample_pos >> 5) + tid] = aux;
            }
        }

        __syncthreads();
    }
}

// Paso 2 del algoritmo.
void bga_model_sampling_mt(struct bga_state *state, mtgp32_status *mt_status, int prob_vector_number) {
    #if defined(DEBUG)
        fprintf(stdout, "[INFO] === Sampling the model =======================\n");
    #endif
    
    #if defined(TIMMING)
        float gputime;
        cudaEvent_t start;
        cudaEvent_t end;

        cudaEvent_t start_inner;
        cudaEvent_t end_inner;
    #endif

    for (int sample_number = 0; sample_number < state->number_of_samples; sample_number++) {
        #if defined(DEBUG)
            fprintf(stdout, "[INFO] > Sample %d ", sample_number);
        #endif
        #if defined(TIMMING)
            ccudaEventCreate(&start);
            ccudaEventCreate(&end);
            ccudaEventRecord(start, 0);

            ccudaEventCreate(&start_inner);
            ccudaEventCreate(&end_inner);
        #endif

        int current_prob_vector_number_of_bits = state->prob_vector_bit_count;
        if (prob_vector_number + 1 == state->number_of_prob_vectors) {
            current_prob_vector_number_of_bits = state->last_prob_vector_bit_count;
        }

        int total_loops;
        total_loops = current_prob_vector_number_of_bits / mt_status->numbers_per_gen;
        if (current_prob_vector_number_of_bits % mt_status->numbers_per_gen > 0) total_loops++;

        int prob_vector_starting_pos;

        for (int loop = 0; loop < total_loops; loop++) {
            prob_vector_starting_pos = mt_status->numbers_per_gen * loop;

            // Genero números aleatorios.
            #if defined(TIMMING)
                fprintf(stdout, "[TIME] Generate mtgp32_generate_float\n", gputime);
                ccudaEventRecord(start_inner, 0);
            #endif

            mtgp32_generate_float(mt_status);

            #if defined(TIMMING)
                ccudaEventRecord(end_inner, 0);
                ccudaEventSynchronize(end_inner);
                ccudaEventElapsedTime(&gputime, start_inner, end_inner);
                fprintf(stdout, "[TIME] Processing time: %f (ms)\n", gputime);
            #endif

            #if defined(DEBUG)
                fprintf(stdout, ".");
            #endif

            #if defined(TIMMING)
                fprintf(stdout, "[TIME] Generate kern_sample_prob_vector\n", gputime);
                ccudaEventRecord(start_inner, 0);
            #endif

            // Sampleo el vector de prob. con los números aleatorios generados.
            kern_sample_prob_vector<<< SAMPLE_PROB_VECTOR_BLOCKS, SAMPLE_PROB_VECTOR_THREADS>>>(
                state->gpu_prob_vectors[prob_vector_number], current_prob_vector_number_of_bits,
                prob_vector_starting_pos, (float*)mt_status->d_data, mt_status->numbers_per_gen,
                state->gpu_samples[sample_number][prob_vector_number], state->population_size);

            #if defined(TIMMING)
                ccudaEventRecord(end_inner, 0);
                ccudaEventSynchronize(end_inner);
                ccudaEventElapsedTime(&gputime, start_inner, end_inner);
                fprintf(stdout, "[TIME] Processing time: %f (ms)\n", gputime);
            #endif
        }

        #if defined(DEBUG)
            fprintf(stdout, "(%d)\n", total_loops);
        #endif
        #if defined(TIMMING)
            ccudaEventRecord(end, 0);
            ccudaEventSynchronize(end);
            ccudaEventElapsedTime(&gputime, start, end);
            fprintf(stdout, "[TIME] Total processing time: %f (ms)\n", gputime);
        #endif
    }

    bga_show_samples(state);

    #if defined(TIMMING)
        ccudaEventDestroy(start_inner);
        ccudaEventDestroy(end_inner);

        ccudaEventDestroy(start);
        ccudaEventDestroy(end);
    #endif
}

void cpu_model_update(int *gpu_prob_vector, int prob_vector_size,
    int *gpu_best_sample, int *gpu_worst_sample, float update_value) {

    int *prob_vector = (int*)malloc(sizeof(int) * prob_vector_size);

    ccudaMemcpy(prob_vector, gpu_prob_vector, sizeof(uint32_t) * prob_vector_size, 
        cudaMemcpyDeviceToHost);

    long current_acc_prob = 0, new_acc_prob = 0;
    for (int i = 0; i < prob_vector_size; i++) {
        current_acc_prob += prob_vector[i];
    }

    int sample_size = prob_vector_size >> 5;
    int *best_sample = (int*)malloc(sizeof(int) * sample_size);
    int *worst_sample = (int*)malloc(sizeof(int) * sample_size);

    ccudaMemcpy(best_sample, gpu_best_sample,
        sizeof(uint32_t) * sample_size, cudaMemcpyDeviceToHost);
        
    ccudaMemcpy(worst_sample, gpu_worst_sample,
        sizeof(uint32_t) * sample_size, cudaMemcpyDeviceToHost);

    int best_sample_current_bit_value;
    int worst_sample_current_bit_value;
    int delta;

    for (int i = 0; i < prob_vector_size; i++) {
        int bit_pos = i & ((1 << 5)-1);
        int int_pos = i >> 5;
        
        best_sample_current_bit_value = (best_sample[int_pos] & (1 << bit_pos)) >> bit_pos;
        worst_sample_current_bit_value = (worst_sample[int_pos] & (1 << bit_pos)) >> bit_pos;

        delta = best_sample_current_bit_value - worst_sample_current_bit_value;
        prob_vector[i] += delta;
    }
    
    for (int i = 0; i < prob_vector_size; i++) {
        new_acc_prob += prob_vector[i];
    }
    
    fprintf(stdout, "[DEBUG][CPU] Acc. prob. => Current %ld , New %ld (delta: %ld )\n", 
        current_acc_prob, new_acc_prob, new_acc_prob - current_acc_prob);
        
    free(prob_vector);
    free(best_sample);
    free(worst_sample);
}

__global__ void kern_model_update(int *gpu_prob_vector, int prob_vector_size,
    int *best_sample, int *worst_sample, float update_value) {

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    __shared__ int best_sample_part[UPDATE_PROB_VECTOR_SHMEM];
    __shared__ int worst_sample_part[UPDATE_PROB_VECTOR_SHMEM];

    int loop_size = gridDim.x * blockDim.x;
    int loop_count = prob_vector_size / loop_size;
    if (loop_count % loop_size > 0) loop_count++;

    int prob_vector_position;
    int block_starting_pos;

    const int tid_int = tid >> 5;
    const int tid_bit = tid & ((1 << 5)-1);

    int best_sample_current_bit_value;
    int worst_sample_current_bit_value;
    int delta;

    for (int loop = 0; loop < loop_count; loop++) {
        block_starting_pos = (loop_size * loop) + (bid * blockDim.x);

        if (tid < UPDATE_PROB_VECTOR_SHMEM) {
            if ((block_starting_pos + (tid << 5)) < prob_vector_size) {
                best_sample_part[tid] = best_sample[(block_starting_pos >> 5) + tid];
                worst_sample_part[tid] = worst_sample[(block_starting_pos >> 5) + tid];
            }
        }
        __syncthreads();

        prob_vector_position = block_starting_pos + tid;

        if (prob_vector_position < prob_vector_size) {
            best_sample_current_bit_value = (best_sample_part[tid_int] & (1 << tid_bit)) >> tid_bit;
            worst_sample_current_bit_value = (worst_sample_part[tid_int] & (1 << tid_bit)) >> tid_bit;

            delta = best_sample_current_bit_value - worst_sample_current_bit_value;
            
            //atomicAdd(&(gpu_prob_vector[prob_vector_position]), delta);
            
            float aux = gpu_prob_vector[prob_vector_position];
            gpu_prob_vector[prob_vector_position] = aux + (delta * update_value);
        }
    }
}

// Paso 4 y 5 del algoritmo.
void bga_model_update(struct bga_state *state, int prob_vector_number) {
    #if defined(DEBUG)
        fprintf(stdout, "[INFO] === Updating the model =======================\n");
    #endif
    #if defined(TIMMING)
        float gputime;
        cudaEvent_t start;
        cudaEvent_t end;

        ccudaEventCreate(&start);
        ccudaEventCreate(&end);
        ccudaEventRecord(start, 0);
    #endif

    assert(state->number_of_samples == 2);

    int best_sample_index, worst_sample_index;

    int fitness_sample_a, fitness_sample_b;
    #if defined(FULL_FITNESS_UPDATE)
        fitness_sample_a = state->samples_fitness[0];
        fitness_sample_b = state->samples_fitness[1];
    #endif
    #if defined(PARTIAL_FITNESS_UPDATE)
        fitness_sample_a = state->samples_vector_fitness[0][prob_vector_number];
        fitness_sample_b = state->samples_vector_fitness[1][prob_vector_number];
    #endif

    if (fitness_sample_a >= fitness_sample_b) {
        best_sample_index = 0;
        worst_sample_index = 1;
    }
    else {
        best_sample_index = 1;
        worst_sample_index = 0;
    }

    int *best_sample;
    int *worst_sample;

    int current_prob_vector_number_of_bits = state->prob_vector_bit_count;
    if (prob_vector_number + 1 == state->number_of_prob_vectors) {
        current_prob_vector_number_of_bits = state->last_prob_vector_bit_count;
    }

    best_sample = state->gpu_samples[best_sample_index][prob_vector_number];
    worst_sample = state->gpu_samples[worst_sample_index][prob_vector_number];

    cpu_model_update(state->gpu_prob_vectors[prob_vector_number], current_prob_vector_number_of_bits,
        best_sample, worst_sample, state->update_value);

    kern_model_update <<< UPDATE_PROB_VECTOR_BLOCKS, UPDATE_PROB_VECTOR_THREADS >>>(
        state->gpu_prob_vectors[prob_vector_number], current_prob_vector_number_of_bits,
        best_sample, worst_sample, state->update_value);

    #if defined(TIMMING)
        ccudaEventRecord(end, 0);
        ccudaEventSynchronize(end);
        ccudaEventElapsedTime(&gputime, start, end);
        fprintf(stdout, "[TIME] Processing time: %f (ms)\n", gputime);

        ccudaEventDestroy(start);
        ccudaEventDestroy(end);
    #endif
}

// Libera la memoria pedida para de estado.
void bga_free(struct bga_state *state) {
    #ifdef INFO
    fprintf(stdout, "[INFO] Freeing memory\n");
    #endif

    for (int vector_number = 0; vector_number < state->number_of_prob_vectors; vector_number++) {
        fprintf(stderr, "[INFO] Freeing gpu_prob_vectors[%d]\n", vector_number);
        cudaFree(state->gpu_prob_vectors[vector_number]);
    }

    free(state->gpu_prob_vectors);

    for (int sample_number = 0; sample_number < state->number_of_samples; sample_number++) {
        for (int vector_number = 0; vector_number < state->number_of_prob_vectors; vector_number++) {
            fprintf(stderr, "[INFO] Freeing gpu_samples[%d][%d]\n", sample_number, vector_number);
            cudaFree(state->gpu_samples[sample_number][vector_number]);
        }
        free(state->gpu_samples[sample_number]);
    }
       
    free(state->gpu_samples);
    free(state->samples_fitness);

    for (int vector_number = 0; vector_number < state->number_of_prob_vectors; vector_number++) {
        fprintf(stderr, "[INFO] Freeing vector_sum_float_free[%d]\n", vector_number);
        vector_sum_int_free(
            state->gpu_int32_vector_sum[vector_number], 
            state->cpu_int32_vector_sum[vector_number]);
            
        fprintf(stderr, "[INFO] Freeing vector_sum_bit_free[%d]\n", vector_number);
        vector_sum_bit_free(
            state->gpu_bit_vector_sum[vector_number], 
            state->cpu_bit_vector_sum[vector_number]);
    }
    
    free(state->gpu_int32_vector_sum);
    free(state->cpu_int32_vector_sum);
    free(state->gpu_bit_vector_sum);
    free(state->cpu_bit_vector_sum);
}
