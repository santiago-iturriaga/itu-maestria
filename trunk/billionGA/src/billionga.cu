#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

#include "config.h"
#include "cuda-util.h"
#include "billionga.h"

#define INIT_PROB_VECTOR_VALUE      0.5
#define INIT_PROB_VECTOR_BLOCKS     128
#define INIT_PROB_VECTOR_THREADS    256

#define SHOW_PROB_VECTOR_STARTING_BITS      10
#define SHOW_PROB_VECTOR_ENDING_BITS        10

#define SUM_PROB_VECTOR_BLOCKS      128
#define SUM_PROB_VECTOR_THREADS     512
#define SUM_PROB_VECTOR_SHARED_MEM  512

/*
 * Establece el valor de todos los elementos de un vector a "value"
 */
__global__ void kern_vector_set(float *gpu_prob_vector, int max_size, float value) {
    int bits_per_loop = gridDim.x * blockDim.x;
    
    int loop_count = max_size / bits_per_loop;
    if (max_size % bits_per_loop > 0) loop_count++;
        
    for (int i = 0; i < loop_count; i++) {
        int current_position = (i * bits_per_loop) + (blockIdx.x * blockDim.x + threadIdx.x);
        
        if (current_position < max_size) {
            gpu_prob_vector[current_position] = INIT_PROB_VECTOR_VALUE;
        }
        
        __syncthreads();
    }
}

/*__global__ void kern_sum_prob_vector(float *gpu_partial_sum, float *gpu_prob_vectors, 
    int max_size, int starting_position) {
    
    __shared__ float local_memory[SUM_PROB_VECTOR_SHARED_MEM];
    
    int current_position = starting_position + (blockIdx.x * blockDim.x + threadIdx.x);
    local_memory[threadIdx.x] = gpu_prob_vectors[current_position];
    local_memory[threadIdx.x] += gpu_prob_vectors[current_position + blockDim.x];
    
    unsigned int t = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride > 1; stride >> 1)
    {
        __syncthreads();
        if (t < stride) gpu_partial_sum[t] += gpu_partial_sum[t+stride];
    }
    
    __syncthreads();
    if (t == 0) gpu_prob_vectors[starting_position] = gpu_partial_sum[0];
}*/

__global__ void kern_sum_prob_vector(float *g_idata, float *g_odata, unsigned int max_size)
{
    __shared__ float sdata[SUM_PROB_VECTOR_SHARED_MEM];

    unsigned int tid = threadIdx.x;
    
    unsigned int adds_per_loop = gridDim.x * blockDim.x * 2;
    unsigned int loops_count = max_size / adds_per_loop;
    if (max_size % adds_per_loop > 0) loops_count++;

    unsigned int starting_position;
    
    for (unsigned int loop = 0; loop < loops_count; loop++) {
        // Perform first level of reduction, reading from global memory, writing to shared memory
        starting_position = adds_per_loop * loop;
        
        unsigned int i = starting_position + (blockIdx.x * (blockDim.x * 2) + threadIdx.x);

        float mySum = (i < max_size) ? g_idata[i] : 0;
        if (i + blockDim.x < max_size) 
            mySum += g_idata[i + blockDim.x];  

        sdata[tid] = mySum;
        __syncthreads();

        // do reduction in shared mem
        for(unsigned int s = blockDim.x/2; s > 0; s >>= 1) 
        {
            if (tid < s) 
            {
                sdata[tid] = mySum = mySum + sdata[tid + s];
            }
            __syncthreads();
        }

        // write result for this block to global mem 
        if (tid == 0) g_odata[blockIdx.x] += sdata[0];
    
        __syncthreads();
    }
}

// Paso 1 del algoritmo.
void bga_initialization(struct bga_state *state, long number_of_bits, int number_of_samples) {
    state->number_of_bits = number_of_bits;
    state->number_of_samples = number_of_samples;
    
    // === Pido la memoria =============================================================
    #if defined(INFO) || defined(DEBUG)
    fprintf(stdout, "[INFO] === Solicitando memoria =======================\n");
    #endif

    cudaError_t error;

    #if defined(DEBUG)
    float gputime;
    cudaEvent_t start;
    cudaEvent_t end;
    
    ccudaEventCreate(&start);
    ccudaEventCreate(&end);

    ccudaEventRecord(start, 0);
    #endif
    
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
    
    #if defined(DEBUG)
    ccudaEventRecord(end, 0);
    ccudaEventSynchronize(end);
    ccudaEventElapsedTime(&gputime, start, end);
    fprintf(stdout, "TIME] Processing time: %f (ms)\n", gputime);
    #endif
    
    // === Inicializo el vector de probabilidades ============================================
    #if defined(INFO) || defined(DEBUG)
    fprintf(stdout, "[INFO] === Inicializando memoria =======================\n");
    #endif

    #if defined(DEBUG)   
    ccudaEventCreate(&start);
    ccudaEventCreate(&end);

    ccudaEventRecord(start, 0);
    #endif
    
    for (int prob_vector_number = 0; prob_vector_number < state->number_of_prob_vectors; prob_vector_number++) {
        int current_prob_vector_number_of_bits = MAX_PROB_VECTOR_BITS;
        if (prob_vector_number + 1 == state->number_of_prob_vectors) {
            current_prob_vector_number_of_bits = state->last_prob_vector_bit_count;
        }

        #ifdef INFO
        fprintf(stdout, "[INFO] Inicializando GPU memory of prob_vector %d (%d bits)\n", 
            prob_vector_number, current_prob_vector_number_of_bits);
        #endif

        kern_vector_set<<< INIT_PROB_VECTOR_BLOCKS, INIT_PROB_VECTOR_THREADS >>>(
            state->gpu_prob_vectors[prob_vector_number], 
            current_prob_vector_number_of_bits, INIT_PROB_VECTOR_VALUE);
    }
    
    #if defined(DEBUG)
    ccudaEventRecord(end, 0);
    ccudaEventSynchronize(end);
    ccudaEventElapsedTime(&gputime, start, end);
    fprintf(stdout, "[TIME] Processing time: %f (ms)\n", gputime);
        
    ccudaEventDestroy(start);
    ccudaEventDestroy(end);
    #endif
}

void bga_show_prob_vector_state(struct bga_state *state) {
    double accumulated_probability = 0.0;

    #if defined(DEBUG)
    float gputime;
    cudaEvent_t start;
    cudaEvent_t end;
    
    ccudaEventCreate(&start);
    ccudaEventCreate(&end);

    ccudaEventRecord(start, 0);
    #endif

    fprintf(stdout, "[INFO] === Probability vector status =======================\n");
    const int max_partial_mem = SUM_PROB_VECTOR_BLOCKS * SUM_PROB_VECTOR_THREADS;
       
    float *partial_sum;
    ccudaMalloc((void**)&(partial_sum), sizeof(float) * max_partial_mem);

    kern_vector_set<<< INIT_PROB_VECTOR_BLOCKS, INIT_PROB_VECTOR_THREADS >>>(
        partial_sum, max_partial_mem, 0.0);

    fprintf(stdout, "[INFO] Prob. vector sample:");
    
    for (int prob_vector_number = 0; prob_vector_number < state->number_of_prob_vectors; prob_vector_number++) {
        int current_prob_vector_number_of_bits = MAX_PROB_VECTOR_BITS;
        
        if (prob_vector_number + 1 == state->number_of_prob_vectors) {
            if (prob_vector_number != 0) {
                fprintf(stdout, "...");
            }
            
            current_prob_vector_number_of_bits = state->last_prob_vector_bit_count;
            
            int probs_to_show_count = SHOW_PROB_VECTOR_STARTING_BITS;
            if (current_prob_vector_number_of_bits < SHOW_PROB_VECTOR_STARTING_BITS) 
                probs_to_show_count = current_prob_vector_number_of_bits;
            
            float *probs_to_show = (float*)malloc(sizeof(float) * probs_to_show_count);
            ccudaMemcpy(probs_to_show, state->gpu_prob_vectors[prob_vector_number], 
                sizeof(uint32_t) * probs_to_show_count, cudaMemcpyDeviceToHost);
            
            for (int i = 0; i < probs_to_show_count; i++) {
                fprintf(stdout, " %.4f", probs_to_show[i]);
            }
        } if (prob_vector_number == 0) {
            int probs_to_show_count = SHOW_PROB_VECTOR_STARTING_BITS;
            if (MAX_PROB_VECTOR_BITS < SHOW_PROB_VECTOR_STARTING_BITS) 
                probs_to_show_count = MAX_PROB_VECTOR_BITS;
            
            float *probs_to_show = (float*)malloc(sizeof(float) * probs_to_show_count);
            ccudaMemcpy(probs_to_show, state->gpu_prob_vectors[prob_vector_number], 
                sizeof(uint32_t) * probs_to_show_count, cudaMemcpyDeviceToHost);
            
            for (int i = 0; i < probs_to_show_count; i++) {
                fprintf(stdout, " %.4f", probs_to_show[i]);
            }
        }
                   
        kern_sum_prob_vector<<< SUM_PROB_VECTOR_BLOCKS, SUM_PROB_VECTOR_THREADS >>>( 
            state->gpu_prob_vectors[prob_vector_number], partial_sum,
            current_prob_vector_number_of_bits);
    }
    
    kern_sum_prob_vector<<< SUM_PROB_VECTOR_BLOCKS, SUM_PROB_VECTOR_THREADS >>>( 
        partial_sum, partial_sum, max_partial_mem);
    
    ccudaMemcpy(&accumulated_probability, partial_sum, sizeof(float), cudaMemcpyDeviceToHost);
    
    fprintf(stdout, "\n[INFO] Prob. vector accumulated probability: %.4f\n", accumulated_probability);
    
    #if defined(DEBUG)
    ccudaEventRecord(end, 0);
    ccudaEventSynchronize(end);
    ccudaEventElapsedTime(&gputime, start, end);
    fprintf(stdout, "[TIME] Processing time: %f (ms)\n", gputime);
        
    ccudaEventDestroy(start);
    ccudaEventDestroy(end);
    #endif
}

// Paso 2 del algoritmo.
void bga_model_sampling_mt(struct bga_state *state) {
    //mtgp32_generate_float(&mt_status);
    //mtgp32_print_generated_floats(&mt_status);

    //mtgp32_generate_uint32(&mt_status);
    //mtgp32_print_generated_uint32(&mt_status);
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
