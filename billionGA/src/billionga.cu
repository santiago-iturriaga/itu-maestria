#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

#include "config.h"
#include "util.h"
#include "cuda-util.h"
#include "mtgp-1.1/mtgp32-cuda.h"
#include "billionga.h"

#define VECTOR_SET_BLOCKS       128
#define VECTOR_SET_THREADS      256

#define VECTOR_SUM_BLOCKS       128
#define VECTOR_SUM_THREADS      512
#define VECTOR_SUM_SHARED_MEM   512

#define SHOW_PROB_VECTOR_BITS   10
#define SHOW_SAMPLE_BITS        32

#define SAMPLE_PROB_VECTOR_BLOCKS    128
#define SAMPLE_PROB_VECTOR_THREADS   256
#define SAMPLE_PROB_VECTOR_SHMEM     (SAMPLE_PROB_VECTOR_THREADS >> 3)

/*
 * Establece el valor de todos los elementos de un vector a "value".
 */
__global__ void kern_vector_set(float *gpu_vector, int size, float value) {
    int bits_per_loop = gridDim.x * blockDim.x;
    
    int loop_count = size / bits_per_loop;
    if (size % bits_per_loop > 0) loop_count++;
        
    for (int i = 0; i < loop_count; i++) {
        int current_position = (i * bits_per_loop) + (blockIdx.x * blockDim.x + threadIdx.x);
        
        if (current_position < size) {
            gpu_vector[current_position] = value;
        }
        
        __syncthreads();
    }
}

void vector_sum_init(float **partial_sum) {      
    ccudaMalloc((void**)partial_sum, sizeof(float) * VECTOR_SUM_BLOCKS);

    kern_vector_set<<< 1, VECTOR_SUM_BLOCKS >>>(
        *partial_sum, VECTOR_SUM_BLOCKS, 0.0);
}

float vector_sum_free(float *partial_sum) {
    float accumulated_probability = 0.0;
    
    float *cpu_partial_sum;
    cpu_partial_sum = (float*)malloc(sizeof(float) * VECTOR_SUM_BLOCKS);
    
    ccudaMemcpy(cpu_partial_sum, partial_sum, sizeof(float) * VECTOR_SUM_BLOCKS, cudaMemcpyDeviceToHost);
    for (int i = 0; i < VECTOR_SUM_BLOCKS; i++) {
        //fprintf(stdout, "%f ", cpu_partial_sum[i]);
        accumulated_probability += cpu_partial_sum[i];
    }

    ccudaFree(partial_sum);
    return accumulated_probability;
}

/*
 * Reduce un array sumando cada uno de sus elementos.
 * gpu_output_data debe tener un elemento por bloque del kernel.
 */
__global__ void kern_vector_sum(float *gpu_input_data, float *gpu_output_data, unsigned int size)
{
    __shared__ float sdata[VECTOR_SUM_SHARED_MEM];

    unsigned int tid = threadIdx.x;
    
    unsigned int adds_per_loop = gridDim.x * blockDim.x * 2;
    unsigned int loops_count = size / adds_per_loop;
    if (size % adds_per_loop > 0) loops_count++;

    unsigned int starting_position;
    
    for (unsigned int loop = 0; loop < loops_count; loop++) {
        // Perform first level of reduction, reading from global memory, writing to shared memory
        starting_position = adds_per_loop * loop;
        
        unsigned int i = starting_position + (blockIdx.x * (blockDim.x * 2) + threadIdx.x);

        float mySum;
        if (i < size) {
            mySum = gpu_input_data[i];
            
            if (i + blockDim.x < size) {
                mySum += gpu_input_data[i + blockDim.x];  
            }
        } else {
            mySum = 0;
        }

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
        if (tid == 0) gpu_output_data[blockIdx.x] += sdata[0];
    
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
        fprintf(stdout, "[INFO] > Requesting %d bits GPU memory for prob_vector %d\n", 
            current_prob_vector_number_of_bits, prob_vector_number);
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
    
        for (int prob_vector_number = 0; prob_vector_number < state->number_of_prob_vectors; prob_vector_number++) {
            int current_prob_vector_number_of_bits = MAX_PROB_VECTOR_BITS;
            if (prob_vector_number + 1 == state->number_of_prob_vectors) {
                current_prob_vector_number_of_bits = state->last_prob_vector_bit_count;
            }
            size_t sample_vector_size = sizeof(int) * (current_prob_vector_number_of_bits >> 5);
            fprintf(stdout, "current_prob_vector_number_of_bits = %d, current_prob_vector_number_of_bits >> 5 = %d\n",
                current_prob_vector_number_of_bits, current_prob_vector_number_of_bits >> 5);
            assert(current_prob_vector_number_of_bits & 0x11111 == 0);

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

        kern_vector_set<<< VECTOR_SET_BLOCKS, VECTOR_SET_THREADS >>>(
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
    #if defined(DEBUG)
    float gputime;
    cudaEvent_t start;
    cudaEvent_t end;
    
    ccudaEventCreate(&start);
    ccudaEventCreate(&end);

    ccudaEventRecord(start, 0);
    #endif

    fprintf(stdout, "[INFO] === Probability vector status =======================\n");

    float *partial_sum;
    vector_sum_init(&partial_sum);

    fprintf(stdout, "[INFO] Prob. vector sample:");
    
    for (int prob_vector_number = 0; prob_vector_number < state->number_of_prob_vectors; prob_vector_number++) {
        int current_prob_vector_number_of_bits = MAX_PROB_VECTOR_BITS;
        if (prob_vector_number + 1 == state->number_of_prob_vectors) {
            current_prob_vector_number_of_bits = state->last_prob_vector_bit_count;
        }        

        if (prob_vector_number == 0) {
            int probs_to_show_count = SHOW_PROB_VECTOR_BITS;
            if (current_prob_vector_number_of_bits < SHOW_PROB_VECTOR_BITS) 
                probs_to_show_count = MAX_PROB_VECTOR_BITS;
            
            float *probs_to_show = (float*)malloc(sizeof(float) * probs_to_show_count);
            ccudaMemcpy(probs_to_show, state->gpu_prob_vectors[prob_vector_number], 
                sizeof(uint32_t) * probs_to_show_count, cudaMemcpyDeviceToHost);
            
            for (int i = 0; i < probs_to_show_count; i++) {
                fprintf(stdout, " %.4f", probs_to_show[i]);
            }
            fprintf(stdout, "...\n");
        }
        
        kern_vector_sum<<< VECTOR_SUM_BLOCKS, VECTOR_SUM_THREADS >>>( 
            state->gpu_prob_vectors[prob_vector_number], partial_sum,
            current_prob_vector_number_of_bits);
    }

    double accumulated_probability = 0.0;
    accumulated_probability = vector_sum_free(partial_sum);
    fprintf(stdout, "[INFO] Prob. vector accumulated probability: %f\n", accumulated_probability);
    
    #if defined(DEBUG)
    ccudaEventRecord(end, 0);
    ccudaEventSynchronize(end);
    ccudaEventElapsedTime(&gputime, start, end);
    fprintf(stdout, "[TIME] Processing time: %f (ms)\n", gputime);
        
    ccudaEventDestroy(start);
    ccudaEventDestroy(end);
    #endif
}

void bga_compute_sample_fitness(struct bga_state *state) {
    
}

void bga_show_samples(struct bga_state *state) {
    #if defined(DEBUG)
    float gputime;
    cudaEvent_t start;
    cudaEvent_t end;
    
    ccudaEventCreate(&start);
    ccudaEventCreate(&end);

    ccudaEventRecord(start, 0);
    #endif

    fprintf(stdout, "[INFO] === Sample vectors =====================================\n");

    float *partial_sum;
    vector_sum_init(&partial_sum);
    
    for (int sample_number = 0; sample_number < state->number_of_samples; sample_number++) {
        fprintf(stdout, "[INFO] Sample vector sample (%d):", sample_number);
        
        for (int prob_vector_number = 0; prob_vector_number < state->number_of_prob_vectors; prob_vector_number++) {
            int current_prob_vector_number_of_bits = MAX_PROB_VECTOR_BITS;
            if (prob_vector_number + 1 == state->number_of_prob_vectors) {
                current_prob_vector_number_of_bits = state->last_prob_vector_bit_count;
            }

            if (prob_vector_number == 0) {
                int bits_to_show_count = SHOW_SAMPLE_BITS;
                if (current_prob_vector_number_of_bits < SHOW_SAMPLE_BITS) 
                    bits_to_show_count = MAX_PROB_VECTOR_BITS;
                
                int bytes_to_show_count = bits_to_show_count >> 5;
                int *bytes_to_show = (int*)malloc(sizeof(int) * bytes_to_show_count);
                
                ccudaMemcpy(bytes_to_show, state->gpu_samples[sample_number][prob_vector_number], 
                    sizeof(uint32_t) * bytes_to_show_count, cudaMemcpyDeviceToHost);
                
                for (int i = 0; i < bytes_to_show_count; i++) {
                    fprintf(stdout, " %s", int_to_binary(bytes_to_show[i]));
                }
                
                fprintf(stdout, "...\n");
            }
            
            /*kern_vector_sum<<< VECTOR_SUM_BLOCKS, VECTOR_SUM_THREADS >>>( 
                state->gpu_prob_vectors[prob_vector_number], partial_sum,
                current_prob_vector_number_of_bits);*/
        }

        /*double accumulated_probability = 0.0;
        accumulated_probability = vector_sum_free(partial_sum);
        fprintf(stdout, "[INFO] Prob. vector accumulated probability: %f\n", accumulated_probability);*/
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

__global__ void kern_sample_prob_vector(float *gpu_prob_vector, int prob_vector_size, 
    int prob_vector_starting_pos, float *prng_vector, int prng_vector_size, int *gpu_sample) {
        
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int samples_per_loop = gridDim.x * blockDim.x;
    
    __shared__ char current_block_sample[SAMPLE_PROB_VECTOR_SHMEM];

    const int shmem_int_size = SAMPLE_PROB_VECTOR_SHMEM >> 2;
    const int bytes_samples_per_loop = gridDim.x * shmem_int_size;
        
    int max_samples_doable = prob_vector_size - prob_vector_starting_pos;
    if (max_samples_doable > prng_vector_size) max_samples_doable = prng_vector_size;
    
    int loops_count = max_samples_doable / samples_per_loop;
    if (max_samples_doable % samples_per_loop > 0) loops_count++;

    int prob_vector_position;
    int sample_position;
    
    const int block = tid >> 3;
    const int offset = tid & ((1 << 3)-1);
    
    for (int loop = 0; loop < loops_count; loop++) {
        // Cada loop genera blockDim.x bits y los guarda en el array de __shared__ memory.
        sample_position = (samples_per_loop * loop) + (bid * blockDim.x) + tid;
        prob_vector_position = prob_vector_starting_pos + sample_position;
        
        if ((sample_position < max_samples_doable) && (prob_vector_position < prob_vector_size)) {
            if (gpu_prob_vector[prob_vector_position] >= prng_vector[sample_position]) {
                // 1
                current_block_sample[block] = current_block_sample[block] | (1 << offset);
            } else {
                // 0
                current_block_sample[block] = current_block_sample[block] & ~(1 << offset);
            }            
        }

        __syncthreads();
        
        // Una vez generados los bits, copio los bytes de shared memory a la global memory.
        // Convierto los char a int.
        if (tid < shmem_int_size) {
            gpu_sample[(bytes_samples_per_loop * loop) + (shmem_int_size * bid) + tid] = ((int*)current_block_sample)[tid];
        }
        
        __syncthreads();
    }
}

// Paso 2 del algoritmo.
void bga_model_sampling_mt(struct bga_state *state, mtgp32_status *mt_status) {   
    #if defined(DEBUG)
    fprintf(stdout, "[INFO] === Sampling the model =======================\n");

    float gputime;
    cudaEvent_t start;
    cudaEvent_t end;
    #endif
    
    for (int sample_number = 0; sample_number < state->number_of_samples; sample_number++) {
        #if defined(DEBUG)
        fprintf(stdout, "[INFO] > Sample %d\n", sample_number);
        
        ccudaEventCreate(&start);
        ccudaEventCreate(&end);
        ccudaEventRecord(start, 0);
        #endif
                
        for (int prob_vector_number = 0; prob_vector_number < state->number_of_prob_vectors; prob_vector_number++) {
            int current_prob_vector_number_of_bits = MAX_PROB_VECTOR_BITS;
            if (prob_vector_number + 1 == state->number_of_prob_vectors) {
                current_prob_vector_number_of_bits = state->last_prob_vector_bit_count;
            }
            
            int total_loops;
            total_loops = current_prob_vector_number_of_bits / RNUMBERS_PER_GEN;
            if (current_prob_vector_number_of_bits % RNUMBERS_PER_GEN > 0) total_loops++;
            
            int prob_vector_starting_pos;
            
            for (int loop = 0; loop < total_loops; loop++) {
                prob_vector_starting_pos = RNUMBERS_PER_GEN * loop;
                
                // Genero RNUMBERS_PER_GEN números aleatorios.
                mtgp32_generate_float(mt_status);
                
                // Sampleo el vector de prob. con los números aleatorios generados.               
                kern_sample_prob_vector<<< SAMPLE_PROB_VECTOR_BLOCKS, SAMPLE_PROB_VECTOR_THREADS>>>(
                    state->gpu_prob_vectors[prob_vector_number], current_prob_vector_number_of_bits, 
                    prob_vector_starting_pos, (float*)mt_status->d_data, RNUMBERS_PER_GEN, 
                    state->gpu_samples[sample_number][prob_vector_number]);
            }
        }

        #if defined(DEBUG)
        ccudaEventRecord(end, 0);
        ccudaEventSynchronize(end);
        ccudaEventElapsedTime(&gputime, start, end);
        fprintf(stdout, "[TIME] Processing time: %f (ms)\n", gputime);
        #endif
    }

    #if defined(DEBUG)
    ccudaEventDestroy(start);
    ccudaEventDestroy(end);
    #endif
    
    #if defined(DEBUG)
    // Muestro los samples recien generados.
    bga_show_samples(state);
    #endif
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
