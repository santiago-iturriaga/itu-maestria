#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

#include "config.h"
#include "cuda-util.h"

/*
 * Establece el valor de todos los elementos de un vector a "value".
 */
__global__ void kern_vector_set_float(float *gpu_vector, int size, float value) {
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
void vector_set_float(float *gpu_vector, int size, float value) {
    kern_vector_set_float<<< VECTOR_SET_BLOCKS, VECTOR_SET_THREADS >>>(gpu_vector, size, value);
}

/*
 * Establece el valor de todos los elementos de un vector a "value".
 */
__global__ void kern_vector_set_int(int *gpu_vector, int size, int value) {
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
void vector_set_int(int *gpu_vector, int size, int value) {
    kern_vector_set_int<<< VECTOR_SET_BLOCKS, VECTOR_SET_THREADS >>>(gpu_vector, size, value);
}

// ------------------------------------------------------------------

void vector_sum_float_init(float **partial_sum) {      
    ccudaMalloc((void**)partial_sum, sizeof(float) * VECTOR_SUM_BLOCKS);

    kern_vector_set_float<<< 1, VECTOR_SUM_BLOCKS >>>(
        *partial_sum, VECTOR_SUM_BLOCKS, 0.0);
}

float vector_sum_float_free(float *partial_sum) {
    float accumulated_sum = 0.0;
    
    float *cpu_partial_sum;
    cpu_partial_sum = (float*)malloc(sizeof(float) * VECTOR_SUM_BLOCKS);
    
    ccudaMemcpy(cpu_partial_sum, partial_sum, sizeof(float) * VECTOR_SUM_BLOCKS, cudaMemcpyDeviceToHost);
    for (int i = 0; i < VECTOR_SUM_BLOCKS; i++) {
        accumulated_sum += cpu_partial_sum[i];
    }

    ccudaFree(partial_sum);
    return accumulated_sum;
}

/*
 * Reduce un array sumando cada uno de sus elementos.
 * gpu_output_data debe tener un elemento por bloque del kernel.
 */
__global__ void kern_vector_sum_float(float *gpu_input_data, float *gpu_output_data, unsigned int size)
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
void vector_sum_float(float *gpu_input_data, float *gpu_output_data, unsigned int size) {
    kern_vector_sum_float<<< VECTOR_SUM_BLOCKS, VECTOR_SUM_THREADS >>>(gpu_input_data, gpu_output_data, size);
}

// ------------------------------------------------------------------

__device__ int sum_bits_from_int(int data) {
    /*unsigned int sum = 0;
    unsigned int starting = 1 << ((sizeof(int) * 8)-1);
    int shifts = 32;
    
    for (unsigned int z = starting; z > 0; z >>= 1)
    {
        shifts--;
        sum += (data & z) >> shifts;
    }
    
    return sum;*/
    return 1;
}

/*
 * Reduce un array sumando cada uno de los bits de cada int por separado.
 * gpu_output_data debe tener un elemento por bloque del kernel.
 */
__global__ void kern_vector_sum_bit(int *gpu_input_data, int *gpu_output_data, unsigned int bit_size)
{
    __shared__ int sdata[VECTOR_SUM_SHARED_MEM];

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int int_size = bit_size >> 5;
    
    unsigned int adds_per_loop = gridDim.x * blockDim.x;
    unsigned int loops_count = int_size / adds_per_loop;
    if (int_size % adds_per_loop > 0) loops_count++;

    unsigned int starting_position;
    
    for (unsigned int loop = 0; loop < loops_count; loop++) {
        // Perform first level of reduction, reading from global memory, writing to shared memory
        starting_position = adds_per_loop * loop;
        
        unsigned int i = starting_position + (bid * blockDim.x) + tid;

        int sum = 0;
        
        if (i < int_size) {
            int int_data = gpu_input_data[i];        
            sum = sum_bits_from_int(int_data);
            
            sdata[tid] = sum;
        } else {
            sdata[tid] = 0;
        }
        
        __syncthreads();

        // do reduction in shared mem
        for(unsigned int s = blockDim.x/2; s > 0; s >>= 1) 
        {
            if (tid < s) 
            {
                sdata[tid] = sum = sum + sdata[tid + s];
            }
            __syncthreads();
        }

        // write result for this block to global mem 
        if (tid == 0) gpu_output_data[blockIdx.x] += sdata[0];
    
        __syncthreads();
    }
}
void vector_sum_bit(int *gpu_input_data, int *gpu_output_data, unsigned int size) {
    kern_vector_sum_bit<<< VECTOR_SUM_BLOCKS, VECTOR_SUM_THREADS >>>(gpu_input_data, gpu_output_data, size);
}

void vector_sum_bit_init(int **partial_sum) {
    ccudaMalloc((void**)partial_sum, sizeof(int) * VECTOR_SUM_BLOCKS);

    kern_vector_set_int<<< 1, VECTOR_SUM_BLOCKS >>>(
        *partial_sum, VECTOR_SUM_BLOCKS, 0);
}

int  vector_sum_bit_free(int *partial_sum) {
    int accumulated_sum = 0;
    
    int *cpu_partial_sum;
    cpu_partial_sum = (int*)malloc(sizeof(int) * VECTOR_SUM_BLOCKS);
    
    ccudaMemcpy(cpu_partial_sum, partial_sum, sizeof(int) * VECTOR_SUM_BLOCKS, cudaMemcpyDeviceToHost);
    for (int i = 0; i < VECTOR_SUM_BLOCKS; i++) {
        accumulated_sum += cpu_partial_sum[i];
    }

    ccudaFree(partial_sum);
    return accumulated_sum;
}
