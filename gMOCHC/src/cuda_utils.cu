#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

#include "config.h"
#include "cuda_utils.h"

int gpu_get_devices_count() {
    int deviceCount;

    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Obteniendo la cantidad de dispositivos disponibles\n");
        exit(EXIT_FAILURE);
    }

    return deviceCount;
}

void gpu_get_devices(cudaDeviceProp devices[], int &size) {
    size = gpu_get_devices_count();

    for (int device = 0; device < size; ++device) {
        if (cudaGetDeviceProperties(&(devices[device]), device) != cudaSuccess) {
            fprintf(stderr, "[ERROR] Obteniendo las propiedades de dispositivo %d.\n", device);
            exit(EXIT_FAILURE);
        }
    }
}

void gpu_show_devices() {
    /*
    struct cudaDeviceProp {
        char name[256];
        size_t totalGlobalMem;
        size_t sharedMemPerBlock;
        int regsPerBlock;
        int warpSize;
        size_t memPitch;
        int maxThreadsPerBlock;
        int maxThreadsDim[3];
        int maxGridSize[3];
        size_t totalConstMem;
        int major;
        int minor;
        int clockRate;
        size_t textureAlignment;
        int deviceOverlap;
        int multiProcessorCount;
    }
    */

    int size = gpu_get_devices_count();
    cudaDeviceProp devices[size];

    gpu_get_devices(devices, size);

    for (int i = 0; i < size; i++) {
        fprintf(stdout, "[[-- DEVICE %d --]]", i);
        fprintf(stdout, " Name: %s\n", devices[i].name);
    }
}

void gpu_set_device(int device) {
    if (cudaSetDevice(device) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Estableciendo el dispositivo %d\n", device);
        exit(EXIT_FAILURE);
    }
}

// -----------------------------------------------------------------------

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

void vector_sum_float_alloc(float **gpu_partial_sum, float **cpu_partial_sum) {      
    ccudaMalloc((void**)gpu_partial_sum, sizeof(float) * VECTOR_SUM_BLOCKS);
    *cpu_partial_sum = (float*)malloc(sizeof(float) * VECTOR_SUM_BLOCKS);
}

void vector_sum_float_init(float *gpu_partial_sum) {      
    kern_vector_set_float<<< 1, VECTOR_SUM_BLOCKS >>>(
        gpu_partial_sum, VECTOR_SUM_BLOCKS, 0.0);
}

float vector_sum_float_get(float *gpu_partial_sum, float *cpu_partial_sum) {
    float accumulated_sum = 0.0;
    
    ccudaMemcpy(cpu_partial_sum, gpu_partial_sum, sizeof(float) * VECTOR_SUM_BLOCKS, cudaMemcpyDeviceToHost);
    for (int i = 0; i < VECTOR_SUM_BLOCKS; i++) {
        accumulated_sum += cpu_partial_sum[i];
    }
    
    return accumulated_sum;
}

void vector_sum_float_free(float *gpu_partial_sum, float *cpu_partial_sum) {
    ccudaFree(gpu_partial_sum);
    free(cpu_partial_sum);
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
    unsigned int sum = 0;
    unsigned int starting = 1 << ((sizeof(int) * 8)-1);
    int shifts = 32;
    
    for (unsigned int z = starting; z > 0; z >>= 1)
    {
        shifts--;
        sum += (data & z) >> shifts;
    }
    
    return sum;
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

void vector_sum_bit_alloc(int **gpu_partial_sum, int **cpu_partial_sum) {      
    ccudaMalloc((void**)gpu_partial_sum, sizeof(int) * VECTOR_SUM_BLOCKS);
    *cpu_partial_sum = (int*)malloc(sizeof(int) * VECTOR_SUM_BLOCKS);
}

void vector_sum_bit_init(int *gpu_partial_sum) {      
    kern_vector_set_int<<< 1, VECTOR_SUM_BLOCKS >>>(
        gpu_partial_sum, VECTOR_SUM_BLOCKS, 0.0);
}

int vector_sum_bit_get(int *gpu_partial_sum, int *cpu_partial_sum) {   
    int accumulated_sum = 0;
    
    ccudaMemcpy(cpu_partial_sum, gpu_partial_sum, sizeof(int) * VECTOR_SUM_BLOCKS, cudaMemcpyDeviceToHost);
    for (int i = 0; i < VECTOR_SUM_BLOCKS; i++) {
        accumulated_sum += cpu_partial_sum[i];
    }
    
    return accumulated_sum;
}

void vector_sum_bit_free(int *gpu_partial_sum, int *cpu_partial_sum) {
    ccudaFree(gpu_partial_sum);
    free(cpu_partial_sum);
}
