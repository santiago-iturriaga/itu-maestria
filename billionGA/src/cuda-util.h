#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#define VECTOR_SET_BLOCKS       128
#define VECTOR_SET_THREADS      256

#define VECTOR_SUM_BLOCKS       128
#define VECTOR_SUM_THREADS      512
#define VECTOR_SUM_SHARED_MEM   512

//#define VECTOR_SUM_BLOCKS       128
//#define VECTOR_SUM_THREADS      256
//#define VECTOR_SUM_SHARED_MEM   256

/*
 * Establece el valor de todos los elementos de un vector a "value".
 */
void vector_set_float(float *gpu_vector, int size, float value);
void vector_set_int(int *gpu_vector, int size, int value);

/*
 * Reduce un array sumando cada uno de sus elementos.
 * gpu_output_data debe tener un elemento por bloque del kernel.
 */
void  vector_sum_float(float *gpu_input_data, float *gpu_output_data, unsigned int size);
void  vector_sum_float_alloc(float **gpu_partial_sum, float **cpu_partial_sum);
void  vector_sum_float_init(float *gpu_partial_sum);
float vector_sum_float_get(float *gpu_partial_sum, float *cpu_partial_sum);
void  vector_sum_float_free(float *gpu_partial_sum, float *cpu_partial_sum);

/*
 * Reduce un array sumando cada uno de los bits de cada int por separado.
 * gpu_output_data debe tener un elemento por bloque del kernel.
 */
void vector_sum_bit(int *gpu_input_data, int *gpu_output_data, unsigned int bit_size);
void vector_sum_bit_alloc(int **gpu_partial_sum, int **cpu_partial_sum);
void vector_sum_bit_init(int *gpu_partial_sum);
int  vector_sum_bit_get(int *gpu_partial_sum, int *cpu_partial_sum);
void vector_sum_bit_free(int *gpu_partial_sum, int *cpu_partial_sum);

/*
 * Reduce un array sumando cada uno de los bits de cada int por separado.
 * gpu_output_data debe tener un elemento por bloque del kernel.
 */
void vector_sum_int(int *gpu_input_data, long *gpu_output_data, unsigned int size);
void vector_sum_int_alloc(long **gpu_partial_sum, long **cpu_partial_sum);
void vector_sum_int_init(long *gpu_partial_sum);
long vector_sum_int_get(long *gpu_partial_sum, long *cpu_partial_sum);
void vector_sum_int_free(long *gpu_partial_sum, long *cpu_partial_sum);
void vector_sum_int_show(long *gpu_partial_sum, long *cpu_partial_sum);

void vector_sp_int(int *gpu_input_data, long *gpu_output_data, unsigned int size, int op, int value);
void vector_sp_int_alloc(long **gpu_partial_sum, long **cpu_partial_sum);
void vector_sp_int_init(long *gpu_partial_sum);
long vector_sp_int_get(long *gpu_partial_sum, long *cpu_partial_sum);
void vector_sp_int_free(long *gpu_partial_sum, long *cpu_partial_sum);
void vector_sp_int_show(long *gpu_partial_sum, long *cpu_partial_sum);

// -----------------------------------------------------------------

inline void exception_maker(cudaError rc, const char * funcname)
{
    if (rc != cudaSuccess) {
        const char * message = cudaGetErrorString(rc);
        fprintf(stderr, "[ERROR] In %s Error(%d):%s\n", funcname, rc, message);
        exit(EXIT_FAILURE);
    }
}

inline int ccudaGetDeviceCount(int * num)
{
    cudaError rc = cudaGetDeviceCount(num);
    exception_maker(rc, "ccudaGetDeviceCount");
    return cudaSuccess;
}

inline int ccudaSetDevice(int dev)
{
    cudaError rc = cudaSetDevice(dev);
    exception_maker(rc, "ccudaSetDevice");
    return cudaSuccess;
}

inline int ccudaMalloc(void **devPtr, size_t size)
{
    cudaError rc = cudaMalloc((void **)(void*)devPtr, size);
    exception_maker(rc, "ccudaMalloc");
    return cudaSuccess;
}

inline int ccudaFree(void *devPtr)
{
    cudaError rc = cudaFree(devPtr);
    exception_maker(rc, "ccudaFree");
    return cudaSuccess;
}

inline int ccudaMemcpy(void *dest, void *src, size_t size,
              enum cudaMemcpyKind kind)
{
    cudaError rc = cudaMemcpy(dest, src, size, kind);
    exception_maker(rc, "ccudaMemcpy");
    return cudaSuccess;
}

inline int ccudaEventCreate(cudaEvent_t * event)
{
    cudaError rc = cudaEventCreate(event);
    exception_maker(rc, "ccudaEventCreate");
    return cudaSuccess;
}

inline int ccudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    cudaError rc = cudaEventRecord(event, stream);
    exception_maker(rc, "ccudaEventRecord");
    return cudaSuccess;
}

inline int ccudaEventSynchronize(cudaEvent_t event)
{
    cudaError rc = cudaEventSynchronize(event);
    exception_maker(rc, "ccudaEventSynchronize");
    return cudaSuccess;
}

inline int ccudaThreadSynchronize()
{
    cudaError rc = cudaThreadSynchronize();
    exception_maker(rc, "ccudaThreadSynchronize");
    return cudaSuccess;
}

inline int ccudaEventElapsedTime(float * ms,
                 cudaEvent_t start, cudaEvent_t end)
{
    cudaError rc = cudaEventElapsedTime(ms, start, end);
    exception_maker(rc, "ccudaEventElapsedTime");
    return cudaSuccess;
}

inline int ccudaEventDestroy(cudaEvent_t event)
{
    cudaError rc = cudaEventDestroy(event);
    exception_maker(rc, "ccudaEventDestroy");
    return cudaSuccess;
}

inline int ccudaMemcpyToSymbol(const void * symbol,
                   const void * src,
                   size_t count,
                   size_t offset = 0,
                   enum cudaMemcpyKind kind
                   = cudaMemcpyHostToDevice)
{
    cudaError rc = cudaMemcpyToSymbol((const char *)symbol,
                    src, count, offset, kind);
    exception_maker(rc, "ccudaMemcpyToSymbol");
    return cudaSuccess;
}

inline int ccudaGetDeviceProperties(struct cudaDeviceProp * prop, int device)
{
    cudaError rc = cudaGetDeviceProperties(prop, device);
    exception_maker(rc, "ccudaGetDeviceProperties");
    return cudaSuccess;
}

#endif
