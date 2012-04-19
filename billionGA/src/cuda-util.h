#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

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
