/*
 * mtgp-util.cu
 *
 * Some utility functions for Sample Programs
 *
 */
#include <stdio.h>
#include <cuda.h>
#include <stdint.h>
#include <inttypes.h>

#include "../cuda-util.h"
#include "mtgp-util.h"

int get_suitable_block_num(int device,
               int *max_block_num,
               int *mp_num,
               int word_size,
               int thread_num,
               int large_size)
{
    cudaDeviceProp dev;
    CUdevice cuDevice;
    int max_thread_dev;
    int max_block, max_block_mem, max_block_dev;
    int major, minor, ver;
    //int regs, max_block_regs;

    ccudaGetDeviceProperties(&dev, device);
    cuDeviceGet(&cuDevice, device);
    cuDeviceComputeCapability(&major, &minor, cuDevice);
    //cudaFuncGetAttributes()
#if 0
    if (word_size == 4) {
    regs = 14;
    } else {
    regs = 16;
    }
    max_block_regs = dev.regsPerBlock / (regs * thread_num);
#endif
    max_block_mem = dev.sharedMemPerBlock / (large_size * word_size + 16);
    if (major == 9999 && minor == 9999) {
    return -1;
    }
    ver = major * 100 + minor;
    if (ver <= 101) {
    max_thread_dev = 768;
    } else if (ver <= 103) {
    max_thread_dev = 1024;
    } else if (ver <= 200) {
    max_thread_dev = 1536;
    } else {
    max_thread_dev = 1536;
    }
    max_block_dev = max_thread_dev / thread_num;
    if (max_block_mem < max_block_dev) {
    max_block = max_block_mem;
    } else {
    max_block = max_block_dev;
    }
#if 0
    if (max_block_regs < max_block) {
    max_block = max_block_regs;
    }
#endif
    *max_block_num = max_block;
    *mp_num = dev.multiProcessorCount;
    return max_block * dev.multiProcessorCount;
}
