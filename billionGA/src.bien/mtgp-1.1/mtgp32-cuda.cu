/*
 * @file mtgp32-cuda.cu
 *
 * @brief Sample Program for CUDA 3.2 and 4.0
 *
 * MTGP32-11213
 * This program generates 32-bit unsigned integers.
 * The period of generated integers is 2<sup>11213</sup>-1.
 *
 * This also generates single precision floating point numbers
 * uniformly distributed in the range [1, 2). (float r; 1.0 <= r < 2.0)
 */
#include <stdio.h>
#include <cuda.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <stdlib.h>

#include "../config.h"
#include "../cuda-util.h"
#include "mtgp-util.h"
#include "mtgp32-fast.h"
#include "mtgp32-cuda.h"

#define MTGPDC_MEXP 11213
#define MTGPDC_FLOOR_2P 256
#define MTGPDC_CEIL_2P 512
#define MTGPDC_PARAM_TABLE mtgp32dc_params_fast_11213
#define MEXP 11213
#define THREAD_NUM MTGPDC_FLOOR_2P
#define LARGE_SIZE (THREAD_NUM * 3)
#define BLOCK_NUM_MAX 200
#define TBL_SIZE 16
#define N MTGPDC_N

extern mtgp32_params_fast_t mtgp32dc_params_fast_11213[];

/*
 * Generator Parameters.
 */
__constant__ unsigned int pos_tbl[BLOCK_NUM_MAX];
__constant__ uint32_t param_tbl[BLOCK_NUM_MAX][TBL_SIZE];
__constant__ uint32_t temper_tbl[BLOCK_NUM_MAX][TBL_SIZE];
__constant__ uint32_t single_temper_tbl[BLOCK_NUM_MAX][TBL_SIZE];
__constant__ uint32_t sh1_tbl[BLOCK_NUM_MAX];
__constant__ uint32_t sh2_tbl[BLOCK_NUM_MAX];
__constant__ uint32_t mask[1];

/**
 * Shared memory
 * The generator's internal status vector.
 */
__shared__ uint32_t status[LARGE_SIZE];

/**
 * The function of the recursion formula calculation.
 *
 * @param[in] X1 the farthest part of state array.
 * @param[in] X2 the second farthest part of state array.
 * @param[in] Y a part of state array.
 * @param[in] bid block id.
 * @return output
 */
__device__ uint32_t para_rec(uint32_t X1, uint32_t X2, uint32_t Y, int bid) {
    uint32_t X = (X1 & mask[0]) ^ X2;
    uint32_t MAT;

    X ^= X << sh1_tbl[bid];
    Y = X ^ (Y >> sh2_tbl[bid]);
    MAT = param_tbl[bid][Y & 0x0f];
    return Y ^ MAT;
}

/**
 * The tempering function.
 *
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @param[in] bid block id.
 * @return the tempered value.
 */
__device__ uint32_t temper(uint32_t V, uint32_t T, int bid) {
    uint32_t MAT;

    T ^= T >> 16;
    T ^= T >> 8;
    MAT = temper_tbl[bid][T & 0x0f];
    return V ^ MAT;
}

/**
 * The tempering and converting function.
 * By using the preset-ted table, converting to IEEE format
 * and tempering are done simultaneously.
 *
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @param[in] bid block id.
 * @return the tempered and converted value.
 */
__device__ uint32_t temper_single(uint32_t V, uint32_t T, int bid) {
    uint32_t MAT;
    uint32_t r;

    T ^= T >> 16;
    T ^= T >> 8;
    MAT = single_temper_tbl[bid][T & 0x0f];
    r = (V >> 9) ^ MAT;
    return r;
}

/**
 * Read the internal state vector from kernel I/O data, and
 * put them into shared memory.
 *
 * @param[out] status shared memory.
 * @param[in] d_status kernel I/O data
 * @param[in] bid block id
 * @param[in] tid thread id
 */
__device__ void status_read(uint32_t status[LARGE_SIZE],
                const mtgp32_kernel_status_t *d_status,
                int bid,
                int tid) {
    status[LARGE_SIZE - N + tid] = d_status[bid].status[tid];
    if (tid < N - THREAD_NUM) {
    status[LARGE_SIZE - N + THREAD_NUM + tid]
        = d_status[bid].status[THREAD_NUM + tid];
    }
    __syncthreads();
}

/**
 * Read the internal state vector from shared memory, and
 * write them into kernel I/O data.
 *
 * @param[out] d_status kernel I/O data
 * @param[in] status shared memory.
 * @param[in] bid block id
 * @param[in] tid thread id
 */
__device__ void status_write(mtgp32_kernel_status_t *d_status,
                 const uint32_t status[LARGE_SIZE],
                 int bid,
                 int tid) {
    d_status[bid].status[tid] = status[LARGE_SIZE - N + tid];
    if (tid < N - THREAD_NUM) {
    d_status[bid].status[THREAD_NUM + tid]
        = status[4 * THREAD_NUM - N + tid];
    }
    __syncthreads();
}

/**
 * kernel function.
 * This function generates 32-bit unsigned integers in d_data
 *
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output
 * @param[in] size number of output data requested.
 */
__global__ void mtgp32_uint32_kernel(mtgp32_kernel_status_t* d_status,
                     uint32_t* d_data, int size) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    int pos = pos_tbl[bid];
    uint32_t r;
    uint32_t o;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, bid, tid);

    // main loop
    for (int i = 0; i < size; i += LARGE_SIZE) {

#if defined(DEBUG) && defined(__DEVICE_EMULATION__)
    if ((i == 0) && (bid == 0) && (tid <= 1)) {
        printf("status[LARGE_SIZE - N + tid]:%08x\n",
           status[LARGE_SIZE - N + tid]);
        printf("status[LARGE_SIZE - N + tid + 1]:%08x\n",
           status[LARGE_SIZE - N + tid + 1]);
        printf("status[LARGE_SIZE - N + tid + pos]:%08x\n",
           status[LARGE_SIZE - N + tid + pos]);
        printf("sh1:%d\n", sh1_tbl[bid]);
        printf("sh2:%d\n", sh2_tbl[bid]);
        printf("mask:%08x\n", mask[0]);
        for (int j = 0; j < 16; j++) {
        printf("tbl[%d]:%08x\n", j, param_tbl[0][j]);
        }
    }
#endif
    r = para_rec(status[LARGE_SIZE - N + tid],
         status[LARGE_SIZE - N + tid + 1],
         status[LARGE_SIZE - N + tid + pos],
         bid);
    status[tid] = r;
#if defined(DEBUG) && defined(__DEVICE_EMULATION__)
    if ((i == 0) && (bid == 0) && (tid <= 1)) {
        printf("status[tid]:%08x\n", status[tid]);
    }
#endif
    o = temper(r, status[LARGE_SIZE - N + tid + pos - 1], bid);
#if defined(DEBUG) && defined(__DEVICE_EMULATION__)
    if ((i == 0) && (bid == 0) && (tid <= 1)) {
        printf("r:%08" PRIx32 "\n", r);
    }
#endif
    d_data[size * bid + i + tid] = o;
    __syncthreads();
    r = para_rec(status[(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
             status[(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
             status[(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
             bid);
    status[tid + THREAD_NUM] = r;
    o = temper(r,
           status[(4 * THREAD_NUM - N + tid + pos - 1) % LARGE_SIZE],
           bid);
    d_data[size * bid + THREAD_NUM + i + tid] = o;
    __syncthreads();
    r = para_rec(status[2 * THREAD_NUM - N + tid],
             status[2 * THREAD_NUM - N + tid + 1],
             status[2 * THREAD_NUM - N + tid + pos],
             bid);
    status[tid + 2 * THREAD_NUM] = r;
    o = temper(r, status[tid + pos - 1 + 2 * THREAD_NUM - N], bid);
    d_data[size * bid + 2 * THREAD_NUM + i + tid] = o;
    __syncthreads();
    }
    // write back status for next call
    status_write(d_status, status, bid, tid);
}

/**
 * kernel function.
 * This function generates single precision floating point numbers in d_data.
 *
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output. IEEE single precision format.
 * @param[in] size number of output data requested.
 */
__global__ void mtgp32_single_kernel(mtgp32_kernel_status_t* d_status,
                     uint32_t* d_data, int size)
{

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    int pos = pos_tbl[bid];
    uint32_t r;
    uint32_t o;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, bid, tid);

    // main loop
    for (int i = 0; i < size; i += LARGE_SIZE) {
    r = para_rec(status[LARGE_SIZE - N + tid],
             status[LARGE_SIZE - N + tid + 1],
             status[LARGE_SIZE - N + tid + pos],
             bid);
    status[tid] = r;
    o = temper_single(r, status[LARGE_SIZE - N + tid + pos - 1], bid);
    d_data[size * bid + i + tid] = o;
    __syncthreads();
    r = para_rec(status[(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
             status[(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
             status[(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
             bid);
    status[tid + THREAD_NUM] = r;
    o = temper_single(
        r,
        status[(4 * THREAD_NUM - N + tid + pos - 1) % LARGE_SIZE],
        bid);
    d_data[size * bid + THREAD_NUM + i + tid] = o;
    __syncthreads();
    r = para_rec(status[2 * THREAD_NUM - N + tid],
             status[2 * THREAD_NUM - N + tid + 1],
             status[2 * THREAD_NUM - N + tid + pos],
             bid);
    status[tid + 2 * THREAD_NUM] = r;
    o = temper_single(r,
              status[tid + pos - 1 + 2 * THREAD_NUM - N],
              bid);
    d_data[size * bid + 2 * THREAD_NUM + i + tid] = o;
    __syncthreads();
    }
    // write back status for next call
    status_write(d_status, status, bid, tid);
}

/**
 * This function initializes kernel I/O data.
 * @param d_status output kernel I/O data.
 * @param params MTGP32 parameters. needed for the initialization.
 */
void make_kernel_data32(mtgp32_kernel_status_t * d_status,
    mtgp32_params_fast_t params[], int block_num, unsigned int seed)
{
    int i;
    mtgp32_kernel_status_t* h_status
    = (mtgp32_kernel_status_t *) malloc(
        sizeof(mtgp32_kernel_status_t) * block_num);

    if (h_status == NULL) {
        fprintf(stderr, "[ERROR] Failure in allocating host memory for kernel I/O data.\n");
        exit(EXIT_FAILURE);
    }
    for (i = 0; i < block_num; i++) {
        mtgp32_init_state(&(h_status[i].status[0]), &params[i], seed + i + 1);
    }
    /*
#if defined(DEBUG)
    printf("h_status[0].status[0]:%08"PRIx32"\n", h_status[0].status[0]);
    printf("h_status[0].status[1]:%08"PRIx32"\n", h_status[0].status[1]);
    printf("h_status[0].status[2]:%08"PRIx32"\n", h_status[0].status[2]);
    printf("h_status[0].status[3]:%08"PRIx32"\n", h_status[0].status[3]);
#endif
* */
    ccudaMemcpy(d_status, h_status,
        sizeof(mtgp32_kernel_status_t) * block_num,
        cudaMemcpyHostToDevice);
    free(h_status);
}

/**
 * This function sets constants in device memory.
 * @param[in] params input, MTGP32 parameters.
 */
void make_constant(const mtgp32_params_fast_t params[],
    int block_num) {
    const int size1 = sizeof(uint32_t) * block_num;
    const int size2 = sizeof(uint32_t) * block_num * TBL_SIZE;
    uint32_t *h_pos_tbl;
    uint32_t *h_sh1_tbl;
    uint32_t *h_sh2_tbl;
    uint32_t *h_param_tbl;
    uint32_t *h_temper_tbl;
    uint32_t *h_single_temper_tbl;
    uint32_t *h_mask;
    h_pos_tbl = (uint32_t *)malloc(size1);
    h_sh1_tbl = (uint32_t *)malloc(size1);
    h_sh2_tbl = (uint32_t *)malloc(size1);
    h_param_tbl = (uint32_t *)malloc(size2);
    h_temper_tbl = (uint32_t *)malloc(size2);
    h_single_temper_tbl = (uint32_t *)malloc(size2);
    h_mask = (uint32_t *)malloc(sizeof(uint32_t));
    if (h_pos_tbl == NULL
    || h_sh1_tbl == NULL
    || h_sh2_tbl == NULL
    || h_param_tbl == NULL
    || h_temper_tbl == NULL
    || h_single_temper_tbl == NULL
    || h_mask == NULL
    ) {
        fprintf(stderr, "[ERROR] Failure in allocating host memory for constant table.\n");
        exit(1);
    }
    h_mask[0] = params[0].mask;
    for (int i = 0; i < block_num; i++) {
        h_pos_tbl[i] = params[i].pos;
        h_sh1_tbl[i] = params[i].sh1;
        h_sh2_tbl[i] = params[i].sh2;
        for (int j = 0; j < TBL_SIZE; j++) {
            h_param_tbl[i * TBL_SIZE + j] = params[i].tbl[j];
            h_temper_tbl[i * TBL_SIZE + j] = params[i].tmp_tbl[j];
            h_single_temper_tbl[i * TBL_SIZE + j] = params[i].flt_tmp_tbl[j];
        }
    }
    ccudaMemcpyToSymbol(pos_tbl, h_pos_tbl, size1);
    ccudaMemcpyToSymbol(sh1_tbl, h_sh1_tbl, size1);
    ccudaMemcpyToSymbol(sh2_tbl, h_sh2_tbl, size1);
    ccudaMemcpyToSymbol(param_tbl, h_param_tbl, size2);
    ccudaMemcpyToSymbol(temper_tbl, h_temper_tbl, size2);
    ccudaMemcpyToSymbol(single_temper_tbl, h_single_temper_tbl, size2);
    ccudaMemcpyToSymbol(mask, h_mask, sizeof(uint32_t));
    free(h_pos_tbl);
    free(h_sh1_tbl);
    free(h_sh2_tbl);
    free(h_param_tbl);
    free(h_temper_tbl);
    free(h_single_temper_tbl);
    free(h_mask);
}

void mtgp32_print_generated_uint32(struct mtgp32_status *status) {
    uint32_t* h_data;

    h_data = (uint32_t *) malloc(sizeof(uint32_t) * status->num_data);
    if (h_data == NULL) {
        fprintf(stderr, "[ERROR] Failure in allocating host memory for output data.\n");
        exit(EXIT_FAILURE);
    }

    ccudaMemcpy(h_data, status->d_data, sizeof(uint32_t) * status->num_data, cudaMemcpyDeviceToHost);
      
    for (int i = 0; i < status->num_data; i++) {
        fprintf(stdout, "%u\n", h_data[i]);
    }
    fprintf(stdout, "[DEBUG] Generated numbers: %d\n", status->num_data);
        
    //free memories
    free(h_data);
}

void mtgp32_generate_uint32(struct mtgp32_status *status) {
    cudaError_t e;

    #if defined(DEBUG)
    //fprintf(stdout, "[DEBUG] Generating single precision floating point random numbers.\n");
    #endif
        
    /* kernel call */
    mtgp32_uint32_kernel<<< status->block_num, THREAD_NUM >>>(
        status->d_status, status->d_data, status->num_data / status->block_num);
    cudaThreadSynchronize();

    e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[ERROR] Failure in kernel call.\n%s\n", cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}

void mtgp32_print_generated_floats(struct mtgp32_status *status) {
    float* h_data;

    h_data = (float *) malloc(sizeof(float) * status->num_data);
    if (h_data == NULL) {
        fprintf(stderr, "[ERROR] Failure in allocating host memory for output data.\n");
        exit(EXIT_FAILURE);
    }

    ccudaMemcpy(h_data, status->d_data, sizeof(uint32_t) * status->num_data, cudaMemcpyDeviceToHost);
      
    for (int i = 0; i < status->num_data; i++) {
        fprintf(stdout, "%f\n", h_data[i]);
    }
    fprintf(stdout, "[DEBUG] Generated numbers: %d\n", status->num_data);
        
    //free memories
    free(h_data);
}

void mtgp32_generate_float(struct mtgp32_status *status) {                
    cudaError_t e;

    #if defined(DEBUG)
    //fprintf(stdout, "[DEBUG] Generating single precision floating point random numbers.\n");
    #endif
        
    /* kernel call */
    mtgp32_single_kernel<<< status->block_num, THREAD_NUM >>>(
        status->d_status, status->d_data, status->num_data / status->block_num);
    cudaThreadSynchronize();

    e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[ERROR] Failure in kernel call.\n%s\n", cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}

int mtgp32_get_suitable_block_num() {
    int mb, mp;
    return get_suitable_block_num(0,
       &mb, &mp, sizeof(uint32_t), THREAD_NUM, LARGE_SIZE);
}

// Se generan (768 * block_num) números aleatorios por cada vez.
void mtgp32_initialize(struct mtgp32_status *status, int numbers_per_gen, unsigned int seed) {
    #if defined(DEBUG)
    float gputime;
    cudaEvent_t start;
    cudaEvent_t end;
    
    ccudaEventCreate(&start);
    ccudaEventCreate(&end);

    ccudaEventRecord(start, 0);
    #endif
    
    #if defined(INFO) || defined(DEBUG)
    fprintf(stdout, "[INFO] === Initializing Mersenne Twister =======================\n");
    #endif
    
    status->numbers_per_gen = numbers_per_gen;
    status->num_data = numbers_per_gen;

    int mb, mp;
    status->block_num = get_suitable_block_num(0,
       &mb, &mp, sizeof(uint32_t), THREAD_NUM, LARGE_SIZE);
       
    if (status->block_num <= 0) {
        fprintf(stderr, "[ERROR] Can't calculate sutable number of blocks.\n");
        exit(EXIT_FAILURE);
    }
       
    if (status->block_num < 1 || status->block_num > BLOCK_NUM_MAX) {
        fprintf(stderr, "[ERROR] Block_num should be between 1 and %d\n", BLOCK_NUM_MAX);
        exit(EXIT_FAILURE);
    }

    status->num_unit = LARGE_SIZE * status->block_num;
    ccudaMalloc((void**)&(status->d_status), sizeof(mtgp32_kernel_status_t) * status->block_num);
    
    int r;
    r = status->num_data % status->num_unit;
    if (r != 0) {
        status->num_data = status->num_data + status->num_unit - r;
    }
    
    #if defined(INFO) || defined(DEBUG)
    fprintf(stdout, "[DEBUG] block_num: %d, num_unit: %d, num_data: %d (size %lu Mb)\n", 
        status->block_num, status->num_unit, status->num_data, (status->num_data * sizeof(uint32_t)) / (1024*1024));
    #endif
    
    make_constant(MTGPDC_PARAM_TABLE, status->block_num);
    make_kernel_data32(status->d_status, MTGPDC_PARAM_TABLE, status->block_num, seed);
    
    ccudaMalloc((void**)&(status->d_data), sizeof(uint32_t) * status->num_data);
    
    #if defined(DEBUG)
    ccudaEventRecord(end, 0);
    ccudaEventSynchronize(end);
    ccudaEventElapsedTime(&gputime, start, end);
    fprintf(stdout, "[TIME] Processing time: %f (ms)\n", gputime);
        
    ccudaEventDestroy(start);
    ccudaEventDestroy(end);
    #endif
}

void mtgp32_free(struct mtgp32_status *status) {
    ccudaFree(status->d_status);
    ccudaFree(status->d_data);
}
