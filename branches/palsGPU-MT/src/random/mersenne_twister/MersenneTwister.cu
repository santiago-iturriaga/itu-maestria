/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/*
 * This sample implements Mersenne Twister random number generator 
 * and Cartesian Box-Muller transformation on the GPU.
 * See supplied whitepaper for more explanations.
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#include "MersenneTwister.h"

///////////////////////////////////////////////////////////////////////////////
// Common host and device function 
///////////////////////////////////////////////////////////////////////////////
//ceil(a / b)
extern "C" int iDivUp(int a, int b){
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

//floor(a / b)
extern "C" int iDivDown(int a, int b){
    return a / b;
}

//Align a to nearest higher multiple of b
extern "C" int iAlignUp(int a, int b){
    return ((a % b) != 0) ?  (a - a % b + b) : a;
}

//Align a to nearest lower multiple of b
extern "C" int iAlignDown(int a, int b){
    return a - a % b;
}

///////////////////////////////////////////////////////////////////////////////
// Mersenne twister functions
///////////////////////////////////////////////////////////////////////////////
__device__ static mt_struct_stripped ds_MT[MT_RNG_COUNT];

static mt_struct_stripped h_MT[MT_RNG_COUNT];
static mt_struct MT[MT_RNG_COUNT];

void initMTRef(const char *fname){   
    FILE *fd = fopen(fname, "rb");
    if(!fd){
        printf("initMTRef(): failed to open %s\n", fname);
        printf("TEST FAILED\n");
        exit(0);
    }

    for (int i = 0; i < MT_RNG_COUNT; i++){
        //Inline structure size for compatibility,
        //since pointer types are 8-byte on 64-bit systems (unused *state variable)
        if( !fread(MT + i, 16 /* sizeof(mt_struct) */ * sizeof(int), 1, fd) ){
            printf("initMTRef(): failed to load %s\n", fname);
            printf("TEST FAILED\n");
            exit(0);
        }
    }

    fclose(fd);
}

//Load twister configurations
void loadMTGPU(const char *fname){
    FILE *fd = fopen(fname, "rb");
    if(!fd){
        printf("initMTGPU(): failed to open %s\n", fname);
        printf("TEST FAILED\n");
        exit(0);
    }
    if( !fread(h_MT, sizeof(h_MT), 1, fd) ){
        printf("initMTGPU(): failed to load %s\n", fname);
        printf("TEST FAILED\n");
        exit(0);
    }
    fclose(fd);
}

//Initialize/seed twister for current GPU context
void seedMTGPU(unsigned int seed){
    int i;
    //Need to be thread-safe
    mt_struct_stripped *MT = (mt_struct_stripped *)malloc(MT_RNG_COUNT * sizeof(mt_struct_stripped));

    for(i = 0; i < MT_RNG_COUNT; i++){
        MT[i]      = h_MT[i];
        MT[i].seed = seed;
    }
    cudaMemcpyToSymbol(ds_MT, MT, sizeof(h_MT));

    free(MT);
}

////////////////////////////////////////////////////////////////////////////////
// Write MT_RNG_COUNT vertical lanes of NPerRng random numbers to *d_Random.
// For coalesced global writes MT_RNG_COUNT should be a multiple of warp size.
// Initial states for each generator are the same, since the states are
// initialized from the global seed. In order to improve distribution properties
// on small NPerRng supply dedicated (local) seed to each twister.
// The local seeds, in their turn, can be extracted from global seed
// by means of any simple random number generator, like LCG.
////////////////////////////////////////////////////////////////////////////////
__global__ void RandomGPU(
    float *d_Random,
    int NPerRng
){
    const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int THREAD_N = blockDim.x * gridDim.x;

    int iState, iState1, iStateM, iOut;
    unsigned int mti, mti1, mtiM, x;
    unsigned int mt[MT_NN];

    for(int iRng = tid; iRng < MT_RNG_COUNT; iRng += THREAD_N){
        //Load bit-vector Mersenne Twister parameters
        mt_struct_stripped config = ds_MT[iRng];

        //Initialize current state
        mt[0] = config.seed;
        for(iState = 1; iState < MT_NN; iState++)
            mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;

        iState = 0;
        mti1 = mt[0];
        for(iOut = 0; iOut < NPerRng; iOut++){
            //iState1 = (iState +     1) % MT_NN
            //iStateM = (iState + MT_MM) % MT_NN
            iState1 = iState + 1;
            iStateM = iState + MT_MM;
            if(iState1 >= MT_NN) iState1 -= MT_NN;
            if(iStateM >= MT_NN) iStateM -= MT_NN;
            mti  = mti1;
            mti1 = mt[iState1];
            mtiM = mt[iStateM];

            x    = (mti & MT_UMASK) | (mti1 & MT_LMASK);
            x    =  mtiM ^ (x >> 1) ^ ((x & 1) ? config.matrix_a : 0);
            mt[iState] = x;
            iState = iState1;

            //Tempering transformation
            x ^= (x >> MT_SHIFT0);
            x ^= (x << MT_SHIFTB) & config.mask_b;
            x ^= (x << MT_SHIFTC) & config.mask_c;
            x ^= (x >> MT_SHIFT1);

            //Convert to (0, 1] float and write to global memory
            d_Random[iRng + iOut * MT_RNG_COUNT] = ((float)x + 1.0f) / 4294967296.0f;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Transform each of MT_RNG_COUNT lanes of NPerRng uniformly distributed 
// random samples, produced by RandomGPU(), to normally distributed lanes
// using Cartesian form of Box-Muller transformation.
// NPerRng must be even.
////////////////////////////////////////////////////////////////////////////////
#define PI 3.14159265358979f
__device__ void BoxMuller(float& u1, float& u2){
    float   r = sqrtf(-2.0f * logf(u1));
    float phi = 2 * PI * u2;
    u1 = r * __cosf(phi);
    u2 = r * __sinf(phi);
}

__global__ void BoxMullerGPU(float *d_Random, int NPerRng){
    const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int THREAD_N = blockDim.x * gridDim.x;

    for(int iRng = tid; iRng < MT_RNG_COUNT; iRng += THREAD_N)
        for(int iOut = 0; iOut < NPerRng; iOut += 2)
            BoxMuller(
                d_Random[iRng + (iOut + 0) * MT_RNG_COUNT],
                d_Random[iRng + (iOut + 1) * MT_RNG_COUNT]
            );
}

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    float
        *d_Rand;

    float
        *h_RandGPU;

    ///////////////////////////////////////////////////////////////////////////////
    // Data configuration
    ///////////////////////////////////////////////////////////////////////////////
    int PATH_N = 80; //atoi(argv[1]); //24000000;
    int N_PER_RNG = iAlignUp(iDivUp(PATH_N, MT_RNG_COUNT), 2);
    int RAND_N = MT_RNG_COUNT * N_PER_RNG;
    unsigned int SEED = 777; //atoi(argv[2]);

    fprintf(stdout, "[DEBUG] PATH_N: %d\n", PATH_N);
    fprintf(stdout, "[DEBUG] SEED  : %d\n", SEED);

    cudaSetDevice(2);

    printf("Initializing data for %i samples...\n", PATH_N);
    h_RandGPU  = (float *)malloc(RAND_N * sizeof(float));
    cudaMalloc((void **)&d_Rand, RAND_N * sizeof(float));

    printf("Loading CPU and GPU twisters configurations...\n");

	char *data_path = "/home/siturria/cuda/palsGPU-MT/src/random/mersenne_twister/data/";
    
    char raw_path[2048] = "";
    strcat(raw_path, data_path);
    strcat(raw_path, "MersenneTwister.raw");
    
    char dat_path[2048];
    strcat(dat_path, data_path);
    strcat(dat_path, "MersenneTwister.dat");
        
    initMTRef(raw_path);
    loadMTGPU(dat_path);
    seedMTGPU(SEED);

    printf("Generating random numbers on GPU...\n");

    //CUDA_SAFE_CALL( 
    cudaThreadSynchronize();

    RandomGPU<<<32, 128>>>(d_Rand, N_PER_RNG);

    //CUT_CHECK_ERROR("RandomGPU() execution failed\n");
    cudaThreadSynchronize();
 
    printf("Generated samples : %i \n", RAND_N);

    printf("Applying Box-Muller transformation on GPU...\n");

    cudaThreadSynchronize();

    BoxMullerGPU<<<32, 128>>>(d_Rand, N_PER_RNG);

    //CUT_CHECK_ERROR("BoxMullerGPU() execution failed\n");
    cudaThreadSynchronize();

    printf("Transformed samples : %i \n", RAND_N);

    printf("Reading back the results...\n");
    cudaMemcpy(h_RandGPU, d_Rand, RAND_N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < PATH_N; i++) {
        fprintf(stdout, "[%d] %f\n", i, h_RandGPU[i]);
    }

	cudaFree(d_Rand);
    free(h_RandGPU);
}
