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
#include "dci.h"

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
// Reference MT front-end and Box-Muller transform
///////////////////////////////////////////////////////////////////////////////
static mt_struct MT[MT_RNG_COUNT];
static uint32_t state[MT_NN];

#define SHIFT1 18

void sgenrand_mt(uint32_t seed, mt_struct *mts){
    int i;

    mts->state[0] = seed;

    for(i = 1; i < mts->nn; i++){
        mts->state[i] = (UINT32_C(1812433253) * (mts->state[i - 1] ^ (mts->state[i - 1] >> 30)) + i) & mts->wmask;
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
    }
    mts->i = mts->nn;
}

uint32_t genrand_mt(mt_struct *mts){
    uint32_t *st, uuu, lll, aa, x;
    int k,n,m,lim;

    if(mts->i >= mts->nn ){
        n = mts->nn; m = mts->mm;
        aa = mts->aaa;
        st = mts->state;
        uuu = mts->umask; lll = mts->lmask;

        lim = n - m;
        for(k = 0; k < lim; k++){
            x = (st[k]&uuu)|(st[k+1]&lll);
            st[k] = st[k + m] ^ (x >> 1) ^ (x&1U ? aa : 0U);
        }

        lim = n - 1;
        for(; k < lim; k++){
            x = (st[k] & uuu)|(st[k + 1] & lll);
            st[k] = st[k + m - n] ^ (x >> 1) ^ (x & 1U ? aa : 0U);
        }

        x = (st[n - 1] & uuu)|(st[0] & lll);
        st[n - 1] = st[m - 1] ^ (x >> 1) ^ (x&1U ? aa : 0U);
        mts->i=0;
    }

    x = mts->state[mts->i];
    mts->i += 1;
    x ^= x >> mts->shift0;
    x ^= (x << mts->shiftB) & mts->maskB;
    x ^= (x << mts->shiftC) & mts->maskC;
    x ^= x >> mts->shift1;

    return x;
}

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

void RandomRef(
    float *h_Random,
    int NPerRng,
    unsigned int seed
){
    int iRng, iOut;

    for(iRng = 0; iRng < MT_RNG_COUNT; iRng++){
        MT[iRng].state = state;
        sgenrand_mt(seed, &MT[iRng]);

        for(iOut = 0; iOut < NPerRng; iOut++)
           h_Random[iRng * NPerRng + iOut] = ((float)genrand_mt(&MT[iRng]) + 1.0f) / 4294967296.0f;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Fast GPU random number generator and Box-Muller transform
///////////////////////////////////////////////////////////////////////////////
__device__ static mt_struct_stripped ds_MT[MT_RNG_COUNT];
static mt_struct_stripped h_MT[MT_RNG_COUNT];


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


///////////////////////////////////////////////////////////////////////////////
// Data configuration
///////////////////////////////////////////////////////////////////////////////
const int    PATH_N = 80; //24000000;
const int N_PER_RNG = iAlignUp(iDivUp(PATH_N, MT_RNG_COUNT), 2);
const int    RAND_N = MT_RNG_COUNT * N_PER_RNG;

const unsigned int SEED = 777;



//#define DO_BOXMULLER
///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    float
        *d_Rand;

    float
        *h_RandCPU,
        *h_RandGPU;

    double
        rCPU, rGPU, delta, sum_delta, max_delta, sum_ref, L1norm, gpuTime;

    int i, j;
    unsigned int hTimer;

    cudaSetDevice(2);

    printf("Initializing data for %i samples...\n", PATH_N);
        h_RandCPU  = (float *)malloc(RAND_N * sizeof(float));
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
        cudaThreadSynchronize();
        RandomGPU<<<32, 128>>>(d_Rand, N_PER_RNG);
        cudaThreadSynchronize();

    printf("Generated samples : %i \n", RAND_N);
 
    printf("Reading back the results...\n");
        cudaMemcpy(h_RandGPU, d_Rand, RAND_N * sizeof(float), cudaMemcpyDeviceToHost);

    for(i = 0; i < MT_RNG_COUNT; i++) {
        for(j = 0; j < N_PER_RNG; j++){
           rGPU = h_RandGPU[i + j * MT_RNG_COUNT];
           printf("%f\n", rGPU);
        }
    }

    printf("Shutting down...\n");
        cudaFree(d_Rand);
        free(h_RandGPU);
        free(h_RandCPU);
}
