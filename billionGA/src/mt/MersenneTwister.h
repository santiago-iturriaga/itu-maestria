#ifndef MERSENNETWISTER_H
#define MERSENNETWISTER_H

#define      DCMT_SEED 4172
#define  MT_RNG_PERIOD 607

typedef struct {
    unsigned int matrix_a;
    unsigned int mask_b;
    unsigned int mask_c;
    unsigned int seed;
} mt_struct_stripped;

#define   MT_RNG_COUNT 4096
#define          MT_MM 9
#define          MT_NN 19
#define       MT_WMASK 0xFFFFFFFFU
#define       MT_UMASK 0xFFFFFFFEU
#define       MT_LMASK 0x1U
#define      MT_SHIFT0 12
#define      MT_SHIFTB 7
#define      MT_SHIFTC 15
#define      MT_SHIFT1 18

typedef struct {
    int count;
    float *gpu_Rand;
    int N_PER_RNG;
    int RAND_N;
} mersenne_twister_init_data;

void mersenne_twister_init(char *data_path, int count, mersenne_twister_init_data &empty_init_data);
void mersenne_twister_generate(mersenne_twister_init_data &init_data, int seed);
void mersenne_twister_read_results(mersenne_twister_init_data &init_data, float *results);
void mersenne_twister_free(mersenne_twister_init_data &init_data);

#endif
