#ifndef CPU_MT_H__
#define CPU_MT_H__

#define NN 312
#define MM 156
#define MATRIX_A 0xB5026F5AA96619E9ULL
#define UM 0xFFFFFFFF80000000ULL /* Most significant 33 bits */
#define LM 0x7FFFFFFFULL /* Least significant 31 bits */

struct cpu_mt_state {
    unsigned int seed;
    
	/* The array for the state vector */
	unsigned long long mt[NN]; 
	/* mti==NN+1 means mt[NN] is not initialized */
	int mti; 

};

void cpu_mt_init(unsigned int seed, struct cpu_mt_state &empty_state);
double cpu_mt_generate(struct cpu_mt_state &state);
void cpu_mt_free(struct cpu_mt_state &state);

#endif
