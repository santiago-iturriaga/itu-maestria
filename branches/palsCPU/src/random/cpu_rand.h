#ifndef CPU_RAND_H__
#define CPU_RAND_H__

struct cpu_rand_state {
    unsigned int seed;
};

void cpu_rand_init(unsigned int seed, struct cpu_rand_state &empty_state);
void cpu_rand_generate_array(struct cpu_rand_state &state, int count, double *result);
double cpu_rand_generate(struct cpu_rand_state &state);
void cpu_rand_free(struct cpu_rand_state &state);

#endif
