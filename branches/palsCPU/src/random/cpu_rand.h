#ifndef CPU_RAND_H__
#define CPU_RAND_H__

struct cpu_rand_state {
    struct drand48_data *buffer;
};

void cpu_rand_init(long int seed, struct cpu_rand_state &empty_state);
void cpu_rand_generate(struct cpu_rand_state &state, int count, double *result);
void cpu_rand_free(struct cpu_rand_state &state);

#endif
