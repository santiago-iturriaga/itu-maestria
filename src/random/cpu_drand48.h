#ifndef CPU_DRAND48_H__
#define CPU_DRAND48_H__

struct cpu_drand48_state {
    struct drand48_data *buffer;
};

void cpu_drand48_init(unsigned int seed, struct cpu_drand48_state &empty_state);
double cpu_drand48_generate(struct cpu_drand48_state &state);
void cpu_drand48_free(struct cpu_drand48_state &state);

#endif
