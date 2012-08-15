#ifndef CPU_RAND_H__
#define CPU_RAND_H__

struct cpu_rand_state {
    unsigned int seed;
};

inline void cpu_rand_init(unsigned int seed, struct cpu_rand_state &empty_state) {
    empty_state.seed = seed;
    srand(seed);
}

inline double cpu_rand_generate(struct cpu_rand_state &state) {
    return (((double)rand_r(&(state.seed))) / (((double)RAND_MAX)+1));
}

inline void cpu_rand_free(struct cpu_rand_state &state) { }

#endif
