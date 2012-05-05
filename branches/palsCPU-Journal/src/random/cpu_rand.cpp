#include <stdlib.h>

#include "cpu_rand.h"

void cpu_rand_init(unsigned int seed, struct cpu_rand_state &empty_state) {
    empty_state.seed = seed;
    srand(seed);
}

double cpu_rand_generate(struct cpu_rand_state &state) {
    return (((double)rand_r(&(state.seed))) / (double)RAND_MAX);
}

void cpu_rand_free(struct cpu_rand_state &state) {

}
