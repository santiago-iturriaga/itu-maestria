#include <stdlib.h>

#include "cpu_drand48.h"

void cpu_drand48_init(unsigned int seed, struct cpu_drand48_state &empty_state) {
    empty_state.buffer = (drand48_data*)malloc(sizeof(drand48_data));
    srand48_r(seed, empty_state.buffer);
}

double cpu_drand48_generate(struct cpu_drand48_state &state) {
     double aux;
     drand48_r(state.buffer, &aux);
     return aux;
}

void cpu_drand48_free(struct cpu_drand48_state &state) {
     free(state.buffer);
}
