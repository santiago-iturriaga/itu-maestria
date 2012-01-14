#include <stdlib.h>

#include "cpu_rand.h"

void cpu_rand_init(long int seed, struct cpu_rand_state &empty_state) {
    empty_state.buffer = (struct drand48_data*)malloc(sizeof(struct drand48_data));
	srand48_r(seed, empty_state.buffer);
}

void cpu_rand_generate(struct cpu_rand_state &state, int count, double *result) {
	for (int i = 0; i < count; i++) {
		drand48_r(state.buffer, &(result[i]));
	}
}

void cpu_rand_free(struct cpu_rand_state &state) {
    free(state.buffer);
}
