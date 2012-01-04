#include <stdlib.h>
#include <cuda.h>

#include "cpu_rand.h"

void cpu_rand_init(int seed) {
	srand (seed);
}

void cpu_rand_generate(int *gpu_destination, int size) {
	int aux[size];

	for (int i = 0; i < size; i++) {
		aux[i] = rand();
	}
	
	cudaMemcpy(gpu_destination, aux, sizeof(int) * size, cudaMemcpyHostToDevice);
}
