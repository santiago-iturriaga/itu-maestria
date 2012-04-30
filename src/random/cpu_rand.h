#ifndef CPU_RAND_H__
#define CPU_RAND_H__

void cpu_rand_init(int seed);
void cpu_rand_generate(int *gpu_destination, int size);

#endif
