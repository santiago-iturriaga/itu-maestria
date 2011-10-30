/*
 * pals.h
 *
 *  Created on: Jul 27, 2011
 *      Author: santiago
 */

#include "etc_matrix.h"
#include "solution.h"

#ifndef PALS_GPU_H_
#define PALS_GPU_H_

struct pals_gpu_instance {
	float *gpu_etc_matrix;
	int *gpu_task_assignment;
	unsigned long int *gpu_best_swaps;
	float *gpu_best_swaps_delta;
	
	int block_size;
	int tasks_per_thread;
	unsigned long total_tasks;
	int number_of_blocks;
};

/*
 * Reserva e inicializa la memoria del dispositivo con los datos del problema.
 */
void pals_gpu_init(struct matrix *etc_matrix, struct solution *s, struct pals_gpu_instance *instance);

/*
 * Libera la memoria del dispositivo.
 */
void pals_gpu_finalize(struct pals_gpu_instance *instance);

/*
 * Ejecuta PALS en el dispositivo.
 */
void pals_gpu_wrapper(struct matrix *etc_matrix, struct solution *s, struct pals_gpu_instance *instance, 
	int &best_swaps_count, unsigned long int best_swaps[], float best_swaps_delta[]);

#endif /* PALS_GPU_H_ */
