/*
 * pals.h
 *
 *  Created on: Jul 27, 2011
 *      Author: santiago
 */

#include "../etc_matrix.h"
#include "../solution.h"

#ifndef PALS_GPU_RTASK_H_
#define PALS_GPU_RTASK_H_

struct pals_gpu_rtask_instance {
	float *gpu_etc_matrix;
	int *gpu_task_assignment;
	
	int *gpu_best_swaps;
	float *gpu_best_swaps_delta;
	
	int number_of_blocks;
	int threads_per_block;
	int tasks_per_thread;
	int total_tasks;
};

struct pals_gpu_rtask_result {
	int *best_swaps;
	float *best_swaps_delta;
	int *rands_nums;

	// Debug...	
	/*int *taskx;
	int *tasky;
	int *loop;
	int *thread;*/
};

/*
 * Reserva e inicializa la memoria del dispositivo con los datos del problema.
 */
void pals_gpu_rtask_init(struct matrix *etc_matrix, struct solution *s, 
	struct pals_gpu_rtask_instance *instance);

/*
 * Libera la memoria del dispositivo.
 */
void pals_gpu_rtask_finalize(struct pals_gpu_rtask_instance *instance);

/*
 * Ejecuta PALS en el dispositivo.
 */
void pals_gpu_rtask_wrapper(struct matrix *etc_matrix, struct solution *s, 
	struct pals_gpu_rtask_instance &instance, int seed, 
	struct pals_gpu_rtask_result &result);

#endif /* PALS_GPU_H_ */
