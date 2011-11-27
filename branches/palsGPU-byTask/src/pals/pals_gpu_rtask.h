/*
 * pals.h
 *
 *  Created on: Jul 27, 2011
 *      Author: santiago
 */

#include "../etc_matrix.h"
#include "../solution.h"

#define PALS_GPU_RTASK_SWAP 0
#define PALS_GPU_RTASK_MOVE 1

#ifndef PALS_GPU_RTASK_H_
#define PALS_GPU_RTASK_H_

struct pals_gpu_rtask_instance {
	float *gpu_etc_matrix;
	int *gpu_task_assignment;
	
	short *gpu_best_swaps;
	float *gpu_best_swaps_delta;
	
	int number_of_blocks;
	int threads_per_block;
	int tasks_per_thread;
	int total_tasks;
	
	short result_count;
};

struct pals_gpu_rtask_result {
	short move_count;

	char *move_type;
	int *origin;
	int *destination;
	float *delta;
};

/*
 * Reserva e inicializa la memoria del dispositivo con los datos del problema.
 */
void pals_gpu_rtask_init(struct matrix *etc_matrix, struct solution *s, 
	struct pals_gpu_rtask_instance *instance);

/*
 * Libera la memoria del dispositivo.
 */
void pals_gpu_rtask_finalize(struct pals_gpu_rtask_instance &instance);

/*
 * Limpia la memoria pedida para un resultado.
 */
void pals_gpu_rtask_clean_result(struct pals_gpu_rtask_result &result);

/*
 * Ejecuta PALS en el dispositivo.
 */
void pals_gpu_rtask_wrapper(struct matrix *etc_matrix, struct solution *s, 
	struct pals_gpu_rtask_instance &instance, int seed, 
	struct pals_gpu_rtask_result &result);

/*
 * Mueve una tarea en la memoria del dispositivo.
 */
void pals_gpu_rtask_move(struct pals_gpu_rtask_instance &instance, int task, int to_machine);

#endif /* PALS_GPU_H_ */
