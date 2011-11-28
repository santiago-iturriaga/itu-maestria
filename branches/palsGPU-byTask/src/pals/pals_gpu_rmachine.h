/*
 * pals.h
 *
 *  Created on: Jul 27, 2011
 *      Author: santiago
 */

#include "../etc_matrix.h"
#include "../solution.h"

#define PALS_GPU_RMACHINE_SWAP 0
#define PALS_GPU_RMACHINE_MOVE 1

#ifndef PALS_GPU_RMACHINE_H_
#define PALS_GPU_RMACHINE_H_

struct pals_gpu_rmachine_instance {
	float *gpu_etc_matrix;
	int *gpu_task_assignment;
	
	ushort *gpu_best_swaps;
	float *gpu_best_swaps_delta;
	
	int number_of_blocks;
	int threads_per_block;
	int tasks_per_thread;
	int total_tasks;
	
	short result_count;
};

struct pals_gpu_rmachine_result {
	short move_count;

	char *move_type;
	int *origin;
	int *destination;
	float *delta;
};

/*
 * Reserva e inicializa la memoria del dispositivo con los datos del problema.
 */
void pals_gpu_rmachine_init(struct matrix *etc_matrix, struct solution *s, 
	struct pals_gpu_rmachine_instance *instance);

/*
 * Libera la memoria del dispositivo.
 */
void pals_gpu_rmachine_finalize(struct pals_gpu_rmachine_instance &instance);

/*
 * Limpia la memoria pedida para un resultado.
 */
void pals_gpu_rmachine_clean_result(struct pals_gpu_rmachine_result &result);

/*
 * Ejecuta PALS en el dispositivo.
 */
void pals_gpu_rmachine_wrapper(struct matrix *etc_matrix, struct solution *s, 
	struct pals_gpu_rmachine_instance &instance, int seed, 
	struct pals_gpu_rmachine_result &result);

#endif /* PALS_GPU_H_ */
