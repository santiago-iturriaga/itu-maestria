/*
 * pals.h
 *
 *  Created on: Jul 27, 2011
 *      Author: santiago
 */

#include "etc_matrix.h"
#include "solution.h"

#ifndef PALS_H_
#define PALS_H_

struct pals_instance {
	float *gpu_etc_matrix;
	int *gpu_task_assignment;
	float *gpu_machine_compute_time;
	int *gpu_best_swaps;
	float *gpu_best_swaps_delta;
	
	int block_size;
	int tasks_per_thread;
	int total_tasks;
	int number_of_blocks;
};

/*
 * Reserva e inicializa la memoria del dispositivo con los datos del problema.
 */
void pals_init(struct matrix *etc_matrix, struct solution *s, struct pals_instance *instance);

/*
 * Libera la memoria del dispositivo.
 */
void pals_finalize(struct pals_instance *instance);

/*
 * Ejecuta PALS en el dispositivo.
 */
void pals_wrapper(struct matrix *etc_matrix, struct solution *s, struct pals_instance *instance);

#endif /* PALS_H_ */
