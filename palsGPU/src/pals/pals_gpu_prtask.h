/*
 * pals.h
 *
 *  Created on: Jul 27, 2011
 *      Author: santiago
 */

#include "../etc_matrix.h"
#include "../solution.h"

#define PALS_GPU_PRTASK_SWAP 0
#define PALS_GPU_PRTASK_MOVE 1

#ifndef PALS_GPU_PRTASK_H_
#define PALS_GPU_PRTASK_H_

struct pals_gpu_prtask_instance {
	float *gpu_etc_matrix;
	int *gpu_task_assignment;
	float *gpu_machine_compute_time;
	
	int *gpu_best_movements;
	float *gpu_best_deltas;
	
	int blocks;
	int threads;
	int loops;
	int total_tasks;
	
	short result_count;
};

struct pals_gpu_prtask_result {
	short move_count;

	char *move_type;
	int *origin;
	int *destination;
	float *delta;
};

/*
 * Ejecuta el algoritmo.
 * Búsqueda masivamente paralela sobre un subdominio del problema. 
 * Se sortea el subdominio por tarea.
 */
void pals_gpu_prtask(struct params &input, struct matrix *etc_matrix, struct solution *current_solution);

/*
 * Reserva e inicializa la memoria del dispositivo con los datos del problema.
 */
void pals_gpu_prtask_init(struct matrix *etc_matrix, struct solution *s, 
	struct pals_gpu_prtask_instance &instance, struct pals_gpu_prtask_result &result);

/*
 * Libera la memoria del dispositivo.
 */
void pals_gpu_prtask_finalize(struct pals_gpu_prtask_instance &instance);

/*
 * Limpia la memoria pedida para un resultado.
 */
void pals_gpu_prtask_clean_result(struct pals_gpu_prtask_result &result);

/*
 * Ejecuta PALS en el dispositivo.
 */
void pals_gpu_prtask_wrapper(struct matrix *etc_matrix, struct solution *s, 
	struct pals_gpu_rtask_instance &instance, int *gpu_random_numbers, 
	struct pals_gpu_rtask_result &result);

/*
 * Mueve una tarea en la memoria del dispositivo.
 */
void pals_gpu_prtask_move(struct pals_gpu_rtask_instance &instance, int task, int to_machine);

/*
 * Actualiza el completion time de una máquina.
 */
void pals_gpu_prtask_update_machine(struct pals_gpu_rtask_instance &instance, int machine, float compute_time);

#endif /* PALS_GPU_H_ */
