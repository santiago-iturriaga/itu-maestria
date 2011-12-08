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
	
	int blocks;
	int threads;
	int loops;
	unsigned long total_tasks;
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
void pals_gpu_prtask_init(struct matrix *etc_matrix, struct solution *s, struct pals_gpu_prtask_instance &instance);

/*
 * Libera la memoria del dispositivo.
 */
void pals_gpu_prtask_finalize(struct pals_gpu_prtask_instance &instance);

/*
 * Ejecuta PALS en el dispositivo.
 */
void pals_gpu_prtask_wrapper(struct matrix *etc_matrix, struct solution *s, 
	struct pals_gpu_rtask_instance &instance, int *gpu_random_numbers);

/*
 * Obtiene todas las soluciones desde el dispositivo a la memoria del huesped.
 */
void pals_gpu_prtask_get_solutions(struct matrix *etc_matrix, struct pals_gpu_prtask_instance &instance, 
	int *gpu_task_assignment, float *gpu_machine_compute_time);

/*
 * Busca en el dispositivo la mejor solución de las halladas hasta el momento, 
 * descarta las peores y crea tantas copias de la mejor como sea necesario.
 */
void pals_gpu_prtask_join_solutions(struct pals_gpu_prtask_instance &instance,
	struct matrix *etc_matrix);

#endif /* PALS_GPU_H_ */
