//============================================================================
// Name        : palsGPU.cu
// Author      : Santiago
// Version     : 1.0
// Copyright   : 
// Description : 
//============================================================================

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <limits.h>
#include <unistd.h>

#include "load_params.h"
#include "load_instance.h"
#include "etc_matrix.h"
#include "mct.h"
#include "solution.h"
#include "config.h"
#include "utils.h"

#include "pals/pals_serial.h"
#include "pals/pals_gpu.h"
#include "pals/pals_gpu_rtask.h"
#include "pals/pals_gpu_rmachine.h"

/// Búsqueda serial sobre el todo el dominio del problema.
void pals_serial(struct params &input, struct matrix *etc_matrix, struct solution *current_solution);

/// Búsqueda masivamente paralela sobre todo el dominio del problema.
void pals_gpu(struct params &input, struct matrix *etc_matrix, struct solution *current_solution);

/// Búsqueda masivamente paralela sobre un subdominio del problema. 
/// Se sortea el subdominio por tarea.
void pals_gpu_rtask(struct params &input, struct matrix *etc_matrix, struct solution *current_solution);

/// Búsqueda masivamente paralela sobre un subdominio del problema. 
/// Se sortea el subdominio por máquina y se evalúan todas las tareas de esa máquina.
void pals_gpu_rmachine(struct params &input, struct matrix *etc_matrix, struct solution *current_solution);

int main(int argc, char** argv)
{
	// =============================================================
	// Loading input parameters
	// =============================================================
	struct params input;
	if (load_params(argc, argv, &input) == EXIT_FAILURE) {
		fprintf(stderr, "[ERROR] Ocurrió un error leyendo los parametros de entrada.\n");
		return EXIT_FAILURE;
	}

	// =============================================================
	// Loading problem instance
	// =============================================================
	if (DEBUG) fprintf(stdout, "[DEBUG] Loading problem instance...\n");
	
	// Se pide el espacio de memoria para la matriz de ETC.
	struct matrix *etc_matrix = create_etc_matrix(&input);

	// Se carga la matriz de ETC.
	if (load_instance(&input, etc_matrix) == EXIT_FAILURE) {
		fprintf(stderr, "[ERROR] Ocurrió un error leyendo el archivo de instancia.\n");
		return EXIT_FAILURE;
	}

	//show_etc_matrix(etc_matrix);

	// =============================================================
	// Candidate solution
	// =============================================================
	if (DEBUG) fprintf(stdout, "[DEBUG] Creating initial candiate solution...\n");

	struct solution *current_solution = create_empty_solution(etc_matrix);
	compute_mct(etc_matrix, current_solution);
	
	validate_solution(etc_matrix, current_solution);

	// =============================================================
	// PALS
	// =============================================================
	if (DEBUG) fprintf(stdout, "[DEBUG] Executing PALS...\n");
	
	// Timming -----------------------------------------------------
	timespec ts;
	timming_start(ts);
	// Timming -----------------------------------------------------
	
	if (input.pals_flavour == PALS_Serial) {
		// =============================================================
		// Serial. Versión de búsqueda completa.
		// =============================================================
		
		pals_serial(input, etc_matrix, current_solution);
		
	} else if (input.pals_flavour == PALS_GPU) {
		// =============================================================
		// CUDA. Versión de búsqueda completa.
		// =============================================================		
		
		pals_gpu(input, etc_matrix, current_solution);
		
	} else if (input.pals_flavour == PALS_GPU_randTask) {
		// =============================================================
		// CUDA. Búsqueda aleatoria por tarea.
		// =============================================================
			
		pals_gpu_rtask(input, etc_matrix, current_solution);
			
	} else if (input.pals_flavour == PALS_GPU_randMachine) {
		// =============================================================
		// CUDA. Búsqueda aleatoria por máquina.
		// =============================================================
			
		pals_gpu_rmachine(input, etc_matrix, current_solution);
		
	}
	
	// Timming -----------------------------------------------------
	timming_end("FULL Elapsed PALS time", ts);
	// Timming -----------------------------------------------------

	// =============================================================
	// Release memory
	// =============================================================
	free_etc_matrix(etc_matrix);
	free_solution(current_solution);

	return EXIT_SUCCESS;
}

void pals_serial(struct params &input, struct matrix *etc_matrix, struct solution *current_solution) {
	int best_swap_task_a;
	int best_swap_task_b;
	float best_swap_delta;
	
	for (int i = 0; i < PALS_COUNT; i++) {
		pals_serial(etc_matrix, current_solution, best_swap_task_a, best_swap_task_b, best_swap_delta);
	}
	
	fprintf(stdout, "[DEBUG] Best swap: task %d for task %d. Gain %f.\n", best_swap_task_a, best_swap_task_b, best_swap_delta);
}

void pals_gpu(struct params &input, struct matrix *etc_matrix, struct solution *current_solution) {
	struct pals_gpu_instance instance;

	// Timming -----------------------------------------------------
	timespec ts_init;
	timming_start(ts_init);
	// Timming -----------------------------------------------------
			
	// Inicializo la memoria en el dispositivo.
	pals_gpu_init(etc_matrix, current_solution, &instance);

	// Timming -----------------------------------------------------
	timming_end("pals_gpu_init", ts_init);
	// Timming -----------------------------------------------------

	int best_swap_count;
	int best_swaps[instance.number_of_blocks];
	float best_swaps_delta[instance.number_of_blocks];

	// Timming -----------------------------------------------------
	timespec ts_wrapper;
	timming_start(ts_wrapper);
	// Timming -----------------------------------------------------
	
	// Ejecuto GPUPALS.
	// for (int i = 0; i < PALS_COUNT; i++) {
	pals_gpu_wrapper(etc_matrix, current_solution, &instance, best_swap_count, best_swaps, best_swaps_delta);
	// }
	
	// Timming -----------------------------------------------------
	timming_end("pals_gpu_wrapper", ts_wrapper);
	// Timming -----------------------------------------------------

	// Debug ------------------------------------------------------------------------------------------
	if (DEBUG) {
		unsigned long current_swap;
		int task_x, task_y;
		int machine_a, machine_b;

		fprintf(stdout, "[DEBUG] Mejores swaps:\n");
		for (int i = 0; i < instance.number_of_blocks; i++) {
			int block_idx = i;
			int thread_idx = best_swaps[i] / instance.tasks_per_thread;
			int task_idx = best_swaps[i] % instance.tasks_per_thread;
		
			current_swap = ((unsigned long)instance.block_size * (unsigned long)instance.tasks_per_thread * (unsigned long)block_idx) 
				+ ((unsigned long)instance.block_size * (unsigned long)task_idx) + (unsigned long)thread_idx;

			float block_offset_start = instance.block_size * instance.tasks_per_thread * block_idx;											
			float auxf = (block_offset_start  + (instance.block_size * task_idx) + thread_idx) / etc_matrix->tasks_count;
			task_x = (int)auxf;
			task_y = (int)((auxf - task_x) * etc_matrix->tasks_count);
			
			if (task_x >= etc_matrix->tasks_count) task_x = etc_matrix->tasks_count - 1;
			if (task_y >= etc_matrix->tasks_count) task_y = etc_matrix->tasks_count - 1;
			if (task_x < 0) task_x = 0;
			if (task_y < 0) task_y = 0;

			machine_a = current_solution->task_assignment[task_x];
			machine_b = current_solution->task_assignment[task_y];

			float swap_delta = 0.0;
			swap_delta -= get_etc_value(etc_matrix, machine_a, task_x); // Resto del ETC de x en a.
			swap_delta += get_etc_value(etc_matrix, machine_a, task_y); // Sumo el ETC de y en a.
			swap_delta -= get_etc_value(etc_matrix, machine_b, task_y); // Resto el ETC de y en b.
			swap_delta += get_etc_value(etc_matrix, machine_b, task_x); // Sumo el ETC de x en b.

			fprintf(stdout, "   GPU Result %d. Swap ID %ld. Task x %d, Task y %d. Delta %f (%f). Task %d in %d swaps with task %d in %d.\n", 
				best_swaps[i], current_swap, (int)auxf, (int)((auxf - task_x) * etc_matrix->tasks_count), 
				best_swaps_delta[i], swap_delta, task_x, machine_a, task_y, machine_b);
		}
	}
	// Debug ------------------------------------------------------------------------------------------

	// Timming -----------------------------------------------------
	timespec ts_finalize;
	timming_start(ts_finalize);
	// Timming -----------------------------------------------------

	// Libero la memoria del dispositivo.
	pals_gpu_finalize(&instance);
	
	// Timming -----------------------------------------------------
	timming_end("pals_gpu_finalize", ts_finalize);
	// Timming -----------------------------------------------------	
}

void pals_gpu_rtask(struct params &input, struct matrix *etc_matrix, struct solution *current_solution) {	
	// ==============================================================================
	// PALS aleatorio por tarea.
	// ==============================================================================
	
	struct pals_gpu_rtask_instance instance;

	// Timming -----------------------------------------------------
	timespec ts_init;
	timming_start(ts_init);
	// Timming -----------------------------------------------------
			
	// Inicializo la memoria en el dispositivo.
	pals_gpu_rtask_init(etc_matrix, current_solution, &instance);
	instance.result_count = PALS_RTASK_RESULT_COUNT;

	// Timming -----------------------------------------------------
	timming_end(">> pals_gpu_rtask_init", ts_init);
	// Timming -----------------------------------------------------

	// Timming -----------------------------------------------------
	timespec ts_wrapper;
	timming_start(ts_wrapper);
	// Timming -----------------------------------------------------

	// ===========> DEBUG
	/*if (DEBUG) {
		validate_solution(etc_matrix, current_solution);
	}*/
	// <=========== DEBUG
	
	// Ejecuto GPUPALS.
	int seed = input.seed;
	struct pals_gpu_rtask_result result;
	
	for (int i = 0; i < PALS_COUNT; i++) {
		pals_gpu_rtask_wrapper(etc_matrix, current_solution, instance, seed, result);

		// Debug ------------------------------------------------------------------------------------------
		if (DEBUG) {
			fprintf(stdout, "[DEBUG] Mejores movimientos:\n");
			for (int i = 0; i < result.move_count; i++) {
				if (result.move_type[i] == PALS_GPU_RTASK_SWAP) {
					int machine_a = current_solution->task_assignment[result.origin[i]];
					int machine_b = current_solution->task_assignment[result.destination[i]];
			
					fprintf(stdout, "        (swap) Task %d in %d swaps with task %d in %d. Delta %f.\n",
						result.origin[i], machine_a, result.destination[i], machine_b, result.delta[i]);
				} else if (result.move_type[i] == PALS_GPU_RTASK_MOVE) {
					int machine_a = current_solution->task_assignment[result.origin[i]];
			
					fprintf(stdout, "        (move) Task %d in %d is moved to machine %d. Delta %f.\n",
						result.origin[i], machine_a, result.destination[i], result.delta[i]);
				}
			}
		}
		// Debug ------------------------------------------------------------------------------------------

		// Aplico el mejor movimiento.
		if (result.move_type[0] == PALS_GPU_RTASK_SWAP) {
			int task_x = result.origin[0];
			int task_y = result.destination[0];
			
			int machine_a = current_solution->task_assignment[result.origin[0]];
			int machine_b = current_solution->task_assignment[result.destination[0]];
	
			//========> DEBUG
			/*if (DEBUG) {
				fprintf(stdout, "[DEBUG] PRE Machine A CT: %f\n", current_solution->machine_compute_time[machine_a]);
				fprintf(stdout, "[DEBUG] PRE Machine B CT: %f\n", current_solution->machine_compute_time[machine_b]);
			}*/
			//<======== DEBUG
	
			// Actualizo la asignación de cada tarea en el dispositivo.
			pals_gpu_rtask_move(instance, task_x, machine_b);
			pals_gpu_rtask_move(instance, task_y, machine_a);
			
			// Actualizo la asignación de cada tarea en el host.
			current_solution->task_assignment[task_x] = machine_b;
			current_solution->task_assignment[task_y] = machine_a;
			
			// Actualizo los compute time de cada máquina luego del move en el host.
			current_solution->machine_compute_time[machine_a] = 
				current_solution->machine_compute_time[machine_a] +
				get_etc_value(etc_matrix, machine_a, task_y) - 
				get_etc_value(etc_matrix, machine_a, task_x);

			current_solution->machine_compute_time[machine_b] = 
				current_solution->machine_compute_time[machine_b] +
				get_etc_value(etc_matrix, machine_b, task_x) - 
				get_etc_value(etc_matrix, machine_b, task_y);
				
			//========> DEBUG
			/*if (DEBUG) {
				fprintf(stdout, "[DEBUG] AFTER Machine A CT: %f\n", current_solution->machine_compute_time[machine_a]);
				fprintf(stdout, "[DEBUG] AFETR Machine B CT: %f\n", current_solution->machine_compute_time[machine_b]);
			}*/
			//<======== DEBUG
		} else if (result.move_type[0] == PALS_GPU_RTASK_MOVE) {
			int task_x = result.origin[0];
			
			int machine_a = current_solution->task_assignment[task_x];
			int machine_b = result.destination[0];
	
			// Actualizo la asignación de la tarea en el device.
			pals_gpu_rtask_move(instance, task_x, machine_b);
			
			// Actualizo la asignación de la tarea en el host.
			current_solution->task_assignment[task_x] = machine_b;
			
			// Actualizo los compute time de cada máquina luego del move en el host.
			current_solution->machine_compute_time[machine_a] = 
				current_solution->machine_compute_time[machine_a] - 
				get_etc_value(etc_matrix, machine_a, task_x);

			current_solution->machine_compute_time[machine_b] = 
				current_solution->machine_compute_time[machine_b] +
				get_etc_value(etc_matrix, machine_b, task_x);
		}

		// Limpio el objeto resultado.
		pals_gpu_rtask_clean_result(result);

		// Nuevo seed.		
		seed++;
	}

	// Reconstruye el compute time de cada máquina.
	// NOTA: tengo que hacer esto cada tanto por errores acumulados en el redondeo.
	for (int i = 0; i < etc_matrix->machines_count; i++) {
		current_solution->machine_compute_time[i] = 0.0;
	}
	
	for (int i = 0; i < etc_matrix->tasks_count; i++) {
		int assigned_machine = current_solution->task_assignment[i];
	
		current_solution->machine_compute_time[assigned_machine] =
			current_solution->machine_compute_time[assigned_machine] + 
			get_etc_value(etc_matrix, assigned_machine, i);
	}	
	
	// Actualiza el makespan de la solución.
	current_solution->makespan = current_solution->machine_compute_time[0];
	for (int i = 1; i < etc_matrix->machines_count; i++) {
		if (current_solution->makespan < current_solution->machine_compute_time[i]) {
			current_solution->makespan = current_solution->machine_compute_time[i];
		}
	}
	
	// ===========> DEBUG
	/*if (DEBUG) {
		validate_solution(etc_matrix, current_solution);
	}*/
	// <=========== DEBUG
	
	if (DEBUG) {
		fprintf(stdout, "[DEBUG] Nuevo makespan: %f\n", current_solution->makespan);
	}
	
	// Timming -----------------------------------------------------
	timming_end(">> pals_gpu_rtask_wrapper", ts_wrapper);
	// Timming -----------------------------------------------------

	// Timming -----------------------------------------------------
	timespec ts_finalize;
	timming_start(ts_finalize);
	// Timming -----------------------------------------------------

	// Libero la memoria del dispositivo.
	pals_gpu_rtask_finalize(instance);
	
	// Timming -----------------------------------------------------
	timming_end(">> pals_gpu_rmachine_finalize", ts_finalize);
	// Timming -----------------------------------------------------		
}

void pals_gpu_rmachine(struct params &input, struct matrix *etc_matrix, struct solution *current_solution) {
	// ==============================================================================
	// PALS aleatorio por tarea.
	// ==============================================================================
	
	struct pals_gpu_rmachine_instance instance;

	// Timming -----------------------------------------------------
	timespec ts_init;
	timming_start(ts_init);
	// Timming -----------------------------------------------------
			
	// Inicializo la memoria en el dispositivo.
	pals_gpu_rmachine_init(etc_matrix, current_solution, &instance);
	//instance.result_count = PALS_RTASK_RESULT_COUNT;

	// Timming -----------------------------------------------------
	timming_end(">> pals_gpu_rmachine_init", ts_init);
	// Timming -----------------------------------------------------

	// Timming -----------------------------------------------------
	timespec ts_wrapper;
	timming_start(ts_wrapper);
	// Timming -----------------------------------------------------

	// ===========> DEBUG
	/*if (DEBUG) {
		validate_solution(etc_matrix, current_solution);
	}*/
	// <=========== DEBUG
	
	// Ejecuto GPUPALS.
	int seed = input.seed;
	struct pals_gpu_rmachine_result result;
	
	for (int i = 0; i < PALS_COUNT; i++) {
		pals_gpu_rmachine_wrapper(etc_matrix, current_solution, instance, seed, result);

		// Debug ------------------------------------------------------------------------------------------
		if (DEBUG) {
			fprintf(stdout, "[DEBUG] Mejores movimientos:\n");
			for (int i = 0; i < result.move_count; i++) {
				if (result.move_type[i] == PALS_GPU_RMACHINE_SWAP) {
					int machine_a = current_solution->task_assignment[result.origin[i]];
					int machine_b = current_solution->task_assignment[result.destination[i]];
			
					fprintf(stdout, "        (swap) Task %d in %d swaps with task %d in %d. Delta %f.\n",
						result.origin[i], machine_a, result.destination[i], machine_b, result.delta[i]);
				} else if (result.move_type[i] == PALS_GPU_RMACHINE_MOVE) {
					int machine_a = current_solution->task_assignment[result.origin[i]];
			
					fprintf(stdout, "        (move) Task %d in %d is moved to machine %d. Delta %f.\n",
						result.origin[i], machine_a, result.destination[i], result.delta[i]);
				}
			}
		}
		// Debug ------------------------------------------------------------------------------------------

		// Aplico el mejor movimiento.
		if (result.move_type[0] == PALS_GPU_RMACHINE_SWAP) {
			int task_x = result.origin[0];
			int task_y = result.destination[0];
			
			int machine_a = current_solution->task_assignment[result.origin[0]];
			int machine_b = current_solution->task_assignment[result.destination[0]];
	
			//========> DEBUG
			/*if (DEBUG) {
				fprintf(stdout, "[DEBUG] PRE Machine A CT: %f\n", current_solution->machine_compute_time[machine_a]);
				fprintf(stdout, "[DEBUG] PRE Machine B CT: %f\n", current_solution->machine_compute_time[machine_b]);
			}*/
			//<======== DEBUG
	
			// Actualizo la asignación de cada tarea en el dispositivo.
			// pals_gpu_rmachine_move(instance, task_x, machine_b);
			// pals_gpu_rmachine_move(instance, task_y, machine_a);
			
			// Actualizo la asignación de cada tarea en el host.
			current_solution->task_assignment[task_x] = machine_b;
			current_solution->task_assignment[task_y] = machine_a;
			
			// Actualizo los compute time de cada máquina luego del move en el host.
			current_solution->machine_compute_time[machine_a] = 
				current_solution->machine_compute_time[machine_a] +
				get_etc_value(etc_matrix, machine_a, task_y) - 
				get_etc_value(etc_matrix, machine_a, task_x);

			current_solution->machine_compute_time[machine_b] = 
				current_solution->machine_compute_time[machine_b] +
				get_etc_value(etc_matrix, machine_b, task_x) - 
				get_etc_value(etc_matrix, machine_b, task_y);
				
			//========> DEBUG
			/*if (DEBUG) {
				fprintf(stdout, "[DEBUG] AFTER Machine A CT: %f\n", current_solution->machine_compute_time[machine_a]);
				fprintf(stdout, "[DEBUG] AFETR Machine B CT: %f\n", current_solution->machine_compute_time[machine_b]);
			}*/
			//<======== DEBUG
		} else if (result.move_type[0] == PALS_GPU_RMACHINE_MOVE) {
			int task_x = result.origin[0];
			
			int machine_a = current_solution->task_assignment[task_x];
			int machine_b = result.destination[0];
	
			// Actualizo la asignación de la tarea en el device.
			// pals_gpu_rmachine_move(instance, task_x, machine_b);
			
			// Actualizo la asignación de la tarea en el host.
			current_solution->task_assignment[task_x] = machine_b;
			
			// Actualizo los compute time de cada máquina luego del move en el host.
			current_solution->machine_compute_time[machine_a] = 
				current_solution->machine_compute_time[machine_a] - 
				get_etc_value(etc_matrix, machine_a, task_x);

			current_solution->machine_compute_time[machine_b] = 
				current_solution->machine_compute_time[machine_b] +
				get_etc_value(etc_matrix, machine_b, task_x);
		}

		// Limpio el objeto resultado.
		pals_gpu_rmachine_clean_result(result);

		// Nuevo seed.		
		seed++;
	}

	// Reconstruye el compute time de cada máquina.
	// NOTA: tengo que hacer esto cada tanto por errores acumulados en el redondeo.
	for (int i = 0; i < etc_matrix->machines_count; i++) {
		current_solution->machine_compute_time[i] = 0.0;
	}
	
	for (int i = 0; i < etc_matrix->tasks_count; i++) {
		int assigned_machine = current_solution->task_assignment[i];
	
		current_solution->machine_compute_time[assigned_machine] =
			current_solution->machine_compute_time[assigned_machine] + 
			get_etc_value(etc_matrix, assigned_machine, i);
	}	
	
	// Actualiza el makespan de la solución.
	current_solution->makespan = current_solution->machine_compute_time[0];
	for (int i = 1; i < etc_matrix->machines_count; i++) {
		if (current_solution->makespan < current_solution->machine_compute_time[i]) {
			current_solution->makespan = current_solution->machine_compute_time[i];
		}
	}
	
	// ===========> DEBUG
	/*if (DEBUG) {
		validate_solution(etc_matrix, current_solution);
	}*/
	// <=========== DEBUG
	
	if (DEBUG) {
		fprintf(stdout, "[DEBUG] Nuevo makespan: %f\n", current_solution->makespan);
	}
	
	// Timming -----------------------------------------------------
	timming_end(">> pals_gpu_rmachine_wrapper", ts_wrapper);
	// Timming -----------------------------------------------------

	// Timming -----------------------------------------------------
	timespec ts_finalize;
	timming_start(ts_finalize);
	// Timming -----------------------------------------------------

	// Libero la memoria del dispositivo.
	pals_gpu_rmachine_finalize(instance);
	
	// Timming -----------------------------------------------------
	timming_end(">> pals_gpu_rmachine_finalize", ts_finalize);
	// Timming -----------------------------------------------------		
}

