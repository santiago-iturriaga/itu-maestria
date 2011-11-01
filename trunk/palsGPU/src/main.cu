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

#include "load_params.h"
#include "load_instance.h"
#include "etc_matrix.h"
#include "pals_gpu.h"
#include "pals_serial.h"
#include "mct.h"
#include "solution.h"
#include "config.h"
#include "utils.h"

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
		// Serial
		// =============================================================
		int best_swap_task_a;
		int best_swap_task_b;
		float best_swap_delta;
		
		for (int i = 0; i < PALS_COUNT; i++) {
			pals_serial(etc_matrix, current_solution, best_swap_task_a, best_swap_task_b, best_swap_delta);
		}
		
		fprintf(stdout, "[DEBUG] Best swap: task %d for task %d. Gain %f.\n", best_swap_task_a, best_swap_task_b, best_swap_delta);
	} else if (input.pals_flavour == PALS_GPU) {
		// =============================================================
		// CUDA
		// =============================================================
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
		for (int i = 0; i < PALS_COUNT; i++) {
			pals_gpu_wrapper(etc_matrix, current_solution, &instance, best_swap_count, best_swaps, best_swaps_delta);
		}
		
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
	
				fprintf(stdout, "   GPU Result %d. Swap ID %ld. Delta %f (%f). Task %d in %d swaps with task %d in %d.\n", 
					best_swaps[i], current_swap, best_swaps_delta[i], swap_delta, task_x, machine_a, task_y, machine_b);
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
	// Timming -----------------------------------------------------
	timming_end("Elapsed PALS time", ts);
	// Timming -----------------------------------------------------

	// =============================================================
	// Free memory
	// =============================================================
	free_etc_matrix(etc_matrix);
	free_solution(current_solution);

	return EXIT_SUCCESS;
}
