//============================================================================
// Name        : palsGPU.cu
// Author      : Santiago
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in CUDA
//============================================================================

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#include "load_params.h"
#include "load_instance.h"
#include "etc_matrix.h"
#include "pals.h"
#include "mct.h"
#include "solution.h"

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
	struct solution *current_solution = create_empty_solution(etc_matrix);
	compute_mct(etc_matrix, current_solution);
	
	validate_solution(etc_matrix, current_solution);

	// =============================================================
	// CUDA
	// =============================================================
	struct pals_instance instance;
		
	// Inicializo la memoria en el dispositivo.
	pals_init(etc_matrix, current_solution, &instance);

	int best_swaps[instance.number_of_blocks];
	float best_swaps_delta[instance.number_of_blocks];
		
	// Ejecuto GPUPALS.
	//for () {
	pals_wrapper(etc_matrix, current_solution, &instance, best_swaps, best_swaps_delta);
	//}
	
	// No es necesario --------------------------------------
	int current_swap;
	int task_x;
	int machine_a;
	int task_y;
	int machine_b;
	
	fprintf(stdout, "[DEBUG] Mejores swaps:\n");
	for (int i = 0; i < instance.number_of_blocks; i++) {
		current_swap = best_swaps[i];
		task_x = (int)floor((float)current_swap / (float)etc_matrix->tasks_count);
		machine_a = current_solution->task_assignment[task_x];
		task_y = (int)fmod((float)current_swap, (float)etc_matrix->tasks_count);
		machine_b = current_solution->task_assignment[task_y];
	
		fprintf(stdout, "   Swap ID %d. Delta %f. Task %d in %d swaps with task %d in %d.\n", 
			current_swap, best_swaps_delta[i], task_x, machine_a, task_y, machine_b);
	}
	fprintf(stdout, "[DEBUG] Ejecuto el primer swap:\n");
	
	int index = instance.number_of_blocks - 1;
	current_swap = best_swaps[index];
		
	task_x = (int)floor((float)current_swap / (float)etc_matrix->tasks_count);
	machine_a = current_solution->task_assignment[task_x];
	task_y = (int)fmod((float)current_swap, (float)etc_matrix->tasks_count);
	machine_b = current_solution->task_assignment[task_y];

	fprintf(stdout, "   Swap ID %d. Delta %f. Task %d in %d swaps with task %d in %d.\n", 
		current_swap, best_swaps_delta[index], task_x, machine_a, task_y, machine_b);
	
	float swap_delta = 0.0;
	swap_delta -= get_etc_value(etc_matrix, machine_a, task_x); // Resto del ETC de x en a.
	swap_delta += get_etc_value(etc_matrix, machine_a, task_y); // Sumo el ETC de y en a.
	swap_delta -= get_etc_value(etc_matrix, machine_b, task_y); // Resto el ETC de y en b.
	swap_delta += get_etc_value(etc_matrix, machine_b, task_x); // Sumo el ETC de x en b.
	fprintf(stdout, "   Computed Delta %f.\n", swap_delta);
	// No es necesario --------------------------------------
	
	// Libero la memoria del dispositivo.
	pals_finalize(&instance);

	// =============================================================
	// Free memory
	// =============================================================
	free_etc_matrix(etc_matrix);
	free_solution(current_solution);

	return EXIT_SUCCESS;
}
