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
#include <limits.h>
#include <unistd.h>

#include "load_params.h"
#include "load_instance.h"
#include "etc_matrix.h"
#include "solution.h"
#include "config.h"

#include "utils.h"

#include "basic/mct.h"
#include "basic/minmin.h"

#include "pals/pals_serial.h"
#include "pals/pals_cpu_rtask.h"

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
	// Create empty solution
	// =============================================================
	if (DEBUG) fprintf(stdout, "[DEBUG] Creating empty solution...\n");
	struct solution *current_solution = create_empty_solution(etc_matrix);

	// =============================================================
	// Solving the problem.
	// =============================================================
	if (DEBUG) fprintf(stdout, "[DEBUG] Executing algorithm...\n");
	
	// Timming -----------------------------------------------------
	timespec ts;
	timming_start(ts);
	// Timming -----------------------------------------------------

	if (input.algorithm == PALS_Serial) {	
		// =============================================================
		// Candidate solution
		// =============================================================
		if (DEBUG) fprintf(stdout, "[DEBUG] Creating initial candiate solution...\n");

		// Timming -----------------------------------------------------
		timespec ts_mct;
		timming_start(ts_mct);
		// Timming -----------------------------------------------------

		compute_mct(etc_matrix, current_solution);
	
		// Timming -----------------------------------------------------
		timming_end(">> MCT Time", ts_mct);
		// Timming -----------------------------------------------------
	
		if (DEBUG) validate_solution(current_solution);
		
		// =============================================================
		// Serial. Versión de búsqueda completa.
		// =============================================================
		pals_serial(input, etc_matrix, current_solution);
		
	} else if (input.algorithm == PALS_GPU) {

        fprintf(stderr, "ERROR!! no es posible ejecutar!!!\n");
		
	} else if (input.algorithm == PALS_GPU_randTask) {

        fprintf(stderr, "ERROR!! no es posible ejecutar!!!\n");
			
	} else if (input.algorithm == PALS_CPU_randTask) {
		// =============================================================
		// Candidate solution
		// =============================================================
		if (DEBUG) fprintf(stdout, "[DEBUG] Creating initial candiate solution...\n");

		// Timming -----------------------------------------------------
		timespec ts_mct;
		timming_start(ts_mct);
		// Timming -----------------------------------------------------

		compute_mct(etc_matrix, current_solution);
	
		// Timming -----------------------------------------------------
		timming_end(">> MCT Time", ts_mct);
		// Timming -----------------------------------------------------
	
		if (DEBUG) validate_solution(current_solution);
	
		// =============================================================
		// CUDA. Búsqueda aleatoria por tarea.
		// =============================================================
			
		pals_cpu_rtask(input, etc_matrix, current_solution);		
		
	} else if (input.algorithm == MinMin) {
		
		compute_minmin(etc_matrix, current_solution);
		if (!OUTPUT_SOLUTION) fprintf(stdout, "%f\n", get_makespan(current_solution));
		
	} else if (input.algorithm == MCT) {
		
		compute_mct(etc_matrix, current_solution);
		
	} else if (input.algorithm == PALS_GPU_randParallelTask) {

        fprintf(stderr, "ERROR!! no es posible ejecutar!!!\n");

	}

	if (OUTPUT_SOLUTION) {
		//fprintf(stdout, "%d %d\n", etc_matrix->tasks_count, etc_matrix->machines_count);
		for (int task_id = 0; task_id < etc_matrix->tasks_count; task_id++) {
			fprintf(stdout, "%d\n", get_task_assigned_machine_id(current_solution,task_id));
		}
	}
	
	// Timming -----------------------------------------------------
	timming_end("Elapsed algorithm total time", ts);
	// Timming -----------------------------------------------------

	// =============================================================
	// Release memory
	// =============================================================
	free_etc_matrix(etc_matrix);
	
	free_solution(current_solution);
	free(current_solution);

	return EXIT_SUCCESS;
}
