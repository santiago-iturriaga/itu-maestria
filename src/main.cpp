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
#include "energy_matrix.h"
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
	
	// Se pide el espacio de memoria para la matriz de ETC y de energía.
	struct etc_matrix etc;
	struct energy_matrix energy;

	// Se carga la matriz de ETC.
	if (load_instance(&input, &etc, &energy) == EXIT_FAILURE) {
		fprintf(stderr, "[ERROR] Ocurrió un error leyendo el archivo de instancia.\n");
		return EXIT_FAILURE;
	}
	
	// =============================================================
	// Solving the problem.
	// =============================================================
	if (DEBUG) fprintf(stdout, "[DEBUG] Executing algorithm...\n");
	
	// Timming -----------------------------------------------------
	timespec ts;
	timming_start(ts);
	// Timming -----------------------------------------------------

	if (input.algorithm == PALS_Serial) {	
	
		fprintf(stderr, "ERROR!! no es posible ejecutar!!!\n");
		
	} else if (input.algorithm == PALS_GPU) {

        fprintf(stderr, "ERROR!! no es posible ejecutar!!!\n");
		
	} else if (input.algorithm == PALS_GPU_randTask) {

        fprintf(stderr, "ERROR!! no es posible ejecutar!!!\n");
			
	} else if (input.algorithm == PALS_CPU_randTask) {
		// =============================================================
		// Búsqueda aleatoria por tarea.
		// =============================================================
			
		pals_cpu_rtask(input, &etc, &energy);
		
	} else if (input.algorithm == MinMin) {
	
		struct solution *current_solution = create_empty_solution(&etc, &energy);
		compute_minmin(current_solution);
		
		if (!OUTPUT_SOLUTION) fprintf(stdout, "%f|%f\n", get_makespan(current_solution), get_energy(current_solution));
		
		free_solution(current_solution);
		free(current_solution);
		
	} else if (input.algorithm == MCT) {
		
		struct solution *current_solution = create_empty_solution(&etc, &energy);
		compute_mct(current_solution);
		
		if (!OUTPUT_SOLUTION) fprintf(stdout, "%f|%f\n", get_makespan(current_solution), get_energy(current_solution));
		
		free_solution(current_solution);
		free(current_solution);
		
	} else if (input.algorithm == PALS_GPU_randParallelTask) {

        fprintf(stderr, "ERROR!! no es posible ejecutar!!!\n");

	}
	
	// Timming -----------------------------------------------------
	timming_end("Elapsed algorithm total time", ts);
	// Timming -----------------------------------------------------

	// =============================================================
	// Release memory
	// =============================================================
	free_etc_matrix(&etc);
	free_energy_matrix(&energy);

	return EXIT_SUCCESS;
}
