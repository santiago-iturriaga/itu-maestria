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
#include "basic/pminmin.h"

#include "pals/pals_cpu_1pop.h"

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
    #if defined(DEBUG) 
    fprintf(stdout, "[DEBUG] Loading problem instance...\n");
    #endif
    
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
    #if defined(DEBUG) 
    fprintf(stdout, "[DEBUG] Executing algorithm...\n");
    #endif
    
    // Timming -----------------------------------------------------
    timespec ts;
    timming_start(ts);
    // Timming -----------------------------------------------------

    if (input.algorithm == PALS_1POP) {
        // =============================================================
        // PALS de 1 poblacion
        // =============================================================
            
        pals_cpu_1pop(input, &etc, &energy);
        
    } else if (input.algorithm == MINMIN) {
        struct solution *current_solution = create_empty_solution(&etc, &energy);
        compute_minmin(current_solution);
        
        if (!OUTPUT_SOLUTION) {
            fprintf(stdout, "%f %f\n", get_makespan(current_solution), get_energy(current_solution));
        } else {
            show_solution(current_solution);
        }
        
        free_solution(current_solution);
        free(current_solution);
            
    } else if (input.algorithm == MCT) {
        struct solution *current_solution = create_empty_solution(&etc, &energy);
        compute_mct(current_solution);
        
        if (!OUTPUT_SOLUTION) {
            fprintf(stdout, "%f %f\n", get_makespan(current_solution), get_energy(current_solution));
        } else {
            show_solution(current_solution);
        }
        
        free_solution(current_solution);
        free(current_solution);
        
    } else if (input.algorithm == pMINMIN) {
        struct solution *current_solution = create_empty_solution(&etc, &energy);
        compute_pminmin(&etc, current_solution, input.thread_count);
        
        if (!OUTPUT_SOLUTION) {
            fprintf(stdout, "%f %f\n", get_makespan(current_solution), get_energy(current_solution));
        } else {
            show_solution(current_solution);
        }
        
        free_solution(current_solution);
        free(current_solution);
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
