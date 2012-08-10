#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <limits.h>
#include <unistd.h>

#include "config.h"

#include "load_params.h"
#include "load_instance.h"

#include "scenario.h"
#include "etc_matrix.h"
#include "energy_matrix.h"
#include "solution.h"

#include "utils.h"
#include "cuda_utils.h"

#include "basic/mct.h"
#include "basic/minmin.h"
#include "basic/pminmin.h"

#include "random/RNG_rand48.h"

int main(int argc, char** argv)
{
    // =============================================================
    // Loading input parameters
    // =============================================================
    struct params input;
    if (load_params(argc, argv, &input) == EXIT_FAILURE) {
        fprintf(stderr, "[ERROR] ocurrió un error leyendo los parametros de entrada.\n");
        return EXIT_FAILURE;
    }

    // =============================================================
    // Loading problem instance
    // =============================================================
    #if defined(DEBUG_0)
        fprintf(stderr, "[DEBUG] cargando la instancia del problema...\n");
    #endif

    // Timming -----------------------------------------------------
    timespec ts_loading;
    TIMMING_START(ts_loading)
    // Timming -----------------------------------------------------

    // Se pide el espacio de memoria para la instancia del problema.
    struct scenario *current_scenario = NULL;
    init_scenario(&input, current_scenario);
    
    #if defined(DEBUG_2)
        show_scenario(current_scenario);
    #endif
    
    struct etc_matrix *etc = NULL;
    init_etc_matrix(&input, etc);

    struct energy_matrix *energy = NULL;
    init_energy_matrix(&input, energy);  

    // Se carga la matriz de ETC.
    if (load_instance(&input, current_scenario, etc, energy) == EXIT_FAILURE) {
        fprintf(stderr, "[ERROR] ocurrió un error leyendo los archivos de instancia.\n");
        return EXIT_FAILURE;
    }

    // Timming -----------------------------------------------------
    TIMMING_END("cargando instancia", ts_loading);
    // Timming -----------------------------------------------------

    // =============================================================
    // Create empty solution
    // =============================================================
    #if defined(DEBUG_0)
        fprintf(stderr, "[DEBUG] creating empty solution...\n");
    #endif
    
    struct solution *current_solution = NULL;
    create_empty_solution(current_solution, current_scenario, etc, energy);

    // =============================================================
    // Solving the problem.
    // =============================================================
    #if defined(DEBUG_0)
        fprintf(stderr, "[DEBUG] executing algorithm...\n");
    #endif
    
    // Timming -----------------------------------------------------
    TIMMING_START(ts)
    // Timming -----------------------------------------------------

    if (input.algorithm == ALGORITHM_MINMIN) {

        compute_minmin(current_solution);

    } else if (input.algorithm == ALGORITHM_PMINMIND) {

        compute_pminmin(current_solution, input.thread_count);

    } else if (input.algorithm == ALGORITHM_MCT) {

        compute_mct(current_solution);

    }

    #if defined(OUTPUT_SOLUTION)
        for (int task_id = 0; task_id < etc->tasks_count; task_id++) {
            fprintf(stdout, "%d\n", current_solution->task_assignment[task_id]);
        }
    #endif

    // Timming -----------------------------------------------------
    TIMMING_END("Elapsed algorithm total time", ts);
    // Timming -----------------------------------------------------

    // =============================================================
    // Release memory
    // =============================================================
    free_solution(current_solution);
    free_etc_matrix(etc);
    free_energy_matrix(energy);
    free_scenario(current_scenario);

    return EXIT_SUCCESS;
}
