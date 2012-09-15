#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>

#if !defined(CPU)
    #include <cuda.h>
    #include "cuda_utils.h"
    #include "random/RNG_rand48.h"
#endif

#include "config.h"

#include "load_params.h"
#include "load_instance.h"

#include "scenario.h"
#include "etc_matrix.h"
#include "energy_matrix.h"
#include "solution.h"

#include "utils.h"

#include "basic/mct.h"
#include "basic/minmin.h"
#include "basic/pminmin.h"

#include "cmochc/cmochc_cell.h"
#include "cmochc/cmochc_island.h"

struct params INPUT;
struct scenario SCENARIO;
struct etc_matrix ETC;
struct energy_matrix ENERGY;

int main(int argc, char** argv)
{
    fprintf(stderr, "[INFO] == Global configuration constants ======================\n");
    fprintf(stderr, "       Debug level                  : %d\n", DEBUG_LEVEL);
    fprintf(stderr, "       Floating point precision     : %s\n", DISPLAY_PRECISION);
    fprintf(stderr, "       Log execution time           : ");
    #ifdef TIMMING
        fprintf(stderr, "YES\n");
    #else
        fprintf(stderr, "NO\n");
    #endif
    fprintf(stderr, "       Random number generator      : ");
    #ifdef CPU_RAND
        fprintf(stderr, "stdlib::rand_r\n");
    #endif
    #ifdef CPU_DRAND48
        fprintf(stderr, "stdlib::drand48_r\n");
    #endif
    #ifdef CPU_MT
        fprintf(stderr, "mersenne twister\n");
    #endif
    fprintf(stderr, "       Output solutions to stdout   : ");
    #ifdef OUTPUT_SOLUTION
        fprintf(stderr, "YES\n");
    #else
        fprintf(stderr, "NO\n");
    #endif
    fprintf(stderr, "       Max. number of threads       : %d\n", MAX_THREADS);
    fprintf(stderr, "[INFO] ========================================================\n");
    
    // =============================================================
    // Loading input parameters
    // =============================================================
    if (load_params(argc, argv) == EXIT_FAILURE) {
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
    TIMMING_START(ts_loading)
    // Timming -----------------------------------------------------

    // Se pide el espacio de memoria para la instancia del problema.
    init_scenario();
    init_etc_matrix();
    init_energy_matrix();  

    // Se carga la matriz de ETC.
    if (load_instance() == EXIT_FAILURE) {
        fprintf(stderr, "[ERROR] ocurrió un error leyendo los archivos de instancia.\n");
        return EXIT_FAILURE;
    }

    #if defined(DEBUG_2)
        show_scenario();
    #endif

    // Timming -----------------------------------------------------
    TIMMING_END("cargando instancia", ts_loading);
    // Timming -----------------------------------------------------

    // =============================================================
    // Solving the problem.
    // =============================================================
    #if defined(DEBUG_0)
        fprintf(stderr, "[DEBUG] executing algorithm...\n");
    #endif
    
    // Timming -----------------------------------------------------
    TIMMING_START(ts)
    // Timming -----------------------------------------------------

    if ((INPUT.algorithm == ALGORITHM_MINMIN) || 
        (INPUT.algorithm == ALGORITHM_PMINMIND) ||
        (INPUT.algorithm == ALGORITHM_MCT)) {

        // =============================================================
        // Trajectory algorithms.
        // =============================================================
        
        // Create empty solution       
        #if defined(DEBUG_0)
            fprintf(stderr, "[DEBUG] creating empty solution...\n");
        #endif
        
        struct solution current_solution;
        create_empty_solution(&current_solution);
            
        if (INPUT.algorithm == ALGORITHM_MINMIN) {

            compute_minmin(&current_solution);

        } else if (INPUT.algorithm == ALGORITHM_PMINMIND) {

            compute_pminmin(&current_solution);

        } else if (INPUT.algorithm == ALGORITHM_MCT) {

            compute_mct(&current_solution);
        } 

        #if defined(OUTPUT_SOLUTION)
            fprintf(stdout, "1\n");
            for (int task_id = 0; task_id < INPUT.tasks_count; task_id++) {
                fprintf(stdout, "%d\n", current_solution.task_assignment[task_id]);
            }
        #endif

        free_solution(&current_solution);
    } else {
        // =============================================================
        // Population algorithms.
        // =============================================================
        
        if (INPUT.algorithm == ALGORITHM_CMOCHCISLAND) {
            
            compute_cmochc_island();
            
        } else if (INPUT.algorithm == ALGORITHM_CMOCHCCELL) {

            compute_cmochc_cell();

        }
    }

    // Timming -----------------------------------------------------
    TIMMING_END("Elapsed algorithm total time", ts);
    // Timming -----------------------------------------------------

    return EXIT_SUCCESS;
}
