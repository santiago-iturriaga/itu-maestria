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
#include "solution.h"
#include "config.h"

#include "utils.h"
#include "gpu_utils.h"

#include "basic/mct.h"
#include "basic/minmin.h"
#include "basic/pminmin.h"

#include "random/RNG_rand48.h"

#include "pals/pals_serial.h"
#include "pals/pals_gpu.h"
#include "pals/pals_gpu_rtask.h"

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

    // Se pide el espacio de memoria para la matriz de ETC.
    struct matrix *etc_matrix = create_etc_matrix(&input);

    // Se carga la matriz de ETC.
    if (load_instance(&input, etc_matrix) == EXIT_FAILURE) {
        fprintf(stderr, "[ERROR] Ocurrió un error leyendo el archivo de instancia.\n");
        return EXIT_FAILURE;
    }

    // =============================================================
    // Create empty solution
    // =============================================================
    #if defined(DEBUG)
        fprintf(stdout, "[DEBUG] Creating empty solution...\n");
    #endif
    
    struct solution *current_solution = create_empty_solution(etc_matrix);

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

    if (input.algorithm == PALS_Serial) {
        // =============================================================
        // Candidate solution
        // =============================================================
        #if defined(DEBUG)
            fprintf(stdout, "[DEBUG] Creating initial candiate solution...\n");
        #endif
        
        // Timming -----------------------------------------------------
        timespec ts_mct;
        timming_start(ts_mct);
        // Timming -----------------------------------------------------

        compute_mct(etc_matrix, current_solution);

        // Timming -----------------------------------------------------
        timming_end(">> MCT Time", ts_mct);
        // Timming -----------------------------------------------------

        #if defined(DEBUG)
            validate_solution(etc_matrix, current_solution);
        #endif

        // =============================================================
        // Serial. Versión de búsqueda completa.
        // =============================================================
        pals_serial(input, etc_matrix, current_solution);

    } else if (input.algorithm == PALS_GPU) {
        // =============================================================
        // Candidate solution
        // =============================================================
        #if defined(DEBUG)
            fprintf(stdout, "[DEBUG] Creating initial candiate solution...\n");
        #endif

        // Timming -----------------------------------------------------
        timespec ts_mct;
        timming_start(ts_mct);
        // Timming -----------------------------------------------------

        compute_mct(etc_matrix, current_solution);

        // Timming -----------------------------------------------------
        timming_end(">> MCT Time", ts_mct);
        // Timming -----------------------------------------------------

        #if defined(DEBUG)
            validate_solution(etc_matrix, current_solution);
        #endif

        // =============================================================
        // CUDA. Versión de búsqueda completa.
        // =============================================================

        gpu_set_device(input.gpu_device);
        pals_gpu(input, etc_matrix, current_solution);

    } else if (input.algorithm == PALS_GPU_randTask) {
        // =============================================================
        // Candidate solution
        // =============================================================
        #if defined(DEBUG)
            fprintf(stdout, "[DEBUG] Creating initial candiate solution...\n");
        #endif

        // Timming -----------------------------------------------------
        timespec ts_mct;
        timming_start(ts_mct);
        // Timming -----------------------------------------------------

        //compute_mct(etc_matrix, current_solution);
        //compute_minmin(etc_matrix, current_solution);
                compute_mct(etc_matrix, current_solution);

        // Timming -----------------------------------------------------
        timming_end(">> MCT Time", ts_mct);
        // Timming -----------------------------------------------------

        #if defined(DEBUG)
            validate_solution(etc_matrix, current_solution);
        #endif

        // =============================================================
        // CUDA. Búsqueda aleatoria por tarea.
        // =============================================================

        gpu_set_device(input.gpu_device);
        pals_gpu_rtask(input, etc_matrix, current_solution);

    } else if (input.algorithm == MinMin) {

        compute_minmin(etc_matrix, current_solution);

    } else if (input.algorithm == pMinMin) {

        int thread_count;
        thread_count = etc_matrix->machines_count >> 5;
        
        if (thread_count == 0) thread_count = 1;
        if (thread_count > MAX_THREAD_COUNT) thread_count = MAX_THREAD_COUNT;

        compute_pminmin(etc_matrix, current_solution, thread_count);

    } else if (input.algorithm == MCT) {

        compute_mct(etc_matrix, current_solution);

    }

    #if defined(OUTPUT_SOLUTION)
        //fprintf(stdout, "%d %d\n", etc_matrix->tasks_count, etc_matrix->machines_count);
        for (int task_id = 0; task_id < etc_matrix->tasks_count; task_id++) {
            fprintf(stdout, "%d\n", current_solution->task_assignment[task_id]);
        }
    #endif

    // Timming -----------------------------------------------------
    timming_end("Elapsed algorithm total time", ts);
    // Timming -----------------------------------------------------

    // =============================================================
    // Release memory
    // =============================================================
    free_etc_matrix(etc_matrix);
    free_solution(current_solution);

    return EXIT_SUCCESS;
}
