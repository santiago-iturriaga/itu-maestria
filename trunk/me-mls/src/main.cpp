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

#include "pals/me_mls_cpu.h"
#include "pals/me_rpals_cpu.h"

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

    #if defined(CPU_RAND)
        fprintf(stderr, "PRNG|randr\n");
    #endif
    #if defined(CPU_DRAND48)
        fprintf(stderr, "PRNG|drand48r\n");
    #endif
    #if defined(CPU_MERSENNE_TWISTER)
        fprintf(stderr, "PRNG|MT\n");
    #endif
    #if !defined(CPU_RAND) && !(CPU_DRAND48) && !(CPU_MERSENNE_TWISTER)
        fprintf(stderr, "No PRNG method is defined.\n");
        return EXIT_FAILURE;
    #endif
    
    #if !defined(OUTPUT_SOLUTION)
        fprintf(stderr, "No output option is defined.\n");
        return EXIT_FAILURE;
    #endif
    fprintf(stderr, "OUTPUT|%d\n", OUTPUT_SOLUTION);

    if (input.algorithm == ME_RPALS) {
        // =============================================================
        // ME-rPALS
        // =============================================================
            
        me_rpals_cpu(input, &etc, &energy);
        
    } else if (input.algorithm == ME_MLS) {
        // =============================================================
        // ME-MLS
        // =============================================================

        #if defined(ARCHIVER_AGA)
            fprintf(stderr, "ARCHIVER|AGA\n");
        #else
            fprintf(stderr, "ARCHIVER|ADHOC\n");
        #endif

        #if defined(INIT_MCT)
            fprintf(stderr, "PRNG|MCT\n");
        #endif
        #if defined(INIT_PMINMIN)
            fprintf(stderr, "PRNG|pMinMinDD\n");
        #endif
        #if defined(INIT_MINMIN)
            fprintf(stderr, "PRNG|MinMin\n");
        #endif
        #if !defined(INIT_MCT) && !(INIT_PMINMIN) && !(INIT_MINMIN)
            fprintf(stderr, "No init method is defined.\n");
            return EXIT_FAILURE;
        #endif
            
        me_mls_cpu(input, &etc, &energy);
        
    } else if (input.algorithm == MINMIN) {
        struct solution *current_solution = create_empty_solution(&etc, &energy);
        compute_minmin(current_solution);
        
        if (OUTPUT_SOLUTION == 0) {
            fprintf(stdout, "%f %f\n", get_makespan(current_solution), get_energy(current_solution));
        } else {
            show_solution(current_solution);
        }
        
        free_solution(current_solution);
        free(current_solution);
            
    } else if (input.algorithm == MCT) {
        struct solution *current_solution = create_empty_solution(&etc, &energy);
        compute_mct(current_solution);
        
        if (OUTPUT_SOLUTION == 0) {
            fprintf(stdout, "%f %f\n", get_makespan(current_solution), get_energy(current_solution));
        } else {
            show_solution(current_solution);
        }
        
        free_solution(current_solution);
        free(current_solution);
        
    } else if (input.algorithm == pMINMIN) {
        struct solution *current_solution = create_empty_solution(&etc, &energy);
        compute_pminmin(&etc, current_solution, input.thread_count);
        
        if (OUTPUT_SOLUTION == 0) {
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
