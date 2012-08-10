#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "config.h"
#include "load_params.h"

int load_params(int argc, char **argv, struct params *input) {
    if (argc == 11) {
        input->scenario_path = argv[1];
        #if defined(DEBUG_0) 
            fprintf(stderr, "[PARAMS] scenario path: %s\n", input->scenario_path);
        #endif

        input->workload_path = argv[2];
        #if defined(DEBUG_0) 
            fprintf(stderr, "[PARAMS] workload path: %s\n", input->workload_path);
        #endif

        input->tasks_count = atoi(argv[3]);
        #if defined(DEBUG_0) 
            fprintf(stderr, "[PARAMS] tasks count: %d\n", input->tasks_count);
        #endif
        
        input->machines_count = atoi(argv[4]);
        #if defined(DEBUG_0) 
            fprintf(stderr, "[PARAMS] machines count: %d\n", input->machines_count);
        #endif

        input->algorithm = atoi(argv[5]);
        #if defined(DEBUG_0) 
            fprintf(stderr, "[PARAMS] algorithm: %d", input->algorithm);
        #endif

        #if defined(DEBUG_0)
            if (input->algorithm == ALGORITHM_MCT) {
                fprintf(stderr, " MCT\n");
            } else if (input->algorithm == ALGORITHM_MINMIN) {
                fprintf(stderr, " MinMin\n");
            } else if (input->algorithm == ALGORITHM_PMINMIND) {
                fprintf(stderr, " pMinMin/D\n");
            } else {
                fprintf(stderr, " unknown!?\n");
            }
        #endif

        input->thread_count = atoi(argv[6]);
        #if defined(DEBUG_0) 
            fprintf(stderr, "[PARAMS] threads count: %d\n", input->thread_count);
        #endif

        input->seed = atoi(argv[7]);
        #if defined(DEBUG_0) 
            fprintf(stderr, "[PARAMS] seed: %d\n", input->seed);
        #endif

        input->max_time_secs = atoi(argv[8]);
        #if defined(DEBUG_0) 
            fprintf(stderr, "[PARAMS] max. time (secs.): %d\n", input->max_time_secs);
        #endif
        
        input->max_iterations = atoi(argv[9]);
        #if defined(DEBUG_0) 
            fprintf(stderr, "[PARAMS] max. iterations: %d\n", input->max_iterations);
        #endif

        input->population_size = atoi(argv[10]);
        #if defined(DEBUG_0) 
            fprintf(stderr, "[PARAMS] population size: %d\n", input->population_size);
        #endif

        if ((input->algorithm < 0)||(input->algorithm > 2)) {
            fprintf(stderr, "[ERROR] invalid algorithm.\n");
            return EXIT_FAILURE;
        }
        
        return EXIT_SUCCESS;
    } else {
        fprintf(stdout, "Usage:\n");    
        fprintf(stdout, "       %s <scenario> <workload> <#tasks> <#machines> <algorithm> <#threads> <seed> <max time (secs)> <max iterations> <population size>\n\n", argv[0]);
        fprintf(stdout, "       Algorithms\n");
        fprintf(stdout, "           0 MCT\n");
        fprintf(stdout, "           1 MinMin\n");
        fprintf(stdout, "           2 pMinMin/D\n");
        fprintf(stdout, "\n");

        return EXIT_FAILURE;
    }
}
