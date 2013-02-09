#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "load_params.h"

int load_params(int argc, char **argv, struct params *input) {
    if (argc >= 11) {
        input->scenario_path = argv[1];
        #if defined(DEBUG) 
        fprintf(stdout, "[PARAMS] scenario path: %s\n", input->scenario_path);
        #endif

        input->workload_path = argv[2];
        #if defined(DEBUG) 
        fprintf(stdout, "[PARAMS] workload path: %s\n", input->workload_path);
        #endif

        input->tasks_count = atoi(argv[3]);
        #if defined(DEBUG) 
        fprintf(stdout, "[PARAMS] tasks count: %d\n", input->tasks_count);
        #endif
        
        input->machines_count = atoi(argv[4]);
        #if defined(DEBUG) 
        fprintf(stdout, "[PARAMS] machines count: %d\n", input->machines_count);
        #endif

        input->algorithm = atoi(argv[5]);
        #if defined(DEBUG) 
        fprintf(stdout, "[PARAMS] algorithm: %d", input->algorithm);
        #endif

        #if defined(DEBUG)
        if (input->algorithm == ME_RPALS) {
            fprintf(stdout, " (ME-rPALS)\n");
        } else if (input->algorithm == ME_MLS) {
            fprintf(stdout, " (ME-MLS)\n");
        } else if (input->algorithm == MINMIN) {
            fprintf(stdout, " (MinMin)\n");
        } else if (input->algorithm == MCT) {
            fprintf(stdout, " (MCT)\n");
        } else if (input->algorithm == pMINMIN) {
            fprintf(stdout, " (pMINMIN)\n");
        }
        #endif

        input->thread_count = atoi(argv[6]);
        //if (input->thread_count < 2) input->thread_count = 2; 
        #if defined(DEBUG) 
        fprintf(stdout, "[PARAMS] threads count: %d\n", input->thread_count);
        #endif

        input->seed = atoi(argv[7]);
        #if defined(DEBUG) 
        fprintf(stdout, "[PARAMS] seed: %d\n", input->seed);
        #endif

        input->max_time_secs = atoi(argv[8]);
        #if defined(DEBUG) 
        fprintf(stdout, "[PARAMS] Max. time (secs.): %d\n", input->max_time_secs);
        #endif
        
        input->max_iterations = atoi(argv[9]);
        #if defined(DEBUG) 
        fprintf(stdout, "[PARAMS] Max. iterations: %d\n", input->max_iterations);
        #endif

        input->population_size = atoi(argv[10]);
        #if defined(DEBUG) 
        fprintf(stdout, "[PARAMS] Population size: %d\n", input->population_size);
        #endif

        if ((input->algorithm < 0)||(input->algorithm > 4)) {
            fprintf(stderr, "[ERROR] Invalid algorithm.\n");
            return EXIT_FAILURE;
        }
        
        return EXIT_SUCCESS;
    } else {
        fprintf(stdout, "Usage:\n");    
        fprintf(stdout, "       %s <scenario> <workload> <#tasks> <#machines> <algorithm> <#threads> <seed> <max time (secs)> <max iterations> <population size>\n\n", argv[0]);
        fprintf(stdout, "       Algorithms\n");
        fprintf(stdout, "           0 ME-rPALS\n");
        fprintf(stdout, "           1 ME-MLS\n");
        fprintf(stdout, "           2 MinMin\n");
        fprintf(stdout, "           3 MCT\n");
        fprintf(stdout, "           4 pMinMin\n");
        
        fprintf(stdout, "\n");

        return EXIT_FAILURE;
    }
}
