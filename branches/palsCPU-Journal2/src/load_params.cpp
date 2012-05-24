#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "load_params.h"

int load_params(int argc, char **argv, struct params *input) {
    if (argc >= 11) {
        input->scenario_path = argv[1];
        if (DEBUG) fprintf(stdout, "[PARAMS] scenario path: %s\n", input->scenario_path);

        input->workload_path = argv[2];
        if (DEBUG) fprintf(stdout, "[PARAMS] workload path: %s\n", input->workload_path);

        input->tasks_count = atoi(argv[3]);
        if (DEBUG) fprintf(stdout, "[PARAMS] tasks count: %d\n", input->tasks_count);
        
        input->machines_count = atoi(argv[4]);
        if (DEBUG) fprintf(stdout, "[PARAMS] machines count: %d\n", input->machines_count);

        input->algorithm = atoi(argv[5]);
        if (DEBUG) fprintf(stdout, "[PARAMS] algorithm: %d", input->algorithm);

        if (DEBUG) {
            if (input->algorithm == PALS_2POP) {
                fprintf(stdout, " (PALS 2-populations)\n");
            } else if (input->algorithm == PALS_1POP) {
                fprintf(stdout, " (PALS 1-population)\n");
            } else if (input->algorithm == MINMIN) {
                fprintf(stdout, " (MinMin)\n");
            } else if (input->algorithm == MCT) {
                fprintf(stdout, " (MCT)\n");
            } else if (input->algorithm == pMINMIN) {
                fprintf(stdout, " (pMINMIN)\n");
            }
        }

        input->thread_count = atoi(argv[6]);
        //if (input->thread_count < 2) input->thread_count = 2; 
        if (DEBUG) fprintf(stdout, "[PARAMS] threads count: %d\n", input->thread_count);

        input->seed = atoi(argv[7]);
        if (DEBUG) fprintf(stdout, "[PARAMS] seed: %d\n", input->seed);

        input->max_time_secs = atoi(argv[8]);
        if (DEBUG) fprintf(stdout, "[PARAMS] Max. time (secs.): %d\n", input->max_time_secs);
        
        input->max_iterations = atoi(argv[9]);
        if (DEBUG) fprintf(stdout, "[PARAMS] Max. iterations: %d\n", input->max_iterations);

        input->population_size = atoi(argv[10]);
        if (DEBUG) fprintf(stdout, "[PARAMS] Population size: %d\n", input->population_size);

        if ((input->algorithm < 0)||(input->algorithm > 4)) {
            fprintf(stderr, "[ERROR] Invalid algorithm.\n");
            return EXIT_FAILURE;
        }
        
        return EXIT_SUCCESS;
    } else {
        fprintf(stdout, "Usage:\n");    
        fprintf(stdout, "       %s <scenario> <workload> <#tasks> <#machines> <algorithm> <#threads> <seed> <max time (secs)> <max iterations> <population size>\n\n", argv[0]);
        fprintf(stdout, "       Algorithms\n");
        fprintf(stdout, "           0 PALS 2-populations\n");
        fprintf(stdout, "           1 PALS 1-population\n");
        fprintf(stdout, "           2 MinMin\n");
        fprintf(stdout, "           3 MCT\n");
        fprintf(stdout, "           4 pMinMin\n");
        
        fprintf(stdout, "\n");

        return EXIT_FAILURE;
    }
}
