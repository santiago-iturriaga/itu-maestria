#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "load_params.h"

#include "config.h"
#include "global.h"

int load_params(int argc, char **argv) {
    if (argc == 10) {
        fprintf(stderr, "[INFO] == Command line arguments ==============================\n");
        
        INPUT.scenario_path = argv[1];
        #if defined(DEBUG_0) 
            fprintf(stderr, "       Scenario path                : %s\n", INPUT.scenario_path);
        #endif

        INPUT.workload_path = argv[2];
        #if defined(DEBUG_0) 
            fprintf(stderr, "       Workload path                : %s\n", INPUT.workload_path);
        #endif

        INPUT.tasks_count = atoi(argv[3]);
        #if defined(DEBUG_0) 
            fprintf(stderr, "       Tasks count                  : %d\n", INPUT.tasks_count);
        #endif
        
        INPUT.machines_count = atoi(argv[4]);
        #if defined(DEBUG_0) 
            fprintf(stderr, "       Machines count               : %d\n", INPUT.machines_count);
        #endif

        INPUT.algorithm = atoi(argv[5]);
        #if defined(DEBUG_0) 
            fprintf(stderr, "       Algorithm                    : %d", INPUT.algorithm);
        #endif

        #if defined(DEBUG_0)
            if (INPUT.algorithm == ALGORITHM_MCT) {
                fprintf(stderr, " MCT\n");
            } else if (INPUT.algorithm == ALGORITHM_MINMIN) {
                fprintf(stderr, " MinMin\n");
            } else if (INPUT.algorithm == ALGORITHM_PMINMIND) {
                fprintf(stderr, " pMinMin/D\n");
            } else if (INPUT.algorithm == ALGORITHM_CMOCHCISLAND) {
                fprintf(stderr, " cMOCHC/islands\n");
            } else if (INPUT.algorithm == ALGORITHM_CMOCHCCELL) {
                fprintf(stderr, " cMOCHC/cellular\n");
            } else {
                fprintf(stderr, " unknown!?\n");
            }
        #endif

        INPUT.thread_count = atoi(argv[6]);
        #if defined(DEBUG_0) 
            fprintf(stderr, "       Thread count                 : %d\n", INPUT.thread_count);
        #endif

        INPUT.seed = atoi(argv[7]);
        #if defined(DEBUG_0) 
            fprintf(stderr, "       Seed                         : %d\n", INPUT.seed);
        #endif

        INPUT.max_time_secs = atoi(argv[8]);
        #if defined(DEBUG_0) 
            fprintf(stderr, "       Max. execution time (s)      : %d\n", INPUT.max_time_secs);
        #endif
        
        /*INPUT.max_iterations = atoi(argv[9]);
        #if defined(DEBUG_0) 
            fprintf(stderr, "       Max. iterations              : %d\n", INPUT.max_iterations);
        #endif*/
        
        INPUT.max_evaluations = atoi(argv[9]);
        #if defined(DEBUG_0) 
            fprintf(stderr, "       Max. evaluations              : %d\n", INPUT.max_evaluations);
        #endif

        if ((INPUT.algorithm < 0)||(INPUT.algorithm > 4)) {
            fprintf(stderr, "[ERROR] invalid algorithm.\n");
            return EXIT_FAILURE;
        }
        
        fprintf(stderr, "[INFO] ========================================================\n");
        
        return EXIT_SUCCESS;
    } else {
        fprintf(stdout, "Usage:\n");    
        fprintf(stdout, "       %s <scenario> <workload> <#tasks> <#machines> <algorithm> <#threads> <seed> <max time (secs)> <max iterations>\n\n", argv[0]);
        fprintf(stdout, "       Algorithms\n");
        fprintf(stdout, "           %d MCT\n",ALGORITHM_MCT);
        fprintf(stdout, "           %d MinMin\n",ALGORITHM_MINMIN);
        fprintf(stdout, "           %d pMinMin/D\n",ALGORITHM_PMINMIND);
        fprintf(stdout, "           %d cMOCHC/islands\n",ALGORITHM_CMOCHCISLAND);
        fprintf(stdout, "           %d cMOCHC/cellular\n",ALGORITHM_CMOCHCCELL);
        fprintf(stdout, "\n");

        return EXIT_FAILURE;
    }
}
