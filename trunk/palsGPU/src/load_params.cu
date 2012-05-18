#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "config.h"
#include "load_params.h"

int load_params(int argc, char **argv, struct params *input) {
    if (argc >= 5) {
        input->instance_path = argv[1];
        //#if defined(DEBUG)
            fprintf(stderr, "[PARAMS] instance path: %s\n", input->instance_path);
        //#endif

        input->tasks_count = atoi(argv[2]);
        //#if defined(DEBUG)
            fprintf(stderr, "[PARAMS] tasks count: %d\n", input->tasks_count);
        //#endif
        
        input->machines_count = atoi(argv[3]);
        //#if defined(DEBUG)
            fprintf(stderr, "[PARAMS] machines count: %d\n", input->machines_count);
        //#endif
        
        input->algorithm = atoi(argv[4]);
        //#if defined(DEBUG)
            fprintf(stderr, "[PARAMS] algorithm: %d", input->algorithm);
        //#endif
        
        //#if defined(DEBUG)
            if (input->algorithm == PALS_Serial) {
                fprintf(stderr, " (PALS_Serial)\n");
            } else if (input->algorithm == PALS_GPU) {
                fprintf(stderr, " (PALS_GPU)\n");
            } else if (input->algorithm == PALS_GPU_randTask) {
                fprintf(stderr, " (PALS_GPU_randTask)\n");
            } else if (input->algorithm == pMinMin) {
                fprintf(stderr, " (pMin-Min)\n");
            } else if (input->algorithm == MinMin) {
                fprintf(stderr, " (Min-Min)\n");
            } else if (input->algorithm == MCT) {
                fprintf(stderr, " (MCT)\n");
            }
        //#endif

        input->seed = 0;
        input->gpu_device = 0;
        input->timeout = INT_MAX;
        input->target_makespan = -1;
        input->init_algorithm = pMinMin;
        input->init_algorithm = 1;

        if (argc >= 6) {
            input->seed = atoi(argv[5]);
        }
        //#if defined(DEBUG) 
            fprintf(stderr, "[PARAMS] seed: %d\n", input->seed);
        //#endif

        if (argc >= 7) {
            input->gpu_device = atoi(argv[6]);
        }
        //#if defined(DEBUG) 
            fprintf(stderr, "[PARAMS] gpu device: %d\n", input->gpu_device);
        //#endif
        
        if (argc >= 8) {
            input->timeout = atoi(argv[7]);
        }
        //#if defined(DEBUG) 
            fprintf(stderr, "[PARAMS] timeout: %d s\n", input->timeout);
        //#endif
        
        if (argc >= 9) {
            input->target_makespan = atof(argv[8]);
        }
        //#if defined(DEBUG) 
            fprintf(stderr, "[PARAMS] target makespan: %f\n", input->target_makespan);
        //#endif

        if (argc >= 10) {
            input->init_algorithm = atoi(argv[9]);
        }
        //#if defined(DEBUG)
            fprintf(stderr, "[PARAMS] init algorithm: %d", input->init_algorithm);
        //#endif
        
        //#if defined(DEBUG)
            if (input->init_algorithm == PALS_Serial) {
                fprintf(stderr, " (PALS_Serial)\n");
            } else if (input->init_algorithm == PALS_GPU) {
                fprintf(stderr, " (PALS_GPU)\n");
            } else if (input->init_algorithm == PALS_GPU_randTask) {
                fprintf(stderr, " (PALS_GPU_randTask)\n");
            } else if (input->init_algorithm == pMinMin) {
                fprintf(stderr, " (pMin-Min)\n");
            } else if (input->init_algorithm == MinMin) {
                fprintf(stderr, " (Min-Min)\n");
            } else if (input->init_algorithm == MCT) {
                fprintf(stderr, " (MCT)\n");
            }
        //#endif

        if (argc >= 11) {
            input->init_algorithm_threads = atoi(argv[10]);
        }
        //#if defined(DEBUG) 
            fprintf(stderr, "[PARAMS] init algorithm threads: %d s\n", input->init_algorithm_threads);
        //#endif
        
        // Input validation.
        if (input->tasks_count < 1) {
            fprintf(stderr, "[ERROR] Invalid tasks count.\n");
            return EXIT_FAILURE;
        }

        if (input->machines_count < 1) {
            fprintf(stderr, "[ERROR] Invalid machines count.\n");
            return EXIT_FAILURE;
        }

        if ((input->algorithm < 0)||(input->algorithm > 6)) {
            fprintf(stderr, "[ERROR] Invalid algorithm.\n");
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    } else {
        fprintf(stdout, "Usage:\n");
        fprintf(stdout, "       %s <instance_path> <tasks count> <machines count> <algorithm> [seed] [gpu device] [timeout] [target makespan] [init algorithm]\n\n", argv[0]);
        fprintf(stdout, "       Algorithm = 0 Serial full\n");
        fprintf(stdout, "                   1 GPU full\n");
        fprintf(stdout, "                   2 GPU rand. task\n");
        fprintf(stdout, "                   3 parallel Min-Min\n");
        fprintf(stdout, "                   4 Min-Min\n");
        fprintf(stdout, "                   5 MCT\n");
        fprintf(stdout, "\n");

        return EXIT_FAILURE;
    }
}
