#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "load_params.h"

int load_params(int argc, char **argv, struct params *input) {
	if (argc >= 8) {
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
			if (input->algorithm == PALS_Serial) {
				fprintf(stdout, " (PALS_Serial)\n");
			} else if (input->algorithm == PALS_GPU) {
				fprintf(stdout, " (PALS_GPU)\n");
			} else if (input->algorithm == PALS_GPU_randTask) {
				fprintf(stdout, " (PALS_GPU_randTask)\n");
			} else if (input->algorithm == PALS_CPU_randTask) {
				fprintf(stdout, " (PALS_CPU_randTask)\n");
			} else if (input->algorithm == MinMin) {
				fprintf(stdout, " (Min-Min)\n");
			} else if (input->algorithm == MCT) {
				fprintf(stdout, " (MCT)\n");
			} else if (input->algorithm == PALS_GPU_randParallelTask) {
				fprintf(stdout, " (PALS_GPU_randParallelTask)\n");
			}
		}

		input->thread_count = atoi(argv[6]);
		if (input->thread_count < 2) input->thread_count = 2;	
		if (DEBUG) fprintf(stdout, "[PARAMS] threads count: %d\n", input->thread_count);

		input->seed = atoi(argv[7]);
		if (DEBUG) fprintf(stdout, "[PARAMS] seed: %d\n", input->seed);

		if ((input->algorithm < 0)||(input->algorithm > 6)) {
			fprintf(stderr, "[ERROR] Invalid algorithm.\n");
			return EXIT_FAILURE;
		}
		
		return EXIT_SUCCESS;
	} else {
		fprintf(stdout, "Usage:\n");	
		fprintf(stdout, "       %s <scenario> <workload> <#tasks> <#machines> <algorithm> <#threads> <seed>\n\n", argv[0]);
		fprintf(stdout, "       Algorithms\n");
		fprintf(stdout, "           0 CPU full (serial)\n");
		fprintf(stdout, "           3 CPU rand. task (multi-threaded)\n");
		fprintf(stdout, "           4 Min-Min\n");
		fprintf(stdout, "           5 MCT\n");
		//fprintf(stdout, "           6 GPU rand. parallel task\n");
		fprintf(stdout, "\n");

		return EXIT_FAILURE;
	}
}
