#include <stdio.h>
#include <stdlib.h>

#include "load_params.h"

int load_params(int argc, char **argv, struct params *input) {
	if (argc >= 5) {
		input->instance_path = argv[1];
		fprintf(stdout, "[PARAMS] instance path: %s\n", input->instance_path);

		input->tasks_count = atoi(argv[2]);
		fprintf(stdout, "[PARAMS] tasks count: %d\n", input->tasks_count);

		input->machines_count = atoi(argv[3]);
		fprintf(stdout, "[PARAMS] machines count: %d\n", input->machines_count);

		input->algorithm = atoi(argv[4]);
		fprintf(stdout, "[PARAMS] Algorithm: %d", input->algorithm);

		if (input->algorithm == PALS_Serial) {
			fprintf(stdout, " (PALS_Serial)\n");
		} else if (input->algorithm == PALS_GPU) {
			fprintf(stdout, " (PALS_GPU)\n");
		} else if (input->algorithm == PALS_GPU_randTask) {
			fprintf(stdout, " (PALS_GPU_randTask)\n");
		} else if (input->algorithm == PALS_GPU_randMachine) {
			fprintf(stdout, " (PALS_GPU_randMachine)\n");
		} else if (input->algorithm == MinMin) {
			fprintf(stdout, " (Min-Min)\n");
		}

		if (argc >= 6) {
			input->seed = atoi(argv[5]);
			fprintf(stdout, "[PARAMS] seed: %d\n", input->seed);
		} else {
			input->seed = 0;
			input->gpu_device = 0;
		}

		if (argc >= 7) {
			input->gpu_device = atoi(argv[6]);
			fprintf(stdout, "[PARAMS] gpu device: %d\n", input->gpu_device);
		} else {
			input->gpu_device = 0;
		}

		// Input validation.
		if (input->tasks_count < 1) {
			fprintf(stderr, "[ERROR] Invalid tasks count.\n");
			return EXIT_FAILURE;
		}
		
		if (input->machines_count < 1) {
			fprintf(stderr, "[ERROR] Invalid machines count.\n");
			return EXIT_FAILURE;
		}

		if ((input->algorithm < 0)||(input->algorithm > 4)) {
			fprintf(stderr, "[ERROR] Invalid algorithm.\n");
			return EXIT_FAILURE;
		}
		
		return EXIT_SUCCESS;
	} else {
		fprintf(stdout, "Usage:\n");	
		fprintf(stdout, "       %s <instance_path> <tasks count> <machines count> <algorithm> [seed] [gpu device]\n\n", argv[0]);
		fprintf(stdout, "       Algorithm = 0 Serial full\n");
		fprintf(stdout, "                   1 GPU full\n");
		fprintf(stdout, "                   2 GPU rand. task\n");
		fprintf(stdout, "                   3 GPU rand. machine\n");
		fprintf(stdout, "                   4 Min-Min\n");
		fprintf(stdout, "\n");

		return EXIT_FAILURE;
	}
}
