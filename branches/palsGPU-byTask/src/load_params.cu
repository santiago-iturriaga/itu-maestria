#include <stdio.h>
#include <stdlib.h>

#include "load_params.h"

int load_params(int argc, char **argv, struct params *input) {
	if (argc == 5) {
		input->instance_path = argv[1];
		fprintf(stdout, "[PARAMS] instance path: %s\n", input->instance_path);

		input->tasks_count = atoi(argv[2]);
		fprintf(stdout, "[PARAMS] tasks count: %d\n", input->tasks_count);

		input->machines_count = atoi(argv[3]);
		fprintf(stdout, "[PARAMS] machines count: %d\n", input->machines_count);

		input->pals_flavour = atoi(argv[4]);
		fprintf(stdout, "[PARAMS] PALS falvour: %d\n", input->pals_flavour);

		// Input validation.
		if (input->tasks_count < 1) {
			fprintf(stderr, "[ERROR] Invalid tasks count.\n");
			return EXIT_FAILURE;
		}
		
		if (input->machines_count < 1) {
			fprintf(stderr, "[ERROR] Invalid machines count.\n");
			return EXIT_FAILURE;
		}

		if ((input->pals_flavour < 0)||(input->pals_flavour > 3)) {
			fprintf(stderr, "[ERROR] Invalid PALS falvour.\n");
			return EXIT_FAILURE;
		}
		
		return EXIT_SUCCESS;
	} else {
		fprintf(stdout, "Usage:\n");	
		fprintf(stdout, "       %s <instance_path> <tasks count> <machines count> <pals flavour>\n\n", argv[0]);
		fprintf(stdout, "       pals flavour = 0 serial full\n");
		fprintf(stdout, "                      1 gpu full\n");
		fprintf(stdout, "                      2 gpu rand task\n");
		fprintf(stdout, "                      3 gpu rand machine\n");
		fprintf(stdout, "\n");

		return EXIT_FAILURE;
	}
}
