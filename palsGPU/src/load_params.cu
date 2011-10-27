#include "load_params.h"

#include <stdio.h>
#include <stdlib.h>

int load_params(int argc, char **argv, struct params *input) {
	if (argc == 4) {
		input->instance_path = argv[1];
		fprintf(stdout, "[PARAMS] instance path: %s\n", input->instance_path);

		input->tasks_count = atoi(argv[2]);
		fprintf(stdout, "[PARAMS] tasks count: %d\n", input->tasks_count);

		input->machines_count = atoi(argv[3]);
		fprintf(stdout, "[PARAMS] machines count: %d\n", input->machines_count);

		return EXIT_SUCCESS;
	} else {
		fprintf(stdout, "Help: %s <instance_path> <tasks count> <machines count>\n", argv[0]);

		return EXIT_FAILURE;
	}
}
