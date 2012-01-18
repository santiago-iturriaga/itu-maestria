#include <stdio.h>
#include <stdlib.h>

#include "load_instance.h"

#ifndef _LOAD_INSTANCE_C_
#define _LOAD_INSTANCE_C_

/* Reads the input file and stores de data in the ETC matrix */
int load_instance(struct params *input, struct etc_matrix *etc) {
	FILE *fi;

	if ((fi = fopen(input->instance_path, "r")) == NULL) {
		fprintf(stderr, "[ERROR] Reading the input file %s.\n", input->instance_path);
		return EXIT_FAILURE;
	}

	float value;
	for (int task = 0; task < input->tasks_count; task++) {
		for (int machine = 0; machine < input->machines_count; machine++) {
			fscanf(fi, "%f", &value);
			
			set_etc_value(etc, machine, task, value); 
		}
	}

	fclose(fi);
	return EXIT_SUCCESS;
}

#endif //_LOAD_INSTANCE_C_
