#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "etc_matrix.h"
#include "energy_matrix.h"

#include "load_instance.h"

#ifndef _LOAD_INSTANCE_C_
#define _LOAD_INSTANCE_C_

/* Reads the input file and stores de data in the ETC matrix */
int load_instance(struct params *input, struct etc_matrix *etc, struct energy_matrix *energy) {
	FILE *workload_file;

	if ((workload_file = fopen(input->workload_path, "r")) == NULL) {
		fprintf(stderr, "[ERROR] Reading the workload file %s.\n", input->workload_path);
		return EXIT_FAILURE;
	}

    int unknown_param;
	if (fscanf(workload_file, "%d %d %d", &(input->tasks_count), &(input->machines_count), &unknown_param) != 3) {
	    return EXIT_FAILURE;	    
	}
	if (DEBUG) fprintf(stdout, "[PARAMS] tasks   : %d", input->tasks_count);
	if (DEBUG) fprintf(stdout, "[PARAMS] machines: %d", input->machines_count);
	if (DEBUG) fprintf(stdout, "[PARAMS] unknown : %d", unknown_param);

	float value;
	for (int task = 0; task < input->tasks_count; task++) {
		for (int machine = 0; machine < input->machines_count; machine++) {
			if (fscanf(workload_file, "%f", &value) != 1) {
			    return EXIT_FAILURE;
			}
			
			set_etc_value(etc, machine, task, value); 
		}
	}

	fclose(workload_file);
	
	// -------------------------------------------------------------------
	
	FILE *scenario_file;

	if ((scenario_file = fopen(input->scenario_path, "r")) == NULL) {
		fprintf(stderr, "[ERROR] Reading the scenario file %s.\n", input->scenario_path);
		return EXIT_FAILURE;
	}

    int cores;
	float ssj;
	float energy_idle;
	float energy_max;
	
	for (int task = 0; task < input->tasks_count; task++) {
		for (int machine = 0; machine < input->machines_count; machine++) {
            if (fscanf(scenario_file,"%d %f %f %f\n",&cores,&ssj,&energy_idle,&energy_max) != 4) {
			    return EXIT_FAILURE;
            }

            // Cambio el energy_max para utilizar solo un core del procesador.
            energy_max = energy_max + ((energy_max - energy_idle) / cores);

			set_energy_value(energy, machine, energy_idle, energy_max); 
		}
	}

	fclose(scenario_file);
	
	return EXIT_SUCCESS;
}

#endif //_LOAD_INSTANCE_C_
