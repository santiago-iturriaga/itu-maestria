/*
 * energy_matrix.c
 *
 *  Created on: Jul 28, 2011
 *      Author: santiago
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "load_params.h"
#include "energy_matrix.h"

void init_energy_matrix(struct params *input, struct energy_matrix *energy) {	
	energy->machines_count = input->machines_count;
	energy->idle_energy = (float*)malloc(sizeof(float) * input->machines_count);
	
	if (energy->idle_energy == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para el energy_matrix->idle_energy.\n");
		exit(EXIT_FAILURE);
	}
	
	energy->max_energy = (float*)malloc(sizeof(float) * input->machines_count);
	
	if (energy->max_energy == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para el energy_matrix->max_energy.\n");
		exit(EXIT_FAILURE);
	}
}

void free_energy_matrix(struct energy_matrix *energy) {
	free(energy->idle_energy);
	free(energy->max_energy);
}

void set_energy_value(struct energy_matrix *energy, int machine, float idle_value, float max_value) {
	assert(machine < energy->machines_count);
	assert(machine >= 0);
	assert(idle_value > 0.0);
	assert(max_value > 0.0);
	
	energy->idle_energy[machine] = idle_value;
	energy->max_energy[machine] = max_value;
}

float get_energy_idle_value(struct energy_matrix *energy, int machine) {
	assert(machine < energy->machines_count);
	assert(machine >= 0);

	return energy->idle_energy[machine];
}

float get_energy_max_value(struct energy_matrix *energy, int machine) {
	assert(machine < energy->machines_count);
	assert(machine >= 0);

	return energy->max_energy[machine];
}

void show_energy_matrix(struct energy_matrix *energy) {
	fprintf(stdout, "[INFO] Energy Matrix =========================== \n");
	for (int machine = 0; machine < energy->machines_count; machine++) {
		fprintf(stdout, "    Machine %d > IDLE=%f MAX=%f\n", machine, energy->idle_energy[machine], energy->max_energy[machine]);
	}
	fprintf(stdout, "[INFO] ====================================== \n");
}
