/*
 * energy_matrix.h
 *
 *  Created on: Jul 28, 2011
 *      Author: santiago
 */

#include "load_params.h"

#ifndef ENERGY_MATRIX_H_
#define ENERGY_MATRIX_H_

struct energy_matrix {
	int machines_count;
	
    int* cores;
	float* ssj;
	float* idle_energy;
	float* max_energy;
};

void init_energy_matrix(struct params *input, struct energy_matrix *energy);
void free_energy_matrix(struct energy_matrix *energy);
void show_energy_matrix(struct energy_matrix *energy);

void set_energy_value(struct energy_matrix *energy, int machine, int cores, float ssj, float idle_value, float max_value);

float get_ssj_value(struct energy_matrix *energy, int machine);
float get_energy_idle_value(struct energy_matrix *energy, int machine);
float get_energy_max_value(struct energy_matrix *energy, int machine);
int get_cores_value(struct energy_matrix *energy, int machine);

#endif /* ENERGY_MATRIX_H_ */
