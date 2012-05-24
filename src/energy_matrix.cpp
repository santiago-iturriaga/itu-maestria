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

    energy->ssj = (float*)malloc(sizeof(float) * input->machines_count);

    if (energy->ssj == NULL) {
        fprintf(stderr, "[ERROR] Solicitando memoria para el energy_matrix->ssj.\n");
        exit(EXIT_FAILURE);
    }

    energy->cores = (int*)malloc(sizeof(int) * input->machines_count);

    if (energy->cores == NULL) {
        fprintf(stderr, "[ERROR] Solicitando memoria para el energy_matrix->cores.\n");
        exit(EXIT_FAILURE);
    }
}

void free_energy_matrix(struct energy_matrix *energy) {
    free(energy->idle_energy);
    free(energy->max_energy);
    free(energy->ssj);
}

void set_energy_value(struct energy_matrix *energy, int machine, int cores, float ssj, float idle_value, float max_value) {
    energy->cores[machine] = cores;
    energy->ssj[machine] = ssj;
    energy->idle_energy[machine] = idle_value;
    energy->max_energy[machine] = max_value;
}

int get_cores_value(struct energy_matrix *energy, int machine) {
    return energy->cores[machine];
}

float get_energy_idle_value(struct energy_matrix *energy, int machine) {
    return energy->idle_energy[machine];
}

float get_energy_max_value(struct energy_matrix *energy, int machine) {
    return energy->max_energy[machine];
}

float get_ssj_value(struct energy_matrix *energy, int machine) {
    return energy->ssj[machine];
}

void show_energy_matrix(struct energy_matrix *energy) {
    fprintf(stdout, "[INFO] Energy Matrix =========================== \n");
    for (int machine = 0; machine < energy->machines_count; machine++) {
        fprintf(stdout, "    Machine %d > IDLE=%f MAX=%f\n", machine, energy->idle_energy[machine], energy->max_energy[machine]);
    }
    fprintf(stdout, "[INFO] ====================================== \n");
}
