#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "load_params.h"
#include "scenario.h"

void init_scenario(struct params *input, struct scenario *s) {
    s->machines_count = input->machines_count;
    s->idle_energy = (float*)malloc(sizeof(float) * input->machines_count);

    if (s->idle_energy == NULL) {
        fprintf(stderr, "[ERROR] solicitando memoria para el scenario->idle_energy.\n");
        exit(EXIT_FAILURE);
    }

    s->max_energy = (float*)malloc(sizeof(float) * input->machines_count);

    if (s->max_energy == NULL) {
        fprintf(stderr, "[ERROR] Solicitando memoria para el scenario->max_energy.\n");
        exit(EXIT_FAILURE);
    }

    s->ssj = (float*)malloc(sizeof(float) * input->machines_count);

    if (s->ssj == NULL) {
        fprintf(stderr, "[ERROR] Solicitando memoria para el scenario->ssj.\n");
        exit(EXIT_FAILURE);
    }

    s->cores = (int*)malloc(sizeof(int) * input->machines_count);

    if (s->cores == NULL) {
        fprintf(stderr, "[ERROR] Solicitando memoria para el scenario->cores.\n");
        exit(EXIT_FAILURE);
    }
}

void free_scenario(struct scenario *s) {
    free(s->cores);
    free(s->idle_energy);
    free(s->max_energy);
    free(s->ssj);
}

void set_scenario_machine(struct scenario *s, int machine, int cores, float ssj, float idle_value, float max_value) {
    s->cores[machine] = cores;
    s->ssj[machine] = ssj;
    s->idle_energy[machine] = idle_value;
    s->max_energy[machine] = max_value;
}

int get_scenario_cores(struct scenario *s, int machine) {
    return s->cores[machine];
}

float get_scenario_energy_idle(struct scenario *s, int machine) {
    return s->idle_energy[machine];
}

float get_scenario_energy_max(struct scenario *s, int machine) {
    return s->max_energy[machine];
}

float get_scenario_ssj(struct scenario *s, int machine) {
    return s->ssj[machine];
}

void show_scenario(struct scenario *s) {
    fprintf(stderr, "[INFO] Scenario =========================== \n");
    for (int machine = 0; machine < s->machines_count; machine++) {
        fprintf(stderr, "    Machine %d > CORES=%d SSJ=%f IDLE=%f MAX=%f\n", machine, 
            s->cores[machine], s->ssj[machine], s->idle_energy[machine], s->max_energy[machine]);
    }
    fprintf(stderr, "[INFO] ==================================== \n");
}
