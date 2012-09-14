#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "scenario.h"

#include "load_params.h"
#include "config.h"

void init_scenario() {
    #if defined(DEBUG_3)
        fprintf(stderr, "[DEBUG] init_scenario\n");
    #endif
    
    SCENARIO.idle_energy = (FLOAT*)malloc(sizeof(FLOAT) * INPUT.machines_count);

    if (SCENARIO.idle_energy == NULL) {
        fprintf(stderr, "[ERROR] solicitando memoria para el scenario->idle_energy.\n");
        exit(EXIT_FAILURE);
    }

    SCENARIO.max_energy = (FLOAT*)malloc(sizeof(FLOAT) * INPUT.machines_count);

    if (SCENARIO.max_energy == NULL) {
        fprintf(stderr, "[ERROR] Solicitando memoria para el scenario->max_energy.\n");
        exit(EXIT_FAILURE);
    }

    SCENARIO.ssj = (FLOAT*)malloc(sizeof(FLOAT) * INPUT.machines_count);

    if (SCENARIO.ssj == NULL) {
        fprintf(stderr, "[ERROR] Solicitando memoria para el scenario->ssj.\n");
        exit(EXIT_FAILURE);
    }

    SCENARIO.cores = (int*)malloc(sizeof(int) * INPUT.machines_count);

    if (SCENARIO.cores == NULL) {
        fprintf(stderr, "[ERROR] Solicitando memoria para el scenario->cores.\n");
        exit(EXIT_FAILURE);
    }
}

void free_scenario() {
    free(SCENARIO.cores);
    free(SCENARIO.idle_energy);
    free(SCENARIO.max_energy);
    free(SCENARIO.ssj);
}

void set_scenario_machine(int machine, int cores, FLOAT ssj, FLOAT idle_value, FLOAT max_value) {
    SCENARIO.cores[machine] = cores;
    SCENARIO.ssj[machine] = ssj;
    SCENARIO.idle_energy[machine] = idle_value;
    SCENARIO.max_energy[machine] = max_value;
}

void show_scenario() {
    fprintf(stderr, "[INFO] Scenario =========================== \n");
    for (int machine = 0; machine < INPUT.machines_count; machine++) {
        fprintf(stderr, "    Machine %d > CORES=%d SSJ=%f IDLE=%f MAX=%f\n", machine, 
            SCENARIO.cores[machine], SCENARIO.ssj[machine], SCENARIO.idle_energy[machine], 
            SCENARIO.max_energy[machine]);
    }
    fprintf(stderr, "[INFO] ==================================== \n");
}
