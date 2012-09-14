#include <stdio.h>
#include <stdlib.h>

#include "energy_matrix.h"

#include "load_params.h"
#include "config.h"

void init_energy_matrix() {
    #if defined(DEBUG_3)
        fprintf(stderr, "[DEBUG] init_energy_matrix\n");
    #endif
    
    ENERGY.data = (FLOAT*)malloc(sizeof(FLOAT) * INPUT.machines_count * INPUT.tasks_count);

    if (ENERGY.data == NULL) {
        fprintf(stderr, "[ERROR] solicitando memoria para el energy_matrix->data.\n");
        exit(EXIT_FAILURE);
    }

    ENERGY.data_machine_index = (int*)malloc(sizeof(int) * INPUT.machines_count);
    ENERGY.data_machine_index[0] = 0;
    ENERGY.data_machine_index[1] = INPUT.tasks_count;
    for (int i = 2; i < INPUT.machines_count; i++) {
        ENERGY.data_machine_index[i] = i * INPUT.tasks_count;
    }
}

void free_energy_matrix() {
    free(ENERGY.data_machine_index);
    free(ENERGY.data);
}

void set_energy_value(int machine, int task, FLOAT value) {
    assert(value > 0.0);
    ENERGY.data[get_energy_coord(machine, task)] = value;
}
