#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "energy_matrix.h"

#include "load_params.h"
#include "config.h"

void init_energy_matrix(struct params *input, struct energy_matrix *energy) {
    #if defined(DEBUG_3)
        fprintf(stderr, "[DEBUG] init_energy_matrix\n");
    #endif
    
    energy->tasks_count = input->tasks_count;
    energy->machines_count = input->machines_count;
    energy->data = (FLOAT*)malloc(sizeof(FLOAT) * input->machines_count * input->tasks_count);

    if (energy->data == NULL) {
        fprintf(stderr, "[ERROR] solicitando memoria para el energy_matrix->data.\n");
        exit(EXIT_FAILURE);
    }

    energy->data_machine_index = (int*)malloc(sizeof(int) * input->machines_count);
    energy->data_machine_index[0] = 0;
    energy->data_machine_index[1] = input->tasks_count;
    for (int i = 2; i < input->machines_count; i++) {
        energy->data_machine_index[i] = i * input->tasks_count;
    }
    
    #if defined(DEBUG_3)
        for (int i = 0; i < input->machines_count; i++) {
            fprintf(stderr, "[DEBUG] data_machine_index[%d]=%d\n", i, energy->data_machine_index[i]);
        }
    #endif
}

void free_energy_matrix(struct energy_matrix *energy) {
    free(energy->data_machine_index);
    free(energy->data);
}

int get_matrix_coord(struct energy_matrix *energy, int machine, int task) {
    #if defined(DEBUG_1)
        assert(machine < energy->machines_count);
        assert(machine >= 0);
        assert(task < energy->tasks_count);
        assert(task >= 0);
    #endif

    return energy->data_machine_index[machine] + task;
}

void set_energy_value(struct energy_matrix *energy, int machine, int task, FLOAT value) {
    // TODO: Ojo! esto no debería ser válido. Pero queda comentado porque la instancia de 65536x2048 tiene 0.0.
    #if defined(DEBUG_1)
        assert(value > 0.0);
    #endif

    energy->data[get_matrix_coord(energy, machine, task)] = value;
}

FLOAT get_energy_value(struct energy_matrix *energy, int machine, int task) {
    return energy->data[get_matrix_coord(energy, machine, task)];
}
