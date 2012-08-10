#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "etc_matrix.h"

#include "load_params.h"
#include "config.h"

void init_etc_matrix(struct params *input, struct etc_matrix *etc) {
    #if defined(DEBUG_3)
        fprintf(stderr, "[DEBUG] init_etc_matrix\n");
    #endif
    
    etc->tasks_count = input->tasks_count;
    etc->machines_count = input->machines_count;
    etc->data = (float*)malloc(sizeof(float) * input->machines_count * input->tasks_count);

    if (etc->data == NULL) {
        fprintf(stderr, "[ERROR] solicitando memoria para el etc_matrix->data.\n");
        exit(EXIT_FAILURE);
    }

    etc->data_machine_index = (int*)malloc(sizeof(int) * input->machines_count);
    etc->data_machine_index[0] = 0;
    etc->data_machine_index[1] = input->tasks_count;
    for (int i = 2; i < input->machines_count; i++) {
        etc->data_machine_index[i] = input->machines_count * input->tasks_count;
    }
}

void free_etc_matrix(struct etc_matrix *etc) {
    free(etc->data_machine_index);
    free(etc->data);
}

int get_matrix_coord(struct etc_matrix *etc, int machine, int task) {
    #if defined(DEBUG_1)
        assert(machine < etc->machines_count);
        assert(machine >= 0);
        assert(task < etc->tasks_count);
        assert(task >= 0);
    #endif

    return etc->data_machine_index[machine] + task;
}

void set_etc_value(struct etc_matrix *etc, int machine, int task, float value) {
    // TODO: Ojo! esto no debería ser válido. Pero queda comentado porque la instancia de 65536x2048 tiene 0.0.
    #if defined(DEBUG_1)
        assert(value > 0.0);
    #endif

    etc->data[get_matrix_coord(etc, machine, task)] = value;
}

float get_etc_value(struct etc_matrix *etc, int machine, int task) {
    return etc->data[get_matrix_coord(etc, machine, task)];
}

void show_etc_matrix(struct etc_matrix *etc) {
    fprintf(stderr, "[INFO] ETC Matrix =========================== \n");
    for (int task = 0; task < etc->tasks_count; task++) {
        for (int machine = 0; machine < etc->machines_count; machine++) {
            fprintf(stderr, "%f\n", etc->data[get_matrix_coord(etc, machine, task)]);
        }
    }
    fprintf(stderr, "[INFO] ====================================== \n");
}
