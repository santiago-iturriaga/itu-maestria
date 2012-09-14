#include <stdio.h>
#include <stdlib.h>

#include "etc_matrix.h"

#include "load_params.h"
#include "config.h"

void init_etc_matrix() {
    #if defined(DEBUG_3)
        fprintf(stderr, "[DEBUG] init_etc_matrix\n");
    #endif
    
    ETC.data = (FLOAT*)malloc(sizeof(FLOAT) * INPUT.machines_count * INPUT.tasks_count);

    if (ETC.data == NULL) {
        fprintf(stderr, "[ERROR] solicitando memoria para el etc_matrix->data.\n");
        exit(EXIT_FAILURE);
    }

    ETC.data_machine_index = (int*)malloc(sizeof(int) * INPUT.machines_count);
    ETC.data_machine_index[0] = 0;
    ETC.data_machine_index[1] = INPUT.tasks_count;
    for (int i = 2; i < INPUT.machines_count; i++) {
        ETC.data_machine_index[i] = i * INPUT.tasks_count;
    }
}

void free_etc_matrix() {
    free(ETC.data_machine_index);
    free(ETC.data);
}

void set_etc_value(int machine, int task, FLOAT value) {
    assert(value > 0.0);
    ETC.data[get_etc_coord(machine, task)] = value;
}

void show_etc_matrix() {
    fprintf(stderr, "[INFO] ETC Matrix =========================== \n");
    for (int task = 0; task < INPUT.tasks_count; task++) {
        for (int machine = 0; machine < INPUT.machines_count; machine++) {
            fprintf(stderr, "%f\n", ETC.data[get_etc_coord(machine, task)]);
        }
    }
    fprintf(stderr, "[INFO] ====================================== \n");
}
