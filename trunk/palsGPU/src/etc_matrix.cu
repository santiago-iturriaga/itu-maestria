/*
 * etc_matrix.c
 *
 *  Created on: Jul 28, 2011
 *      Author: santiago
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "load_params.h"
#include "etc_matrix.h"

struct matrix* create_etc_matrix(struct params *input) {
    struct matrix *etc_matrix;
    etc_matrix = (struct matrix*)malloc(sizeof(struct matrix));

    if (etc_matrix == NULL) {
        fprintf(stderr, "[ERROR] Solicitando memoria para el struct etc_matrix.\n");
        exit(EXIT_FAILURE);
    }

    etc_matrix->tasks_count = input->tasks_count;
    etc_matrix->machines_count = input->machines_count;
    etc_matrix->data = (float*)malloc(sizeof(float) * input->machines_count * input->tasks_count);

    if (etc_matrix->data == NULL) {
        fprintf(stderr, "[ERROR] Solicitando memoria para el etc_matrix->data.\n");
        exit(EXIT_FAILURE);
    }

    etc_matrix->shifts = -1;
    int aux_tasks = etc_matrix->tasks_count;
    while(aux_tasks > 0) {
        aux_tasks = aux_tasks >> 1;
        etc_matrix->shifts++;
    }

    //fprintf(stdout, "SHIFTS: %d\n", etc_matrix->shifts);

    return etc_matrix;
}

void free_etc_matrix(struct matrix *etc_matrix) {
    free(etc_matrix->data);
    free(etc_matrix);
}

int get_matrix_coord(struct matrix *etc_matrix, int machine, int task) {
    //assert(machine < etc_matrix->machines_count);
    //assert(machine >= 0);
    //assert(task < etc_matrix->tasks_count);
    //assert(task >= 0);

    /*fprintf(stdout, "(machine * etc_matrix->tasks_count) = %d\n", (machine * etc_matrix->tasks_count));
    fprintf(stdout, "(machine << etc_matrix->shifts) = %d\n", (machine << etc_matrix->shifts));
    assert((machine * etc_matrix->tasks_count) == (machine << etc_matrix->shifts));*/

    return (machine << etc_matrix->shifts) + task;
    //return (machine * etc_matrix->tasks_count) + task;
}

void set_etc_value(struct matrix *etc_matrix, int machine, int task, float value) {
    // TODO: Ojo! esto no debería ser válido. Pero queda comentado porque la instancia de 65536x2048 tiene 0.0.
    // assert(value > 0.0);

    etc_matrix->data[get_matrix_coord(etc_matrix, machine, task)] = value;
}

float get_etc_value(struct matrix *etc_matrix, int machine, int task) {
    return etc_matrix->data[get_matrix_coord(etc_matrix, machine, task)];
}

void show_etc_matrix(struct matrix *etc_matrix) {
    fprintf(stdout, "[INFO] ETC Matrix =========================== \n");
    for (int task = 0; task < etc_matrix->tasks_count; task++) {
        for (int machine = 0; machine < etc_matrix->machines_count; machine++) {
            fprintf(stdout, "    Task %i on machine %i -> %f\n", task, machine, etc_matrix->data[get_matrix_coord(etc_matrix, machine, task)]);
        }
    }
    fprintf(stdout, "[INFO] ====================================== \n");
}
