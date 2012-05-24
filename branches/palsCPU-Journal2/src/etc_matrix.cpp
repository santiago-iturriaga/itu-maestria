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

void init_etc_matrix(struct params *input, struct etc_matrix *etc) {
    etc->tasks_count = input->tasks_count;
    etc->machines_count = input->machines_count;
    etc->data = (float*)malloc(sizeof(float) * input->machines_count * input->tasks_count);

    if (etc->data == NULL) {
        fprintf(stderr, "[ERROR] Solicitando memoria para el etc_matrix->data.\n");
        exit(EXIT_FAILURE);
    }
    etc->shifts = -1;
    int aux_tasks = etc->tasks_count;
    while(aux_tasks > 0) {
        aux_tasks = aux_tasks >> 1;
        etc->shifts++;
    }
}

void free_etc_matrix(struct etc_matrix *etc) {
    free(etc->data);
}

int get_etc_coord(struct etc_matrix *etc, int machine, int task) {
    /*assert(machine < etc->machines_count);
    assert(machine >= 0);
    assert(task < etc->tasks_count);
    assert(task >= 0);*/

    return (machine << etc->shifts) + task;
    //return (machine * etc->tasks_count) + task;
}

void set_etc_value(struct etc_matrix *etc, int machine, int task, float value) {
    //Soporte para instancias del ruso.
    //Assert(value > 0.0);

    etc->data[get_etc_coord(etc, machine, task)] = value;
}

float get_etc_value(struct etc_matrix *etc, int machine, int task) {
    return etc->data[get_etc_coord(etc, machine, task)];
}

void show_etc_matrix(struct etc_matrix *etc) {
    fprintf(stdout, "[INFO] ETC Matrix =========================== \n");
    for (int task = 0; task < etc->tasks_count; task++) {
        for (int machine = 0; machine < etc->machines_count; machine++) {
            fprintf(stdout, "    Task %i on machine %i -> %f\n", task, machine, etc->data[get_etc_coord(etc, machine, task)]);
        }
    }
    fprintf(stdout, "[INFO] ====================================== \n");
}
