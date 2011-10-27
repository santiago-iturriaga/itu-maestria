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
		
	etc_matrix->tasks_count = input->tasks_count;
	etc_matrix->machines_count = input->machines_count;
	etc_matrix->data = (float*)malloc(sizeof(float) * input->machines_count * input->tasks_count);
	
	return etc_matrix;
}

void free_etc_matrix(struct matrix *etc_matrix) {
	free(etc_matrix->data);
	free(etc_matrix);
}

int get_matrix_coord(struct matrix *etc_matrix, int machine, int task) {
	assert(machine < etc_matrix->machines_count);
	assert(machine >= 0);
	assert(task < etc_matrix->tasks_count);
	assert(task >= 0);
	
	return (machine * etc_matrix->tasks_count) + task;
}

void set_etc_value(struct matrix *etc_matrix, int machine, int task, float value) {
	assert(value > 0.0);
	
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
