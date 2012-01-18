/*
 * etc_matrix.h
 *
 *  Created on: Jul 28, 2011
 *      Author: santiago
 */

#include "load_params.h"

#ifndef ETC_MATRIX_H_
#define ETC_MATRIX_H_

struct etc_matrix {
	int tasks_count;
	int machines_count;
	float* data;
};

struct etc_matrix* create_etc_matrix(struct params *input);
void free_etc_matrix(struct etc_matrix *etc);
void show_etc_matrix(struct etc_matrix *etc);

void set_etc_value(struct etc_matrix *etc, int machine, int task, float value);
float get_etc_value(struct etc_matrix *etc, int machine, int task);

#endif /* ETC_MATRIX_H_ */
