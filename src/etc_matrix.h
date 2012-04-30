/*
 * etc_matrix.h
 *
 *  Created on: Jul 28, 2011
 *      Author: santiago
 */

#include "load_params.h"

#ifndef ETC_MATRIX_H_
#define ETC_MATRIX_H_

struct matrix {
	ushort tasks_count;
	ushort machines_count;
	float* data;
};

struct matrix* create_etc_matrix(struct params *input);
void free_etc_matrix(struct matrix *etc_matrix);
void show_etc_matrix(struct matrix *etc_matrix);

void set_etc_value(struct matrix *etc_matrix, ushort machine, ushort task, float value);
float get_etc_value(struct matrix *etc_matrix, ushort machine, ushort task);

#endif /* ETC_MATRIX_H_ */
