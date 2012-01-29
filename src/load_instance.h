/*
 * load_instance.h
 *
 *  Created on: Jul 28, 2011
 *      Author: santiago
 */

#include "load_params.h"
#include "etc_matrix.h"

#define LOAD_INSTANCE__FORMAT_DIMENSION_FIRST   0

#ifndef LOAD_INSTANCE_H_
#define LOAD_INSTANCE_H_

int load_instance(struct params *input, struct etc_matrix *etc, struct energy_matrix *energy);

#endif /* LOAD_INSTANCE_H_ */
