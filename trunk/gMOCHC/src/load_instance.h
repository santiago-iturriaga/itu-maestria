#include "load_params.h"
#include "scenario.h"
#include "etc_matrix.h"
#include "energy_matrix.h"

#ifndef LOAD_INSTANCE_H_
#define LOAD_INSTANCE_H_

/* Reads the input files and stores de data in memory */
int load_instance(struct params *input, struct scenario *s, struct etc_matrix *etc, struct energy_matrix *energy);

#endif /* LOAD_INSTANCE_H_ */
