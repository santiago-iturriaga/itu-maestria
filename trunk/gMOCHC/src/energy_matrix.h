#include "config.h"
#include "load_params.h"

#ifndef ENERGY_MATRIX_H_
#define ENERGY_MATRIX_H_

struct energy_matrix {
    int tasks_count;
    int machines_count;
    
    FLOAT* data;
    int* data_machine_index;
};

void init_energy_matrix(struct params *input, struct energy_matrix *energy);
void free_energy_matrix(struct energy_matrix *energy);

void set_energy_value(struct energy_matrix *energy, int machine, int task, FLOAT value);
FLOAT get_energy_value(struct energy_matrix *energy, int machine, int task);

#endif /* ENERGY_MATRIX_H_ */
